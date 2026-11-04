"""Milestone 18: Predict-then-retrieve using pre-cached mean deltas.

v10 OOM'd during the per-cell response_role phase (h5py page cache fills 7+ GB
after the first data pass). The mean_delta_cache.npz was saved successfully.

v11 avoids h5ad entirely: loads the cached mean deltas from v10's output,
then runs the full prediction suite including predict-then-retrieve.

This decouples the expensive data-loading from the fast model experiments.

Key new experiments:
  A) Predict-then-retrieve: ridge → cosine NN in delta space → weighted mean
  B) Alpha sweep (0..1) for ensemble: stage1 vs PTR
  C) Stacked ridge: concatenate gene features + stage-1 predicted delta as input
     to a second-pass ridge (uses only training data for calibration)
  D) Direct NN retrieval in delta space (oracle ceiling test with training genes)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from virtualcell.data.target_descriptors import (
    fit_response_role_descriptors,
    load_gene2vec_table,
)

import gc
from collections import defaultdict
import anndata as ad

CACHE_DIR = Path("outputs/replogle_genelevel_v10")
OUT_DIR = Path("outputs/replogle_genelevel_v11")
TARGET_SUM = 10_000.0
CHUNK_SIZE = 6000


def pearson_per_gene(pred_dict, true_dict):
    genes = sorted(set(pred_dict) & set(true_dict))
    vals = []
    for g in genes:
        p, t = pred_dict[g], true_dict[g]
        if np.std(t) < 1e-6 or np.std(p) < 1e-6:
            vals.append(0.0)
        else:
            vals.append(float(np.clip(pearsonr(p, t)[0], -1, 1)))
    return float(np.mean(vals)), float(np.median(vals))


def predict_then_retrieve(stage1_preds, train_gene_list, train_md, K=10):
    D_tr = np.stack([train_md[g] for g in train_gene_list])
    D_tr_norm = D_tr / (np.linalg.norm(D_tr, axis=1, keepdims=True) + 1e-9)
    genes = list(stage1_preds.keys())
    preds_mat = np.stack([stage1_preds[g] for g in genes])
    preds_norm = preds_mat / (np.linalg.norm(preds_mat, axis=1, keepdims=True) + 1e-9)
    sim = preds_norm @ D_tr_norm.T  # (n_test, n_train)
    result = {}
    for i, g in enumerate(genes):
        top_k = np.argsort(sim[i])[::-1][:K]
        weights = np.maximum(sim[i][top_k], 0)
        if weights.sum() < 1e-9:
            weights = np.ones(K, dtype=np.float32)
        weights /= weights.sum()
        result[g] = sum(w * train_md[train_gene_list[j]] for j, w in zip(top_k, weights)).astype(np.float32)
    return result


def lognorm_chunk(X_sparse, top_gene_idx):
    lib = np.asarray(X_sparse.sum(axis=1)).ravel().astype(np.float64)
    lib = np.where(lib == 0, 1.0, lib)
    X_top = X_sparse[:, top_gene_idx]
    X_dense = np.asarray(
        X_top.todense() if hasattr(X_top, "todense") else X_top,
        dtype=np.float32,
    )
    X_dense /= lib[:, None].astype(np.float32)
    X_dense *= TARGET_SUM
    return np.log1p(X_dense)


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load cached mean deltas (raw sums + counts from v10) ---
    print("Loading mean delta cache from v10 ...")
    cache = np.load(str(CACHE_DIR / "mean_delta_cache.npz"))
    valid_sums = cache["valid_sums"]    # (n_genes, N_HVG) - raw log1p expression sums
    valid_counts = cache["valid_counts"]  # (n_genes,)
    N_HVG = valid_sums.shape[1]
    with open(CACHE_DIR / "mean_delta_cache_genes.json") as f:
        raw = json.load(f)
        valid_genes = raw["genes"] if isinstance(raw, dict) else raw
    print(f"  Loaded {len(valid_genes)} genes x {N_HVG} HVGs")

    # --- Load adata briefly just to get control_mean and HVG gene names ---
    # Only reads control cells (~10k), which is fast and memory-safe.
    print("Opening adata for control_mean and HVG names (control cells only) ...")
    adata = ad.read_h5ad("data/replogle/K562_essential_raw_singlecell_01.h5ad", backed="r")
    gene_col = adata.obs["gene"].astype(str).tolist()
    gene_names = (
        adata.var["gene_name"].tolist() if "gene_name" in adata.var.columns
        else adata.var_names.tolist()
    )
    ctrl_positions = sorted([i for i, g in enumerate(gene_col) if g == "non-targeting"])

    # Compute same top_gene_idx as v10 (variance of control cells, same seed)
    rng = np.random.default_rng(0)
    ctrl_sample = sorted(
        rng.choice(ctrl_positions, size=min(8000, len(ctrl_positions)), replace=False).tolist()
    )
    X_cs = adata.X[ctrl_sample]
    X_cs = np.asarray(X_cs.todense() if hasattr(X_cs, "todense") else X_cs, dtype=np.float32)
    lib_cs = X_cs.sum(axis=1, keepdims=True)
    lib_cs = np.where(lib_cs == 0, 1, lib_cs)
    X_cs_norm = np.log1p(X_cs / lib_cs * TARGET_SUM)
    del X_cs
    ctrl_var = X_cs_norm.var(axis=0)
    top_gene_idx = np.sort(np.argsort(ctrl_var)[::-1][:N_HVG])
    top_gene_names = [gene_names[i] for i in top_gene_idx]
    del X_cs_norm, ctrl_var

    # Control mean over ALL control cells (chunked, same as v10)
    ctrl_sum = np.zeros(N_HVG, dtype=np.float64)
    n_ctrl = 0
    for start in range(0, len(ctrl_positions), CHUNK_SIZE):
        chunk = ctrl_positions[start: start + CHUNK_SIZE]
        X_c = lognorm_chunk(adata.X[chunk], top_gene_idx).astype(np.float64)
        ctrl_sum += X_c.sum(axis=0)
        n_ctrl += len(chunk)
    control_mean = (ctrl_sum / n_ctrl).astype(np.float32)
    del adata, ctrl_sum
    gc.collect()
    print(f"  control_mean computed over {n_ctrl} cells. Time: {time.time()-t0:.1f}s")

    # Build deltas from cached sums
    all_mean_deltas = {}
    for i, g in enumerate(valid_genes):
        all_mean_deltas[g] = (valid_sums[i] / valid_counts[i]).astype(np.float32) - control_mean
    del valid_sums, valid_counts
    gc.collect()

    print(f"  {len(all_mean_deltas)} gene mean deltas ready. Time: {time.time()-t0:.1f}s")

    # --- Split ---
    train_t, test_t_full = train_test_split(list(all_mean_deltas.keys()), test_size=0.3, random_state=0)
    val_t, test_t = train_test_split(test_t_full, test_size=0.5, random_state=0)
    print(f"Split: {len(train_t)} train / {len(val_t)} val / {len(test_t)} test")

    train_md = {g: all_mean_deltas[g] for g in train_t}
    val_md   = {g: all_mean_deltas[g] for g in val_t}
    test_md  = {g: all_mean_deltas[g] for g in test_t}

    train_gene_list = sorted(train_md)
    D_train = np.stack([train_md[g] for g in train_gene_list])
    n_out = D_train.shape[1]
    print(f"D_train: {D_train.shape}")

    # --- Descriptors ---
    gene2vec_table = load_gene2vec_table(Path("data/gene2vec/gene2vec_dim_200_iter_9.txt"))
    gene2vec_lookup = {g: gene2vec_table[g] for g in top_gene_names if g in gene2vec_table}
    # Also try using target gene names directly for Gene2Vec lookup
    for g in all_mean_deltas:
        if g not in gene2vec_lookup and g in gene2vec_table:
            gene2vec_lookup[g] = gene2vec_table[g]

    # Response role from mean deltas (1 obs per gene, same as v7)
    rr_dim = min(50, len(train_gene_list))
    train_targets_approx = list(set(train_t))
    train_cell_delta_approx = np.stack([train_md[g] for g in train_targets_approx])
    response_role_table = fit_response_role_descriptors(
        train_targets_approx, train_cell_delta_approx, top_gene_names,
        n_components=rr_dim, seed=0,
    )
    # Also fit response_role using target gene names as gene_names
    rr_table_tgn = fit_response_role_descriptors(
        train_targets_approx, train_cell_delta_approx, list(all_mean_deltas.keys()),
        n_components=rr_dim, seed=0,
    )

    # Concat: Gene2Vec + response_role
    concat_dim = 200 + rr_dim
    concat_table = {}
    for g in all_mean_deltas:
        g2v_vec = gene2vec_lookup.get(g, np.zeros(200, np.float32))
        rr_vec  = rr_table_tgn.get(g, np.zeros(rr_dim, np.float32))
        concat_table[g] = np.concatenate([g2v_vec, rr_vec])

    # --- Baselines ---
    gm = D_train.mean(axis=0)
    no_m, no_med = pearson_per_gene({g: np.zeros(n_out, np.float32) for g in test_t}, test_md)
    gm_m, gm_med = pearson_per_gene({g: gm for g in test_t}, test_md)
    print(f"\nNo-effect: {no_m:.4f};  Global mean: {gm_m:.4f} (median {gm_med:.4f})")

    # --- Stage-1 model builder ---
    def build_stage1_model(name, table, dim):
        """Return (best_val_score, test_preds_dict, model_name) for best alpha/k."""
        X_tr = np.array([table.get(g, np.zeros(dim, np.float32)) for g in train_gene_list])
        X_va = np.array([table.get(g, np.zeros(dim, np.float32)) for g in val_t])
        X_te = np.array([table.get(g, np.zeros(dim, np.float32)) for g in test_t])
        X_all = np.array([table.get(g, np.zeros(dim, np.float32)) for g in train_gene_list + list(val_t) + list(test_t)])
        out = {}

        # Ridge direct
        best_dr = None
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
            m = Ridge(alpha=alpha)
            m.fit(X_tr, D_train)
            va_preds = {g: m.predict(X_va[[i]])[0] for i, g in enumerate(val_t)}
            vr, _ = pearson_per_gene(va_preds, val_md)
            if best_dr is None or vr > best_dr[0]:
                best_dr = (vr, alpha, m)
        vr_dr, alpha_dr, m_dr = best_dr
        te_preds = {g: m_dr.predict(X_te[[i]])[0] for i, g in enumerate(test_t)}
        tr_m, tr_med = pearson_per_gene(te_preds, test_md)
        out["direct_ridge"] = {"alpha": alpha_dr, "test_pearson_mean": tr_m, "test_pearson_median": tr_med}
        print(f"  {name}+ridge:    {tr_m:.4f} (median {tr_med:.4f})")
        # Also get train+val preds for predict-then-retrieve calibration
        all_preds_dr = {g: m_dr.predict(X_all[[i]])[0] for i, g in enumerate(train_gene_list + list(val_t) + list(test_t))}

        # SVD variants
        best_svd_overall = (0.0, None, None, None)  # (val_score, k, alpha, preds_dict)
        for k in (20, 50, 100, 200):
            if k >= min(D_train.shape):
                continue
            svd = TruncatedSVD(n_components=k, random_state=0)
            codes_tr = svd.fit_transform(D_train)
            best_svd = None
            for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
                m = Ridge(alpha=alpha)
                m.fit(X_tr, codes_tr)
                va_codes = m.predict(X_va)
                va_preds = {g: svd.inverse_transform(va_codes[[i]])[0] for i, g in enumerate(val_t)}
                vr, _ = pearson_per_gene(va_preds, val_md)
                if best_svd is None or vr > best_svd[0]:
                    best_svd = (vr, alpha, m, svd)
            vr_svd, alpha_svd, m_svd, svd_best = best_svd
            te_codes = m_svd.predict(X_te)
            te_preds = {g: svd_best.inverse_transform(te_codes[[i]])[0] for i, g in enumerate(test_t)}
            tr_m, tr_med = pearson_per_gene(te_preds, test_md)
            out[f"svd_k{k}"] = {"k": k, "alpha": alpha_svd, "test_pearson_mean": tr_m, "test_pearson_median": tr_med}
            print(f"  {name}+SVD_k={k}: {tr_m:.4f} (median {tr_med:.4f})")
            if vr_svd > best_svd_overall[0]:
                all_codes = m_svd.predict(np.array([table.get(g, np.zeros(dim, np.float32)) for g in train_gene_list + list(val_t) + list(test_t)]))
                all_svd_preds = {g: svd_best.inverse_transform(all_codes[[i]])[0] for i, g in enumerate(train_gene_list + list(val_t) + list(test_t))}
                best_svd_overall = (vr_svd, k, alpha_svd, all_svd_preds)

        return out, all_preds_dr, best_svd_overall[3]  # results, ridge_preds_all, svd_preds_all

    print("\nEvaluating Gene2Vec ...")
    g2v_results, g2v_ridge_all, g2v_svd_all = build_stage1_model("gene2vec", gene2vec_lookup, 200)
    print("\nEvaluating response_role ...")
    rr_results, rr_ridge_all, rr_svd_all = build_stage1_model("response_role", rr_table_tgn, rr_dim)
    print("\nEvaluating Gene2Vec+response_role (concat) ...")
    cc_results, cc_ridge_all, cc_svd_all = build_stage1_model("concat", concat_table, concat_dim)

    # --- Predict-then-retrieve ---
    print("\nPredict-then-retrieve experiments ...")
    ptr_results = {}

    all_gene_keys = train_gene_list + list(val_t) + list(test_t)

    for s1_name, s1_preds_all in [("g2v_ridge", g2v_ridge_all), ("g2v_svd", g2v_svd_all),
                                   ("rr_ridge",  rr_ridge_all),  ("rr_svd",  rr_svd_all),
                                   ("cc_ridge",  cc_ridge_all),  ("cc_svd",  cc_svd_all)]:
        if s1_preds_all is None:
            continue
        for K in (5, 10, 20, 50):
            # Build stage-2 preds for val and test using TRAINING gene deltas as retrieval corpus
            val_s1 = {g: s1_preds_all[g] for g in val_t}
            test_s1 = {g: s1_preds_all[g] for g in test_t}
            val_ptr  = predict_then_retrieve(val_s1,  train_gene_list, train_md, K=K)
            test_ptr = predict_then_retrieve(test_s1, train_gene_list, train_md, K=K)

            # Alpha sweep on validation
            best_alpha = None
            for alpha in np.arange(0.0, 1.05, 0.1):
                va_blend = {g: (1-alpha)*val_s1[g] + alpha*val_ptr[g] for g in val_t}
                vr, _ = pearson_per_gene(va_blend, val_md)
                if best_alpha is None or vr > best_alpha[0]:
                    best_alpha = (vr, alpha)
            _, alpha_best = best_alpha

            te_blend = {g: (1-alpha_best)*test_s1[g] + alpha_best*test_ptr[g] for g in test_t}
            te_m, te_med = pearson_per_gene(te_blend, test_md)

            # Also pure PTR
            pure_te_m, pure_te_med = pearson_per_gene(test_ptr, test_md)

            key = f"{s1_name}_ptr_K{K}"
            ptr_results[key] = {
                "K": K, "alpha_best": float(alpha_best),
                "test_pearson_mean_blend": te_m, "test_pearson_median_blend": te_med,
                "test_pearson_mean_pure_ptr": pure_te_m,
            }
            print(f"  {key}: blend({alpha_best:.1f})={te_m:.4f}  pure_ptr={pure_te_m:.4f}")

    # --- Stacked ridge (use stage-1 predicted delta as additional features) ---
    print("\nStacked ridge experiments ...")
    stacked_results = {}
    for s1_name, s1_preds_all, feat_table, feat_dim in [
        ("g2v_ridge_stacked", g2v_ridge_all, gene2vec_lookup, 200),
        ("cc_svd_stacked",    cc_svd_all,    concat_table,    concat_dim),
    ]:
        if s1_preds_all is None:
            continue
        # Build feature = [gene_features, stage1_pred] for train, val, test
        def stacked_feat(g):
            base = feat_table.get(g, np.zeros(feat_dim, np.float32))
            pred = s1_preds_all.get(g, np.zeros(n_out, np.float32))
            return np.concatenate([base, pred])

        X_tr2 = np.array([stacked_feat(g) for g in train_gene_list])
        X_va2 = np.array([stacked_feat(g) for g in val_t])
        X_te2 = np.array([stacked_feat(g) for g in test_t])
        stacked_dim2 = X_tr2.shape[1]

        best_st = None
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
            m = Ridge(alpha=alpha)
            m.fit(X_tr2, D_train)
            va_preds = {g: m.predict(X_va2[[i]])[0] for i, g in enumerate(val_t)}
            vr, _ = pearson_per_gene(va_preds, val_md)
            if best_st is None or vr > best_st[0]:
                best_st = (vr, alpha, m)
        _, alpha_st, m_st = best_st
        te_preds = {g: m_st.predict(X_te2[[i]])[0] for i, g in enumerate(test_t)}
        st_m, st_med = pearson_per_gene(te_preds, test_md)
        stacked_results[s1_name] = {"alpha": alpha_st, "test_pearson_mean": st_m, "test_pearson_median": st_med}
        print(f"  {s1_name}: {st_m:.4f} (median {st_med:.4f})")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    summary = {
        "dataset": "Replogle et al. 2022 K562 essential RAW SINGLE-CELL",
        "note": "v11: loaded cached mean deltas from v10 (no h5ad re-read)",
        "normalization": "log1p (library-size to 10k, then log1p)",
        "n_hvg": valid_sums.shape[1],
        "n_distinct_targets": len(all_mean_deltas),
        "n_train_targets": len(train_t),
        "n_val_targets": len(val_t),
        "n_test_targets": len(test_t),
        "elapsed_seconds": elapsed,
        "literature_reference_points": {
            "GEARS_unseen_K562": 0.076,
            "mean_training_delta_baseline": 0.267,
            "oracle_nn_retrieval": 0.592,
        },
        "baselines": {
            "no_effect": {"test_pearson_mean": no_m},
            "global_mean": {"test_pearson_mean": gm_m, "test_pearson_median": gm_med},
        },
        "gene2vec": g2v_results,
        "response_role": rr_results,
        "concat_gene2vec_rr": cc_results,
        "predict_then_retrieve": ptr_results,
        "stacked_ridge": stacked_results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_DIR}/metrics.json")


if __name__ == "__main__":
    main()

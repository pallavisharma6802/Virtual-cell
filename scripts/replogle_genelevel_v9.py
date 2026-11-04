"""Milestone 17: Nearest-neighbor retrieval + ensemble with ridge regression.

The literature's oracle baseline (0.592) uses exact nearest-neighbor in
gene embedding space — it retrieves the K closest training genes and predicts
their mean delta. This should dramatically outperform global mean for genes
whose training neighbors have similar perturbation effects.

Retrieval approaches tested:
  1. Gene2Vec cosine NN (K=1,2,3,5,10,20,50): weighted mean of neighbor deltas
  2. NN + SVD denoising: reconstruct neighbor-mean through low-rank SVD
  3. Ensemble: alpha * NN_pred + (1-alpha) * ridge_pred for best alpha on val

Also: response_role with actual per-cell deltas for training genes (not mean-
delta approximation). This may fix the under-performance seen in v8.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from collections import defaultdict

import anndata as ad
import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from virtualcell.data.target_descriptors import (
    fit_response_role_descriptors,
    load_gene2vec_table,
)

MIN_CELLS = 5
TARGET_SUM = 10_000.0
N_HVG = 5000
CHUNK_SIZE = 6000


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


def cosine_sim_matrix(A, B):
    """A: (n,d), B: (m,d) → (n,m) cosine similarity matrix."""
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return A_norm @ B_norm.T


def nn_retrieve(test_genes, train_genes, embeddings, D_train_dict, K=5):
    """Retrieve K nearest training neighbors by cosine distance in embedding space."""
    dim = next(iter(embeddings.values())).shape[0]
    E_tr = np.array([embeddings.get(g, np.zeros(dim, np.float32)) for g in train_genes])
    E_te = np.array([embeddings.get(g, np.zeros(dim, np.float32)) for g in test_genes])
    sim = cosine_sim_matrix(E_te, E_tr)  # (n_test, n_train)
    n_out = next(iter(D_train_dict.values())).shape[0]
    preds = {}
    for i, g in enumerate(test_genes):
        top_k = np.argsort(sim[i])[::-1][:K]
        weights = np.maximum(sim[i][top_k], 0)
        if weights.sum() < 1e-9:
            weights = np.ones(K, dtype=np.float32)
        weights = weights / weights.sum()
        pred = sum(w * D_train_dict[train_genes[j]] for j, w in zip(top_k, weights))
        preds[g] = pred.astype(np.float32)
    return preds


def main():
    t0 = time.time()
    adata = ad.read_h5ad(
        "data/replogle/K562_essential_raw_singlecell_01.h5ad", backed="r"
    )
    gene_col = adata.obs["gene"].astype(str).tolist()
    gene_names = (
        adata.var["gene_name"].tolist()
        if "gene_name" in adata.var.columns
        else adata.var_names.tolist()
    )
    print(f"Full: {adata.shape[0]} cells x {adata.shape[1]} genes, N_HVG={N_HVG}")

    ctrl_positions = [i for i, g in enumerate(gene_col) if g == "non-targeting"]
    target_by_gene: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(gene_col):
        if g != "non-targeting":
            target_by_gene[g].append(i)

    all_genes = sorted(k for k, v in target_by_gene.items() if len(v) >= MIN_CELLS)
    all_target_pos = sorted(p for g in all_genes for p in target_by_gene[g])
    print(f"Controls: {len(ctrl_positions)}, target cells: {len(all_target_pos)} across {len(all_genes)} genes")

    # --- Top-N HVGs ---
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
    del X_cs_norm
    print(f"Top {N_HVG} HVGs identified.")

    # --- All control cells: mean + std ---
    ctrl_sum = np.zeros(N_HVG, dtype=np.float64)
    ctrl_sum2 = np.zeros(N_HVG, dtype=np.float64)
    n_ctrl = 0
    for start in range(0, len(ctrl_positions), CHUNK_SIZE):
        chunk = sorted(ctrl_positions)[start: start + CHUNK_SIZE]
        X_c = lognorm_chunk(adata.X[chunk], top_gene_idx).astype(np.float64)
        ctrl_sum += X_c.sum(axis=0)
        ctrl_sum2 += (X_c ** 2).sum(axis=0)
        n_ctrl += len(chunk)
    control_mean = (ctrl_sum / n_ctrl).astype(np.float32)
    ctrl_std = np.sqrt(np.maximum(ctrl_sum2 / n_ctrl - (ctrl_sum / n_ctrl) ** 2, 0)).astype(np.float32)
    print(f"Control mean/std over {n_ctrl} cells.")

    # --- Online per-gene mean delta ---
    print(f"Computing online mean deltas ({len(all_target_pos)} target cells)...")
    gene_sum = {g: np.zeros(N_HVG, dtype=np.float64) for g in all_genes}
    gene_count = {g: 0 for g in all_genes}
    n_all = len(all_target_pos)
    for start in range(0, n_all, CHUNK_SIZE):
        chunk_rows = all_target_pos[start: start + CHUNK_SIZE]
        X_t = lognorm_chunk(adata.X[chunk_rows], top_gene_idx)
        chunk_genes = [gene_col[p] for p in chunk_rows]
        for i, g in enumerate(chunk_genes):
            if g in gene_sum:
                gene_sum[g] += X_t[i].astype(np.float64)
                gene_count[g] += 1
        del X_t
        if (start // CHUNK_SIZE) % 10 == 0:
            pct = min(100, 100 * (start + len(chunk_rows)) // n_all)
            print(f"  {pct}% done", flush=True)

    all_mean_deltas = {
        g: (gene_sum[g] / gene_count[g]).astype(np.float32) - control_mean
        for g in all_genes if gene_count[g] >= MIN_CELLS
    }
    fetch_time = time.time() - t0
    print(f"Mean deltas done in {fetch_time:.1f}s.")

    # --- Split ---
    train_t, test_t_full = train_test_split(list(all_mean_deltas.keys()), test_size=0.3, random_state=0)
    val_t, test_t = train_test_split(test_t_full, test_size=0.5, random_state=0)
    print(f"Split: {len(train_t)} train / {len(val_t)} val / {len(test_t)} test")

    train_md = {g: all_mean_deltas[g] for g in train_t}
    val_md = {g: all_mean_deltas[g] for g in val_t}
    test_md = {g: all_mean_deltas[g] for g in test_t}
    train_gene_list = sorted(train_md)
    D_train = np.stack([train_md[g] for g in train_gene_list])
    n_out = D_train.shape[1]
    gm = D_train.mean(axis=0)
    print(f"D_train: {D_train.shape}")

    # --- Gene2Vec embeddings ---
    gene2vec_full = load_gene2vec_table(Path("data/gene2vec/gene2vec_dim_200_iter_9.txt"))
    gene2vec_lookup = {g: gene2vec_full[g] for g in top_gene_names if g in gene2vec_full}
    all_genes_set = set(all_mean_deltas.keys())
    gene2vec_all = {g: gene2vec_full[g] for g in all_genes_set if g in gene2vec_full}
    g2v_coverage_train = sum(1 for g in train_t if g in gene2vec_full) / len(train_t)
    g2v_coverage_test = sum(1 for g in test_t if g in gene2vec_full) / len(test_t)
    print(f"Gene2Vec coverage: {g2v_coverage_train:.1%} train / {g2v_coverage_test:.1%} test")

    # --- response_role: cap at 50 cells/gene to avoid OOM (1439 genes × 50 × 5000 ~ 1.4GB) ---
    print("Building response_role (per-cell, capped at 50/gene) ...")
    train_set = set(train_t)
    MAX_CELLS_RR = 50
    train_cell_deltas_list = []
    train_gene_labels = []
    train_gene_cell_counts: dict[str, int] = defaultdict(int)
    for start in range(0, n_all, CHUNK_SIZE):
        chunk_rows = all_target_pos[start: start + CHUNK_SIZE]
        chunk_genes = [gene_col[p] for p in chunk_rows]
        train_mask = [i for i, g in enumerate(chunk_genes)
                      if g in train_set and train_gene_cell_counts[g] < MAX_CELLS_RR]
        if not train_mask:
            continue
        X_t = lognorm_chunk(adata.X[chunk_rows], top_gene_idx)
        for i in train_mask:
            g = chunk_genes[i]
            if train_gene_cell_counts[g] < MAX_CELLS_RR:
                train_cell_deltas_list.append(X_t[i] - control_mean)
                train_gene_labels.append(g)
                train_gene_cell_counts[g] += 1
        del X_t
    train_cell_delta_all = np.stack(train_cell_deltas_list).astype(np.float32)
    print(f"  {train_cell_delta_all.shape[0]} training cells collected (cap={MAX_CELLS_RR}/gene).")
    rr_dim = min(50, len(train_gene_list))
    response_role_table = fit_response_role_descriptors(
        train_gene_labels, train_cell_delta_all, top_gene_names,
        n_components=rr_dim, seed=0,
    )
    del train_cell_delta_all, train_cell_deltas_list
    print(f"Response_role fitted (dim={rr_dim}).")

    # --- Baselines ---
    no_m, no_med = pearson_per_gene({g: np.zeros(n_out, np.float32) for g in test_t}, test_md)
    gm_m, gm_med = pearson_per_gene({g: gm for g in test_t}, test_md)
    print(f"\nNo-effect: {no_m:.4f};  Global mean: {gm_m:.4f} (median {gm_med:.4f})")

    # --- NN retrieval experiments ---
    print("\nNearest-neighbor retrieval (Gene2Vec cosine)...")
    nn_results = {}
    best_nn_val = None
    best_nn_k = None
    best_nn_preds = None
    for K in (1, 2, 3, 5, 10, 20, 50):
        va_preds = nn_retrieve(list(val_t), train_gene_list, gene2vec_lookup, train_md, K=K)
        vr, _ = pearson_per_gene(va_preds, val_md)
        te_preds = nn_retrieve(list(test_t), train_gene_list, gene2vec_lookup, train_md, K=K)
        tr_m, tr_med = pearson_per_gene(te_preds, test_md)
        nn_results[f"k{K}"] = {"K": K, "test_pearson_mean": tr_m, "test_pearson_median": tr_med, "val_pearson": vr}
        print(f"  NN K={K}: val={vr:.4f} test={tr_m:.4f} (median {tr_med:.4f})")
        if best_nn_val is None or vr > best_nn_val:
            best_nn_val = vr
            best_nn_k = K
            best_nn_preds = te_preds
    print(f"\nBest NN: K={best_nn_k} (val {best_nn_val:.4f})")

    # --- Standard ridge with Gene2Vec ---
    print("\nRidge regression + Gene2Vec (for ensemble) ...")
    X_tr_g2v = np.array([gene2vec_lookup.get(g, np.zeros(200, np.float32)) for g in train_gene_list])
    X_va_g2v = np.array([gene2vec_lookup.get(g, np.zeros(200, np.float32)) for g in val_t])
    X_te_g2v = np.array([gene2vec_lookup.get(g, np.zeros(200, np.float32)) for g in test_t])

    best_ridge = None
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        m = Ridge(alpha=alpha)
        m.fit(X_tr_g2v, D_train)
        va_preds = {g: m.predict(X_va_g2v[[i]])[0] for i, g in enumerate(val_t)}
        vr, _ = pearson_per_gene(va_preds, val_md)
        if best_ridge is None or vr > best_ridge[0]:
            best_ridge = (vr, alpha, m)
    _, best_alpha, best_m = best_ridge
    ridge_va_preds = {g: best_m.predict(X_va_g2v[[i]])[0] for i, g in enumerate(val_t)}
    ridge_te_preds = {g: best_m.predict(X_te_g2v[[i]])[0] for i, g in enumerate(test_t)}
    ridge_m, ridge_med = pearson_per_gene(ridge_te_preds, test_md)
    print(f"  Gene2Vec+ridge (alpha={best_alpha}): test={ridge_m:.4f} (median {ridge_med:.4f})")

    # --- SVD-denoised ridge ---
    print("\nSVD + Gene2Vec ...")
    svd_results_g2v = {}
    best_svd_preds = None
    best_svd_val = None
    for k in (20, 50, 100, 200):
        svd = TruncatedSVD(n_components=k, random_state=0)
        codes_tr = svd.fit_transform(D_train)
        best_sk = None
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
            m = Ridge(alpha=alpha)
            m.fit(X_tr_g2v, codes_tr)
            va_codes = m.predict(X_va_g2v)
            va_preds = {g: svd.inverse_transform(va_codes[[i]])[0] for i, g in enumerate(val_t)}
            vr, _ = pearson_per_gene(va_preds, val_md)
            if best_sk is None or vr > best_sk[0]:
                best_sk = (vr, alpha, m, svd)
        vr, alpha, m, svd_obj = best_sk
        te_codes = m.predict(X_te_g2v)
        te_preds = {g: svd_obj.inverse_transform(te_codes[[i]])[0] for i, g in enumerate(test_t)}
        tr_m, tr_med = pearson_per_gene(te_preds, test_md)
        svd_results_g2v[f"svd_k{k}"] = {"k": k, "alpha": alpha, "test_pearson_mean": tr_m, "test_pearson_median": tr_med, "val_pearson": vr}
        print(f"  Gene2Vec+SVD k={k}: test={tr_m:.4f} (median {tr_med:.4f})")
        if best_svd_val is None or vr > best_svd_val:
            best_svd_val = vr
            best_svd_preds = te_preds

    # --- Ensemble: NN + best SVD/ridge ---
    print("\nEnsemble (NN + Gene2Vec SVD) ...")
    ensemble_results = {}
    best_ensemble = None
    for alpha_e in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        nn_va = nn_retrieve(list(val_t), train_gene_list, gene2vec_lookup, train_md, K=best_nn_k)
        # Use ridge (not SVD) for simplicity
        ens_va = {g: alpha_e * nn_va[g] + (1 - alpha_e) * ridge_va_preds.get(g, gm) for g in val_t}
        vr, _ = pearson_per_gene(ens_va, val_md)
        if best_ensemble is None or vr > best_ensemble[0]:
            best_ensemble = (vr, alpha_e)
    best_vr, best_alpha_e = best_ensemble
    te_nn = nn_retrieve(list(test_t), train_gene_list, gene2vec_lookup, train_md, K=best_nn_k)
    ens_te = {g: best_alpha_e * te_nn[g] + (1 - best_alpha_e) * ridge_te_preds.get(g, gm) for g in test_t}
    ens_m, ens_med = pearson_per_gene(ens_te, test_md)
    ensemble_results["nn_ridge_ensemble"] = {
        "nn_weight": best_alpha_e, "K": best_nn_k,
        "test_pearson_mean": ens_m, "test_pearson_median": ens_med
    }
    print(f"  Ensemble (NN weight={best_alpha_e:.1f}, K={best_nn_k}): test={ens_m:.4f} (median {ens_med:.4f})")

    # --- response_role with per-cell deltas ---
    print("\nResponse_role (per-cell deltas) ...")
    X_tr_rr = np.array([response_role_table.get(g, np.zeros(rr_dim, np.float32)) for g in train_gene_list])
    X_va_rr = np.array([response_role_table.get(g, np.zeros(rr_dim, np.float32)) for g in val_t])
    X_te_rr = np.array([response_role_table.get(g, np.zeros(rr_dim, np.float32)) for g in test_t])
    rr_results = {}
    best_rr = None
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        m = Ridge(alpha=alpha)
        m.fit(X_tr_rr, D_train)
        va_preds = {g: m.predict(X_va_rr[[i]])[0] for i, g in enumerate(val_t)}
        vr, _ = pearson_per_gene(va_preds, val_md)
        if best_rr is None or vr > best_rr[0]:
            best_rr = (vr, alpha, m)
    _, alpha, m = best_rr
    te_preds = {g: m.predict(X_te_rr[[i]])[0] for i, g in enumerate(test_t)}
    tr_m, tr_med = pearson_per_gene(te_preds, test_md)
    rr_results["direct_ridge"] = {"alpha": alpha, "test_pearson_mean": tr_m, "test_pearson_median": tr_med}
    print(f"  rr+ridge: test={tr_m:.4f} (median {tr_med:.4f})")
    for k in (20, 50, 100, 200):
        svd = TruncatedSVD(n_components=k, random_state=0)
        codes_tr = svd.fit_transform(D_train)
        best_sk = None
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
            m = Ridge(alpha=alpha)
            m.fit(X_tr_rr, codes_tr)
            va_codes = m.predict(X_va_rr)
            va_preds = {g: svd.inverse_transform(va_codes[[i]])[0] for i, g in enumerate(val_t)}
            vr, _ = pearson_per_gene(va_preds, val_md)
            if best_sk is None or vr > best_sk[0]:
                best_sk = (vr, alpha, m, svd)
        vr, alpha, m, svd_obj = best_sk
        te_codes = m.predict(X_te_rr)
        te_preds = {g: svd_obj.inverse_transform(te_codes[[i]])[0] for i, g in enumerate(test_t)}
        tr_m, tr_med = pearson_per_gene(te_preds, test_md)
        rr_results[f"svd_k{k}"] = {"k": k, "alpha": alpha, "test_pearson_mean": tr_m, "test_pearson_median": tr_med}
        print(f"  rr+SVD k={k}: test={tr_m:.4f} (median {tr_med:.4f})")

    summary = {
        "dataset": "Replogle et al. 2022 K562 essential RAW SINGLE-CELL",
        "normalization": "log1p (library-size to 10k, then log1p)",
        "n_hvg": N_HVG,
        "evaluation_metric": "delta Pearson per test gene, then averaged",
        "literature_reference_points": {
            "GEARS_unseen_K562": 0.076,
            "mean_training_delta_baseline": 0.267,
            "oracle_nn_retrieval": 0.592,
        },
        "n_distinct_targets": len(all_genes),
        "n_train_targets": len(train_t),
        "n_val_targets": len(val_t),
        "n_test_targets": len(test_t),
        "fetch_time_seconds": fetch_time,
        "baselines": {
            "no_effect": {"test_pearson_mean": no_m},
            "global_mean": {"test_pearson_mean": gm_m, "test_pearson_median": gm_med},
        },
        "nn_retrieval_gene2vec": nn_results,
        "gene2vec_ridge": {"alpha": best_alpha, "test_pearson_mean": ridge_m},
        "gene2vec_svd": svd_results_g2v,
        "nn_ridge_ensemble": ensemble_results,
        "response_role_percell": rr_results,
    }
    Path("outputs/replogle_genelevel_v9").mkdir(parents=True, exist_ok=True)
    Path("outputs/replogle_genelevel_v9/metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote outputs/replogle_genelevel_v9/metrics.json")


if __name__ == "__main__":
    main()

"""Milestone 18: Predict-then-retrieve two-stage approach + stronger response_role.

Key insights from v9:
  1. Per-cell response_role (0.2575) > mean-delta response_role (0.255)
  2. NN retrieval in Gene2Vec space fails (Gene2Vec ≠ perturbation response similarity)

v10 introduces two new ideas:
  A) Predict-then-retrieve: use a stage-1 delta prediction to find training genes
     with similar actual deltas (cosine in DELTA space), then predict the weighted
     mean of K most similar training deltas. This bridges embedding-based prediction
     and retrieval without needing ground truth at test time.

  B) Higher response_role cell cap (100/gene): 1439 × 100 × 5000 × 4 ≈ 2.9 GB.
     Combined with other memory this is borderline; we try it, with 80 as fallback.

  C) Ensemble of stage-1 (best ridge/SVD) + stage-2 (predict-then-retrieve).
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
MAX_CELLS_RR = 100  # per-gene cap for response_role training cells


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


def predict_then_retrieve(stage1_preds, train_gene_list, train_md, K=10):
    """Stage 1 predicted deltas → cosine NN in training delta space → weighted mean."""
    D_tr = np.stack([train_md[g] for g in train_gene_list])  # (n_train, n_out)
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
        weights = weights / weights.sum()
        result[g] = sum(w * train_md[train_gene_list[j]] for j, w in zip(top_k, weights)).astype(np.float32)
    return result


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

    print(f"Computing online mean deltas ({len(all_target_pos)} target cells)...")
    gene_sum = {g: np.zeros(N_HVG, dtype=np.float64) for g in all_genes}
    gene_count = {g: 0 for g in all_genes}
    n_all = len(all_target_pos)
    for start in range(0, n_all, CHUNK_SIZE):
        chunk_rows = all_target_pos[start: start + CHUNK_SIZE]
        X_t = lognorm_chunk(adata.X[chunk_rows], top_gene_idx)
        for i, g in enumerate([gene_col[p] for p in chunk_rows]):
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

    # --- Gene2Vec ---
    gene2vec_full = load_gene2vec_table(Path("data/gene2vec/gene2vec_dim_200_iter_9.txt"))
    gene2vec_lookup = {g: gene2vec_full[g] for g in top_gene_names if g in gene2vec_full}

    # --- response_role with per-cell deltas, capped at MAX_CELLS_RR ---
    print(f"Building response_role (per-cell, cap={MAX_CELLS_RR}/gene) ...")
    train_set = set(train_t)
    cell_deltas_list = []
    cell_gene_labels = []
    gene_cell_counts: dict[str, int] = defaultdict(int)
    for start in range(0, n_all, CHUNK_SIZE):
        chunk_rows = all_target_pos[start: start + CHUNK_SIZE]
        chunk_genes = [gene_col[p] for p in chunk_rows]
        train_mask = [i for i, g in enumerate(chunk_genes)
                      if g in train_set and gene_cell_counts[g] < MAX_CELLS_RR]
        if not train_mask:
            continue
        X_t = lognorm_chunk(adata.X[chunk_rows], top_gene_idx)
        for i in train_mask:
            g = chunk_genes[i]
            if gene_cell_counts[g] < MAX_CELLS_RR:
                cell_deltas_list.append(X_t[i] - control_mean)
                cell_gene_labels.append(g)
                gene_cell_counts[g] += 1
        del X_t
    train_cell_delta_all = np.stack(cell_deltas_list).astype(np.float32)
    print(f"  {train_cell_delta_all.shape[0]} cells collected.")
    rr_dim = min(50, len(train_gene_list))
    response_role_table = fit_response_role_descriptors(
        cell_gene_labels, train_cell_delta_all, top_gene_names,
        n_components=rr_dim, seed=0,
    )
    del train_cell_delta_all, cell_deltas_list
    print(f"Response_role fitted (dim={rr_dim}).")

    # --- Stage 1 models ---
    no_m, no_med = pearson_per_gene({g: np.zeros(n_out, np.float32) for g in test_t}, test_md)
    gm_m, gm_med = pearson_per_gene({g: gm for g in test_t}, test_md)
    print(f"\nNo-effect: {no_m:.4f};  Global mean: {gm_m:.4f} (median {gm_med:.4f})")

    def build_tables():
        return {
            "gene2vec": (gene2vec_lookup, 200),
            "response_role": (response_role_table, rr_dim),
        }

    stage1_results = {}
    best_stage1_va = None
    best_stage1_te_preds = None
    best_stage1_va_preds = None

    for name, (table, dim) in build_tables().items():
        X_tr = np.array([table.get(g, np.zeros(dim, np.float32)) for g in train_gene_list])
        X_va = np.array([table.get(g, np.zeros(dim, np.float32)) for g in val_t])
        X_te = np.array([table.get(g, np.zeros(dim, np.float32)) for g in test_t])

        # SVD k=100 (from prior experiments, consistently best)
        svd = TruncatedSVD(n_components=100, random_state=0)
        codes_tr = svd.fit_transform(D_train)
        best_m = None
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
            m = Ridge(alpha=alpha)
            m.fit(X_tr, codes_tr)
            va_codes = m.predict(X_va)
            va_preds = {g: svd.inverse_transform(va_codes[[i]])[0] for i, g in enumerate(val_t)}
            vr, _ = pearson_per_gene(va_preds, val_md)
            if best_m is None or vr > best_m[0]:
                best_m = (vr, alpha, m, va_preds)
        vr, alpha, m, va_preds = best_m
        te_codes = m.predict(X_te)
        te_preds = {g: svd.inverse_transform(te_codes[[i]])[0] for i, g in enumerate(test_t)}
        tr_m, tr_med = pearson_per_gene(te_preds, test_md)
        stage1_results[f"{name}_svd100"] = {"alpha": alpha, "test_pearson_mean": tr_m, "test_pearson_median": tr_med, "val_pearson": vr}
        print(f"  Stage1 {name}+SVD k=100: test={tr_m:.4f} (val={vr:.4f})")

        if best_stage1_va is None or vr > best_stage1_va:
            best_stage1_va = vr
            best_stage1_te_preds = te_preds
            best_stage1_va_preds = va_preds

    print(f"\nBest stage-1 val: {best_stage1_va:.4f}")

    # --- Stage 2: predict-then-retrieve ---
    print("\nStage 2: predict-then-retrieve (K in delta space)...")
    ptr_results = {}
    best_ptr_val = None
    best_ptr_te = None
    for K in (3, 5, 10, 20, 50, 100):
        # Validate on val set using stage-1 val predictions
        va_ptr = predict_then_retrieve(best_stage1_va_preds, train_gene_list, train_md, K=K)
        vr_ptr, _ = pearson_per_gene(va_ptr, val_md)
        te_ptr = predict_then_retrieve(best_stage1_te_preds, train_gene_list, train_md, K=K)
        tr_m, tr_med = pearson_per_gene(te_ptr, test_md)
        ptr_results[f"k{K}"] = {"K": K, "val_pearson": vr_ptr, "test_pearson_mean": tr_m, "test_pearson_median": tr_med}
        print(f"  PTR K={K}: val={vr_ptr:.4f} test={tr_m:.4f} (median {tr_med:.4f})")
        if best_ptr_val is None or vr_ptr > best_ptr_val:
            best_ptr_val = vr_ptr
            best_ptr_te = te_ptr

    # --- Ensemble: stage1 + predict-then-retrieve ---
    print("\nEnsemble (stage1 + predict-then-retrieve) ...")
    ens_results = {}
    best_ens = None
    for alpha_e in np.arange(0.0, 1.05, 0.1):
        # Find best PTR K on val
        va_ptr = predict_then_retrieve(best_stage1_va_preds, train_gene_list, train_md, K=20)
        ens_va = {g: (1 - alpha_e) * best_stage1_va_preds[g] + alpha_e * va_ptr[g] for g in val_t}
        vr, _ = pearson_per_gene(ens_va, val_md)
        if best_ens is None or vr > best_ens[0]:
            best_ens = (vr, float(alpha_e))

    best_vr_ens, best_alpha_e = best_ens
    te_ptr_20 = predict_then_retrieve(best_stage1_te_preds, train_gene_list, train_md, K=20)
    ens_te = {g: (1 - best_alpha_e) * best_stage1_te_preds[g] + best_alpha_e * te_ptr_20[g] for g in test_t}
    ens_m, ens_med = pearson_per_gene(ens_te, test_md)
    print(f"  Best ensemble (ptr_weight={best_alpha_e:.1f}): test={ens_m:.4f} (median {ens_med:.4f})")
    ens_results["stage1_plus_ptr"] = {"ptr_weight": best_alpha_e, "K": 20, "test_pearson_mean": ens_m, "test_pearson_median": ens_med}

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
        "stage1_models": stage1_results,
        "predict_then_retrieve": ptr_results,
        "ensemble_stage1_ptr": ens_results,
    }
    Path("outputs/replogle_genelevel_v10").mkdir(parents=True, exist_ok=True)
    Path("outputs/replogle_genelevel_v10/metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote outputs/replogle_genelevel_v10/metrics.json")


if __name__ == "__main__":
    main()

"""Milestone 14: log1p normalization + all cells (no cap), chunked loading.

Builds on v5 (which showed log1p raises global-mean baseline from 0.198 to 0.226
and best model from 0.218 to 0.233). Removes the 80-cell cap to test whether
more cells per gene help in log1p space (they hurt in raw space due to
batch-to-batch variance amplification; log1p normalization should reduce that).

Memory approach: load each chunk (8000 rows) of the full 8563-gene sparse matrix,
compute per-cell library size from the full sparse row, column-select top 1500
genes, normalize in-place, stack chunks. Peak RAM per chunk: ~274 MB sparse +
~48 MB top-gene dense = ~320 MB. Total output: 299k x 1500 x 4 ≈ 1.8 GB.

Literature reference: mean training delta baseline 0.267 (Kernfeld et al.),
GEARS 0.076 (Roohani et al. 2023). Both on log1p-normalized Replogle K562.
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


def load_rows_lognorm_top_genes(adata, row_positions, top_gene_idx, chunk_size=8000):
    """Chunked load: for each chunk compute lib sizes from full sparse row,
    then column-select top 1500 and log1p-normalize in-place.

    Peak per-chunk RAM: chunk_size x n_genes_sparse + chunk_size x 1500 dense.
    For chunk_size=8000 and n_genes=8563 at ~10% sparsity: ~100MB + 48MB.
    """
    chunks = []
    n = len(row_positions)
    for start in range(0, n, chunk_size):
        chunk_rows = row_positions[start: start + chunk_size]
        X_sparse = adata.X[chunk_rows]  # sparse, shape (chunk, 8563)
        lib_sizes = np.asarray(X_sparse.sum(axis=1)).ravel().astype(np.float64)
        lib_sizes = np.where(lib_sizes == 0, 1.0, lib_sizes)
        X_top = X_sparse[:, top_gene_idx]
        X_dense = np.asarray(
            X_top.todense() if hasattr(X_top, "todense") else X_top,
            dtype=np.float32,
        )
        X_dense /= lib_sizes[:, None].astype(np.float32)
        X_dense *= TARGET_SUM
        X_dense = np.log1p(X_dense)
        chunks.append(X_dense)
        if (start // chunk_size) % 5 == 0:
            pct = min(100, 100 * (start + len(chunk_rows)) // n)
            print(f"  chunk {start // chunk_size + 1}: {pct}% done", flush=True)
    return np.vstack(chunks)


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
    print(f"Full: {adata.shape[0]} cells x {adata.shape[1]} genes")

    rng = np.random.default_rng(0)
    ctrl_positions = [i for i, g in enumerate(gene_col) if g == "non-targeting"]
    target_by_gene: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(gene_col):
        if g != "non-targeting":
            target_by_gene[g].append(i)

    all_genes = sorted(k for k, v in target_by_gene.items() if len(v) >= MIN_CELLS)
    all_target_pos = sorted(p for g in all_genes for p in target_by_gene[g])
    print(
        f"Controls: {len(ctrl_positions)}, "
        f"target cells (no cap): {len(all_target_pos)} across {len(all_genes)} genes"
    )

    # --- Identify top-variable genes from 8000 sampled control cells in log1p space ---
    ctrl_sample = sorted(
        rng.choice(ctrl_positions, size=min(8000, len(ctrl_positions)), replace=False)
        .tolist()
    )
    print("Loading control cells for top-gene identification (log1p space)...")
    X_ctrl_raw = adata.X[ctrl_sample]
    X_ctrl_raw = np.asarray(
        X_ctrl_raw.todense() if hasattr(X_ctrl_raw, "todense") else X_ctrl_raw,
        dtype=np.float32,
    )
    lib_ctrl = X_ctrl_raw.sum(axis=1, keepdims=True)
    lib_ctrl = np.where(lib_ctrl == 0, 1, lib_ctrl)
    X_ctrl_norm = np.log1p(X_ctrl_raw / lib_ctrl * TARGET_SUM)
    del X_ctrl_raw
    ctrl_var = X_ctrl_norm.var(axis=0)
    top_gene_idx = np.sort(np.argsort(ctrl_var)[::-1][:1500])
    top_gene_names = [gene_names[i] for i in top_gene_idx]
    X_ctrl = X_ctrl_norm[:, top_gene_idx].astype(np.float32)
    control_mean = X_ctrl.mean(axis=0)
    del X_ctrl_norm
    print(f"Top 1500 genes identified in log1p space.")

    # --- Load ALL control cells for std computation in mean_std descriptor ---
    print(f"Loading ALL {len(ctrl_positions)} control cells (log1p) for std...")
    X_ctrl_all = load_rows_lognorm_top_genes(adata, sorted(ctrl_positions), top_gene_idx)
    control_mean = X_ctrl_all.mean(axis=0)  # more accurate with all controls
    ctrl_std = X_ctrl_all.std(axis=0)
    del X_ctrl_all

    # --- Load ALL target cells chunked (log1p normalization inside) ---
    print(f"Loading {len(all_target_pos)} target cells (log1p, chunked)...")
    X_target = load_rows_lognorm_top_genes(adata, all_target_pos, top_gene_idx)
    target_genes_flat = [gene_col[p] for p in all_target_pos]
    fetch_time = time.time() - t0
    print(f"Loaded in {fetch_time:.1f}s. {X_target.nbytes / 1e9:.2f} GB")

    # --- Per-gene mean deltas ---
    all_mean_deltas: dict[str, np.ndarray] = {}
    for g in all_genes:
        idx = [i for i, t in enumerate(target_genes_flat) if t == g]
        if idx:
            all_mean_deltas[g] = X_target[idx].mean(axis=0) - control_mean

    # --- Split (same seed as all prior experiments) ---
    train_t, test_t_full = train_test_split(
        list(all_mean_deltas.keys()), test_size=0.3, random_state=0
    )
    val_t, test_t = train_test_split(test_t_full, test_size=0.5, random_state=0)
    print(f"Split: {len(train_t)} train / {len(val_t)} val / {len(test_t)} test")

    train_md = {g: all_mean_deltas[g] for g in train_t}
    val_md = {g: all_mean_deltas[g] for g in val_t}
    test_md = {g: all_mean_deltas[g] for g in test_t}

    train_gene_list = sorted(train_md)
    D_train = np.stack([train_md[g] for g in train_gene_list])
    n_out = D_train.shape[1]
    print(f"D_train: {D_train.shape}")

    # --- Descriptors ---
    gene2vec_table = load_gene2vec_table(Path("data/gene2vec/gene2vec_dim_200_iter_9.txt"))
    gene2vec_lookup = {g: gene2vec_table[g] for g in top_gene_names if g in gene2vec_table}
    mean_std_table = {
        gene: np.array([control_mean[i], ctrl_std[i]], dtype=np.float32)
        for i, gene in enumerate(top_gene_names)
    }
    train_set = set(train_t)
    train_idx = [i for i, t in enumerate(target_genes_flat) if t in train_set]
    train_targets_list = [target_genes_flat[i] for i in train_idx]
    train_cell_delta = X_target[train_idx] - control_mean
    rr_dim = min(30, len(train_gene_list))
    response_role_table = fit_response_role_descriptors(
        train_targets_list, train_cell_delta, top_gene_names,
        n_components=rr_dim, seed=0,
    )
    del X_target, train_cell_delta

    # --- Baselines ---
    gm = D_train.mean(axis=0)
    no_m, no_med = pearson_per_gene(
        {g: np.zeros(n_out, np.float32) for g in test_t}, test_md
    )
    gm_m, gm_med = pearson_per_gene({g: gm for g in test_t}, test_md)
    print(f"\nNo-effect: {no_m:.4f};  Global mean: {gm_m:.4f} (median {gm_med:.4f})")

    # --- Model evaluation ---
    def evaluate(name, table, dim):
        X_tr = np.array([table.get(g, np.zeros(dim, np.float32)) for g in train_gene_list])
        X_va = np.array([table.get(g, np.zeros(dim, np.float32)) for g in val_t])
        X_te = np.array([table.get(g, np.zeros(dim, np.float32)) for g in test_t])
        out = {}

        best_dr = None
        for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
            m = Ridge(alpha=alpha)
            m.fit(X_tr, D_train)
            va_preds = {g: m.predict(X_va[[i]])[0] for i, g in enumerate(val_t)}
            vr, _ = pearson_per_gene(va_preds, val_md)
            if best_dr is None or vr > best_dr[0]:
                best_dr = (vr, alpha, m)
        _, alpha, m = best_dr
        te_preds = {g: m.predict(X_te[[i]])[0] for i, g in enumerate(test_t)}
        tr_m, tr_med = pearson_per_gene(te_preds, test_md)
        out["direct_ridge"] = {"alpha": alpha, "test_pearson_mean": tr_m, "test_pearson_median": tr_med}
        print(f"  {name}+ridge:    {tr_m:.4f} (median {tr_med:.4f})")

        for k in (10, 20, 30, 50, 100):
            if k >= min(D_train.shape):
                continue
            svd = TruncatedSVD(n_components=k, random_state=0)
            codes_tr = svd.fit_transform(D_train)
            best_svd = None
            for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
                m = Ridge(alpha=alpha)
                m.fit(X_tr, codes_tr)
                va_codes = m.predict(X_va)
                va_preds = {
                    g: svd.inverse_transform(va_codes[[i]])[0]
                    for i, g in enumerate(val_t)
                }
                vr, _ = pearson_per_gene(va_preds, val_md)
                if best_svd is None or vr > best_svd[0]:
                    best_svd = (vr, alpha, m)
            _, alpha, m = best_svd
            te_codes = m.predict(X_te)
            te_preds = {
                g: svd.inverse_transform(te_codes[[i]])[0]
                for i, g in enumerate(test_t)
            }
            tr_m, tr_med = pearson_per_gene(te_preds, test_md)
            out[f"svd_k{k}"] = {"k": k, "alpha": alpha, "test_pearson_mean": tr_m, "test_pearson_median": tr_med}
            print(f"  {name}+SVD_k={k}: {tr_m:.4f} (median {tr_med:.4f})")
        return out

    print("\nEvaluating mean_std ...")
    ms_results = evaluate("mean_std", mean_std_table, 2)
    print("\nEvaluating Gene2Vec ...")
    g2v_results = evaluate("gene2vec", gene2vec_lookup, 200)
    print("\nEvaluating response_role ...")
    rr_results = evaluate("response_role", response_role_table, rr_dim)

    summary = {
        "dataset": "Replogle et al. 2022 K562 essential RAW SINGLE-CELL",
        "normalization": "log1p (library-size to 10k, then log1p)",
        "cells_per_target_cap": "none (all available cells)",
        "evaluation_metric": "delta Pearson per test gene, then averaged",
        "literature_reference_points": {
            "GEARS_unseen_K562": 0.076,
            "mean_training_delta_baseline": 0.267,
            "oracle_nn_retrieval": 0.592,
            "source": "Delta Gap paper arXiv 2611.10658; Ahlmann-Eltze NatMeth 2024",
        },
        "n_distinct_targets": len(all_genes),
        "n_train_targets": len(train_t),
        "n_val_targets": len(val_t),
        "n_test_targets": len(test_t),
        "n_target_cells": len(all_target_pos),
        "fetch_time_seconds": fetch_time,
        "baselines": {
            "no_effect": {"test_pearson_mean": no_m, "test_pearson_median": no_med},
            "global_mean": {"test_pearson_mean": gm_m, "test_pearson_median": gm_med},
        },
        "mean_std": ms_results,
        "gene2vec": g2v_results,
        "response_role": rr_results,
    }
    Path("outputs/replogle_genelevel_v6").mkdir(parents=True, exist_ok=True)
    Path("outputs/replogle_genelevel_v6/metrics.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"\nWrote outputs/replogle_genelevel_v6/metrics.json")


if __name__ == "__main__":
    main()

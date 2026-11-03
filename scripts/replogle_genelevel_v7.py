"""Milestone 15: 5000 HVGs + all cells + log1p + online mean delta.

The literature (Delta Gap paper) uses 5000 HVGs. We've been using 1500,
which likely explains why our global-mean baseline (0.237) lags the
literature's 0.267. This script computes per-gene mean deltas *online*
during chunked loading so we never need to hold all 299k × 5000 in RAM.

Memory budget:
  - Per chunk: 8000 × 8563 sparse (~100 MB) + 8000 × 5000 dense (~160 MB)
  - Accumulation: 2057 × 5000 × float64 sum + int32 count = ~82 MB
  - Total peak: ~350 MB, comfortably within 8 GB EC2 RAM.

We also run v5-style 80-cell cap for direct comparison on 5000 HVGs.
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
N_HVG = 5000  # ← key change from v5/v6 which used 1500
CHUNK_SIZE = 6000


def lognorm_chunk(X_sparse, top_gene_idx):
    """Compute library size from full sparse row, column-select, log1p normalize."""
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
    print(
        f"Controls: {len(ctrl_positions)}, "
        f"target cells (no cap): {len(all_target_pos)} across {len(all_genes)} genes"
    )

    # --- Top-N HVGs from sampled control cells ---
    rng = np.random.default_rng(0)
    ctrl_sample = sorted(
        rng.choice(ctrl_positions, size=min(8000, len(ctrl_positions)), replace=False).tolist()
    )
    print(f"Computing HVGs from {len(ctrl_sample)} sampled control cells...")
    X_cs = adata.X[ctrl_sample]
    X_cs = np.asarray(
        X_cs.todense() if hasattr(X_cs, "todense") else X_cs, dtype=np.float32
    )
    lib_cs = X_cs.sum(axis=1, keepdims=True)
    lib_cs = np.where(lib_cs == 0, 1, lib_cs)
    X_cs_norm = np.log1p(X_cs / lib_cs * TARGET_SUM)
    del X_cs
    ctrl_var = X_cs_norm.var(axis=0)
    top_gene_idx = np.sort(np.argsort(ctrl_var)[::-1][:N_HVG])
    top_gene_names = [gene_names[i] for i in top_gene_idx]
    del X_cs_norm
    print(f"Top {N_HVG} HVGs identified.")

    # --- Load all control cells chunked, compute control_mean & std ---
    print(f"Loading ALL {len(ctrl_positions)} control cells (log1p) for mean/std...")
    ctrl_sum = np.zeros(N_HVG, dtype=np.float64)
    ctrl_sum2 = np.zeros(N_HVG, dtype=np.float64)
    n_ctrl = 0
    ctrl_pos_list = sorted(ctrl_positions)
    for start in range(0, len(ctrl_pos_list), CHUNK_SIZE):
        chunk = ctrl_pos_list[start: start + CHUNK_SIZE]
        X_c = lognorm_chunk(adata.X[chunk], top_gene_idx).astype(np.float64)
        ctrl_sum += X_c.sum(axis=0)
        ctrl_sum2 += (X_c ** 2).sum(axis=0)
        n_ctrl += len(chunk)
    control_mean = (ctrl_sum / n_ctrl).astype(np.float32)
    ctrl_std = np.sqrt(np.maximum(ctrl_sum2 / n_ctrl - (ctrl_sum / n_ctrl) ** 2, 0)).astype(np.float32)
    print(f"Control mean computed over {n_ctrl} cells.")

    # --- Online mean delta for all target cells ---
    print(f"Computing per-gene mean deltas online ({len(all_target_pos)} target cells)...")
    gene_sum: dict[str, np.ndarray] = {g: np.zeros(N_HVG, dtype=np.float64) for g in all_genes}
    gene_count: dict[str, int] = {g: 0 for g in all_genes}

    n_all = len(all_target_pos)
    for start in range(0, n_all, CHUNK_SIZE):
        chunk_rows = all_target_pos[start: start + CHUNK_SIZE]
        X_t = lognorm_chunk(adata.X[chunk_rows], top_gene_idx)
        chunk_genes = [gene_col[p] for p in chunk_rows]
        for i, g in enumerate(chunk_genes):
            if g in gene_sum:
                gene_sum[g] += X_t[i].astype(np.float64)
                gene_count[g] += 1
        if (start // CHUNK_SIZE) % 10 == 0:
            pct = min(100, 100 * (start + len(chunk_rows)) // n_all)
            print(f"  {pct}% done ({start + len(chunk_rows)} / {n_all} cells)", flush=True)
        del X_t

    all_mean_deltas = {}
    for g in all_genes:
        if gene_count[g] >= MIN_CELLS:
            all_mean_deltas[g] = (gene_sum[g] / gene_count[g]).astype(np.float32) - control_mean

    fetch_time = time.time() - t0
    print(f"Online mean delta done in {fetch_time:.1f}s. {len(all_mean_deltas)} genes.")

    # --- Split ---
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
    # For response_role we need per-cell deltas for training genes.
    # Build from per-gene mean delta × gene_count (approximation) — or load training cells.
    # Use mean-delta stacked as "per-cell" (1 observation per gene, weighted).
    # This is a simplified response_role that just uses the mean delta as a single observation.
    train_set = set(train_t)
    # Approximate fit_response_role with mean deltas (each gene = 1 obs)
    train_targets_approx = list(train_set)
    train_cell_delta_approx = np.stack([train_md[g] for g in train_targets_approx])
    rr_dim = min(30, len(train_gene_list))
    response_role_table = fit_response_role_descriptors(
        train_targets_approx, train_cell_delta_approx, top_gene_names,
        n_components=rr_dim, seed=0,
    )

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
        "n_hvg": N_HVG,
        "cells_per_target_cap": "none (all available, online mean delta)",
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
        "n_target_cells": n_all,
        "fetch_time_seconds": fetch_time,
        "baselines": {
            "no_effect": {"test_pearson_mean": no_m, "test_pearson_median": no_med},
            "global_mean": {"test_pearson_mean": gm_m, "test_pearson_median": gm_med},
        },
        "mean_std": ms_results,
        "gene2vec": g2v_results,
        "response_role": rr_results,
    }
    Path("outputs/replogle_genelevel_v7").mkdir(parents=True, exist_ok=True)
    Path("outputs/replogle_genelevel_v7/metrics.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"\nWrote outputs/replogle_genelevel_v7/metrics.json")


if __name__ == "__main__":
    main()

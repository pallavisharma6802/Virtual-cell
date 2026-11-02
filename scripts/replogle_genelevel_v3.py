"""Milestone 11: batch-matched controls + all available cells (no cap).

Two fixes over v2 (which got delta Pearson 0.218 mean):

1. Batch-matched controls: for each target cell, subtract the control mean
   from its own gem_group (10x Chromium batch), not a global mean.
   TxPert (2025) showed significant batch-perturbation confounding in
   Replogle data (p as low as 1e-98). Using globally-sampled controls adds
   noise that is entirely removable by matching within gem_group.

2. All cells, no cap: use every available cell per target gene, which
   should raise the global-mean baseline toward the literature's 0.267
   (we got 0.198 at 80 cells/gene cap vs. 0.267 from Kernfeld et al.).

Together these two fixes should substantially reduce per-gene mean delta
noise, narrowing the gap to the literature reference.

Literature reference points (Replogle K562 essential, unseen perturbations):
  - GEARS (NatBiotech 2023): delta Pearson ≈ 0.076
  - Mean training delta (Kernfeld et al.): ≈ 0.267
  - Oracle upper bound: ≈ 0.592
  (From: Delta Gap paper arXiv 2611.10658; Ahlmann-Eltze NatMeth 2024)
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from virtualcell.data.target_descriptors import (
    descriptor_for_targets,
    fit_response_role_descriptors,
    load_gene2vec_table,
)

MIN_CELLS_PER_TARGET = 5


def load_rows_top_genes(adata, row_positions, top_gene_idx):
    """Load specific rows, selecting gene columns while still sparse."""
    X_sparse = adata.X[row_positions]
    X_sub = X_sparse[:, top_gene_idx]
    return np.asarray(
        X_sub.todense() if hasattr(X_sub, "todense") else X_sub,
        dtype=np.float32,
    )


def pearson_per_gene(pred_dict, true_dict):
    genes = sorted(set(pred_dict) & set(true_dict))
    vals = []
    for g in genes:
        p, t = pred_dict[g], true_dict[g]
        if np.std(t) < 1e-6 or np.std(p) < 1e-6:
            vals.append(0.0)
        else:
            r, _ = pearsonr(p, t)
            vals.append(float(np.clip(r, -1, 1)))
    return float(np.mean(vals)), float(np.median(vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad-path", default="data/replogle/K562_essential_raw_singlecell_01.h5ad")
    parser.add_argument("--gene2vec-path", default="data/gene2vec/gene2vec_dim_200_iter_9.txt")
    parser.add_argument("--output", default="outputs/replogle_genelevel_v3/metrics.json")
    parser.add_argument("--n-top-genes", type=int, default=1500)
    parser.add_argument("--ctrl-sample-for-var", type=int, default=8000,
                        help="Controls to use for top-gene variance estimation")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    t0 = time.time()
    print(f"Opening {args.h5ad_path} backed ...")
    adata = ad.read_h5ad(args.h5ad_path, backed="r")
    n_cells, n_genes_full = adata.shape
    gene_names = (adata.var["gene_name"].tolist()
                  if "gene_name" in adata.var.columns
                  else adata.var_names.tolist())
    print(f"Full file: {n_cells} cells x {n_genes_full} genes")
    print(f"obs columns: {list(adata.obs.columns)}")

    # Load obs metadata (no X)
    gene_col = adata.obs["gene"].astype(str).tolist()
    gem_col = adata.obs["gem_group"].astype(str).tolist() if "gem_group" in adata.obs.columns else ["1"] * n_cells
    print(f"Unique gem_groups: {sorted(set(gem_col))}")

    rng = np.random.default_rng(args.seed)

    # Separate control vs target positions
    ctrl_positions = [i for i, g in enumerate(gene_col) if g == "non-targeting"]
    target_positions_by_gene: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(gene_col):
        if g != "non-targeting":
            target_positions_by_gene[g].append(i)

    all_target_genes = sorted(k for k, v in target_positions_by_gene.items()
                               if len(v) >= MIN_CELLS_PER_TARGET)
    all_target_positions = sorted(p for g in all_target_genes
                                   for p in target_positions_by_gene[g])

    print(f"Control cells: {len(ctrl_positions)}")
    print(f"Target genes (>={MIN_CELLS_PER_TARGET} cells): {len(all_target_genes)}")
    print(f"Target cells total (no cap): {len(all_target_positions)}")

    # --- Step 1: Identify top variable genes from a sample of control cells ---
    ctrl_sample_for_var = rng.choice(ctrl_positions,
                                      size=min(args.ctrl_sample_for_var, len(ctrl_positions)),
                                      replace=False)
    ctrl_sample_for_var = sorted(ctrl_sample_for_var.tolist())
    print(f"Loading {len(ctrl_sample_for_var)} control cells for top-gene identification ...")
    X_ctrl_full = adata.X[ctrl_sample_for_var]
    X_ctrl_full = np.asarray(
        X_ctrl_full.todense() if hasattr(X_ctrl_full, "todense") else X_ctrl_full,
        dtype=np.float32,
    )  # shape: (ctrl_sample, n_genes_full)
    ctrl_var = X_ctrl_full.var(axis=0)
    top_gene_idx = np.sort(np.argsort(ctrl_var)[::-1][: args.n_top_genes])
    top_gene_names = [gene_names[i] for i in top_gene_idx]
    del X_ctrl_full
    print(f"Top {args.n_top_genes} variable genes identified.")

    # --- Step 2: Load ALL control cells (top genes only) for batch-matched means ---
    print(f"Loading ALL {len(ctrl_positions)} control cells (top genes) for batch means ...")
    X_ctrl = load_rows_top_genes(adata, sorted(ctrl_positions), top_gene_idx)
    # shape: (n_ctrl, 1500)
    ctrl_gemgroups = [gem_col[p] for p in sorted(ctrl_positions)]

    # Compute per-gem_group control mean
    ctrl_mean_by_gg: dict[str, np.ndarray] = {}
    for gg in sorted(set(ctrl_gemgroups)):
        idx = [i for i, g in enumerate(ctrl_gemgroups) if g == gg]
        ctrl_mean_by_gg[gg] = X_ctrl[idx].mean(axis=0)
        print(f"  gem_group {gg}: {len(idx)} control cells, mean norm={np.linalg.norm(ctrl_mean_by_gg[gg]):.2f}")
    # Global fallback for any unseen gem_group
    global_ctrl_mean = X_ctrl.mean(axis=0)
    print(f"Control loading done. {len(ctrl_mean_by_gg)} gem_groups.")

    # --- Step 3: Load ALL target cells (top genes only) ---
    print(f"Loading {len(all_target_positions)} target cells (top genes, sparse select) ...")
    X_target = load_rows_top_genes(adata, all_target_positions, top_gene_idx)
    # shape: (n_target, 1500)
    target_genes_flat = [gene_col[p] for p in all_target_positions]
    target_gg_flat = [gem_col[p] for p in all_target_positions]
    fetch_time = time.time() - t0
    print(f"Loaded in {fetch_time:.1f}s. "
          f"target {X_target.nbytes/1e9:.2f}GB, ctrl {X_ctrl.nbytes/1e9:.2f}GB")

    # --- Step 4: Compute batch-matched deltas per cell ---
    print("Computing batch-matched deltas ...")
    deltas = np.empty_like(X_target)
    for i, gg in enumerate(target_gg_flat):
        cm = ctrl_mean_by_gg.get(gg, global_ctrl_mean)
        deltas[i] = X_target[i] - cm
    del X_target  # free memory

    # --- Step 5: Compute per-gene mean deltas ---
    all_mean_deltas: dict[str, np.ndarray] = {}
    for g in all_target_genes:
        idx = [i for i, t in enumerate(target_genes_flat) if t == g]
        if idx:
            all_mean_deltas[g] = deltas[idx].mean(axis=0)

    # --- Split ---
    train_t, test_t_full = train_test_split(list(all_mean_deltas.keys()),
                                            test_size=0.3, random_state=args.seed)
    val_t, test_t = train_test_split(test_t_full, test_size=0.5, random_state=args.seed)
    print(f"Split: {len(train_t)} train / {len(val_t)} val / {len(test_t)} test")

    train_mean_deltas = {g: all_mean_deltas[g] for g in train_t}
    val_mean_deltas = {g: all_mean_deltas[g] for g in val_t}
    test_mean_deltas = {g: all_mean_deltas[g] for g in test_t}

    train_gene_list = sorted(train_mean_deltas)
    D_train = np.stack([train_mean_deltas[g] for g in train_gene_list])
    n_genes_out = D_train.shape[1]
    print(f"Training perturbation matrix: {D_train.shape}")

    # Per-cell training data for response_role descriptor
    train_set = set(train_t)
    train_idx = [i for i, t in enumerate(target_genes_flat) if t in train_set]
    train_targets_list = [target_genes_flat[i] for i in train_idx]
    train_cell_delta = deltas[train_idx]  # (n_train_cells, 1500)

    # --- Descriptors ---
    print("Loading Gene2Vec ...")
    gene2vec_table = load_gene2vec_table(Path(args.gene2vec_path))
    gene2vec_lookup = {g: gene2vec_table[g] for g in top_gene_names if g in gene2vec_table}

    mean_std_table = {
        gene: np.array([global_ctrl_mean[i], X_ctrl[:, i].std()], dtype=np.float32)
        for i, gene in enumerate(top_gene_names)
    }

    response_role_dim = min(30, len(train_gene_list), D_train.shape[1])
    response_role_table = fit_response_role_descriptors(
        train_targets_list, train_cell_delta, top_gene_names,
        n_components=response_role_dim, seed=args.seed,
    )

    del deltas, train_cell_delta  # free memory before model training

    # --- Baselines ---
    zero_preds = {g: np.zeros(n_genes_out, dtype=np.float32) for g in test_t}
    no_r_mean, no_r_med = pearson_per_gene(zero_preds, test_mean_deltas)
    no_mae = float(np.mean([np.mean(np.abs(test_mean_deltas[g])) for g in test_t]))

    global_mean_delta = D_train.mean(axis=0)
    gm_preds = {g: global_mean_delta for g in test_t}
    gm_r_mean, gm_r_med = pearson_per_gene(gm_preds, test_mean_deltas)
    gm_mae = float(np.mean([np.mean(np.abs(global_mean_delta - test_mean_deltas[g])) for g in test_t]))

    print(f"\nNo-effect:   Pearson mean={no_r_mean:.4f}, median={no_r_med:.4f}, MAE={no_mae:.4f}")
    print(f"Global mean: Pearson mean={gm_r_mean:.4f}, median={gm_r_med:.4f}, MAE={gm_mae:.4f}")

    # --- Model evaluation ---
    def evaluate(name, table, dim):
        X_tr = np.array([table.get(g, np.zeros(dim, np.float32)) for g in train_gene_list])
        X_va = np.array([table.get(g, np.zeros(dim, np.float32)) for g in val_t])
        X_te = np.array([table.get(g, np.zeros(dim, np.float32)) for g in test_t])

        out = {}
        # Direct ridge
        best_dr = None
        for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
            m = Ridge(alpha=alpha)
            m.fit(X_tr, D_train)
            va_preds = {g: m.predict(X_va[[i]])[0] for i, g in enumerate(val_t)}
            vr, _ = pearson_per_gene(va_preds, val_mean_deltas)
            if best_dr is None or vr > best_dr[0]:
                best_dr = (vr, alpha, m)
        _, alpha, m = best_dr
        te_preds = {g: m.predict(X_te[[i]])[0] for i, g in enumerate(test_t)}
        tr_m, tr_med = pearson_per_gene(te_preds, test_mean_deltas)
        te_mae = float(np.mean([np.mean(np.abs(te_preds[g] - test_mean_deltas[g])) for g in test_t]))
        out["direct_ridge"] = {"alpha": alpha, "val_pearson": best_dr[0],
                               "test_pearson_mean": tr_m, "test_pearson_median": tr_med, "test_mae": te_mae}
        print(f"  {name}+ridge: test Pearson={tr_m:.4f} (median {tr_med:.4f})")

        # SVD latent factor
        for k in (10, 20, 30, 50, 100):
            if k >= min(D_train.shape):
                continue
            svd = TruncatedSVD(n_components=k, random_state=args.seed)
            codes_tr = svd.fit_transform(D_train)
            best_svd = None
            for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
                m = Ridge(alpha=alpha)
                m.fit(X_tr, codes_tr)
                va_codes = m.predict(X_va)
                va_preds = {g: svd.inverse_transform(va_codes[[i]])[0] for i, g in enumerate(val_t)}
                vr, _ = pearson_per_gene(va_preds, val_mean_deltas)
                if best_svd is None or vr > best_svd[0]:
                    best_svd = (vr, alpha, m)
            _, alpha, m = best_svd
            te_codes = m.predict(X_te)
            te_preds = {g: svd.inverse_transform(te_codes[[i]])[0] for i, g in enumerate(test_t)}
            tr_m, tr_med = pearson_per_gene(te_preds, test_mean_deltas)
            te_mae = float(np.mean([np.mean(np.abs(te_preds[g] - test_mean_deltas[g])) for g in test_t]))
            out[f"svd_k{k}"] = {"k": k, "alpha": alpha, "val_pearson": best_svd[0],
                                 "test_pearson_mean": tr_m, "test_pearson_median": tr_med, "test_mae": te_mae}
            print(f"  {name}+SVD_k={k}: test Pearson={tr_m:.4f} (median {tr_med:.4f})")
        return out

    print("\nEvaluating mean_std ...")
    ms_results = evaluate("mean_std", mean_std_table, 2)
    print("\nEvaluating Gene2Vec ...")
    g2v_results = evaluate("gene2vec", gene2vec_lookup, 200)
    print("\nEvaluating response_role ...")
    rr_results = evaluate("response_role", response_role_table, response_role_dim)

    # --- Save ---
    summary = {
        "dataset": "Replogle et al. 2022 K562 essential RAW SINGLE-CELL",
        "evaluation_metric": "delta Pearson (Pearson of predicted vs actual mean delta, per test gene, then averaged)",
        "control_matching": "batch-matched (per gem_group)",
        "cells_per_target_cap": "none (all available cells used)",
        "literature_reference_points": {
            "GEARS_unseen_K562": 0.076,
            "mean_training_delta_baseline": 0.267,
            "oracle_nn_retrieval": 0.592,
            "source": "Delta Gap paper arXiv 2611.10658; Ahlmann-Eltze NatMeth 2024",
        },
        "n_distinct_targets": len(all_target_genes),
        "n_train_targets": len(train_t), "n_val_targets": len(val_t), "n_test_targets": len(test_t),
        "n_gem_groups": len(ctrl_mean_by_gg),
        "n_ctrl_cells": len(ctrl_positions),
        "n_target_cells": len(all_target_positions),
        "fetch_time_seconds": fetch_time,
        "baselines": {
            "no_effect": {"test_pearson_mean": no_r_mean, "test_pearson_median": no_r_med, "test_mae": no_mae},
            "global_mean": {"test_pearson_mean": gm_r_mean, "test_pearson_median": gm_r_med, "test_mae": gm_mae},
        },
        "mean_std": ms_results,
        "gene2vec": g2v_results,
        "response_role": rr_results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

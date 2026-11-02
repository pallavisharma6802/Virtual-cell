"""Milestone 10 v2: gene-level evaluation with more cells per gene.

Key fix over v1: load more cells per target gene (cap 80 instead of 40)
to reduce noise in per-gene mean delta estimates. Also fixes the memory
bottleneck by selecting the top-1500-variable-gene columns WHILE THE DATA
IS STILL SPARSE (before densifying), so peak RAM stays ~3GB even with
~168k total cells instead of Milestone 8/9's 84k.

Motivation: our v1 global-mean baseline got delta Pearson=0.157, while
the literature reports 0.267 for the same trivial baseline on Replogle
K562. The gap is almost certainly due to noisy per-gene mean estimates
from too few cells (40 cap → SE ≈ 0.16σ). With 80 cells/gene, SE drops
to ~0.11σ — a meaningful noise reduction.

Outputs Pearson correlation (the GEARS-paper metric, also called "delta
Pearson" to distinguish from inflated global Pearson). Literature
reference points on Replogle K562 essential unseen perturbations:
  - GEARS (NatBiotech 2023): delta Pearson ≈ 0.076
  - Mean training delta (trivial baseline, Kernfeld et al.): ≈ 0.267
  - Oracle upper bound (NN retrieval on true deltas): ≈ 0.592
  (From: Delta Gap paper, arXiv 2611.10658; Ahlmann-Eltze NatMeth 2024)
"""
from __future__ import annotations

import argparse
import json
import time
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


def subsample_positions(gene_col, cells_per_target_cap, control_sample_size, seed):
    rng = np.random.default_rng(seed)
    targets_by_row = [None if g == "non-targeting" else g for g in gene_col]
    ctrl_positions = [i for i, t in enumerate(targets_by_row) if t is None]
    ctrl_sample = rng.choice(ctrl_positions, size=min(control_sample_size, len(ctrl_positions)), replace=False)
    by_target: dict[str, list[int]] = {}
    for i, t in enumerate(targets_by_row):
        if t is not None:
            by_target.setdefault(t, []).append(i)
    target_positions, kept_targets = [], []
    for t, positions in by_target.items():
        if len(positions) < MIN_CELLS_PER_TARGET:
            continue
        chosen = rng.choice(positions, size=min(cells_per_target_cap, len(positions)), replace=False)
        target_positions.extend(chosen.tolist())
        kept_targets.append(t)
    return sorted(ctrl_sample.tolist()), sorted(target_positions), sorted(kept_targets)


def load_rows_top_genes(adata, row_positions, top_gene_idx):
    """Load specific rows from backed h5ad, selecting gene columns WHILE
    STILL SPARSE to avoid a full n_cells x 8563 dense intermediate."""
    X_sparse = adata.X[row_positions]
    # column-select while sparse: much cheaper than densifying all 8563 cols
    X_sub_sparse = X_sparse[:, top_gene_idx]
    return np.asarray(
        X_sub_sparse.todense() if hasattr(X_sub_sparse, "todense") else X_sub_sparse,
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


def compute_mean_deltas(delta, targets_list, target_set, control_mean):
    result = {}
    for g in target_set:
        mask = [i for i, t in enumerate(targets_list) if t == g]
        if mask:
            result[g] = delta[mask].mean(axis=0) - control_mean
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad-path", default="data/replogle/K562_essential_raw_singlecell_01.h5ad")
    parser.add_argument("--gene2vec-path", default="data/gene2vec/gene2vec_dim_200_iter_9.txt")
    parser.add_argument("--output", default="outputs/replogle_genelevel_v2/metrics.json")
    parser.add_argument("--cells-per-target-cap", type=int, default=80)
    parser.add_argument("--control-sample-size", type=int, default=4000)
    parser.add_argument("--n-top-genes", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    t0 = time.time()
    print(f"Opening {args.h5ad_path} backed ...")
    adata = ad.read_h5ad(args.h5ad_path, backed="r")
    gene_col = adata.obs["gene"].astype(str).tolist()
    gene_names = adata.var["gene_name"].tolist() if "gene_name" in adata.var.columns else adata.var_names.tolist()
    print(f"Full file: {adata.shape[0]} cells x {adata.shape[1]} genes")
    print(f"Cells per target cap: {args.cells_per_target_cap}")

    # Also print obs columns to identify batch info
    print(f"obs columns: {list(adata.obs.columns)}")

    ctrl_pos, target_pos, _ = subsample_positions(
        gene_col, args.cells_per_target_cap, args.control_sample_size, args.seed,
    )
    cell_targets = [gene_col[p] for p in target_pos]
    unique_targets = sorted(set(cell_targets))
    print(f"Control cells: {len(ctrl_pos)}, target cells: {len(target_pos)} "
          f"across {len(unique_targets)} distinct targets")

    # Step 1: Load control cells (4000) to identify top-variable genes
    print("Loading control cells to identify top-variable genes ...")
    X_ctrl_full = adata.X[ctrl_pos]
    X_ctrl_full = np.asarray(
        X_ctrl_full.todense() if hasattr(X_ctrl_full, "todense") else X_ctrl_full,
        dtype=np.float32,
    )  # (4000, 8563) = 137 MB — fine
    ctrl_var = X_ctrl_full.var(axis=0)
    top_gene_idx = np.sort(np.argsort(ctrl_var)[::-1][: args.n_top_genes])
    top_gene_names = [gene_names[i] for i in top_gene_idx]
    X_ctrl = X_ctrl_full[:, top_gene_idx]  # (4000, 1500)
    control_mean = X_ctrl.mean(axis=0)
    del X_ctrl_full
    print(f"Top {args.n_top_genes} genes identified.")

    # Step 2: Load target cells — select top genes while sparse
    print(f"Loading {len(target_pos)} target cells (selecting 1500 columns while sparse) ...")
    X_target = load_rows_top_genes(adata, target_pos, top_gene_idx)  # (n_target, 1500)
    fetch_time = time.time() - t0
    print(f"Loaded in {fetch_time:.1f}s. Memory: target {X_target.nbytes/1e9:.2f}GB, ctrl {X_ctrl.nbytes/1e9:.2f}GB")

    # Split targets
    train_t, test_t_full = train_test_split(unique_targets, test_size=0.3, random_state=args.seed)
    val_t, test_t = train_test_split(test_t_full, test_size=0.5, random_state=args.seed)
    print(f"Split: {len(train_t)} train / {len(val_t)} val / {len(test_t)} test targets")

    def delta_for_split(t_set):
        idx = [i for i, t in enumerate(cell_targets) if t in t_set]
        split_targets = [cell_targets[i] for i in idx]
        split_X = X_target[idx]  # (n_cells_in_split, 1500)
        mean_deltas = {}
        for g in t_set:
            mask = [j for j, t in enumerate(split_targets) if t == g]
            if mask:
                mean_deltas[g] = split_X[mask].mean(axis=0) - control_mean
        return mean_deltas

    # Compute per-gene mean deltas (ground truth for each split)
    train_mean_deltas = delta_for_split(set(train_t))
    val_mean_deltas = delta_for_split(set(val_t))
    test_mean_deltas = delta_for_split(set(test_t))

    # Training matrix: (n_train_genes, 1500)
    train_gene_list = sorted(train_mean_deltas.keys())
    D_train = np.stack([train_mean_deltas[g] for g in train_gene_list])
    n_genes_out = D_train.shape[1]
    print(f"Training perturbation matrix: {D_train.shape}")

    # Also need per-cell training data for response-role descriptor
    train_idx = [i for i, t in enumerate(cell_targets) if t in set(train_t)]
    train_targets_list = [cell_targets[i] for i in train_idx]
    train_cell_delta = X_target[train_idx] - control_mean

    # ------------------------------------------------------------------ #
    # Descriptors
    # ------------------------------------------------------------------ #
    print("Loading Gene2Vec ...")
    gene2vec_table = load_gene2vec_table(Path(args.gene2vec_path))
    gene2vec_lookup = {g: gene2vec_table[g] for g in top_gene_names if g in gene2vec_table}

    mean_std_table = {
        gene: np.array([control_mean[i], X_ctrl[:, i].std()], dtype=np.float32)
        for i, gene in enumerate(top_gene_names)
    }
    response_role_dim = min(30, len(train_gene_list), D_train.shape[1])
    response_role_table = fit_response_role_descriptors(
        train_targets_list, train_cell_delta, top_gene_names,
        n_components=response_role_dim, seed=args.seed,
    )

    # ------------------------------------------------------------------ #
    # Baselines
    # ------------------------------------------------------------------ #
    # No-effect
    zero_preds = {g: np.zeros(n_genes_out, dtype=np.float32) for g in test_t}
    no_r_mean, no_r_med = pearson_per_gene(zero_preds, test_mean_deltas)
    no_mae = float(np.mean([np.mean(np.abs(test_mean_deltas[g])) for g in test_t]))

    # Global mean training delta
    global_mean = D_train.mean(axis=0)
    gm_preds = {g: global_mean for g in test_t}
    gm_r_mean, gm_r_med = pearson_per_gene(gm_preds, test_mean_deltas)
    gm_mae = float(np.mean([np.mean(np.abs(global_mean - test_mean_deltas[g])) for g in test_t]))

    print(f"No-effect:   Pearson mean={no_r_mean:.4f}, median={no_r_med:.4f}, MAE={no_mae:.4f}")
    print(f"Global mean: Pearson mean={gm_r_mean:.4f}, median={gm_r_med:.4f}, MAE={gm_mae:.4f}")

    # ------------------------------------------------------------------ #
    # Model evaluation
    # ------------------------------------------------------------------ #
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
        print(f"  {name}+ridge: alpha={alpha}, test Pearson={tr_m:.4f} (median {tr_med:.4f}), MAE={te_mae:.4f}")

        # SVD
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
            print(f"  {name}+SVD_k={k}: alpha={alpha}, test Pearson={tr_m:.4f} (median {tr_med:.4f})")
        return out

    print("\nEvaluating mean_std ...")
    ms_results = evaluate("mean_std", mean_std_table, 2)
    print("\nEvaluating Gene2Vec ...")
    g2v_results = evaluate("gene2vec", gene2vec_lookup, 200)
    print("\nEvaluating response_role ...")
    rr_results = evaluate("response_role", response_role_table, response_role_dim)

    summary = {
        "dataset": "Replogle et al. 2022 K562 essential RAW SINGLE-CELL",
        "evaluation_metric": "delta Pearson (Pearson of predicted vs actual mean delta, per test gene, then averaged)",
        "literature_reference_points": {
            "GEARS_unseen_K562": 0.076,
            "mean_training_delta_baseline": 0.267,
            "oracle_nn_retrieval": 0.592,
            "source": "Delta Gap paper arXiv 2611.10658; Ahlmann-Eltze NatMeth 2024",
        },
        "n_distinct_targets": len(unique_targets),
        "n_train_targets": len(train_t), "n_val_targets": len(val_t), "n_test_targets": len(test_t),
        "cells_per_target_cap": args.cells_per_target_cap,
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

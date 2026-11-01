"""Milestone 8: the per-cell, many-target experiment Milestone 6/7's
caveat called for. Replogle et al. 2022 (Cell) K562 "essential"
genome-scale Perturb-seq, RAW SINGLE-CELL release this time (not
pseudobulk) -- ~2,000 distinct targets (like Milestone 6/7) AND real
per-cell control variation (unlike Milestone 6/7, which only had one
constant global control vector). This is the first experiment in the
project with both properties at once, so it is the cleanest test yet of
whether more distinct targets fixes the architecture gap, and whether
per-cell control signal recovers the better relative improvements the
Arc pilots got.

Source: same Figshare+ deposit as Milestone 6/7 (doi
10.25452/figshare.plus.20029387), file
K562_essential_raw_singlecell_01.h5ad (~10.6GB, CC BY-NC-SA 4.0,
non-commercial research use). Downloaded once to EC2 local disk (not
read via scattered remote access -- the scattered-read lesson from the
Arc pilots applies here too).

Memory-bounded like scale_up_real_arc.py: open backed, two-pass
(1) sample row positions per target + controls, capped, (2) fetch only
those rows, reduce to top-variable genes, assemble a dense in-memory
array. No GCS concurrency machinery needed here (local disk, not remote
HTTP range requests), so this is simpler than the Arc scale-up script.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.model_selection import train_test_split

from virtualcell.data.target_descriptors import (
    descriptor_for_targets,
    fit_gene2vec_descriptors,
    fit_response_role_descriptors,
    load_gene2vec_table,
)
from virtualcell.evaluation.metrics import mae
from virtualcell.models.baselines import NoEffectBaseline, PCARidgeBaseline, SmallMLPBaseline

MIN_CELLS_PER_TARGET = 5


def subsample_positions(gene_col: list[str], cells_per_target_cap: int, control_sample_size: int, seed: int):
    """`gene_col` is this file's own `obs['gene']` column: 'non-targeting'
    for control cells, a gene symbol otherwise -- a real obs column here,
    unlike the pseudobulk release's index-embedded format."""
    rng = np.random.default_rng(seed)
    targets_by_row = [None if g == "non-targeting" else g for g in gene_col]

    ctrl_positions = [i for i, t in enumerate(targets_by_row) if t is None]
    ctrl_sample = rng.choice(ctrl_positions, size=min(control_sample_size, len(ctrl_positions)), replace=False)

    by_target: dict[str, list[int]] = {}
    for i, t in enumerate(targets_by_row):
        if t is not None:
            by_target.setdefault(t, []).append(i)

    target_positions = []
    kept_targets = []
    for t, positions in by_target.items():
        if len(positions) < MIN_CELLS_PER_TARGET:
            continue
        chosen = rng.choice(positions, size=min(cells_per_target_cap, len(positions)), replace=False)
        target_positions.extend(chosen.tolist())
        kept_targets.append(t)

    return sorted(ctrl_sample.tolist()), sorted(target_positions), sorted(kept_targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad-path", default="data/replogle/K562_essential_raw_singlecell_01.h5ad")
    parser.add_argument("--gene2vec-path", default="data/gene2vec/gene2vec_dim_200_iter_9.txt")
    parser.add_argument("--output", default="outputs/replogle_percell_scale/metrics.json")
    parser.add_argument("--cells-per-target-cap", type=int, default=40)
    parser.add_argument("--control-sample-size", type=int, default=4000)
    parser.add_argument("--n-top-genes", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    t0 = time.time()
    print(f"Opening {args.h5ad_path} backed (memory-mapped) ...")
    adata = ad.read_h5ad(args.h5ad_path, backed="r")
    gene_col = adata.obs["gene"].astype(str).tolist()
    gene_names = adata.var["gene_name"].tolist() if "gene_name" in adata.var.columns else adata.var_names.tolist()
    print(f"Full file: {adata.shape[0]} cells x {adata.shape[1]} genes")

    print("Pass 1/2: selecting row positions to sample ...")
    ctrl_pos, target_pos, kept_targets = subsample_positions(
        gene_col, args.cells_per_target_cap, args.control_sample_size, args.seed,
    )
    print(f"Selected {len(ctrl_pos)} control cells, {len(target_pos)} perturbation cells across "
          f"{len(kept_targets)} distinct targets (>= {MIN_CELLS_PER_TARGET} cells each)")

    all_pos = sorted(set(ctrl_pos) | set(target_pos))
    print(f"Pass 2/2: fetching {len(all_pos)} rows from local disk ...")
    X_full = adata.X[all_pos]
    X_full = np.asarray(X_full.todense() if hasattr(X_full, "todense") else X_full, dtype=np.float32)
    pos_to_row = {p: i for i, p in enumerate(all_pos)}
    fetch_time = time.time() - t0
    print(f"Fetched in {fetch_time:.1f}s")

    ctrl_rows = [pos_to_row[p] for p in ctrl_pos]
    target_rows = [pos_to_row[p] for p in target_pos]
    cell_targets = [gene_col[p] for p in target_pos]

    ctrl_var = X_full[ctrl_rows].var(axis=0)
    top_gene_idx = np.argsort(ctrl_var)[::-1][: args.n_top_genes]
    top_gene_idx = np.sort(top_gene_idx)
    top_gene_names = [gene_names[i] for i in top_gene_idx]

    X_ctrl = X_full[ctrl_rows][:, top_gene_idx]
    X_target = X_full[target_rows][:, top_gene_idx]
    control_mean = X_ctrl.mean(axis=0)

    unique_targets = sorted(set(cell_targets))
    train_t, test_t = train_test_split(unique_targets, test_size=0.3, random_state=args.seed)
    val_t, test_t = train_test_split(test_t, test_size=0.5, random_state=args.seed)
    print(f"Held-out-target split: {len(train_t)} train / {len(val_t)} val / {len(test_t)} test targets")

    def rows_for(target_set):
        return [i for i, t in enumerate(cell_targets) if t in target_set]

    train_idx, val_idx, test_idx = rows_for(set(train_t)), rows_for(set(val_t)), rows_for(set(test_t))

    rng = np.random.default_rng(args.seed)

    def matched_control(n, offset_seed):
        local_rng = np.random.default_rng(args.seed + offset_seed)
        idx = local_rng.integers(0, len(X_ctrl), size=n)
        return X_ctrl[idx]

    train_ctrl = matched_control(len(train_idx), 1)
    val_ctrl = matched_control(len(val_idx), 2)
    test_ctrl = matched_control(len(test_idx), 3)

    train_delta = X_target[train_idx] - train_ctrl
    val_delta = X_target[val_idx] - val_ctrl
    test_delta = X_target[test_idx] - test_ctrl

    train_targets_list = [cell_targets[i] for i in train_idx]
    val_targets_list = [cell_targets[i] for i in val_idx]
    test_targets_list = [cell_targets[i] for i in test_idx]

    print("Loading Gene2Vec table ...")
    gene2vec_table = load_gene2vec_table(Path(args.gene2vec_path))

    mean_std_table = {
        gene: np.array([control_mean[i], X_ctrl[:, i].std()], dtype=np.float32)
        for i, gene in enumerate(top_gene_names)
    }
    gene2vec_lookup = {g: gene2vec_table[g] for g in top_gene_names if g in gene2vec_table}
    gene2vec_coverage = len(gene2vec_lookup) / len(top_gene_names)
    print(f"Gene2Vec coverage: {gene2vec_coverage:.1%} of {len(top_gene_names)} top-variable genes")

    response_role_dim = min(30, len(set(train_targets_list)), train_delta.shape[1])
    response_role_table = fit_response_role_descriptors(
        train_targets_list, train_delta, top_gene_names, n_components=response_role_dim, seed=args.seed,
    )
    response_role_coverage = sum(1 for t in unique_targets if t in response_role_table) / len(unique_targets)
    print(f"Response-role descriptor coverage: {response_role_coverage:.1%} ({response_role_dim} components)")

    def run_ladder(name, train_desc, val_desc, test_desc):
        no_effect = NoEffectBaseline(n_genes=test_delta.shape[1])
        no_effect_mae = mae(test_delta, no_effect.predict_delta(len(test_delta)))

        best_ridge, best_ridge_val = None, np.inf
        for alpha in (0.1, 1.0, 10.0, 100.0):
            n_comp = min(20, train_ctrl.shape[0], train_ctrl.shape[1])
            m = PCARidgeBaseline(n_components=n_comp, alpha=alpha)
            m.fit(train_ctrl, train_desc, train_delta)
            v = mae(val_delta, m.predict_delta(val_ctrl, val_desc))
            if v < best_ridge_val:
                best_ridge_val, best_ridge = v, (m, alpha)
        ridge_model, ridge_alpha = best_ridge
        ridge_test_mae = mae(test_delta, ridge_model.predict_delta(test_ctrl, test_desc))

        best_mlp, best_mlp_val = None, np.inf
        for hidden_dim in (32, 64, 128, 256):
            m = SmallMLPBaseline(hidden_dim=hidden_dim, seed=0, max_iter=300)
            m.fit(train_ctrl, train_desc, train_delta)
            v = mae(val_delta, m.predict_delta(val_ctrl, val_desc))
            if v < best_mlp_val:
                best_mlp_val, best_mlp = v, (m, hidden_dim)
        mlp_model, mlp_hidden = best_mlp
        mlp_test_mae = mae(test_delta, mlp_model.predict_delta(test_ctrl, test_desc))

        return {
            "descriptor": name,
            "no_effect_mae": float(no_effect_mae),
            "pca_ridge": {"best_alpha": ridge_alpha, "val_mae": float(best_ridge_val), "test_mae": float(ridge_test_mae)},
            "small_mlp": {"best_hidden_dim": mlp_hidden, "val_mae": float(best_mlp_val), "test_mae": float(mlp_test_mae)},
        }

    results = {}
    for desc_name, table, dim in (
        ("mean_std", mean_std_table, 2),
        ("gene2vec", gene2vec_lookup, 200),
        ("response_role", response_role_table, response_role_dim),
    ):
        train_desc = descriptor_for_targets(train_targets_list, table, dim)
        val_desc = descriptor_for_targets(val_targets_list, table, dim)
        test_desc = descriptor_for_targets(test_targets_list, table, dim)
        results[desc_name] = run_ladder(desc_name, train_desc, val_desc, test_desc)
        print(json.dumps(results[desc_name], indent=2))

    summary = {
        "dataset": "Replogle et al. 2022 Cell, K562 essential Perturb-seq RAW SINGLE-CELL "
                   "(CC BY-NC-SA 4.0, https://doi.org/10.25452/figshare.plus.20029387)",
        "n_cells_sampled": len(all_pos),
        "n_control_cells": len(ctrl_pos),
        "n_target_cells": len(target_pos),
        "n_genes_profiled_full": int(adata.shape[1]),
        "n_top_genes_used": args.n_top_genes,
        "n_distinct_targets": len(unique_targets),
        "n_train_targets": len(train_t), "n_val_targets": len(val_t), "n_test_targets": len(test_t),
        "fetch_time_seconds": fetch_time,
        "gene2vec_coverage": gene2vec_coverage,
        "response_role_coverage": response_role_coverage,
        "response_role_dim": response_role_dim,
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

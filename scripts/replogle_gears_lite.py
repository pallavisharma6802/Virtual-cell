"""Milestone 9: GEARS-lite -- test the one mechanistically different
hypothesis identified after Milestones 1-8: that a gene's perturbation
effect can be predicted by borrowing from its functionally similar
neighbors (GO Biological Process overlap), rather than by describing the
gene's own identity as a flat feature vector fed into a generic
regressor (mean/std, Gene2Vec, response-role -- all tried, all tied).

This is a lite reimplementation of the core GEARS (Roohani et al. 2023)
idea -- GO-similarity-based effect propagation -- not the official
package (which needs torch_geometric and is impractical on a CPU-only
free-tier EC2 box). The predictor here is direct effect retrieval: for
each held-out target gene, find its k nearest neighbors among TRAIN
targets by GO-BP Jaccard similarity, and predict its effect as a
similarity-weighted average of those neighbors' observed mean deltas.
No neural net, no generic per-gene descriptor -- the graph does the
work instead of a feature vector.

Reuses the exact same per-cell sampling/split pipeline as Milestone 8
(replogle_percell_scale.py) so results are directly comparable to the
mean_std / gene2vec / response_role results already on record.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.model_selection import train_test_split

from go_similarity import load_go_bp_annotations, top_k_neighbors
from virtualcell.evaluation.metrics import mae
from virtualcell.models.baselines import NoEffectBaseline

MIN_CELLS_PER_TARGET = 5


def subsample_positions(gene_col: list[str], cells_per_target_cap: int, control_sample_size: int, seed: int):
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


def predict_via_go_retrieval(
    query_targets: list[str],
    train_mean_delta: dict[str, np.ndarray],
    train_target_list: list[str],
    gene_to_go: dict[str, set[str]],
    k: int,
    n_genes: int,
) -> np.ndarray:
    """One prediction vector per query target gene, retrieved from its
    k nearest GO-similar TRAIN targets. Falls back to the global
    train-average delta (closest thing to a no-effect-ish prior) when a
    query gene has no GO annotation or no GO-similar train neighbor."""
    global_mean = np.mean(list(train_mean_delta.values()), axis=0) if train_mean_delta else np.zeros(n_genes, dtype=np.float32)
    preds = np.zeros((len(query_targets), n_genes), dtype=np.float32)
    for i, gene in enumerate(query_targets):
        neighbors = top_k_neighbors(gene, train_target_list, gene_to_go, k)
        if not neighbors:
            preds[i] = global_mean
            continue
        weights = np.array([sim for _, sim in neighbors], dtype=np.float32)
        weights /= weights.sum()
        vecs = np.stack([train_mean_delta[g] for g, _ in neighbors])
        preds[i] = (weights[:, None] * vecs).sum(axis=0)
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad-path", default="data/replogle/K562_essential_raw_singlecell_01.h5ad")
    parser.add_argument("--gaf-path", default="data/go/goa_human.gaf.gz")
    parser.add_argument("--output", default="outputs/replogle_gears_lite/metrics.json")
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

    unique_targets = sorted(set(cell_targets))
    train_t, test_t = train_test_split(unique_targets, test_size=0.3, random_state=args.seed)
    val_t, test_t = train_test_split(test_t, test_size=0.5, random_state=args.seed)
    print(f"Held-out-target split: {len(train_t)} train / {len(val_t)} val / {len(test_t)} test targets")

    def rows_for(target_set):
        return [i for i, t in enumerate(cell_targets) if t in target_set]

    train_idx, val_idx, test_idx = rows_for(set(train_t)), rows_for(set(val_t)), rows_for(set(test_t))

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

    print(f"Loading GO-BP annotations from {args.gaf_path} ...")
    gene_to_go = load_go_bp_annotations(args.gaf_path)
    go_coverage = sum(1 for t in unique_targets if t in gene_to_go and gene_to_go[t]) / len(unique_targets)
    print(f"GO-BP annotation coverage: {go_coverage:.1%} of {len(unique_targets)} distinct targets")

    train_mean_delta = {}
    for t in set(train_targets_list):
        mask = [i for i, g in enumerate(train_targets_list) if g == t]
        train_mean_delta[t] = train_delta[mask].mean(axis=0)

    no_effect = NoEffectBaseline(n_genes=test_delta.shape[1])
    no_effect_mae = mae(test_delta, no_effect.predict_delta(len(test_delta)))

    best_k, best_val_mae = None, np.inf
    val_maes_by_k = {}
    for k in (1, 3, 5, 10, 20, 50):
        val_target_preds = predict_via_go_retrieval(
            val_t, train_mean_delta, list(set(train_targets_list)), gene_to_go, k, val_delta.shape[1],
        )
        target_to_pred = dict(zip(val_t, val_target_preds))
        val_preds = np.stack([target_to_pred[g] for g in val_targets_list])
        v = mae(val_delta, val_preds)
        val_maes_by_k[k] = float(v)
        if v < best_val_mae:
            best_val_mae, best_k = v, k

    test_target_preds = predict_via_go_retrieval(
        test_t, train_mean_delta, list(set(train_targets_list)), gene_to_go, best_k, test_delta.shape[1],
    )
    target_to_pred = dict(zip(test_t, test_target_preds))
    test_preds = np.stack([target_to_pred[g] for g in test_targets_list])
    test_go_mae = mae(test_delta, test_preds)

    result = {
        "descriptor": "go_similarity_retrieval",
        "no_effect_mae": float(no_effect_mae),
        "go_retrieval": {
            "best_k": best_k,
            "val_maes_by_k": val_maes_by_k,
            "val_mae": float(best_val_mae),
            "test_mae": float(test_go_mae),
        },
        "go_bp_coverage": go_coverage,
    }
    print(json.dumps(result, indent=2))

    summary = {
        "dataset": "Replogle et al. 2022 Cell, K562 essential Perturb-seq RAW SINGLE-CELL "
                   "(CC BY-NC-SA 4.0, https://doi.org/10.25452/figshare.plus.20029387)",
        "approach": "GEARS-lite: GO-BP Jaccard similarity-weighted kNN effect retrieval from train "
                    "targets, in place of a generic descriptor fed into a regressor",
        "n_cells_sampled": len(all_pos),
        "n_control_cells": len(ctrl_pos),
        "n_target_cells": len(target_pos),
        "n_genes_profiled_full": int(adata.shape[1]),
        "n_top_genes_used": args.n_top_genes,
        "n_distinct_targets": len(unique_targets),
        "n_train_targets": len(train_t), "n_val_targets": len(val_t), "n_test_targets": len(test_t),
        "fetch_time_seconds": fetch_time,
        "result": result,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

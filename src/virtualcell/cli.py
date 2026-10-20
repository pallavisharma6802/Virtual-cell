"""CLI entry point for the Milestone 1 CPU smoke test.

Generates the synthetic fixture, builds a held-out-target split, fits the
no-effect baseline, evaluates it, and writes metrics to outputs/RUN_ID.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from virtualcell.data.adapters import control_reference_mean, from_anndata
from virtualcell.data.fixtures import make_synthetic_dataset
from virtualcell.data.splits import held_out_target_split
from virtualcell.evaluation.metrics import (
    grouped_bootstrap_ci,
    improvement_over_no_effect,
    mae,
    per_target_metrics,
    rmse,
)
from virtualcell.models.baselines import NoEffectBaseline, SeenTargetMeanBaseline


def run_smoke_test(output_dir: Path, seed: int = 0) -> dict:
    adata = make_synthetic_dataset(seed=seed)
    dataset = from_anndata(adata, source_name="synthetic_fixture", is_synthetic=True)
    manifest = held_out_target_split(dataset, seed=seed)

    obs = dataset.adata.obs
    test_mask = obs.index.isin(manifest.test_cell_ids)
    ctrl_mask = dataset.control_mask()
    test_pert_mask = test_mask & (~ctrl_mask)

    ctrl_ref = control_reference_mean(dataset)
    global_control_mean = ctrl_ref["__global__"]

    X = np.asarray(dataset.adata.X)
    y_true = X[test_pert_mask] - global_control_mean
    test_targets = obs.loc[test_pert_mask, "perturbation_id"].tolist()

    no_effect = NoEffectBaseline(n_genes=dataset.n_genes)
    y_pred_no_effect = no_effect.predict_delta(n_cells=y_true.shape[0])

    seen_mean_baseline = SeenTargetMeanBaseline().fit(dataset, manifest)
    y_pred_seen_mean = seen_mean_baseline.predict_delta(test_targets)

    no_effect_mae = mae(y_true, y_pred_no_effect)
    no_effect_rmse = rmse(y_true, y_pred_no_effect)
    seen_mean_mae = mae(y_true, y_pred_seen_mean)
    seen_mean_rmse = rmse(y_true, y_pred_seen_mean)

    per_target_no_effect = per_target_metrics(y_true, y_pred_no_effect, test_targets)
    per_target_mae_by_target = {t: m["mae"] for t, m in per_target_no_effect.items()}
    ci = grouped_bootstrap_ci(per_target_mae_by_target, seed=seed)

    metrics = {
        "run_id": f"smoke_{int(time.time())}",
        "seed": seed,
        "dataset": {
            "source": dataset.provenance.source_name,
            "is_synthetic": dataset.provenance.is_synthetic,
            "n_cells": dataset.n_cells,
            "n_genes": dataset.n_genes,
            "n_targets_total": len(dataset.targets()),
        },
        "split": {
            "name": manifest.name,
            "n_train_targets": len(manifest.train_targets),
            "n_val_targets": len(manifest.val_targets),
            "n_test_targets": len(manifest.test_targets),
        },
        "baselines": {
            "no_effect": {"mae": no_effect_mae, "rmse": no_effect_rmse},
            "seen_target_mean_on_held_out_targets": {
                "mae": seen_mean_mae,
                "rmse": seen_mean_rmse,
                "note": (
                    "This baseline cannot represent held-out targets; its "
                    "predictions for unseen targets fall back to zero effect, "
                    "so comparing it here is a known-limitation sanity check, "
                    "not a valid held-out-target benchmark."
                ),
            },
            "no_effect_per_target_mae_bootstrap_ci": ci,
        },
        "improvement_seen_mean_over_no_effect": improvement_over_no_effect(
            seen_mean_mae, no_effect_mae
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="VirtualCell Milestone 1 CPU smoke test")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/milestone1_smoke"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    metrics = run_smoke_test(args.output_dir, seed=args.seed)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

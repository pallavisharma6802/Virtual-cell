"""Milestone 4: research-grade evaluation of the conditional model (the
"full" variant from Milestone 3) on the synthetic fixture's held-out-target
test split. Produces per-target metrics, grouped bootstrap CIs, per-batch
breakdown, DE-gene overlap, worst-case targets, and figures.

Still synthetic-fixture only — see docs/experiment_log.md / results.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from virtualcell.data.adapters import from_anndata
from virtualcell.data.control_sampling import sample_matched_controls
from virtualcell.data.fixtures import make_synthetic_dataset
from virtualcell.data.preprocess import fit_scaler
from virtualcell.data.splits import held_out_target_split
from virtualcell.data.target_descriptors import descriptor_for_targets, fit_control_mean_std_descriptors
from virtualcell.evaluation.report import build_full_report, de_overlap_by_target
from virtualcell.models.conditional import ConditionalModelConfig
from virtualcell.training.trainer import Trainer


def _cell_positions(obs, cell_ids):
    index_list = list(obs.index)
    return [index_list.index(c) for c in cell_ids]


def run(checkpoint_dir: Path, output_dir: Path, seed: int = 0) -> dict:
    adata = make_synthetic_dataset(seed=seed)
    dataset = from_anndata(adata, source_name="synthetic_fixture", is_synthetic=True)
    manifest = held_out_target_split(dataset, seed=seed)
    obs = dataset.adata.obs
    ctrl_mask = dataset.control_mask()

    scaler = fit_scaler(dataset, manifest)
    gene_descriptors = fit_control_mean_std_descriptors(dataset, manifest)

    test_mask = obs.index.isin(manifest.test_cell_ids) & (~ctrl_mask)
    test_cell_ids = list(obs.index[test_mask])
    test_positions = _cell_positions(obs, test_cell_ids)

    X = np.asarray(dataset.adata.X)
    X_test_raw = X[test_positions]
    X_test_scaled = scaler.transform(X_test_raw)

    control_pool = [
        c for c in manifest.train_cell_ids if ctrl_mask[list(obs.index).index(c)]
    ]
    matched_control_raw = sample_matched_controls(dataset, test_cell_ids, control_pool, seed=seed)
    matched_control_scaled = scaler.transform(matched_control_raw)

    true_delta_scaled = X_test_scaled - matched_control_scaled
    test_targets = obs.loc[test_cell_ids, "perturbation_id"].tolist()
    test_batches = obs.loc[test_cell_ids, "batch_id"].tolist()
    target_desc = descriptor_for_targets(test_targets, gene_descriptors)

    trainer = Trainer.load(checkpoint_dir)
    pred_delta_scaled = trainer.predict_delta(matched_control_scaled, target_desc)

    report = build_full_report(true_delta_scaled, pred_delta_scaled, test_targets, test_batches)

    control_pool_positions = _cell_positions(obs, control_pool)
    control_raw_pool = X[control_pool_positions]
    de_report = de_overlap_by_target(
        X_test_raw, control_raw_pool, test_targets, true_delta_scaled, pred_delta_scaled, k=20
    )
    report["de_overlap_top20"] = de_report
    report["n_test_cells"] = len(test_cell_ids)
    report["n_test_targets"] = len(set(test_targets))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2, default=str))

    # Observed-vs-predicted figure for the single worst target.
    worst_target = report["worst_3_targets_by_mae"][0]["target"]
    mask = np.asarray(test_targets) == worst_target
    true_mean = true_delta_scaled[mask].mean(axis=0)
    pred_mean = pred_delta_scaled[mask].mean(axis=0)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true_mean, pred_mean, s=8, alpha=0.6)
    lim = max(np.abs(true_mean).max(), np.abs(pred_mean).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8)
    ax.set_xlabel("True mean delta (scaled)")
    ax.set_ylabel("Predicted mean delta (scaled)")
    ax.set_title(f"Observed vs predicted delta — worst target: {worst_target}")
    fig.tight_layout()
    fig.savefig(output_dir / "observed_vs_predicted_worst_target.png", dpi=150)
    plt.close(fig)

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/conditional_mlp/full"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation_milestone4"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = run(args.checkpoint_dir, args.output_dir, seed=args.seed)
    print(json.dumps({k: v for k, v in report.items() if k != "per_target"}, indent=2, default=str))


if __name__ == "__main__":
    main()

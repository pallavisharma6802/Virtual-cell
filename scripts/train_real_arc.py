"""Real-data pilot run: conditional model on a subsample of the Arc
Virtual Cell Challenge 2025 dataset (verified in Milestone 2).

This is explicitly a SUBSAMPLED pilot, not the full 221,273-cell / 18,080
gene dataset: running on constrained EC2 memory (see docs/experiment_log.md
Milestone 5 entry for the exact instance type and why). To keep this
feasible:
  - a fixed number of cells is sampled per target (documented below),
  - genes are restricted to the top-K by control-population variance,
    computed on TRAIN controls only (non-leaking, same principle as
    fit_scaler/fit_control_mean_std_descriptors).

Report any numbers from this script as a small-scale real-data pilot, not
as a claim about full-dataset performance.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from virtualcell.data.arc_adapter import ARC_CHALLENGE_TRAIN_URL, load_arc_challenge_dataset
from virtualcell.data.adapters import CanonicalDataset, DatasetProvenance, control_reference_mean
from virtualcell.data.control_sampling import sample_matched_controls
from virtualcell.data.preprocess import fit_scaler
from virtualcell.data.splits import held_out_target_split
from virtualcell.data.target_descriptors import descriptor_for_targets, fit_control_mean_std_descriptors
from virtualcell.evaluation.metrics import improvement_over_no_effect, mae, rmse
from virtualcell.evaluation.report import build_full_report
from virtualcell.models.conditional import ConditionalModelConfig
from virtualcell.training.trainer import TrainConfig, Trainer

CELLS_PER_TARGET_CAP = 60
N_TOP_VARIABLE_GENES = 1000
MIN_CELLS_PER_TARGET = 10


def subsample_and_materialize(dataset: CanonicalDataset, seed: int) -> CanonicalDataset:
    """Load a manageable in-memory subset: capped cells/target + top-K
    variable genes by control variance (fit on controls only, before any
    split, so it is a dataset-level dimensionality reduction decision,
    not a leak of held-out-target response signal)."""
    rng = np.random.default_rng(seed)
    obs = dataset.adata.obs
    ctrl_mask = dataset.control_mask()

    # Cap cells per target, keep targets with enough cells for a test.
    keep_cell_positions = []
    for target in dataset.targets():
        positions = np.where((obs["perturbation_id"] == target).to_numpy())[0]
        if len(positions) < MIN_CELLS_PER_TARGET:
            continue
        chosen = rng.choice(positions, size=min(CELLS_PER_TARGET_CAP, len(positions)), replace=False)
        keep_cell_positions.extend(chosen.tolist())

    ctrl_positions = np.where(ctrl_mask)[0]
    ctrl_sample = rng.choice(ctrl_positions, size=min(2000, len(ctrl_positions)), replace=False)
    keep_cell_positions.extend(ctrl_sample.tolist())

    keep_cell_positions = sorted(set(keep_cell_positions))
    print(f"Materializing {len(keep_cell_positions)} cells from the backed h5ad (this reads from GCS)...")
    t0 = time.time()
    sub = dataset.adata[keep_cell_positions, :].to_memory()
    print(f"  done in {time.time() - t0:.1f}s")

    X_dense = np.asarray(sub.X.todense()) if hasattr(sub.X, "todense") else np.asarray(sub.X)
    sub_ctrl_mask = sub.obs["is_control"].to_numpy()
    gene_var = X_dense[sub_ctrl_mask].var(axis=0)
    top_gene_idx = np.sort(np.argsort(-gene_var)[:N_TOP_VARIABLE_GENES])

    sub = sub[:, top_gene_idx].copy()
    sub.X = X_dense[:, top_gene_idx]

    return CanonicalDataset(
        adata=sub,
        gene_ids=list(sub.var.index),
        provenance=DatasetProvenance(
            source_name=dataset.provenance.source_name,
            source_version=dataset.provenance.source_version,
            source_url=dataset.provenance.source_url,
            is_synthetic=False,
            notes=(
                dataset.provenance.notes
                + f" | SUBSAMPLED pilot: <=~{CELLS_PER_TARGET_CAP} cells/target, "
                f"{N_TOP_VARIABLE_GENES} top-variable genes (by train-control variance)."
            ),
        ),
    )


def run(output_dir: Path, seed: int = 0) -> dict:
    print("Loading Arc Virtual Cell Challenge dataset (backed mode)...")
    full_dataset = load_arc_challenge_dataset(ARC_CHALLENGE_TRAIN_URL, backed="r")
    dataset = subsample_and_materialize(full_dataset, seed=seed)
    print(f"Pilot dataset: {dataset.n_cells} cells x {dataset.n_genes} genes, "
          f"{len(dataset.targets())} targets")

    manifest = held_out_target_split(dataset, seed=seed)
    scaler = fit_scaler(dataset, manifest)
    gene_descriptors = fit_control_mean_std_descriptors(dataset, manifest)

    obs = dataset.adata.obs
    ctrl_mask = dataset.control_mask()
    X = np.asarray(dataset.adata.X)

    def build(cell_ids):
        pert_mask = obs.index.isin(cell_ids) & (~ctrl_mask)
        pert_cell_ids = list(obs.index[pert_mask])
        if not pert_cell_ids:
            return (np.zeros((0, dataset.n_genes), dtype=np.float32),
                     np.zeros((0, 2), dtype=np.float32),
                     np.zeros((0, dataset.n_genes), dtype=np.float32), [])
        index_list = list(obs.index)
        positions = [index_list.index(c) for c in pert_cell_ids]
        X_scaled = scaler.transform(X[positions])

        control_pool = [c for c in manifest.train_cell_ids if ctrl_mask[index_list.index(c)]]
        matched_raw = sample_matched_controls(dataset, pert_cell_ids, control_pool, seed=seed)
        matched_scaled = scaler.transform(matched_raw)

        delta = X_scaled - matched_scaled
        targets = obs.loc[pert_cell_ids, "perturbation_id"].tolist()
        desc = descriptor_for_targets(targets, gene_descriptors)
        return matched_scaled, desc, delta, targets

    train_ctrl, train_desc, train_delta, _ = build(manifest.train_cell_ids)
    val_ctrl, val_desc, val_delta, _ = build(manifest.val_cell_ids)
    test_ctrl, test_desc, test_delta, test_targets = build(manifest.test_cell_ids)

    model_config = ConditionalModelConfig(n_genes=dataset.n_genes, target_descriptor_dim=2)
    train_config = TrainConfig(seed=seed, max_epochs=150, patience=15)
    trainer = Trainer(model_config, train_config)
    print("Training...")
    train_result = trainer.fit(train_ctrl, train_desc, train_delta, val_ctrl, val_desc, val_delta)

    pred_test_delta = trainer.predict_delta(test_ctrl, test_desc)
    model_mae = mae(test_delta, pred_test_delta)
    no_effect_pred = np.zeros_like(test_delta)
    no_effect_mae = mae(test_delta, no_effect_pred)

    report = build_full_report(test_delta, pred_test_delta, test_targets)

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(output_dir / "checkpoint")

    summary = {
        "pilot_note": "SUBSAMPLED real Arc data, not the full 221k-cell dataset.",
        "dataset": {
            "source": dataset.provenance.source_name,
            "source_url": dataset.provenance.source_url,
            "notes": dataset.provenance.notes,
            "n_cells_pilot": dataset.n_cells,
            "n_genes_pilot": dataset.n_genes,
            "n_targets_total": len(dataset.targets()),
        },
        "split": {
            "n_train_targets": len(manifest.train_targets),
            "n_val_targets": len(manifest.val_targets),
            "n_test_targets": len(manifest.test_targets),
            "n_test_cells": len(test_targets),
        },
        "training": {"best_epoch": train_result.best_epoch, "best_val_loss": train_result.best_val_loss},
        "test_mae_scaled": model_mae,
        "test_rmse_scaled": rmse(test_delta, pred_test_delta),
        "no_effect_mae_scaled": no_effect_mae,
        "improvement_over_no_effect": improvement_over_no_effect(model_mae, no_effect_mae),
        "grouped_bootstrap_ci_mae": report["grouped_bootstrap_ci_mae"],
        "worst_3_targets_by_mae": report["worst_3_targets_by_mae"],
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/real_arc_pilot"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    summary = run(args.output_dir, seed=args.seed)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

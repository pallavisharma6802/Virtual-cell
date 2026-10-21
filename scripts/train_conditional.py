"""Milestone 3: train the conditional model + ablations on the synthetic
fixture, compare against the no-effect baseline on held-out targets.

This runs entirely on the CPU synthetic fixture (same one used in
Milestone 1). It is NOT a real-data result — see docs/experiment_log.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from virtualcell.data.adapters import control_reference_mean, from_anndata
from virtualcell.data.control_sampling import sample_matched_controls
from virtualcell.data.fixtures import make_synthetic_dataset
from virtualcell.data.preprocess import fit_scaler
from virtualcell.data.splits import held_out_target_split
from virtualcell.data.target_descriptors import descriptor_for_targets, fit_control_mean_std_descriptors
from virtualcell.evaluation.metrics import improvement_over_no_effect, mae, rmse
from virtualcell.models.conditional import ConditionalModelConfig
from virtualcell.training.trainer import TrainConfig, Trainer


def build_arrays(dataset, manifest, cell_ids, scaler, gene_descriptors, seed):
    obs = dataset.adata.obs
    ctrl_mask = dataset.control_mask()
    pert_mask = obs.index.isin(cell_ids) & (~ctrl_mask)
    pert_cell_ids = list(obs.index[pert_mask])
    if len(pert_cell_ids) == 0:
        n_genes = dataset.n_genes
        return (
            np.zeros((0, n_genes), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, n_genes), dtype=np.float32),
            [],
        )

    X = np.asarray(dataset.adata.X)
    pert_positions = [list(obs.index).index(c) for c in pert_cell_ids]
    X_pert_scaled = scaler.transform(X[pert_positions])

    control_pool = [c for c in manifest.train_cell_ids if ctrl_mask[list(obs.index).index(c)]]
    matched_control_raw = sample_matched_controls(dataset, pert_cell_ids, control_pool, seed=seed)
    matched_control_scaled = scaler.transform(matched_control_raw)

    delta_scaled = X_pert_scaled - matched_control_scaled

    targets = obs.loc[pert_cell_ids, "perturbation_id"].tolist()
    target_desc = descriptor_for_targets(targets, gene_descriptors)

    return matched_control_scaled, target_desc, delta_scaled, targets


def run(output_dir: Path, seed: int = 0) -> dict:
    adata = make_synthetic_dataset(seed=seed)
    dataset = from_anndata(adata, source_name="synthetic_fixture", is_synthetic=True)
    manifest = held_out_target_split(dataset, seed=seed)

    scaler = fit_scaler(dataset, manifest)
    gene_descriptors = fit_control_mean_std_descriptors(dataset, manifest)

    train_ctrl, train_desc, train_delta, _ = build_arrays(
        dataset, manifest, manifest.train_cell_ids, scaler, gene_descriptors, seed
    )
    val_ctrl, val_desc, val_delta, _ = build_arrays(
        dataset, manifest, manifest.val_cell_ids, scaler, gene_descriptors, seed
    )
    test_ctrl, test_desc, test_delta, test_targets = build_arrays(
        dataset, manifest, manifest.test_cell_ids, scaler, gene_descriptors, seed
    )

    results = {}
    variants = {
        "full": {"use_control_context": True, "use_target_descriptor": True},
        "ablation_no_control": {"use_control_context": False, "use_target_descriptor": True},
        "ablation_no_target": {"use_control_context": True, "use_target_descriptor": False},
    }

    for name, flags in variants.items():
        model_config = ConditionalModelConfig(
            n_genes=dataset.n_genes,
            target_descriptor_dim=2,
            **flags,
        )
        train_config = TrainConfig(seed=seed)
        trainer = Trainer(model_config, train_config)
        train_result = trainer.fit(
            train_ctrl, train_desc, train_delta, val_ctrl, val_desc, val_delta
        )

        pred_test_delta_scaled = trainer.predict_delta(test_ctrl, test_desc)

        checkpoint_dir = output_dir / name
        trainer.save(checkpoint_dir)
        reloaded = Trainer.load(checkpoint_dir)
        pred_reloaded = reloaded.predict_delta(test_ctrl, test_desc)
        reload_matches = bool(np.allclose(pred_test_delta_scaled, pred_reloaded, atol=1e-5))

        model_mae = mae(test_delta, pred_test_delta_scaled)
        model_rmse = rmse(test_delta, pred_test_delta_scaled)
        no_effect_pred = np.zeros_like(test_delta)
        no_effect_mae = mae(test_delta, no_effect_pred)

        results[name] = {
            "best_epoch": train_result.best_epoch,
            "best_val_loss": train_result.best_val_loss,
            "test_mae_scaled": model_mae,
            "test_rmse_scaled": model_rmse,
            "no_effect_mae_scaled": no_effect_mae,
            "improvement_over_no_effect": improvement_over_no_effect(model_mae, no_effect_mae),
            "checkpoint_reload_matches_predictions": reload_matches,
        }

    summary = {
        "dataset": {"n_cells": dataset.n_cells, "n_genes": dataset.n_genes},
        "split": {
            "n_train_targets": len(manifest.train_targets),
            "n_val_targets": len(manifest.val_targets),
            "n_test_targets": len(manifest.test_targets),
            "n_test_cells": len(test_targets),
        },
        "variants": results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/conditional_mlp"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    summary = run(args.output_dir, seed=args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Tests whether the conditional model's gap vs. the small MLP baseline
(docs/results.md "Fair retuned comparison") is a training-budget/
optimization issue rather than an architecture issue — the one
hypothesis left untested after ruling out the target descriptor and
overfitting (docs/experiment_log.md).

Reuses the same 45,584-cell subsample as all prior scale-up experiments
(same seed/cap/control-size/n-top-genes, so it hits the local disk cache
added to subsample_and_materialize rather than re-fetching from GCS).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from virtualcell.data.arc_adapter import ARC_CHALLENGE_TRAIN_URL, load_arc_challenge_dataset
from virtualcell.data.preprocess import fit_scaler
from virtualcell.data.splits import held_out_target_split
from virtualcell.data.target_descriptors import fit_control_mean_std_descriptors
from virtualcell.evaluation.metrics import improvement_over_no_effect, mae, rmse
from virtualcell.models.conditional import ConditionalModelConfig
from virtualcell.training.trainer import TrainConfig, Trainer
from scale_up_real_arc import build_arrays, subsample_and_materialize

CELLS_PER_TARGET_CAP = 300
CONTROL_SAMPLE_SIZE = 3000
N_TOP_GENES = 1500


def run(output_dir: Path, seed: int = 0) -> dict:
    print("Loading Arc Virtual Cell Challenge dataset (backed mode)...")
    full_dataset = load_arc_challenge_dataset(ARC_CHALLENGE_TRAIN_URL, backed="r")
    dataset = subsample_and_materialize(
        full_dataset, seed=seed, cells_per_target_cap=CELLS_PER_TARGET_CAP,
        control_sample_size=CONTROL_SAMPLE_SIZE, n_top_genes=N_TOP_GENES, n_workers=1,
    )
    print(f"Dataset: {dataset.n_cells} cells x {dataset.n_genes} genes, {len(dataset.targets())} targets")

    manifest = held_out_target_split(dataset, seed=seed)
    scaler = fit_scaler(dataset, manifest)
    gene_descriptors = fit_control_mean_std_descriptors(dataset, manifest)
    descriptor_dim = 2

    train_ctrl, train_desc, train_delta, _ = build_arrays(dataset, manifest, manifest.train_cell_ids, scaler, gene_descriptors, descriptor_dim, seed)
    val_ctrl, val_desc, val_delta, _ = build_arrays(dataset, manifest, manifest.val_cell_ids, scaler, gene_descriptors, descriptor_dim, seed)
    test_ctrl, test_desc, test_delta, test_targets = build_arrays(dataset, manifest, manifest.test_cell_ids, scaler, gene_descriptors, descriptor_dim, seed)
    no_effect_mae = mae(test_delta, np.zeros_like(test_delta))

    # Reference point: the ORIGINAL short-training config (100 epochs,
    # patience 12) at the best known capacity (256/256), for direct
    # before/after comparison on the identical data/split.
    configs = [
        {"name": "original_short", "max_epochs": 100, "patience": 12, "weight_decay": 1e-5, "lr": 1e-3},
        {"name": "longer_training", "max_epochs": 500, "patience": 50, "weight_decay": 1e-5, "lr": 1e-3},
        {"name": "longer_training_more_reg", "max_epochs": 500, "patience": 50, "weight_decay": 1e-4, "lr": 1e-3},
        {"name": "longer_training_less_reg", "max_epochs": 500, "patience": 50, "weight_decay": 1e-6, "lr": 1e-3},
        {"name": "longer_training_lower_lr", "max_epochs": 800, "patience": 80, "weight_decay": 1e-5, "lr": 3e-4},
    ]

    results = []
    for cfg in configs:
        model_config = ConditionalModelConfig(
            n_genes=dataset.n_genes, target_descriptor_dim=descriptor_dim,
            control_hidden_dim=256, fusion_hidden_dim=256,
        )
        train_config = TrainConfig(
            seed=seed, max_epochs=cfg["max_epochs"], patience=cfg["patience"],
            weight_decay=cfg["weight_decay"], lr=cfg["lr"], batch_size=256,
        )
        trainer = Trainer(model_config, train_config)
        train_result = trainer.fit(train_ctrl, train_desc, train_delta, val_ctrl, val_desc, val_delta)

        train_pred = trainer.predict_delta(train_ctrl, train_desc)
        val_pred = trainer.predict_delta(val_ctrl, val_desc)
        test_pred = trainer.predict_delta(test_ctrl, test_desc)
        row = {
            "config": cfg,
            "best_epoch": train_result.best_epoch,
            "best_val_loss": train_result.best_val_loss,
            "train_mae": mae(train_delta, train_pred),
            "val_mae": mae(val_delta, val_pred),
            "test_mae": mae(test_delta, test_pred),
            "test_rmse": rmse(test_delta, test_pred),
            "improvement_over_no_effect": improvement_over_no_effect(mae(test_delta, test_pred), no_effect_mae),
        }
        results.append(row)
        print(f"  {cfg['name']}: best_epoch={train_result.best_epoch}, "
              f"train_mae={row['train_mae']:.4f}, val_mae={row['val_mae']:.4f}, "
              f"test_mae={row['test_mae']:.4f}, improvement={row['improvement_over_no_effect']:.4f}")

    summary = {
        "reference_small_mlp_test_mae": 0.8592525124549866,
        "reference_small_mlp_improvement": 0.19082783308894039,
        "no_effect_mae_reference": no_effect_mae,
        "dataset": {"n_cells": dataset.n_cells, "n_genes": dataset.n_genes, "n_targets": len(dataset.targets())},
        "configs": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/extended_training"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    summary = run(args.output_dir, seed=args.seed)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

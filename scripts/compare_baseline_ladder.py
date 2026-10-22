"""Milestone 2/3 baseline ladder, strictly in plan order (section 4A-B):
no-effect -> seen-target mean -> PCA+ridge -> small MLP -> conditional
model. All share the same held-out-target split, scaler, target
descriptor, and matched-control sampling, so the comparison is apples to
apples. Runs on the synthetic fixture (fast, offline, no cloud needed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from virtualcell.data.adapters import from_anndata
from virtualcell.data.control_sampling import sample_matched_controls
from virtualcell.data.fixtures import make_synthetic_dataset
from virtualcell.data.preprocess import fit_scaler
from virtualcell.data.splits import held_out_target_split
from virtualcell.data.target_descriptors import descriptor_for_targets, fit_control_mean_std_descriptors
from virtualcell.evaluation.metrics import grouped_bootstrap_ci, improvement_over_no_effect, mae, rmse
from virtualcell.models.baselines import (
    NoEffectBaseline,
    PCARidgeBaseline,
    SeenTargetMeanBaseline,
    SmallMLPBaseline,
)
from virtualcell.models.conditional import ConditionalModelConfig
from virtualcell.training.trainer import TrainConfig, Trainer


def build_arrays(dataset, manifest, cell_ids, scaler, gene_descriptors, seed):
    obs = dataset.adata.obs
    ctrl_mask = dataset.control_mask()
    pert_mask = obs.index.isin(cell_ids) & (~ctrl_mask)
    pert_cell_ids = list(obs.index[pert_mask])
    if not pert_cell_ids:
        n_genes = dataset.n_genes
        return (np.zeros((0, n_genes), dtype=np.float32), np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, n_genes), dtype=np.float32), [])

    X = np.asarray(dataset.adata.X)
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

    no_effect_mae = mae(test_delta, np.zeros_like(test_delta))
    results = {}

    # 1. No-effect
    no_effect_pred = NoEffectBaseline(n_genes=dataset.n_genes).predict_delta(n_cells=test_delta.shape[0])
    results["1_no_effect"] = {"mae": mae(test_delta, no_effect_pred), "rmse": rmse(test_delta, no_effect_pred)}

    # 2. Seen-target mean (documented to fail on held-out targets)
    seen_mean = SeenTargetMeanBaseline().fit(dataset, manifest)
    seen_mean_pred = seen_mean.predict_delta(test_targets)
    results["2_seen_target_mean"] = {
        "mae": mae(test_delta, seen_mean_pred),
        "rmse": rmse(test_delta, seen_mean_pred),
        "note": "Falls back to zero-delta for every held-out target by design; matches no-effect here.",
    }

    # 3. PCA + ridge
    pca_ridge = PCARidgeBaseline(n_components=20, alpha=1.0).fit(train_ctrl, train_desc, train_delta)
    pca_ridge_pred = pca_ridge.predict_delta(test_ctrl, test_desc)
    results["3_pca_ridge"] = {"mae": mae(test_delta, pca_ridge_pred), "rmse": rmse(test_delta, pca_ridge_pred)}

    # 4. Small MLP baseline
    small_mlp = SmallMLPBaseline(hidden_dim=64, seed=seed).fit(train_ctrl, train_desc, train_delta)
    small_mlp_pred = small_mlp.predict_delta(test_ctrl, test_desc)
    results["4_small_mlp"] = {"mae": mae(test_delta, small_mlp_pred), "rmse": rmse(test_delta, small_mlp_pred)}

    # 5. Conditional model (full)
    model_config = ConditionalModelConfig(n_genes=dataset.n_genes, target_descriptor_dim=2)
    train_config = TrainConfig(seed=seed)
    trainer = Trainer(model_config, train_config)
    trainer.fit(train_ctrl, train_desc, train_delta, val_ctrl, val_desc, val_delta)
    conditional_pred = trainer.predict_delta(test_ctrl, test_desc)
    results["5_conditional_model"] = {
        "mae": mae(test_delta, conditional_pred), "rmse": rmse(test_delta, conditional_pred)
    }

    for name, r in results.items():
        r["improvement_over_no_effect"] = improvement_over_no_effect(r["mae"], no_effect_mae)

    per_target_mae_by_model = {}
    labels_arr = np.asarray(test_targets)
    for name, pred in [
        ("1_no_effect", no_effect_pred),
        ("2_seen_target_mean", seen_mean_pred),
        ("3_pca_ridge", pca_ridge_pred),
        ("4_small_mlp", small_mlp_pred),
        ("5_conditional_model", conditional_pred),
    ]:
        per_target = {}
        for t in sorted(set(test_targets)):
            mask = labels_arr == t
            per_target[t] = mae(test_delta[mask], pred[mask])
        results[name]["grouped_bootstrap_ci_mae"] = grouped_bootstrap_ci(per_target, seed=seed)

    summary = {
        "dataset": {"n_cells": dataset.n_cells, "n_genes": dataset.n_genes},
        "split": {
            "n_train_targets": len(manifest.train_targets),
            "n_val_targets": len(manifest.val_targets),
            "n_test_targets": len(manifest.test_targets),
            "n_test_cells": len(test_targets),
        },
        "no_effect_mae_reference": no_effect_mae,
        "models": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_ladder.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline_ladder"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    summary = run(args.output_dir, seed=args.seed)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

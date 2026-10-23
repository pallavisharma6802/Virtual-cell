"""Full baseline ladder (no-effect, seen-target mean, PCA+ridge, small MLP,
conditional model) on a subsample of the real Arc Virtual Cell Challenge
data, under BOTH target descriptors (weak mean/std and richer
co-expression). This fixture (synthetic) version showed the richer
descriptor hurting simpler models, plausibly because of only 14 training
targets; the real data has ~150 targets, a fairer test of whether that
holds up with more training signal.

Still a SUBSAMPLED pilot (same subsampling as scripts/train_real_arc.py),
not the full 221k-cell dataset — see docs/experiment_log.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from virtualcell.data.arc_adapter import ARC_CHALLENGE_TRAIN_URL, load_arc_challenge_dataset
from virtualcell.data.control_sampling import sample_matched_controls
from virtualcell.data.preprocess import fit_scaler
from virtualcell.data.splits import held_out_target_split
from virtualcell.data.target_descriptors import (
    descriptor_for_targets,
    fit_coexpression_descriptors,
    fit_control_mean_std_descriptors,
)
from virtualcell.evaluation.metrics import grouped_bootstrap_ci, improvement_over_no_effect, mae, rmse
from virtualcell.models.baselines import (
    NoEffectBaseline,
    PCARidgeBaseline,
    SeenTargetMeanBaseline,
    SmallMLPBaseline,
)
from virtualcell.models.conditional import ConditionalModelConfig
from virtualcell.training.trainer import TrainConfig, Trainer
from train_real_arc import subsample_and_materialize


def build_arrays(dataset, manifest, cell_ids, scaler, gene_descriptors, descriptor_dim, seed):
    obs = dataset.adata.obs
    ctrl_mask = dataset.control_mask()
    pert_mask = obs.index.isin(cell_ids) & (~ctrl_mask)
    pert_cell_ids = list(obs.index[pert_mask])
    if not pert_cell_ids:
        n_genes = dataset.n_genes
        return (np.zeros((0, n_genes), dtype=np.float32), np.zeros((0, descriptor_dim), dtype=np.float32),
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
    desc = descriptor_for_targets(targets, gene_descriptors, descriptor_dim=descriptor_dim)
    return matched_scaled, desc, delta, targets


def run_one_descriptor(dataset, manifest, descriptor_name, n_components, seed):
    scaler = fit_scaler(dataset, manifest)
    if descriptor_name == "coexpression":
        gene_descriptors = fit_coexpression_descriptors(dataset, manifest, n_components=n_components, seed=seed)
        descriptor_dim = min(n_components, dataset.n_genes)
    else:
        gene_descriptors = fit_control_mean_std_descriptors(dataset, manifest)
        descriptor_dim = 2

    train_ctrl, train_desc, train_delta, _ = build_arrays(
        dataset, manifest, manifest.train_cell_ids, scaler, gene_descriptors, descriptor_dim, seed
    )
    val_ctrl, val_desc, val_delta, _ = build_arrays(
        dataset, manifest, manifest.val_cell_ids, scaler, gene_descriptors, descriptor_dim, seed
    )
    test_ctrl, test_desc, test_delta, test_targets = build_arrays(
        dataset, manifest, manifest.test_cell_ids, scaler, gene_descriptors, descriptor_dim, seed
    )

    no_effect_mae = mae(test_delta, np.zeros_like(test_delta))
    results = {}

    no_effect_pred = NoEffectBaseline(n_genes=dataset.n_genes).predict_delta(n_cells=test_delta.shape[0])
    results["1_no_effect"] = {"mae": mae(test_delta, no_effect_pred)}

    seen_mean = SeenTargetMeanBaseline().fit(dataset, manifest)
    seen_mean_pred = seen_mean.predict_delta(test_targets)
    results["2_seen_target_mean"] = {"mae": mae(test_delta, seen_mean_pred)}

    pca_n = min(20, train_ctrl.shape[0], train_ctrl.shape[1])
    pca_ridge = PCARidgeBaseline(n_components=pca_n, alpha=1.0).fit(train_ctrl, train_desc, train_delta)
    pca_ridge_pred = pca_ridge.predict_delta(test_ctrl, test_desc)
    results["3_pca_ridge"] = {"mae": mae(test_delta, pca_ridge_pred)}

    small_mlp = SmallMLPBaseline(hidden_dim=64, seed=seed).fit(train_ctrl, train_desc, train_delta)
    small_mlp_pred = small_mlp.predict_delta(test_ctrl, test_desc)
    results["4_small_mlp"] = {"mae": mae(test_delta, small_mlp_pred)}

    model_config = ConditionalModelConfig(n_genes=dataset.n_genes, target_descriptor_dim=descriptor_dim)
    train_config = TrainConfig(seed=seed, max_epochs=150, patience=15)
    trainer = Trainer(model_config, train_config)
    trainer.fit(train_ctrl, train_desc, train_delta, val_ctrl, val_desc, val_delta)
    conditional_pred = trainer.predict_delta(test_ctrl, test_desc)
    results["5_conditional_model"] = {"mae": mae(test_delta, conditional_pred)}

    labels_arr = np.asarray(test_targets)
    for name, pred in [
        ("1_no_effect", no_effect_pred), ("2_seen_target_mean", seen_mean_pred),
        ("3_pca_ridge", pca_ridge_pred), ("4_small_mlp", small_mlp_pred),
        ("5_conditional_model", conditional_pred),
    ]:
        results[name]["improvement_over_no_effect"] = improvement_over_no_effect(results[name]["mae"], no_effect_mae)
        per_target = {t: mae(test_delta[labels_arr == t], pred[labels_arr == t]) for t in sorted(set(test_targets))}
        results[name]["grouped_bootstrap_ci_mae"] = grouped_bootstrap_ci(per_target, seed=seed)

    return {
        "descriptor": descriptor_name,
        "descriptor_dim": descriptor_dim,
        "no_effect_mae_reference": no_effect_mae,
        "n_test_cells": len(test_targets),
        "n_test_targets": len(set(test_targets)),
        "models": results,
    }


def run(output_dir: Path, seed: int = 0) -> dict:
    print("Loading Arc Virtual Cell Challenge dataset (backed mode)...")
    full_dataset = load_arc_challenge_dataset(ARC_CHALLENGE_TRAIN_URL, backed="r")
    dataset = subsample_and_materialize(full_dataset, seed=seed)
    print(f"Pilot dataset: {dataset.n_cells} cells x {dataset.n_genes} genes, {len(dataset.targets())} targets")

    manifest = held_out_target_split(dataset, seed=seed)
    print(f"Split: {len(manifest.train_targets)} train / {len(manifest.val_targets)} val / {len(manifest.test_targets)} test targets")

    summary = {
        "dataset": {"n_cells": dataset.n_cells, "n_genes": dataset.n_genes, "n_targets": len(dataset.targets())},
        "split": {
            "n_train_targets": len(manifest.train_targets),
            "n_val_targets": len(manifest.val_targets),
            "n_test_targets": len(manifest.test_targets),
        },
        "by_descriptor": {},
    }

    for descriptor_name in ["mean_std", "coexpression"]:
        print(f"Running ladder with descriptor={descriptor_name}...")
        summary["by_descriptor"][descriptor_name] = run_one_descriptor(
            dataset, manifest, descriptor_name, n_components=20, seed=seed
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_ladder_real.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline_ladder_real"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    summary = run(args.output_dir, seed=args.seed)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

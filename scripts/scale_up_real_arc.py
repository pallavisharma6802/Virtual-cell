"""Milestone 5 (continued): scaled-up real-data training with memory-
efficient, concurrent data loading + mini-batch training, plus a small
hyperparameter sweep for the conditional model. Keeps the full baseline
ladder (no-effect, seen-target mean, PCA+ridge, small MLP) for comparison
— NOT a pivot away from the conditional model, just an honest comparison
point as before.

Still a SUBSAMPLED pilot relative to the full 221,273-cell dataset
(configurable scale via CLI flags), but substantially larger than the
first real-data pilot (10,960 cells) via:
  - concurrent/parallel GCS reads (arc_adapter.materialize_cells_concurrent)
    instead of one sequential connection,
  - mini-batch training (Trainer now supports TrainConfig.batch_size)
    instead of full-batch gradient descent.

Single cell type (same Arc Virtual Cell Challenge file as before) —
multi-cell-type generalization is a separate, later, explicitly deferred
phase (requires verifying a different dataset's schema first).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from anndata import AnnData

from virtualcell.data.adapters import CanonicalDataset, DatasetProvenance
from virtualcell.data.arc_adapter import ARC_CHALLENGE_TRAIN_URL, load_arc_challenge_dataset
from virtualcell.data.control_sampling import sample_matched_controls
from virtualcell.data.preprocess import fit_scaler
from virtualcell.data.splits import held_out_target_split
from virtualcell.data.target_descriptors import descriptor_for_targets, fit_control_mean_std_descriptors
from virtualcell.evaluation.metrics import grouped_bootstrap_ci, improvement_over_no_effect, mae, rmse
from virtualcell.evaluation.report import build_full_report
from virtualcell.models.baselines import NoEffectBaseline, PCARidgeBaseline, SeenTargetMeanBaseline, SmallMLPBaseline
from virtualcell.models.conditional import ConditionalModelConfig
from virtualcell.training.trainer import TrainConfig, Trainer

MIN_CELLS_PER_TARGET = 10


def subsample_and_materialize(
    dataset: CanonicalDataset,
    seed: int,
    cells_per_target_cap: int,
    control_sample_size: int,
    n_top_genes: int,
    n_workers: int,
    path_or_url: str = ARC_CHALLENGE_TRAIN_URL,
) -> CanonicalDataset:
    """Like the first pilot's subsampling, scaled up via configurable cap
    sizes. Gene selection (top-K by train-control variance) still happens
    AFTER fetching the requested cells' full gene width — CSR row reads
    can't skip columns, so this is a post-hoc reduction, not a
    server-side optimization.

    NOTE on concurrency: a concurrent/multi-connection fetch
    (`arc_adapter.materialize_cells_concurrent`) was built and measured
    against the proven sequential approach on this same instance type
    before this run. It was SLOWER (0.29s/cell vs. 0.127s/cell for the
    sequential anndata-backed `.to_memory()` slice), almost certainly
    because each worker re-opens the full remote HDF5 file's metadata
    independently. That optimization is not used here.

    NOTE on memory: fetching all requested cells in one
    `.to_memory()` call OOM-killed an 8GB instance at ~45,600 cells —
    the sparse CSR slice and its dense `.todense()` copy coexist in
    memory during conversion, roughly doubling peak RAM, and that
    exceeded 8GB well before the target cell count. Fixed here by
    fetching and densifying in bounded-size CHUNKS, writing each chunk
    directly into a preallocated float32 array and freeing the chunk's
    intermediate sparse/dense objects (`del` + `gc.collect()`) before
    the next chunk — peak memory is then bounded by chunk size, not
    total cell count, which is the actual "memory-efficient loading"
    this milestone needs. See docs/experiment_log.md for the OOM and
    this fix.
    """
    import gc

    rng = np.random.default_rng(seed)
    obs = dataset.adata.obs
    ctrl_mask = dataset.control_mask()

    keep_cell_positions = []
    for target in dataset.targets():
        positions = np.where((obs["perturbation_id"] == target).to_numpy())[0]
        if len(positions) < MIN_CELLS_PER_TARGET:
            continue
        chosen = rng.choice(positions, size=min(cells_per_target_cap, len(positions)), replace=False)
        keep_cell_positions.extend(chosen.tolist())

    ctrl_positions = np.where(ctrl_mask)[0]
    ctrl_sample = rng.choice(ctrl_positions, size=min(control_sample_size, len(ctrl_positions)), replace=False)
    keep_cell_positions.extend(ctrl_sample.tolist())

    keep_cell_positions = sorted(set(keep_cell_positions))
    n_cells = len(keep_cell_positions)
    n_genes_full = dataset.adata.n_vars
    print(f"Fetching {n_cells} cells from GCS in memory-bounded chunks...")

    CHUNK = 4000
    X_full_genes = np.empty((n_cells, n_genes_full), dtype=np.float32)
    t_fetch0 = time.time()
    for start in range(0, n_cells, CHUNK):
        chunk_positions = keep_cell_positions[start : start + CHUNK]
        # A backed AnnData's remote file handle is consumed/closed after a
        # .to_memory() call on a view of it — reusing `dataset.adata`
        # across chunks fails on the second chunk with a FileNotFoundError
        # from h5py trying to reopen an already-closed gcsfs file object
        # (confirmed by a live run; see docs/experiment_log.md). Re-open a
        # fresh backed handle per chunk instead — cheap relative to the
        # per-row read cost (confirmed: ~1-2s to reopen vs. the per-chunk
        # multi-second data read).
        fresh = load_arc_challenge_dataset(path_or_url, backed="r")
        sub_chunk = fresh.adata[chunk_positions, :].to_memory()
        chunk_dense = np.asarray(sub_chunk.X.todense()) if hasattr(sub_chunk.X, "todense") else np.asarray(sub_chunk.X)
        X_full_genes[start : start + len(chunk_positions)] = chunk_dense
        del fresh, sub_chunk, chunk_dense
        gc.collect()
        print(f"  fetched {min(start + CHUNK, n_cells)}/{n_cells} cells "
              f"({time.time() - t_fetch0:.1f}s elapsed)")

    sub_obs = obs.iloc[keep_cell_positions].copy()
    sub_var = dataset.adata.var.copy()

    sub_ctrl_mask = sub_obs["is_control"].to_numpy()
    gene_var = X_full_genes[sub_ctrl_mask].var(axis=0)
    top_gene_idx = np.sort(np.argsort(-gene_var)[:n_top_genes])

    X_reduced = X_full_genes[:, top_gene_idx].astype(np.float32)
    sub_var = sub_var.iloc[top_gene_idx]

    sub_adata = AnnData(X=X_reduced, obs=sub_obs, var=sub_var)

    return CanonicalDataset(
        adata=sub_adata,
        gene_ids=list(sub_adata.var.index),
        provenance=DatasetProvenance(
            source_name=dataset.provenance.source_name,
            source_version=dataset.provenance.source_version,
            source_url=dataset.provenance.source_url,
            is_synthetic=False,
            notes=(
                dataset.provenance.notes
                + f" | SCALED-UP pilot: <=~{cells_per_target_cap} cells/target, "
                f"{control_sample_size} controls, {n_top_genes} top-variable genes, "
                f"fetched sequentially (concurrent fetch was measured and found slower, see experiment_log.md)."
            ),
        ),
    )


def build_arrays(dataset, manifest, cell_ids, scaler, gene_descriptors, seed):
    obs = dataset.adata.obs
    ctrl_mask = dataset.control_mask()
    pert_mask = obs.index.isin(cell_ids) & (~ctrl_mask)
    pert_cell_ids = list(obs.index[pert_mask])
    if not pert_cell_ids:
        n_genes = dataset.n_genes
        return (np.zeros((0, n_genes), dtype=np.float32), np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, n_genes), dtype=np.float32), [])
    index_list = list(obs.index)
    positions = [index_list.index(c) for c in pert_cell_ids]
    X = np.asarray(dataset.adata.X)
    X_scaled = scaler.transform(X[positions])

    control_pool = [c for c in manifest.train_cell_ids if ctrl_mask[index_list.index(c)]]
    matched_raw = sample_matched_controls(dataset, pert_cell_ids, control_pool, seed=seed)
    matched_scaled = scaler.transform(matched_raw)

    delta = X_scaled - matched_scaled
    targets = obs.loc[pert_cell_ids, "perturbation_id"].tolist()
    desc = descriptor_for_targets(targets, gene_descriptors)
    return matched_scaled, desc, delta, targets


def hyperparameter_sweep(train_ctrl, train_desc, train_delta, val_ctrl, val_desc, val_delta, n_genes, seed):
    """Small, bounded grid search over the conditional model's capacity
    and learning rate, selected by validation loss. Kept small to stay
    within the agreed compute budget, not exhaustive."""
    configs = [
        {"control_hidden_dim": 128, "fusion_hidden_dim": 128, "lr": 1e-3},
        {"control_hidden_dim": 64, "fusion_hidden_dim": 64, "lr": 1e-3},
        {"control_hidden_dim": 128, "fusion_hidden_dim": 128, "lr": 3e-4},
        {"control_hidden_dim": 256, "fusion_hidden_dim": 256, "lr": 1e-3},
    ]
    results = []
    best = None
    for cfg in configs:
        model_config = ConditionalModelConfig(
            n_genes=n_genes, target_descriptor_dim=2,
            control_hidden_dim=cfg["control_hidden_dim"], fusion_hidden_dim=cfg["fusion_hidden_dim"],
        )
        train_config = TrainConfig(seed=seed, max_epochs=100, patience=12, lr=cfg["lr"], batch_size=256)
        trainer = Trainer(model_config, train_config)
        result = trainer.fit(train_ctrl, train_desc, train_delta, val_ctrl, val_desc, val_delta)
        row = {**cfg, "best_val_loss": result.best_val_loss, "best_epoch": result.best_epoch}
        results.append(row)
        print(f"  sweep config {cfg} -> val_loss {result.best_val_loss:.4f}")
        if best is None or result.best_val_loss < best[0]:
            best = (result.best_val_loss, cfg, trainer)
    return best[1], best[2], results


def run(output_dir: Path, seed: int, cells_per_target_cap: int, control_sample_size: int,
        n_top_genes: int, n_workers: int) -> dict:
    print("Loading Arc Virtual Cell Challenge dataset (backed mode)...")
    full_dataset = load_arc_challenge_dataset(ARC_CHALLENGE_TRAIN_URL, backed="r")
    t0 = time.time()
    dataset = subsample_and_materialize(
        full_dataset, seed=seed, cells_per_target_cap=cells_per_target_cap,
        control_sample_size=control_sample_size, n_top_genes=n_top_genes, n_workers=n_workers,
    )
    fetch_time = time.time() - t0
    print(f"Scaled-up dataset: {dataset.n_cells} cells x {dataset.n_genes} genes, "
          f"{len(dataset.targets())} targets (fetch took {fetch_time:.1f}s)")

    manifest = held_out_target_split(dataset, seed=seed)
    print(f"Split: {len(manifest.train_targets)} train / {len(manifest.val_targets)} val / {len(manifest.test_targets)} test targets")

    scaler = fit_scaler(dataset, manifest)
    gene_descriptors = fit_control_mean_std_descriptors(dataset, manifest)

    train_ctrl, train_desc, train_delta, _ = build_arrays(dataset, manifest, manifest.train_cell_ids, scaler, gene_descriptors, seed)
    val_ctrl, val_desc, val_delta, _ = build_arrays(dataset, manifest, manifest.val_cell_ids, scaler, gene_descriptors, seed)
    test_ctrl, test_desc, test_delta, test_targets = build_arrays(dataset, manifest, manifest.test_cell_ids, scaler, gene_descriptors, seed)
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

    print("Running conditional-model hyperparameter sweep...")
    best_cfg, best_trainer, sweep_results = hyperparameter_sweep(
        train_ctrl, train_desc, train_delta, val_ctrl, val_desc, val_delta, dataset.n_genes, seed
    )
    print(f"Best config: {best_cfg}")
    conditional_pred = best_trainer.predict_delta(test_ctrl, test_desc)
    results["5_conditional_model_tuned"] = {"mae": mae(test_delta, conditional_pred), "best_config": best_cfg}

    for name, r in results.items():
        r["improvement_over_no_effect"] = improvement_over_no_effect(r["mae"], no_effect_mae)

    report = build_full_report(test_delta, conditional_pred, test_targets)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_trainer.save(output_dir / "checkpoint_conditional_tuned")

    summary = {
        "pilot_note": f"SCALED-UP subsample: {dataset.n_cells} cells (full dataset has 221,273).",
        "dataset": {
            "n_cells": dataset.n_cells, "n_genes": dataset.n_genes,
            "n_targets": len(dataset.targets()), "fetch_time_seconds": fetch_time,
            "cells_per_target_cap": cells_per_target_cap, "control_sample_size": control_sample_size,
        },
        "split": {
            "n_train_targets": len(manifest.train_targets), "n_val_targets": len(manifest.val_targets),
            "n_test_targets": len(manifest.test_targets), "n_test_cells": len(test_targets),
        },
        "no_effect_mae_reference": no_effect_mae,
        "models": results,
        "hyperparameter_sweep": sweep_results,
        "conditional_model_grouped_bootstrap_ci_mae": report["grouped_bootstrap_ci_mae"],
        "conditional_model_worst_3_targets": report["worst_3_targets_by_mae"],
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scale_up_real_arc"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cells-per-target-cap", type=int, default=300)
    parser.add_argument("--control-sample-size", type=int, default=3000)
    parser.add_argument("--n-top-genes", type=int, default=1500)
    parser.add_argument("--n-workers", type=int, default=12)
    args = parser.parse_args()
    summary = run(
        args.output_dir, seed=args.seed, cells_per_target_cap=args.cells_per_target_cap,
        control_sample_size=args.control_sample_size, n_top_genes=args.n_top_genes, n_workers=args.n_workers,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "per_target"}, indent=2, default=str))


if __name__ == "__main__":
    main()

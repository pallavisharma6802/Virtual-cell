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
import sys
import time
from pathlib import Path

import numpy as np
from anndata import AnnData

from virtualcell.data.adapters import CanonicalDataset, DatasetProvenance
from virtualcell.data.arc_adapter import ARC_CHALLENGE_TRAIN_URL, load_arc_challenge_dataset
from virtualcell.data.control_sampling import sample_matched_controls
from virtualcell.data.preprocess import fit_scaler
from virtualcell.data.splits import held_out_target_split
from virtualcell.data.target_descriptors import (
    descriptor_for_targets,
    fit_coexpression_descriptors,
    fit_control_mean_std_descriptors,
    fit_gene2vec_descriptors,
)
from virtualcell.evaluation.metrics import grouped_bootstrap_ci, improvement_over_no_effect, mae, rmse
from virtualcell.evaluation.report import build_full_report
from virtualcell.models.baselines import NoEffectBaseline, PCARidgeBaseline, SeenTargetMeanBaseline, SmallMLPBaseline
from virtualcell.models.conditional import ConditionalModelConfig
from virtualcell.training.trainer import TrainConfig, Trainer

MIN_CELLS_PER_TARGET = 10
_WORKER_SCRIPT = Path(__file__).parent / "_fetch_chunk_worker.py"


def fetch_chunk_with_retry(
    path_or_url: str,
    chunk_positions: list[int],
    top_gene_idx: np.ndarray | None = None,
    timeout_s: float = 120.0,
    max_attempts: int = 3,
) -> np.ndarray:
    """Fetch one chunk in a SEPARATE OS PROCESS, with a hard timeout and
    retry.

    Live runs at larger scale (2026-11-10, see docs/experiment_log.md)
    repeatedly stalled indefinitely on a single chunk's fetch. An
    in-process fix (hard timeout + retry in a worker THREAD) was tried
    first and did not work: once gcsfs's background async prefetcher
    failed to tear down cleanly once ("BackgroundPrefetcher teardown did
    not complete... it will keep running in the background"), ALL
    subsequent same-process retries also timed out — the corrupted
    async/event-loop state persists for the rest of that process, so
    "fresh" attempts in a new thread were never actually independent.

    Running each chunk fetch in a genuinely separate OS process (a fresh
    Python interpreter, fresh event loop every time) fixes this: if one
    process hangs, it's killed outright by `subprocess.run`'s timeout and
    the next attempt starts with zero shared state.
    """
    import subprocess
    import tempfile

    for attempt in range(1, max_attempts + 1):
        with tempfile.TemporaryDirectory() as tmpdir:
            positions_path = f"{tmpdir}/positions.npy"
            output_path = f"{tmpdir}/output.npy"
            np.save(positions_path, np.asarray(chunk_positions, dtype=np.int64))

            cmd = [sys.executable, str(_WORKER_SCRIPT), path_or_url, positions_path, output_path]
            if top_gene_idx is not None:
                top_gene_path = f"{tmpdir}/top_gene_idx.npy"
                np.save(top_gene_path, top_gene_idx)
                cmd.append(top_gene_path)

            try:
                subprocess.run(cmd, timeout=timeout_s, check=True, capture_output=True, text=True)
                return np.load(output_path)
            except subprocess.TimeoutExpired:
                print(f"  chunk fetch attempt {attempt}/{max_attempts} timed out after {timeout_s}s "
                      f"(subprocess killed), retrying...")
            except subprocess.CalledProcessError as e:
                print(f"  chunk fetch attempt {attempt}/{max_attempts} failed (exit {e.returncode}): "
                      f"{e.stderr[-500:] if e.stderr else '(no stderr)'}")

    raise RuntimeError(
        f"Failed to fetch chunk of {len(chunk_positions)} cells after {max_attempts} attempts "
        f"(each either timed out at {timeout_s}s or errored). This may indicate a persistent "
        "network/GCS issue, not a transient one."
    )


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

    Results are cached to a local .npz keyed by the scale parameters, so
    repeated experiments on the same subsample (different model
    hyperparameters, training budgets, etc.) don't re-pay the ~15 minute
    GCS fetch each time. Cache lives at
    `.cache/subsample_{seed}_{cells_per_target_cap}_{control_sample_size}_{n_top_genes}.npz`
    relative to the working directory.
    """
    import gc

    cache_dir = Path(".cache")
    cache_path = cache_dir / f"subsample_{seed}_{cells_per_target_cap}_{control_sample_size}_{n_top_genes}.npz"
    if cache_path.exists():
        print(f"Loading cached subsample from {cache_path}...")
        cached = np.load(cache_path, allow_pickle=True)
        X_reduced = cached["X"]
        obs_df = cached["obs"].item()
        var_df = cached["var"].item()
        sub_adata = AnnData(X=X_reduced, obs=obs_df, var=var_df)
        return CanonicalDataset(
            adata=sub_adata,
            gene_ids=list(sub_adata.var.index),
            provenance=DatasetProvenance(
                source_name=dataset.provenance.source_name,
                source_version=dataset.provenance.source_version,
                source_url=dataset.provenance.source_url,
                is_synthetic=False,
                notes=dataset.provenance.notes + " | Loaded from local cache.",
            ),
        )

    rng = np.random.default_rng(seed)
    obs = dataset.adata.obs
    ctrl_mask = dataset.control_mask()

    target_cell_positions = []
    for target in dataset.targets():
        positions = np.where((obs["perturbation_id"] == target).to_numpy())[0]
        if len(positions) < MIN_CELLS_PER_TARGET:
            continue
        chosen = rng.choice(positions, size=min(cells_per_target_cap, len(positions)), replace=False)
        target_cell_positions.extend(chosen.tolist())

    ctrl_positions_all = np.where(ctrl_mask)[0]
    ctrl_positions = sorted(
        rng.choice(ctrl_positions_all, size=min(control_sample_size, len(ctrl_positions_all)), replace=False).tolist()
    )
    target_cell_positions = sorted(set(target_cell_positions))
    n_genes_full = dataset.adata.n_vars
    # Reduced from 4000 after live stalls: fetching a 4000-row chunk of
    # SCATTERED (non-contiguous) positions triggers dozens of concurrent
    # GCS range requests at once (confirmed via `ss -tnp`), which appears
    # to occasionally hang indefinitely with no error (observed twice at
    # CHUNK=4000, always on the very first chunk, with near-zero CPU
    # usage and no data in flight). A smaller chunk bounds how many
    # concurrent range requests fire at once per call.
    CHUNK = 800

    def fetch_chunks(positions: list[int]) -> np.ndarray:
        """Fetch positions in memory-bounded chunks, returning a dense
        float32 array at FULL gene width — caller must reduce/discard
        promptly for large `positions` lists (see note below on why
        reduction can't happen inside this helper for the gene-selection
        pass, but must for the large target-cell pass)."""
        out = np.empty((len(positions), n_genes_full), dtype=np.float32)
        t0 = time.time()
        for start in range(0, len(positions), CHUNK):
            chunk_positions = positions[start : start + CHUNK]
            chunk_dense = fetch_chunk_with_retry(path_or_url, chunk_positions, top_gene_idx=None)
            out[start : start + len(chunk_positions)] = chunk_dense
            del chunk_dense
            gc.collect()
            print(f"  fetched {min(start + CHUNK, len(positions))}/{len(positions)} cells "
                  f"({time.time() - t0:.1f}s elapsed)")
        return out

    # PASS 1: fetch controls only (small enough to hold at full gene
    # width) to pick the top-variable genes BEFORE the large target-cell
    # fetch. This is what lets the large pass reduce each chunk to
    # n_top_genes immediately instead of ever holding
    # (total_cells x 18,080) in memory — the thing that would otherwise
    # OOM at large scale (e.g. 126,000 cells x 18,080 genes x 4 bytes is
    # ~9.1 GB, over this instance's 8GB RAM).
    print(f"Pass 1/2: fetching {len(ctrl_positions)} control cells to select top-{n_top_genes} variable genes...")
    X_ctrl_full = fetch_chunks(ctrl_positions)
    gene_var = X_ctrl_full.var(axis=0)
    top_gene_idx = np.sort(np.argsort(-gene_var)[:n_top_genes])
    X_ctrl_reduced = X_ctrl_full[:, top_gene_idx].copy()
    del X_ctrl_full
    gc.collect()

    # PASS 2: fetch target cells in chunks, reducing each chunk to
    # top_gene_idx immediately (not accumulating full gene width).
    n_target_cells = len(target_cell_positions)
    print(f"Pass 2/2: fetching {n_target_cells} target cells from GCS in memory-bounded, gene-reduced chunks...")
    X_target_reduced = np.empty((n_target_cells, n_top_genes), dtype=np.float32)
    t_fetch0 = time.time()
    for start in range(0, n_target_cells, CHUNK):
        chunk_positions = target_cell_positions[start : start + CHUNK]
        # A backed AnnData's remote file handle is consumed/closed after a
        # .to_memory() call on a view of it — reusing `dataset.adata`
        # across chunks fails on the second chunk with a FileNotFoundError
        # from h5py trying to reopen an already-closed gcsfs file object
        # (confirmed by a live run; see docs/experiment_log.md). Re-open a
        # fresh backed handle per chunk instead. `fetch_chunk_with_retry`
        # also applies a hard timeout + retry, since this fetch has been
        # observed to silently hang indefinitely on some chunks at larger
        # scale (see docs/experiment_log.md).
        chunk_dense = fetch_chunk_with_retry(path_or_url, chunk_positions, top_gene_idx=top_gene_idx)
        X_target_reduced[start : start + len(chunk_positions)] = chunk_dense
        del chunk_dense
        gc.collect()
        print(f"  fetched {min(start + CHUNK, n_target_cells)}/{n_target_cells} cells "
              f"({time.time() - t_fetch0:.1f}s elapsed)")

    keep_cell_positions = ctrl_positions + target_cell_positions
    X_reduced = np.concatenate([X_ctrl_reduced, X_target_reduced], axis=0)
    sub_obs = obs.iloc[keep_cell_positions].copy()
    sub_var = dataset.adata.var.iloc[top_gene_idx].copy()

    sub_adata = AnnData(X=X_reduced, obs=sub_obs, var=sub_var)

    cache_dir.mkdir(parents=True, exist_ok=True)
    obs_holder = np.empty((), dtype=object)
    obs_holder[()] = sub_obs
    var_holder = np.empty((), dtype=object)
    var_holder[()] = sub_var
    np.savez(
        cache_path,
        X=X_reduced,
        obs=obs_holder,
        var=var_holder,
    )
    print(f"Cached subsample to {cache_path}")

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


def build_arrays(dataset, manifest, cell_ids, scaler, gene_descriptors, descriptor_dim, seed):
    obs = dataset.adata.obs
    ctrl_mask = dataset.control_mask()
    pert_mask = obs.index.isin(cell_ids) & (~ctrl_mask)
    pert_cell_ids = list(obs.index[pert_mask])
    if not pert_cell_ids:
        n_genes = dataset.n_genes
        return (np.zeros((0, n_genes), dtype=np.float32), np.zeros((0, descriptor_dim), dtype=np.float32),
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
    desc = descriptor_for_targets(targets, gene_descriptors, descriptor_dim=descriptor_dim)
    return matched_scaled, desc, delta, targets


def hyperparameter_sweep_pca_ridge(train_ctrl, train_desc, train_delta, val_ctrl, val_desc, val_delta):
    """Grid search over PCA+ridge's n_components and alpha, selected by
    validation MAE — same principle as the conditional model's sweep,
    so the comparison in metrics.json is no longer "tuned conditional
    model vs. untouched baselines"."""
    n_comp_options = [10, 20, 50]
    alpha_options = [0.1, 1.0, 10.0]
    results = []
    best = None
    for n_comp in n_comp_options:
        n_comp_capped = min(n_comp, train_ctrl.shape[0], train_ctrl.shape[1])
        for alpha in alpha_options:
            model = PCARidgeBaseline(n_components=n_comp_capped, alpha=alpha).fit(train_ctrl, train_desc, train_delta)
            val_pred = model.predict_delta(val_ctrl, val_desc)
            val_mae = mae(val_delta, val_pred)
            row = {"n_components": n_comp_capped, "alpha": alpha, "val_mae": val_mae}
            results.append(row)
            print(f"  pca_ridge sweep {row}")
            if best is None or val_mae < best[0]:
                best = (val_mae, {"n_components": n_comp_capped, "alpha": alpha}, model)
    return best[1], best[2], results


def hyperparameter_sweep_small_mlp(train_ctrl, train_desc, train_delta, val_ctrl, val_desc, val_delta, seed):
    """Grid search over the small MLP baseline's hidden_dim, selected by
    validation MAE."""
    hidden_dim_options = [32, 64, 128, 256]
    results = []
    best = None
    for hidden_dim in hidden_dim_options:
        model = SmallMLPBaseline(hidden_dim=hidden_dim, seed=seed, max_iter=400).fit(train_ctrl, train_desc, train_delta)
        val_pred = model.predict_delta(val_ctrl, val_desc)
        val_mae = mae(val_delta, val_pred)
        row = {"hidden_dim": hidden_dim, "val_mae": val_mae}
        results.append(row)
        print(f"  small_mlp sweep {row}")
        if best is None or val_mae < best[0]:
            best = (val_mae, {"hidden_dim": hidden_dim}, model)
    return best[1], best[2], results


def hyperparameter_sweep(train_ctrl, train_desc, train_delta, val_ctrl, val_desc, val_delta, n_genes, descriptor_dim, seed):
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
            n_genes=n_genes, target_descriptor_dim=descriptor_dim,
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
        n_top_genes: int, n_workers: int, descriptor: str = "mean_std", n_components: int = 20,
        source_path: str = ARC_CHALLENGE_TRAIN_URL,
        gene2vec_path: str | None = None) -> dict:
    print(f"Loading Arc Virtual Cell Challenge dataset from {source_path} (backed mode)...")
    full_dataset = load_arc_challenge_dataset(source_path, backed="r")
    t0 = time.time()
    dataset = subsample_and_materialize(
        full_dataset, seed=seed, cells_per_target_cap=cells_per_target_cap,
        control_sample_size=control_sample_size, n_top_genes=n_top_genes, n_workers=n_workers,
        path_or_url=source_path,
    )
    fetch_time = time.time() - t0
    print(f"Scaled-up dataset: {dataset.n_cells} cells x {dataset.n_genes} genes, "
          f"{len(dataset.targets())} targets (fetch took {fetch_time:.1f}s)")

    manifest = held_out_target_split(dataset, seed=seed)
    print(f"Split: {len(manifest.train_targets)} train / {len(manifest.val_targets)} val / {len(manifest.test_targets)} test targets")

    scaler = fit_scaler(dataset, manifest)
    gene2vec_coverage = None
    if descriptor == "coexpression":
        print(f"Fitting co-expression target descriptor (n_components={n_components})...")
        gene_descriptors = fit_coexpression_descriptors(dataset, manifest, n_components=n_components, seed=seed)
        descriptor_dim = min(n_components, dataset.n_genes)
    elif descriptor == "gene2vec":
        if not gene2vec_path:
            raise ValueError("descriptor='gene2vec' requires --gene2vec-path")
        print(f"Loading pretrained Gene2Vec target descriptor from {gene2vec_path}...")
        gene_descriptors, gene2vec_coverage = fit_gene2vec_descriptors(dataset, gene2vec_path)
        descriptor_dim = 200
        print(f"Gene2Vec vocabulary coverage: {gene2vec_coverage:.1%} of this dataset's {dataset.n_genes} genes")
    else:
        gene_descriptors = fit_control_mean_std_descriptors(dataset, manifest)
        descriptor_dim = 2

    train_ctrl, train_desc, train_delta, _ = build_arrays(dataset, manifest, manifest.train_cell_ids, scaler, gene_descriptors, descriptor_dim, seed)
    val_ctrl, val_desc, val_delta, _ = build_arrays(dataset, manifest, manifest.val_cell_ids, scaler, gene_descriptors, descriptor_dim, seed)
    test_ctrl, test_desc, test_delta, test_targets = build_arrays(dataset, manifest, manifest.test_cell_ids, scaler, gene_descriptors, descriptor_dim, seed)
    no_effect_mae = mae(test_delta, np.zeros_like(test_delta))

    results = {}
    no_effect_pred = NoEffectBaseline(n_genes=dataset.n_genes).predict_delta(n_cells=test_delta.shape[0])
    results["1_no_effect"] = {"mae": mae(test_delta, no_effect_pred)}

    seen_mean = SeenTargetMeanBaseline().fit(dataset, manifest)
    seen_mean_pred = seen_mean.predict_delta(test_targets)
    results["2_seen_target_mean"] = {"mae": mae(test_delta, seen_mean_pred)}

    print("Running PCA+ridge hyperparameter sweep...")
    pca_ridge_best_cfg, pca_ridge_best_model, pca_ridge_sweep_results = hyperparameter_sweep_pca_ridge(
        train_ctrl, train_desc, train_delta, val_ctrl, val_desc, val_delta
    )
    print(f"Best PCA+ridge config: {pca_ridge_best_cfg}")
    pca_ridge_pred = pca_ridge_best_model.predict_delta(test_ctrl, test_desc)
    pca_ridge_train_mae = mae(train_delta, pca_ridge_best_model.predict_delta(train_ctrl, train_desc))
    pca_ridge_val_mae = mae(val_delta, pca_ridge_best_model.predict_delta(val_ctrl, val_desc))
    results["3_pca_ridge_tuned"] = {
        "mae": mae(test_delta, pca_ridge_pred), "best_config": pca_ridge_best_cfg,
        "train_mae": pca_ridge_train_mae, "val_mae": pca_ridge_val_mae,
    }

    print("Running small-MLP hyperparameter sweep...")
    small_mlp_best_cfg, small_mlp_best_model, small_mlp_sweep_results = hyperparameter_sweep_small_mlp(
        train_ctrl, train_desc, train_delta, val_ctrl, val_desc, val_delta, seed
    )
    print(f"Best small MLP config: {small_mlp_best_cfg}")
    small_mlp_pred = small_mlp_best_model.predict_delta(test_ctrl, test_desc)
    small_mlp_train_mae = mae(train_delta, small_mlp_best_model.predict_delta(train_ctrl, train_desc))
    small_mlp_val_mae = mae(val_delta, small_mlp_best_model.predict_delta(val_ctrl, val_desc))
    results["4_small_mlp_tuned"] = {
        "mae": mae(test_delta, small_mlp_pred), "best_config": small_mlp_best_cfg,
        "train_mae": small_mlp_train_mae, "val_mae": small_mlp_val_mae,
    }

    print("Running conditional-model hyperparameter sweep...")
    best_cfg, best_trainer, sweep_results = hyperparameter_sweep(
        train_ctrl, train_desc, train_delta, val_ctrl, val_desc, val_delta, dataset.n_genes, descriptor_dim, seed
    )
    print(f"Best config: {best_cfg}")
    conditional_pred = best_trainer.predict_delta(test_ctrl, test_desc)
    conditional_train_mae = mae(train_delta, best_trainer.predict_delta(train_ctrl, train_desc))
    conditional_val_mae = mae(val_delta, best_trainer.predict_delta(val_ctrl, val_desc))
    results["5_conditional_model_tuned"] = {
        "mae": mae(test_delta, conditional_pred), "best_config": best_cfg,
        "train_mae": conditional_train_mae, "val_mae": conditional_val_mae,
    }

    for name, r in results.items():
        r["improvement_over_no_effect"] = improvement_over_no_effect(r["mae"], no_effect_mae)

    report = build_full_report(test_delta, conditional_pred, test_targets)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_trainer.save(output_dir / "checkpoint_conditional_tuned")

    summary = {
        "pilot_note": f"SCALED-UP subsample: {dataset.n_cells} cells (full dataset has 221,273).",
        "descriptor": descriptor,
        "descriptor_dim": descriptor_dim,
        "gene2vec_coverage": gene2vec_coverage,
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
        "hyperparameter_sweep_conditional_model": sweep_results,
        "hyperparameter_sweep_pca_ridge": pca_ridge_sweep_results,
        "hyperparameter_sweep_small_mlp": small_mlp_sweep_results,
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
    parser.add_argument("--descriptor", choices=["mean_std", "coexpression", "gene2vec"], default="mean_std")
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument("--gene2vec-path", type=str, default=None,
                         help="Path to the Gene2Vec pretrained embedding text file "
                              "(data/gene2vec/gene2vec_dim_200_iter_9.txt). Required when "
                              "--descriptor gene2vec is used.")
    parser.add_argument("--h5ad-path", type=str, default=ARC_CHALLENGE_TRAIN_URL,
                         help="Local file path or gs:// URL for the Arc h5ad. Use a local path "
                              "(after scripts/download_arc_h5ad.py) to avoid the scattered-remote-"
                              "row-access problem documented in docs/experiment_log.md.")
    args = parser.parse_args()
    summary = run(
        args.output_dir, seed=args.seed, cells_per_target_cap=args.cells_per_target_cap,
        control_sample_size=args.control_sample_size, n_top_genes=args.n_top_genes, n_workers=args.n_workers,
        descriptor=args.descriptor, n_components=args.n_components, source_path=args.h5ad_path,
        gene2vec_path=args.gene2vec_path,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "per_target"}, indent=2, default=str))


if __name__ == "__main__":
    main()

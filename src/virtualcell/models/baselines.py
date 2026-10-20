"""Simple baselines, built before any conditional model.

- NoEffectBaseline: predicted delta = 0 relative to context-matched control.
- SeenTargetMeanBaseline: training-derived per-target mean delta; valid only
  for in-distribution evaluation, cannot predict held-out targets.
"""
from __future__ import annotations

import numpy as np

from virtualcell.data.adapters import CanonicalDataset, control_reference_mean
from virtualcell.data.splits import SplitManifest


class NoEffectBaseline:
    """Predicts zero change from the control reference for any target."""

    def __init__(self, n_genes: int):
        self.n_genes = n_genes

    def predict_delta(self, n_cells: int) -> np.ndarray:
        return np.zeros((n_cells, self.n_genes), dtype=np.float32)


class SeenTargetMeanBaseline:
    """Per-target mean delta learned on train; undefined for unseen targets."""

    def __init__(self):
        self.target_mean_delta_: dict[str, np.ndarray] = {}
        self.global_control_mean_: np.ndarray | None = None

    def fit(self, dataset: CanonicalDataset, manifest: SplitManifest) -> "SeenTargetMeanBaseline":
        obs = dataset.adata.obs
        train_mask = obs.index.isin(manifest.train_cell_ids)
        ctrl_mask = dataset.control_mask()
        ctrl_ref = control_reference_mean(dataset)
        self.global_control_mean_ = ctrl_ref["__global__"]

        X = np.asarray(dataset.adata.X)
        for target in manifest.train_targets:
            mask = train_mask & (obs["perturbation_id"] == target).to_numpy() & (~ctrl_mask)
            if mask.sum() == 0:
                continue
            self.target_mean_delta_[target] = X[mask].mean(axis=0) - self.global_control_mean_
        return self

    def predict_delta(self, targets: list[str]) -> np.ndarray:
        n_genes = len(self.global_control_mean_)
        out = np.zeros((len(targets), n_genes), dtype=np.float32)
        for i, t in enumerate(targets):
            if t in self.target_mean_delta_:
                out[i] = self.target_mean_delta_[t]
            # else: unseen target -> leave as zero-effect fallback; caller
            # should treat these as "cannot predict" for held-out eval.
        return out

    def known_targets(self) -> set[str]:
        return set(self.target_mean_delta_.keys())

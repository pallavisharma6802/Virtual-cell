"""Training-only preprocessing fit to avoid leakage into held-out splits."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from virtualcell.data.adapters import CanonicalDataset
from virtualcell.data.splits import SplitManifest


@dataclass
class Scaler:
    mean_: np.ndarray
    std_: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.std_ + self.mean_


def fit_scaler(dataset: CanonicalDataset, manifest: SplitManifest, eps: float = 1e-6) -> Scaler:
    """Fit mean/std on the TRAIN split only."""
    obs = dataset.adata.obs
    train_mask = obs.index.isin(manifest.train_cell_ids)
    X_raw = dataset.adata.X[train_mask]
    # Sparse matrices don't support np.asarray → use sparse-aware paths
    if hasattr(X_raw, "toarray"):
        mean_ = np.asarray(X_raw.mean(axis=0)).ravel()
        # var = E[x^2] - E[x]^2, computed without materializing the full matrix
        mean_sq = np.asarray(X_raw.power(2).mean(axis=0)).ravel()
        std_ = np.sqrt(np.maximum(mean_sq - mean_ ** 2, 0.0)) + eps
    else:
        X_train = np.asarray(X_raw)
        mean_ = X_train.mean(axis=0)
        std_ = X_train.std(axis=0) + eps
    return Scaler(mean_=mean_, std_=std_)

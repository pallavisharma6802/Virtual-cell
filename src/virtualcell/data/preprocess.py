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
    X_train = np.asarray(dataset.adata.X[train_mask])
    mean_ = X_train.mean(axis=0)
    std_ = X_train.std(axis=0) + eps
    return Scaler(mean_=mean_, std_=std_)

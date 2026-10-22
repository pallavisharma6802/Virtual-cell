"""Simple baselines, built before any conditional model (plan section 4A,
built in order: no-effect -> seen-target mean -> PCA+ridge -> small MLP).

- NoEffectBaseline: predicted delta = 0 relative to context-matched control.
- SeenTargetMeanBaseline: training-derived per-target mean delta; valid only
  for in-distribution evaluation, cannot predict held-out targets.
- PCARidgeBaseline: PCA(control profile) + target descriptor -> ridge
  regression on delta. Uses the same non-leaking control-context and
  target-side-information inputs as the conditional model, so it is a
  fair "is the extra architecture worth it" comparison.
- SmallMLPBaseline: a single-hidden-layer MLP on the same two feature
  blocks, via sklearn's MLPRegressor — deliberately simpler than
  ConditionalDeltaModel's separate encoders + fusion, to test whether
  that extra structure earns its complexity.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

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


class PCARidgeBaseline:
    """PCA(control profile) concatenated with a target descriptor, fed
    into ridge regression predicting the gene-wise delta.

    PCA and ridge are both fit on TRAIN only. `n_components` is capped to
    the number of training samples/features available, so this degrades
    gracefully on small datasets instead of raising.
    """

    def __init__(self, n_components: int = 20, alpha: float = 1.0):
        self.n_components = n_components
        self.alpha = alpha
        self.pca: PCA | None = None
        self.ridge: Ridge | None = None

    def fit(self, control: np.ndarray, target_desc: np.ndarray, delta: np.ndarray) -> "PCARidgeBaseline":
        n_components = min(self.n_components, control.shape[0], control.shape[1])
        self.pca = PCA(n_components=n_components, random_state=0)
        control_pca = self.pca.fit_transform(control)
        features = np.concatenate([control_pca, target_desc], axis=1)
        self.ridge = Ridge(alpha=self.alpha, random_state=0)
        self.ridge.fit(features, delta)
        return self

    def predict_delta(self, control: np.ndarray, target_desc: np.ndarray) -> np.ndarray:
        control_pca = self.pca.transform(control)
        features = np.concatenate([control_pca, target_desc], axis=1)
        return self.ridge.predict(features)


class SmallMLPBaseline:
    """A plain single-hidden-layer MLP on [control profile, target
    descriptor] -> delta, via sklearn. Deliberately simpler than
    ConditionalDeltaModel (no separate encoders, no fusion block), to
    serve as the "does the extra architecture help" comparison point.
    """

    def __init__(self, hidden_dim: int = 64, seed: int = 0, max_iter: int = 500):
        self.model = MLPRegressor(
            hidden_layer_sizes=(hidden_dim,),
            activation="relu",
            random_state=seed,
            max_iter=max_iter,
            early_stopping=True,
            n_iter_no_change=15,
        )

    def fit(self, control: np.ndarray, target_desc: np.ndarray, delta: np.ndarray) -> "SmallMLPBaseline":
        features = np.concatenate([control, target_desc], axis=1)
        self.model.fit(features, delta)
        return self

    def predict_delta(self, control: np.ndarray, target_desc: np.ndarray) -> np.ndarray:
        features = np.concatenate([control, target_desc], axis=1)
        return self.model.predict(features)

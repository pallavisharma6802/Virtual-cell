"""Documented control-context sampling.

Perturbed cells are not 1:1 matched to controls (plan section 2). For each
perturbed cell we sample one control cell from the SAME batch when
available, falling back to a uniformly random control cell from the same
split otherwise. This sampling is documented here rather than left
implicit, and is available at inference (it only uses observed batch
metadata and other control cells, never the perturbed cell's own label).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from virtualcell.data.adapters import CanonicalDataset


def sample_matched_controls(
    dataset: CanonicalDataset,
    cell_ids: list[str],
    control_pool_cell_ids: list[str],
    seed: int = 0,
) -> np.ndarray:
    """Return one sampled control expression profile per cell in `cell_ids`.

    `control_pool_cell_ids` restricts which control cells may be sampled
    from (e.g. the train split's controls), so held-out cells never draw
    on a control that is itself held out.
    """
    rng = np.random.default_rng(seed)
    obs = dataset.adata.obs
    # Keep sparse — materialize only the single row we need per cell
    X = dataset.adata.X

    pool_mask = obs.index.isin(control_pool_cell_ids)
    pool_batches = obs.loc[pool_mask, "batch_id"] if "batch_id" in obs.columns else pd.Series(dtype=object)
    pool_positions = np.where(pool_mask)[0]

    batch_to_positions: dict = {}
    for pos, batch in zip(pool_positions, pool_batches.to_numpy() if len(pool_batches) else []):
        batch_to_positions.setdefault(batch, []).append(pos)

    cell_batches = obs.loc[cell_ids, "batch_id"] if "batch_id" in obs.columns else None

    out = np.zeros((len(cell_ids), X.shape[1]), dtype=np.float32)
    for i, cid in enumerate(cell_ids):
        candidates = None
        if cell_batches is not None:
            batch = cell_batches.loc[cid]
            candidates = batch_to_positions.get(batch)
        if not candidates:
            candidates = pool_positions
        chosen = int(candidates[rng.integers(0, len(candidates))])
        row = X[chosen]
        out[i] = np.asarray(row.todense()).ravel() if hasattr(row, "todense") else np.asarray(row).ravel()
    return out

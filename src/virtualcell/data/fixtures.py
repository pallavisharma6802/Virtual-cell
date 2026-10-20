"""Deterministic synthetic AnnData fixtures for CPU smoke tests.

These fixtures are NOT real biological data. They exist only to validate
schema, split logic, preprocessing, and the training/eval pipeline shape.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

N_GENES_DEFAULT = 200
N_TARGETS_DEFAULT = 24
N_BATCHES_DEFAULT = 3
N_CELL_TYPES_DEFAULT = 2
CELLS_PER_TARGET_DEFAULT = 20
CELLS_PER_CONTROL_DEFAULT = 60


def make_synthetic_dataset(
    n_genes: int = N_GENES_DEFAULT,
    n_targets: int = N_TARGETS_DEFAULT,
    n_batches: int = N_BATCHES_DEFAULT,
    n_cell_types: int = N_CELL_TYPES_DEFAULT,
    cells_per_target: int = CELLS_PER_TARGET_DEFAULT,
    cells_per_control: int = CELLS_PER_CONTROL_DEFAULT,
    seed: int = 0,
) -> AnnData:
    """Build a small synthetic perturbation dataset with a known effect.

    Each perturbation target has a fixed, reproducible "true" effect vector
    added to a control-derived baseline plus noise. Effects are shared by
    target so that a model conditioned on target identity has a learnable
    signal, while held-out targets have effects the model has never seen.
    """
    rng = np.random.default_rng(seed)

    gene_ids = [f"gene_{i:04d}" for i in range(n_genes)]
    target_ids = [f"target_{i:03d}" for i in range(n_targets)]
    batch_ids = [f"batch_{i}" for i in range(n_batches)]
    cell_types = [f"celltype_{i}" for i in range(n_cell_types)]

    base_mean = rng.normal(loc=3.0, scale=1.0, size=n_genes)

    target_effects = {
        t: rng.normal(loc=0.0, scale=0.8, size=n_genes) * (rng.random(n_genes) < 0.15)
        for t in target_ids
    }

    rows = []
    expr_rows = []
    cell_counter = 0

    def sample_cell(is_control: bool, target: str | None, batch: str, cell_type: str):
        nonlocal cell_counter
        batch_shift = rng.normal(loc=0.0, scale=0.2, size=n_genes)
        celltype_shift = rng.normal(loc=0.0, scale=0.3, size=n_genes)
        noise = rng.normal(loc=0.0, scale=0.5, size=n_genes)
        profile = base_mean + batch_shift + celltype_shift + noise
        if not is_control:
            profile = profile + target_effects[target]
        profile = np.clip(profile, a_min=0.0, a_max=None)
        rows.append(
            {
                "cell_id": f"cell_{cell_counter:06d}",
                "perturbation_id": "control" if is_control else target,
                "is_control": is_control,
                "cell_type": cell_type,
                "batch_id": batch,
                "donor_id": f"donor_{cell_counter % 4}",
            }
        )
        expr_rows.append(profile)
        cell_counter += 1

    for batch in batch_ids:
        for cell_type in cell_types:
            for _ in range(cells_per_control // (n_batches * n_cell_types) or 1):
                sample_cell(True, None, batch, cell_type)

    for target in target_ids:
        for _ in range(cells_per_target):
            batch = batch_ids[rng.integers(0, n_batches)]
            cell_type = cell_types[rng.integers(0, n_cell_types)]
            sample_cell(False, target, batch, cell_type)

    obs = pd.DataFrame(rows).set_index("cell_id")
    var = pd.DataFrame(index=gene_ids)
    X = np.vstack(expr_rows).astype(np.float32)

    adata = AnnData(X=X, obs=obs, var=var)
    adata.uns["target_effects_true"] = target_effects
    adata.uns["gene_ids"] = gene_ids
    adata.uns["fixture_seed"] = seed
    return adata

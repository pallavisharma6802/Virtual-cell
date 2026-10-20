"""Canonical dataset adapter interface.

Exposes a stable schema regardless of underlying source:
cell_id, perturbation_id, is_control, cell_type, batch_id, donor_id,
ordered gene_ids, and an expression matrix, plus provenance metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from anndata import AnnData

REQUIRED_OBS_COLUMNS = ("perturbation_id", "is_control")
OPTIONAL_OBS_COLUMNS = ("cell_type", "batch_id", "donor_id")


@dataclass
class DatasetProvenance:
    source_name: str
    source_version: str | None = None
    source_url: str | None = None
    is_synthetic: bool = False
    notes: str = ""


@dataclass
class CanonicalDataset:
    adata: AnnData
    gene_ids: list[str]
    provenance: DatasetProvenance
    extra: dict = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return self.adata.n_obs

    @property
    def n_genes(self) -> int:
        return len(self.gene_ids)

    def control_mask(self) -> np.ndarray:
        return self.adata.obs["is_control"].to_numpy(dtype=bool)

    def targets(self) -> list[str]:
        pert = self.adata.obs.loc[~self.control_mask(), "perturbation_id"]
        return sorted(pert.unique().tolist())


def validate_schema(adata: AnnData) -> None:
    missing = [c for c in REQUIRED_OBS_COLUMNS if c not in adata.obs.columns]
    if missing:
        raise ValueError(f"adata.obs missing required columns: {missing}")
    if adata.obs["is_control"].dtype != bool:
        raise ValueError("adata.obs['is_control'] must be boolean")
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("adata must have at least one cell and one gene")
    if adata.var.index.duplicated().any():
        raise ValueError("duplicate gene_ids in adata.var.index")
    control_targets = adata.obs.loc[adata.obs["is_control"], "perturbation_id"].unique()
    if len(control_targets) > 1 or (len(control_targets) == 1 and control_targets[0] != "control"):
        raise ValueError("control cells must have perturbation_id == 'control'")


def from_anndata(
    adata: AnnData,
    source_name: str,
    is_synthetic: bool = False,
    source_version: str | None = None,
    source_url: str | None = None,
) -> CanonicalDataset:
    validate_schema(adata)
    gene_ids = list(adata.var.index)
    provenance = DatasetProvenance(
        source_name=source_name,
        source_version=source_version,
        source_url=source_url,
        is_synthetic=is_synthetic,
    )
    return CanonicalDataset(adata=adata, gene_ids=gene_ids, provenance=provenance)


def control_reference_mean(dataset: CanonicalDataset, group_obs_keys: tuple[str, ...] = ()) -> dict:
    """Build a context-matched control mean profile.

    Perturbed cells are not one-to-one matched to controls, so we build a
    control distribution per context (e.g. batch, cell_type) and fall back
    to the global control mean when the specific context has no controls.
    """
    obs = dataset.adata.obs
    ctrl_mask = dataset.control_mask()
    X = dataset.adata.X
    global_mean = np.asarray(X[ctrl_mask].mean(axis=0)).ravel()

    result = {"__global__": global_mean}
    if not group_obs_keys:
        return result

    groups = obs.loc[ctrl_mask, list(group_obs_keys)].drop_duplicates()
    for _, row in groups.iterrows():
        key = tuple(row[k] for k in group_obs_keys)
        mask = ctrl_mask.copy()
        for k, v in zip(group_obs_keys, key):
            mask &= (obs[k] == v).to_numpy()
        if mask.sum() == 0:
            continue
        result[key] = np.asarray(X[mask].mean(axis=0)).ravel()
    return result

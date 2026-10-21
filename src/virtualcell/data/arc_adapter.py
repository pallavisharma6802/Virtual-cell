"""Adapter for the Arc Institute Virtual Cell Challenge 2025 dataset.

Schema verified directly against the live `adata_Training.h5ad` file in
`gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/train/`
on 2026-11-05 (see docs/dataset_card.md). That file has NO `is_control`
column and NO `cell_type`/`donor_id` columns — this adapter does not
invent them.

Loads backed (memory-mapped) by default since the file is ~14.4 GiB.
"""
from __future__ import annotations

import anndata as ad

from virtualcell.data.adapters import CanonicalDataset, DatasetProvenance, validate_schema

ARC_CHALLENGE_TRAIN_URL = (
    "gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/train/"
    "adata_Training.h5ad"
)
CONTROL_LABEL = "non-targeting"


def load_arc_challenge_dataset(path_or_url: str, backed: str | None = "r") -> CanonicalDataset:
    """Load the Arc Virtual Cell Challenge training AnnData and map it to
    the canonical schema.

    `path_or_url` may be a local path or a `gs://` URL (requires `gcsfs`
    to be installed for the latter). `backed="r"` avoids loading the full
    matrix into memory; pass `backed=None` to force a full in-memory load.
    """
    if path_or_url.startswith("gs://"):
        import gcsfs

        fs = gcsfs.GCSFileSystem(token="anon")
        file_handle = fs.open(path_or_url, "rb")
        adata = ad.read_h5ad(file_handle, backed=backed)
    else:
        adata = ad.read_h5ad(path_or_url, backed=backed)

    missing = {"target_gene", "batch"} - set(adata.obs.columns)
    if missing:
        raise ValueError(
            f"Arc challenge file is missing expected obs columns: {missing}. "
            "The verified schema in docs/dataset_card.md no longer matches "
            "the live file; re-verify before adapting further."
        )

    obs = adata.obs.copy()
    is_control = (obs["target_gene"] == CONTROL_LABEL).to_numpy()
    perturbation_id = obs["target_gene"].astype(str).where(~is_control, "control")

    obs["perturbation_id"] = perturbation_id
    obs["is_control"] = is_control
    obs["batch_id"] = obs["batch"].astype(str)
    adata.obs = obs

    validate_schema(adata)

    provenance = DatasetProvenance(
        source_name="arc_virtual_cell_challenge_2025",
        source_version="2025/train",
        source_url=ARC_CHALLENGE_TRAIN_URL,
        is_synthetic=False,
        notes=(
            "No cell_type or donor_id in source file; omitted rather than "
            "fabricated. Control label mapped from 'non-targeting' to "
            "'control' to match canonical schema."
        ),
    )
    return CanonicalDataset(adata=adata, gene_ids=list(adata.var.index), provenance=provenance)

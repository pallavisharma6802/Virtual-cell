"""Adapter for the Arc Institute Virtual Cell Challenge 2025 dataset.

Schema verified directly against the live `adata_Training.h5ad` file in
`gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/train/`
on 2026-11-05 (see docs/dataset_card.md). That file has NO `is_control`
column and NO `cell_type`/`donor_id` columns — this adapter does not
invent them.

Loads backed (memory-mapped) by default since the file is ~14.4 GiB.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import anndata as ad
import numpy as np

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


def materialize_cells_concurrent(
    path_or_url: str,
    cell_positions: np.ndarray,
    n_workers: int = 8,
    chunk_size: int = 500,
) -> np.ndarray:
    """Fetch the dense expression rows for `cell_positions` from the
    remote backed h5ad using multiple concurrent GCS connections.

    This is purely a throughput optimization for the network-bound
    materialization step (observed ~23 min for ~11k cells with a single
    sequential connection — see docs/experiment_log.md). It does not
    change any modeling logic: each worker opens its own `gcsfs`/h5py
    file handle (safe — independent handles to the same remote object)
    and reads a disjoint contiguous chunk of `cell_positions`, sorted
    first so each worker's reads stay close together on disk/in the
    remote object for better chunk cache reuse.

    Returns a dense float32 array of shape (len(cell_positions), n_genes)
    in the SAME ORDER as the input `cell_positions` (not sorted order).
    """
    import anndata as ad
    import gcsfs

    order = np.argsort(cell_positions)
    sorted_positions = np.asarray(cell_positions)[order]

    chunks = [sorted_positions[i : i + chunk_size] for i in range(0, len(sorted_positions), chunk_size)]

    def fetch_chunk(chunk: np.ndarray) -> np.ndarray:
        fs = gcsfs.GCSFileSystem(token="anon")
        f = fs.open(path_or_url, "rb")
        adata = ad.read_h5ad(f, backed="r")
        sub = adata[chunk, :].to_memory()
        X = sub.X
        return np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)

    t0 = time.time()
    results = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(fetch_chunk, chunk): i for i, chunk in enumerate(chunks)}
        for future in futures:
            idx = futures[future]
            results[idx] = future.result()
    elapsed = time.time() - t0
    print(f"  materialize_cells_concurrent: {len(cell_positions)} cells, "
          f"{len(chunks)} chunks, {n_workers} workers, {elapsed:.1f}s")

    stacked_sorted = np.concatenate(results, axis=0)
    out = np.empty_like(stacked_sorted)
    out[order] = stacked_sorted
    return out

"""Target-gene descriptors usable for genuinely held-out targets.

A learned per-ID embedding cannot represent a target never seen in
training (plan section 4A, "Unseen-target caveat"). The descriptor here
instead uses each target gene's own control-population statistics (mean,
std of that gene's expression across control cells). This is:
  - generalizable: every target corresponds to a gene that also appears
    in the fixed gene vocabulary, so a descriptor exists for any target,
    seen or unseen;
  - non-leaking: computed only from control cells in the TRAIN split,
    never from perturbed cells or the held-out target's own response.

This is a deliberately simple, weak descriptor (2 scalars per target), not
a pretrained gene-representation model. It is a documented placeholder for
the "appropriately licensed pretrained gene representation" the plan
mentions as the stronger alternative; swapping it in later would mean
adding real coverage/provenance documentation first, not here.
"""
from __future__ import annotations

import numpy as np

from virtualcell.data.adapters import CanonicalDataset
from virtualcell.data.splits import SplitManifest


def fit_control_mean_std_descriptors(
    dataset: CanonicalDataset, manifest: SplitManifest
) -> dict[str, np.ndarray]:
    """One descriptor per gene (usable as a target), fit on TRAIN controls only."""
    obs = dataset.adata.obs
    train_mask = obs.index.isin(manifest.train_cell_ids)
    ctrl_mask = dataset.control_mask()
    mask = train_mask & ctrl_mask

    X_train_ctrl = np.asarray(dataset.adata.X[mask])
    mean_per_gene = X_train_ctrl.mean(axis=0)
    std_per_gene = X_train_ctrl.std(axis=0)

    return {
        gene: np.array([mean_per_gene[i], std_per_gene[i]], dtype=np.float32)
        for i, gene in enumerate(dataset.gene_ids)
    }


def descriptor_for_targets(
    targets: list[str], gene_descriptors: dict[str, np.ndarray], descriptor_dim: int = 2
) -> np.ndarray:
    """Look up each target's descriptor by matching its ID to a gene_id.

    Targets not found in `gene_descriptors` (e.g. a multi-gene or
    synthetic target name that doesn't match a gene_id 1:1) fall back to
    zeros; callers should treat that as a documented limitation, not a
    silent success.
    """
    out = np.zeros((len(targets), descriptor_dim), dtype=np.float32)
    for i, t in enumerate(targets):
        if t in gene_descriptors:
            out[i] = gene_descriptors[t]
    return out

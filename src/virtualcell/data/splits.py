"""Split manifests with leak guards.

Primary claim of this project rests on the held-out-target split: train,
val, and test must have disjoint perturbation target IDs (including all
replicates of a target).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from virtualcell.data.adapters import CanonicalDataset


@dataclass
class SplitManifest:
    name: str
    seed: int
    train_cell_ids: list[str]
    val_cell_ids: list[str]
    test_cell_ids: list[str]
    train_targets: list[str]
    val_targets: list[str]
    test_targets: list[str]

    def assert_disjoint_targets(self) -> None:
        train_set, val_set, test_set = (
            set(self.train_targets),
            set(self.val_targets),
            set(self.test_targets),
        )
        if train_set & val_set or train_set & test_set or val_set & test_set:
            raise AssertionError("split manifest has overlapping targets between splits")


def held_out_target_split(
    dataset: CanonicalDataset,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 0,
    name: str = "held_out_target",
) -> SplitManifest:
    """Group by perturbation target; disjoint target IDs across splits.

    Control cells are distributed into all three splits (they carry no
    target leakage since 'control' is not a held-out target).
    """
    rng = np.random.default_rng(seed)
    obs = dataset.adata.obs
    targets = dataset.targets()
    targets_shuffled = list(targets)
    rng.shuffle(targets_shuffled)

    n = len(targets_shuffled)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    test_targets = targets_shuffled[:n_test]
    val_targets = targets_shuffled[n_test : n_test + n_val]
    train_targets = targets_shuffled[n_test + n_val :]

    def cells_for_targets(target_list: list[str]) -> list[str]:
        mask = obs["perturbation_id"].isin(target_list).to_numpy()
        return list(obs.index[mask])

    train_cells = cells_for_targets(train_targets)
    val_cells = cells_for_targets(val_targets)
    test_cells = cells_for_targets(test_targets)

    control_ids = list(obs.index[dataset.control_mask()])
    rng.shuffle(control_ids)
    n_ctrl = len(control_ids)
    c_test = int(round(n_ctrl * test_frac))
    c_val = int(round(n_ctrl * val_frac))
    train_cells += control_ids[c_test + c_val :]
    val_cells += control_ids[c_test : c_test + c_val]
    test_cells += control_ids[:c_test]

    manifest = SplitManifest(
        name=name,
        seed=seed,
        train_cell_ids=sorted(train_cells),
        val_cell_ids=sorted(val_cells),
        test_cell_ids=sorted(test_cells),
        train_targets=sorted(train_targets),
        val_targets=sorted(val_targets),
        test_targets=sorted(test_targets),
    )
    manifest.assert_disjoint_targets()
    return manifest


def in_distribution_sanity_split(
    dataset: CanonicalDataset,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 0,
    name: str = "in_distribution_sanity",
) -> SplitManifest:
    """Disjoint CELLS from previously seen targets (easiest benchmark).

    Targets are intentionally shared across splits here; this split must
    never be reported as evidence of unseen-target generalization.
    """
    rng = np.random.default_rng(seed)
    obs = dataset.adata.obs
    all_cells = list(obs.index)
    rng.shuffle(all_cells)
    n = len(all_cells)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test_cells = all_cells[:n_test]
    val_cells = all_cells[n_test : n_test + n_val]
    train_cells = all_cells[n_test + n_val :]

    all_targets = sorted(obs["perturbation_id"].unique().tolist())
    return SplitManifest(
        name=name,
        seed=seed,
        train_cell_ids=sorted(train_cells),
        val_cell_ids=sorted(val_cells),
        test_cell_ids=sorted(test_cells),
        train_targets=all_targets,
        val_targets=all_targets,
        test_targets=all_targets,
    )

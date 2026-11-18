import pytest

from virtualcell.data.adapters import from_anndata
from virtualcell.data.fixtures import make_synthetic_dataset
from virtualcell.data.splits import held_out_target_split, in_distribution_sanity_split


def _dataset(seed=2):
    adata = make_synthetic_dataset(seed=seed)
    return from_anndata(adata, source_name="synthetic_fixture", is_synthetic=True)


def test_held_out_target_split_has_no_target_overlap():
    dataset = _dataset()
    manifest = held_out_target_split(dataset, seed=0)
    manifest.assert_disjoint_targets()
    assert len(manifest.train_targets) > 0
    assert len(manifest.val_targets) > 0
    assert len(manifest.test_targets) > 0


def test_held_out_target_split_has_no_cell_overlap():
    dataset = _dataset()
    manifest = held_out_target_split(dataset, seed=0)
    train, val, test = (
        set(manifest.train_cell_ids),
        set(manifest.val_cell_ids),
        set(manifest.test_cell_ids),
    )
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()


def test_held_out_target_split_covers_all_cells():
    dataset = _dataset()
    manifest = held_out_target_split(dataset, seed=0)
    all_split_cells = set(manifest.train_cell_ids) | set(manifest.val_cell_ids) | set(manifest.test_cell_ids)
    assert all_split_cells == set(dataset.adata.obs.index)


def test_held_out_target_split_is_seeded_reproducible():
    dataset = _dataset()
    m1 = held_out_target_split(dataset, seed=42)
    m2 = held_out_target_split(dataset, seed=42)
    assert m1.test_targets == m2.test_targets
    assert m1.train_cell_ids == m2.train_cell_ids


def test_in_distribution_split_shares_targets_but_not_cells():
    dataset = _dataset()
    manifest = in_distribution_sanity_split(dataset, seed=0)
    assert set(manifest.train_targets) == set(manifest.test_targets)
    assert set(manifest.train_cell_ids) & set(manifest.test_cell_ids) == set()


def test_small_positive_frac_still_allocates_at_least_one_target():
    # With 24 targets and frac=0.001, int(round(24*0.001))=0 — must clamp to 1
    adata = make_synthetic_dataset(seed=7)
    dataset = from_anndata(adata, source_name="synthetic_fixture", is_synthetic=True)
    manifest = held_out_target_split(dataset, val_frac=0.001, test_frac=0.001, seed=0)
    assert len(manifest.val_targets) >= 1
    assert len(manifest.test_targets) >= 1
    assert len(manifest.train_targets) >= 1
    manifest.assert_disjoint_targets()


def test_zero_frac_allocates_no_targets():
    adata = make_synthetic_dataset(seed=7)
    dataset = from_anndata(adata, source_name="synthetic_fixture", is_synthetic=True)
    manifest = held_out_target_split(dataset, val_frac=0.0, test_frac=0.2, seed=0)
    assert len(manifest.val_targets) == 0
    assert len(manifest.val_cell_ids) == 0
    assert len(manifest.test_targets) >= 1
    assert len(manifest.train_targets) >= 1


def test_fracs_summing_to_one_raises():
    adata = make_synthetic_dataset(seed=7)
    dataset = from_anndata(adata, source_name="synthetic_fixture", is_synthetic=True)
    with pytest.raises(ValueError, match="no training data"):
        held_out_target_split(dataset, val_frac=0.5, test_frac=0.5, seed=0)


def test_negative_frac_raises():
    adata = make_synthetic_dataset(seed=7)
    dataset = from_anndata(adata, source_name="synthetic_fixture", is_synthetic=True)
    with pytest.raises(ValueError, match="non-negative"):
        held_out_target_split(dataset, val_frac=-0.1, test_frac=0.2, seed=0)

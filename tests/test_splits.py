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

import pytest

from virtualcell.data.adapters import from_anndata, validate_schema
from virtualcell.data.fixtures import make_synthetic_dataset


def test_synthetic_fixture_has_required_columns():
    adata = make_synthetic_dataset(seed=1)
    validate_schema(adata)


def test_control_cells_labeled_control():
    adata = make_synthetic_dataset(seed=1)
    dataset = from_anndata(adata, source_name="synthetic_fixture", is_synthetic=True)
    ctrl = dataset.adata.obs.loc[dataset.control_mask(), "perturbation_id"]
    assert (ctrl == "control").all()


def test_validate_schema_rejects_missing_columns():
    adata = make_synthetic_dataset(seed=1)
    del adata.obs["is_control"]
    with pytest.raises(ValueError):
        validate_schema(adata)


def test_gene_ids_unique_and_ordered():
    adata = make_synthetic_dataset(seed=1)
    dataset = from_anndata(adata, source_name="synthetic_fixture", is_synthetic=True)
    assert len(dataset.gene_ids) == len(set(dataset.gene_ids))
    assert dataset.gene_ids == list(adata.var.index)

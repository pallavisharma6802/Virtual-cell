import numpy as np

from virtualcell.data.adapters import from_anndata
from virtualcell.data.fixtures import make_synthetic_dataset
from virtualcell.data.splits import held_out_target_split
from virtualcell.data.target_descriptors import (
    descriptor_for_targets,
    fit_coexpression_descriptors,
    fit_control_mean_std_descriptors,
)


def _dataset(seed=5):
    adata = make_synthetic_dataset(seed=seed)
    return from_anndata(adata, source_name="synthetic_fixture", is_synthetic=True)


def test_coexpression_descriptor_covers_all_genes():
    dataset = _dataset()
    manifest = held_out_target_split(dataset, seed=0)
    descriptors = fit_coexpression_descriptors(dataset, manifest, n_components=10)
    assert set(descriptors.keys()) == set(dataset.gene_ids)
    some_gene = dataset.gene_ids[0]
    assert descriptors[some_gene].shape == (10,)


def test_coexpression_descriptor_exists_for_held_out_targets():
    dataset = _dataset()
    manifest = held_out_target_split(dataset, seed=0)
    descriptors = fit_coexpression_descriptors(dataset, manifest, n_components=10)
    desc = descriptor_for_targets(manifest.test_targets, descriptors, descriptor_dim=10)
    assert desc.shape == (len(manifest.test_targets), 10)
    assert not np.all(desc == 0)


def test_coexpression_descriptor_caps_n_components():
    dataset = _dataset()
    manifest = held_out_target_split(dataset, seed=0)
    descriptors = fit_coexpression_descriptors(dataset, manifest, n_components=100000)
    some_gene = dataset.gene_ids[0]
    assert descriptors[some_gene].shape[0] <= dataset.n_genes


def test_coexpression_descriptor_differs_from_mean_std():
    dataset = _dataset()
    manifest = held_out_target_split(dataset, seed=0)
    coexpr = fit_coexpression_descriptors(dataset, manifest, n_components=5)
    mean_std = fit_control_mean_std_descriptors(dataset, manifest)
    some_gene = dataset.gene_ids[0]
    assert coexpr[some_gene].shape != mean_std[some_gene].shape

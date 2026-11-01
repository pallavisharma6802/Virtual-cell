import numpy as np

from virtualcell.data.adapters import from_anndata
from virtualcell.data.fixtures import make_synthetic_dataset
from virtualcell.data.splits import held_out_target_split
from virtualcell.data.target_descriptors import (
    descriptor_for_targets,
    fit_coexpression_descriptors,
    fit_control_mean_std_descriptors,
    fit_gene2vec_descriptors,
    fit_response_role_descriptors,
    load_gene2vec_table,
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


def test_load_gene2vec_table_parses_whitespace_separated_rows(tmp_path):
    path = tmp_path / "gene2vec.txt"
    path.write_text("GENE_A " + " ".join(str(float(i)) for i in range(200)) + "\n"
                     "GENE_B " + " ".join(str(float(i) * 2) for i in range(200)) + "\n")
    table = load_gene2vec_table(path)
    assert set(table.keys()) == {"GENE_A", "GENE_B"}
    assert table["GENE_A"].shape == (200,)
    assert np.allclose(table["GENE_B"], np.arange(200, dtype=np.float32) * 2)


def test_load_gene2vec_table_skips_malformed_lines(tmp_path):
    path = tmp_path / "gene2vec.txt"
    path.write_text("GENE_A 1.0 2.0\n"
                     "GENE_B " + " ".join(str(float(i)) for i in range(200)) + "\n")
    table = load_gene2vec_table(path)
    assert set(table.keys()) == {"GENE_B"}


def test_fit_gene2vec_descriptors_reports_coverage_and_falls_back_for_unseen_genes(tmp_path):
    dataset = _dataset()
    known_gene = dataset.gene_ids[0]
    path = tmp_path / "gene2vec.txt"
    path.write_text(f"{known_gene} " + " ".join(str(float(i)) for i in range(200)) + "\n")

    gene_descriptors, coverage = fit_gene2vec_descriptors(dataset, path)
    assert known_gene in gene_descriptors
    assert gene_descriptors[known_gene].shape == (200,)
    assert coverage == 1 / dataset.n_genes

    desc = descriptor_for_targets([known_gene, "NOT_IN_TABLE"], gene_descriptors, descriptor_dim=200)
    assert not np.all(desc[0] == 0)
    assert np.all(desc[1] == 0)


def test_fit_response_role_descriptors_covers_every_profiled_gene_including_non_targets():
    rng = np.random.default_rng(0)
    gene_names = [f"G{i}" for i in range(10)]
    train_targets = ["G0", "G0", "G1", "G2", "G3"]
    train_delta = rng.normal(size=(5, len(gene_names))).astype(np.float32)

    descriptors = fit_response_role_descriptors(train_targets, train_delta, gene_names, n_components=3, seed=0)

    assert set(descriptors.keys()) == set(gene_names)
    assert descriptors["G9"].shape == (3,)  # G9 was never a train target, still gets a descriptor


def test_fit_response_role_descriptors_held_out_target_not_derived_from_its_own_row():
    rng = np.random.default_rng(1)
    gene_names = [f"G{i}" for i in range(6)]
    train_targets = ["G0", "G1", "G2"]
    train_delta = rng.normal(size=(3, len(gene_names))).astype(np.float32)

    descriptors = fit_response_role_descriptors(train_targets, train_delta, gene_names, n_components=2, seed=0)
    held_out_gene = "G5"
    assert held_out_gene not in train_targets
    assert held_out_gene in descriptors
    assert descriptors[held_out_gene].shape == (2,)

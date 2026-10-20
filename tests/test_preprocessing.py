import numpy as np

from virtualcell.data.adapters import from_anndata
from virtualcell.data.fixtures import make_synthetic_dataset
from virtualcell.data.preprocess import fit_scaler
from virtualcell.data.splits import held_out_target_split


def test_scaler_fit_only_on_train_split():
    adata = make_synthetic_dataset(seed=3)
    dataset = from_anndata(adata, source_name="synthetic_fixture", is_synthetic=True)
    manifest = held_out_target_split(dataset, seed=0)

    scaler = fit_scaler(dataset, manifest)

    obs = dataset.adata.obs
    train_mask = obs.index.isin(manifest.train_cell_ids)
    X_train = np.asarray(dataset.adata.X[train_mask])
    expected_mean = X_train.mean(axis=0)

    assert np.allclose(scaler.mean_, expected_mean)


def test_scaler_transform_inverse_roundtrip():
    adata = make_synthetic_dataset(seed=3)
    dataset = from_anndata(adata, source_name="synthetic_fixture", is_synthetic=True)
    manifest = held_out_target_split(dataset, seed=0)
    scaler = fit_scaler(dataset, manifest)

    X = np.asarray(dataset.adata.X)
    X_scaled = scaler.transform(X)
    X_back = scaler.inverse_transform(X_scaled)
    assert np.allclose(X, X_back, atol=1e-4)

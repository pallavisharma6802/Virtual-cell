import numpy as np

from virtualcell.models.baselines import PCARidgeBaseline, SmallMLPBaseline


def _toy_data(n=40, n_genes=10, descriptor_dim=2, seed=0):
    rng = np.random.default_rng(seed)
    control = rng.normal(size=(n, n_genes)).astype(np.float32)
    target_desc = rng.normal(size=(n, descriptor_dim)).astype(np.float32)
    true_w = rng.normal(size=(descriptor_dim, n_genes)).astype(np.float32)
    delta = (target_desc @ true_w).astype(np.float32)
    return control, target_desc, delta


def test_pca_ridge_fits_and_predicts_correct_shape():
    control, desc, delta = _toy_data(n=30, seed=1)
    model = PCARidgeBaseline(n_components=5, alpha=1.0).fit(control, desc, delta)
    pred = model.predict_delta(control[:5], desc[:5])
    assert pred.shape == (5, 10)


def test_pca_ridge_handles_n_components_larger_than_data():
    control, desc, delta = _toy_data(n=3, n_genes=10, seed=1)
    model = PCARidgeBaseline(n_components=50).fit(control, desc, delta)
    pred = model.predict_delta(control, desc)
    assert pred.shape == control.shape[:1] + (10,)


def test_small_mlp_fits_and_predicts_correct_shape():
    control, desc, delta = _toy_data(n=30, seed=2)
    model = SmallMLPBaseline(hidden_dim=8, max_iter=50).fit(control, desc, delta)
    pred = model.predict_delta(control[:5], desc[:5])
    assert pred.shape == (5, 10)


def test_small_mlp_learns_better_than_random_on_simple_signal():
    control, desc, delta = _toy_data(n=200, n_genes=5, seed=3)
    model = SmallMLPBaseline(hidden_dim=16, max_iter=300).fit(control, desc, delta)
    pred = model.predict_delta(control, desc)
    mae_model = np.mean(np.abs(delta - pred))
    mae_zero = np.mean(np.abs(delta))
    assert mae_model < mae_zero

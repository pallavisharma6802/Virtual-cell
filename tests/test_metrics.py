import math

import numpy as np

from virtualcell.evaluation.metrics import (
    delta_correlation,
    grouped_bootstrap_ci,
    improvement_over_no_effect,
    mae,
    per_target_metrics,
    rmse,
)


def test_mae_rmse_basic():
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[1.0, 2.0], [3.0, 5.0]])
    assert math.isclose(mae(y_true, y_pred), 0.25, abs_tol=1e-6)
    assert rmse(y_true, y_pred) > 0


def test_delta_correlation_handles_constant_vector():
    y_true = np.zeros(10)
    y_pred = np.random.default_rng(0).normal(size=10)
    result = delta_correlation(y_true, y_pred)
    assert np.isnan(result["pearson"])
    assert np.isnan(result["spearman"])


def test_per_target_metrics_groups_correctly():
    y_true = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    y_pred = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    labels = ["a", "a", "b"]
    result = per_target_metrics(y_true, y_pred, labels)
    assert set(result.keys()) == {"a", "b"}
    assert result["a"]["n_cells"] == 2
    assert result["a"]["mae"] == 0.0


def test_improvement_over_no_effect():
    assert improvement_over_no_effect(model_mae=0.5, no_effect_mae=1.0) == 0.5
    assert improvement_over_no_effect(model_mae=1.5, no_effect_mae=1.0) == -0.5


def test_grouped_bootstrap_ci_shape():
    values = {"t1": 0.5, "t2": 0.6, "t3": 0.4}
    result = grouped_bootstrap_ci(values, n_boot=200, seed=0)
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]
    assert result["n_groups"] == 3

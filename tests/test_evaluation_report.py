import numpy as np

from virtualcell.evaluation.biological import de_genes_by_ttest, top_k_precision_recall
from virtualcell.evaluation.report import build_full_report, per_context_mae, worst_targets


def test_per_context_mae_breaks_down_by_group():
    y_true = np.array([[1.0], [1.0], [2.0], [2.0]])
    y_pred = np.array([[1.0], [1.1], [2.0], [2.5]])
    contexts = ["a", "a", "b", "b"]
    result = per_context_mae(y_true, y_pred, contexts)
    assert set(result.keys()) == {"a", "b"}
    assert result["a"]["n_cells"] == 2
    assert result["b"]["mae"] > result["a"]["mae"]


def test_worst_targets_ranks_by_metric():
    per_target = {
        "t1": {"mae": 0.1},
        "t2": {"mae": 0.9},
        "t3": {"mae": 0.5},
    }
    worst = worst_targets(per_target, n=2, metric="mae")
    assert [name for name, _ in worst] == ["t2", "t3"]


def test_build_full_report_includes_expected_keys():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=(10, 5))
    y_pred = y_true + rng.normal(scale=0.1, size=(10, 5))
    targets = ["a"] * 5 + ["b"] * 5
    report = build_full_report(y_true, y_pred, targets)
    assert "per_target" in report
    assert "grouped_bootstrap_ci_mae" in report
    assert "worst_3_targets_by_mae" in report


def test_de_genes_by_ttest_detects_shifted_genes():
    rng = np.random.default_rng(0)
    control = rng.normal(loc=0.0, scale=1.0, size=(50, 5))
    perturbed = control.copy()
    perturbed[:, 0] += 5.0  # strong, detectable shift in gene 0 only
    mask = de_genes_by_ttest(perturbed, control)
    assert mask[0]
    assert not mask[1:].any()


def test_top_k_precision_recall_perfect_match():
    true_delta = np.array([0.0, 5.0, 0.0, -5.0, 0.0])
    pred_delta = np.array([0.0, 4.0, 0.0, -4.0, 0.0])
    de_mask = np.array([False, True, False, True, False])
    result = top_k_precision_recall(true_delta, pred_delta, de_mask, k=2)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


def test_top_k_precision_recall_handles_no_de_genes():
    true_delta = np.zeros(5)
    pred_delta = np.zeros(5)
    de_mask = np.zeros(5, dtype=bool)
    result = top_k_precision_recall(true_delta, pred_delta, de_mask, k=2)
    assert np.isnan(result["precision"])
    assert np.isnan(result["recall"])

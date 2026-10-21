"""Research-grade evaluation report: per-target metrics, context/batch
breakdown, worst-case targets, and DE overlap — not just pooled averages
(plan section 5).
"""
from __future__ import annotations

import numpy as np

from virtualcell.evaluation.biological import de_genes_by_ttest, top_k_precision_recall
from virtualcell.evaluation.metrics import grouped_bootstrap_ci, per_target_metrics


def per_context_mae(
    y_true: np.ndarray, y_pred: np.ndarray, context_labels: list[str]
) -> dict[str, dict]:
    """MAE/RMSE broken down by a context variable (e.g. batch_id), not
    just pooled across all cells — exposes context-confounded performance.
    """
    from virtualcell.evaluation.metrics import mae, rmse

    labels_arr = np.asarray(context_labels)
    out = {}
    for ctx in sorted(set(context_labels)):
        mask = labels_arr == ctx
        if mask.sum() == 0:
            continue
        out[ctx] = {
            "n_cells": int(mask.sum()),
            "mae": mae(y_true[mask], y_pred[mask]),
            "rmse": rmse(y_true[mask], y_pred[mask]),
        }
    return out


def worst_targets(per_target: dict[str, dict], n: int = 3, metric: str = "mae") -> list[tuple[str, dict]]:
    """Return the n targets with the highest error, for failure-case review."""
    ranked = sorted(per_target.items(), key=lambda kv: kv[1][metric], reverse=True)
    return ranked[:n]


def de_overlap_by_target(
    perturbed_expr_raw: np.ndarray,
    control_expr_raw: np.ndarray,
    target_labels: list[str],
    true_delta: np.ndarray,
    pred_delta: np.ndarray,
    k: int = 20,
) -> dict[str, dict]:
    """Per-target DE-gene overlap between predicted and statistically
    defined true DE genes. `perturbed_expr_raw`/`control_expr_raw` must be
    in the SAME (e.g. raw or consistently scaled) numeric space used to
    define "differential expression" via the t-test.
    """
    labels_arr = np.asarray(target_labels)
    out = {}
    for t in sorted(set(target_labels)):
        mask = labels_arr == t
        if mask.sum() < 2:
            continue
        de_mask = de_genes_by_ttest(perturbed_expr_raw[mask], control_expr_raw)
        mean_true_delta = true_delta[mask].mean(axis=0)
        mean_pred_delta = pred_delta[mask].mean(axis=0)
        out[t] = top_k_precision_recall(mean_true_delta, mean_pred_delta, de_mask, k=k)
    return out


def build_full_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_labels: list[str],
    context_labels: list[str] | None = None,
) -> dict:
    """Assemble the standard set of research-grade metrics for one model
    on one evaluation set: per-target metrics, grouped bootstrap CI over
    targets, worst-k targets, and (if provided) per-context breakdown.
    """
    per_target = per_target_metrics(y_true, y_pred, target_labels)
    mae_by_target = {t: m["mae"] for t, m in per_target.items()}
    ci = grouped_bootstrap_ci(mae_by_target)

    report = {
        "per_target": per_target,
        "grouped_bootstrap_ci_mae": ci,
        "worst_3_targets_by_mae": [
            {"target": t, **m} for t, m in worst_targets(per_target, n=3, metric="mae")
        ],
    }
    if context_labels is not None:
        report["per_context_mae"] = per_context_mae(y_true, y_pred, context_labels)
    return report

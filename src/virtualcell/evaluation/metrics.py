"""Evaluation metrics for predicted vs. true expression deltas."""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def delta_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Pearson/Spearman correlation between true and predicted delta vectors.

    Handles constant vectors (zero variance) by returning NaN rather than
    raising, since correlation is undefined in that case.
    """
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return {"pearson": float("nan"), "spearman": float("nan")}
    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_r, _ = spearmanr(y_true, y_pred)
    return {"pearson": float(pearson_r), "spearman": float(spearman_r)}


def per_target_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_labels: list[str],
) -> dict[str, dict]:
    """Compute MAE/RMSE/correlation per target, not just pooled averages."""
    results: dict[str, dict] = {}
    targets = sorted(set(target_labels))
    labels_arr = np.asarray(target_labels)
    for t in targets:
        mask = labels_arr == t
        yt, yp = y_true[mask], y_pred[mask]
        if yt.shape[0] == 0:
            continue
        true_mean = yt.mean(axis=0)
        pred_mean = yp.mean(axis=0)
        results[t] = {
            "n_cells": int(mask.sum()),
            "mae": mae(true_mean, pred_mean),
            "rmse": rmse(true_mean, pred_mean),
            **delta_correlation(true_mean, pred_mean),
        }
    return results


def mean_pearson_across_genes(
    pred_mean_deltas: dict[str, np.ndarray],
    true_mean_deltas: dict[str, np.ndarray],
) -> dict:
    """Mean Pearson correlation across genes, the standard metric GEARS reports.

    For each gene, correlate the predicted mean expression delta (over the
    1500-gene output space) with the actual observed mean delta. Then
    average those per-gene correlations. This is the gene-level metric,
    not the per-cell metric: it asks 'does the model predict the right
    *pattern* of change across genes, averaged across cells?'
    """
    genes = sorted(set(pred_mean_deltas) & set(true_mean_deltas))
    if not genes:
        return {"mean_pearson": float("nan"), "n_genes": 0}
    pearson_vals = []
    for g in genes:
        p, t = pred_mean_deltas[g], true_mean_deltas[g]
        if np.std(t) == 0 or np.std(p) == 0:
            pearson_vals.append(0.0)
        else:
            r, _ = pearsonr(p, t)
            pearson_vals.append(float(r))
    arr = np.array(pearson_vals)
    return {
        "mean_pearson": float(arr.mean()),
        "median_pearson": float(np.median(arr)),
        "std_pearson": float(arr.std()),
        "n_genes": len(genes),
    }


def improvement_over_no_effect(model_mae: float, no_effect_mae: float) -> float:
    """Relative MAE improvement vs. the no-effect baseline (positive = better)."""
    if no_effect_mae == 0:
        return float("nan")
    return (no_effect_mae - model_mae) / no_effect_mae


def grouped_bootstrap_ci(
    values_by_group: dict[str, float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Bootstrap CI over groups (e.g. perturbation targets), not raw cells.

    Resampling targets (not individual cells) respects that cells within a
    target are correlated replicates, not independent samples.
    """
    rng = np.random.default_rng(seed)
    groups = list(values_by_group.keys())
    vals = np.array([values_by_group[g] for g in groups])
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = vals[idx].mean()
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return {"mean": float(vals.mean()), "ci_low": float(lo), "ci_high": float(hi), "n_groups": n}

"""Differentially-expressed gene overlap with a documented statistical
procedure (per-gene Welch's t-test, Benjamini-Hochberg FDR correction),
plus precision/recall of the model's predicted top-k delta genes against
that DE set. Gene sets, species, and procedure are stated explicitly
rather than left implicit (plan section 5).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests


def de_genes_by_ttest(
    perturbed_expr: np.ndarray,
    control_expr: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """Boolean mask over genes: True where perturbed vs. control differ
    significantly after Welch's t-test + Benjamini-Hochberg correction.

    This is a simple, documented procedure — not a claim of biological
    mechanism, just a statistically defined "differentially expressed"
    label for benchmarking predicted-vs-true gene rankings.
    """
    if perturbed_expr.shape[0] < 2 or control_expr.shape[0] < 2:
        return np.zeros(perturbed_expr.shape[1], dtype=bool)
    _, p_values = ttest_ind(perturbed_expr, control_expr, axis=0, equal_var=False)
    p_values = np.nan_to_num(p_values, nan=1.0)
    rejected, _, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
    return rejected


def top_k_precision_recall(
    true_delta: np.ndarray,
    pred_delta: np.ndarray,
    de_mask: np.ndarray,
    k: int,
) -> dict:
    """Precision/recall of the model's top-k |predicted delta| genes
    against the statistically-defined DE gene set for one target.

    Returns NaN precision/recall when there is nothing to retrieve
    (de_mask has no True entries) rather than silently returning 0, since
    0 would misleadingly suggest a failed prediction rather than an
    undefined comparison.
    """
    n_genes = true_delta.shape[0]
    k = min(k, n_genes)
    top_k_idx = np.argsort(-np.abs(pred_delta))[:k]
    predicted_mask = np.zeros(n_genes, dtype=bool)
    predicted_mask[top_k_idx] = True

    n_de = int(de_mask.sum())
    if n_de == 0:
        return {"precision": float("nan"), "recall": float("nan"), "n_de_genes": 0}

    true_positives = int((predicted_mask & de_mask).sum())
    precision = true_positives / k if k > 0 else float("nan")
    recall = true_positives / n_de
    return {"precision": precision, "recall": recall, "n_de_genes": n_de}

"""Milestone 10: gene-level evaluation with Pearson correlation.

Reframes the prediction problem as: predict the MEAN perturbation
signature per target gene (not per individual cell), and evaluate using
Pearson correlation between predicted and actual mean gene expression
delta vectors. This is the same metric GEARS (Roohani et al. 2023)
reports, so it makes our results comparable to the published literature.

New approach also tested: SVD latent factor prediction.
  - Factorize training gene mean-deltas into K components via SVD.
  - Regress gene features -> K codes (much smaller regression target).
  - Reconstruct test-gene mean-deltas from predicted codes.
  - Hypothesis: the low-rank structure of the perturbation effect space
    is easier to predict than the full 1500-dim output directly.

Same data/split as Milestones 8-9 (seed=0, same Replogle per-cell file).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from virtualcell.data.target_descriptors import (
    descriptor_for_targets,
    fit_response_role_descriptors,
    fit_gene2vec_descriptors,
    load_gene2vec_table,
)
from virtualcell.evaluation.metrics import mean_pearson_across_genes, mae

MIN_CELLS_PER_TARGET = 5


def subsample_positions(gene_col, cells_per_target_cap, control_sample_size, seed):
    rng = np.random.default_rng(seed)
    targets_by_row = [None if g == "non-targeting" else g for g in gene_col]
    ctrl_positions = [i for i, t in enumerate(targets_by_row) if t is None]
    ctrl_sample = rng.choice(ctrl_positions, size=min(control_sample_size, len(ctrl_positions)), replace=False)
    by_target: dict[str, list[int]] = {}
    for i, t in enumerate(targets_by_row):
        if t is not None:
            by_target.setdefault(t, []).append(i)
    target_positions, kept_targets = [], []
    for t, positions in by_target.items():
        if len(positions) < MIN_CELLS_PER_TARGET:
            continue
        chosen = rng.choice(positions, size=min(cells_per_target_cap, len(positions)), replace=False)
        target_positions.extend(chosen.tolist())
        kept_targets.append(t)
    return sorted(ctrl_sample.tolist()), sorted(target_positions), sorted(kept_targets)


def compute_mean_deltas(
    delta: np.ndarray,
    targets: list[str],
    target_set: list[str],
) -> dict[str, np.ndarray]:
    """Return {gene: mean_delta} for each gene in target_set."""
    result = {}
    for g in target_set:
        mask = [i for i, t in enumerate(targets) if t == g]
        if mask:
            result[g] = delta[mask].mean(axis=0)
    return result


def pearson_all_genes(pred_dict, true_dict):
    genes = sorted(set(pred_dict) & set(true_dict))
    vals = []
    for g in genes:
        p, t = pred_dict[g], true_dict[g]
        if np.std(t) < 1e-6 or np.std(p) < 1e-6:
            vals.append(0.0)
        else:
            r, _ = pearsonr(p, t)
            vals.append(float(np.clip(r, -1, 1)))
    return float(np.mean(vals)), float(np.median(vals)), genes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad-path", default="data/replogle/K562_essential_raw_singlecell_01.h5ad")
    parser.add_argument("--gene2vec-path", default="data/gene2vec/gene2vec_dim_200_iter_9.txt")
    parser.add_argument("--output", default="outputs/replogle_genelevel/metrics.json")
    parser.add_argument("--cells-per-target-cap", type=int, default=40)
    parser.add_argument("--control-sample-size", type=int, default=4000)
    parser.add_argument("--n-top-genes", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    t0 = time.time()
    print(f"Opening {args.h5ad_path} backed ...")
    adata = ad.read_h5ad(args.h5ad_path, backed="r")
    gene_col = adata.obs["gene"].astype(str).tolist()
    gene_names = adata.var["gene_name"].tolist() if "gene_name" in adata.var.columns else adata.var_names.tolist()
    print(f"Full file: {adata.shape[0]} cells x {adata.shape[1]} genes")

    ctrl_pos, target_pos, _ = subsample_positions(
        gene_col, args.cells_per_target_cap, args.control_sample_size, args.seed,
    )
    all_pos = sorted(set(ctrl_pos) | set(target_pos))
    print(f"Fetching {len(all_pos)} rows ...")
    X_full = adata.X[all_pos]
    X_full = np.asarray(X_full.todense() if hasattr(X_full, "todense") else X_full, dtype=np.float32)
    pos_to_row = {p: i for i, p in enumerate(all_pos)}
    fetch_time = time.time() - t0
    print(f"Fetched in {fetch_time:.1f}s")

    ctrl_rows = [pos_to_row[p] for p in ctrl_pos]
    target_rows = [pos_to_row[p] for p in target_pos]
    cell_targets = [gene_col[p] for p in target_pos]

    ctrl_var = X_full[ctrl_rows].var(axis=0)
    top_gene_idx = np.argsort(ctrl_var)[::-1][: args.n_top_genes]
    top_gene_idx = np.sort(top_gene_idx)
    top_gene_names = [gene_names[i] for i in top_gene_idx]

    X_ctrl = X_full[ctrl_rows][:, top_gene_idx]
    X_target = X_full[target_rows][:, top_gene_idx]
    control_mean = X_ctrl.mean(axis=0)

    rng = np.random.default_rng(args.seed)
    ctrl_idx = rng.integers(0, len(X_ctrl), size=len(target_rows))
    matched_ctrl = X_ctrl[ctrl_idx]
    all_delta = X_target - matched_ctrl

    unique_targets = sorted(set(cell_targets))
    train_t, test_t_full = train_test_split(unique_targets, test_size=0.3, random_state=args.seed)
    val_t, test_t = train_test_split(test_t_full, test_size=0.5, random_state=args.seed)
    print(f"Split: {len(train_t)} train / {len(val_t)} val / {len(test_t)} test targets")

    train_targets_list = [cell_targets[i] for i, t in enumerate(cell_targets) if t in set(train_t)]
    train_delta_rows = [all_delta[i] for i, t in enumerate(cell_targets) if t in set(train_t)]
    train_delta = np.stack(train_delta_rows)

    # Compute per-gene mean deltas (ground truth)
    train_mean_deltas = compute_mean_deltas(train_delta, train_targets_list, train_t)

    # For val and test: need their actual per-cell deltas too
    val_delta_rows = [all_delta[i] for i, t in enumerate(cell_targets) if t in set(val_t)]
    val_targets_list = [cell_targets[i] for i, t in enumerate(cell_targets) if t in set(val_t)]
    val_delta = np.stack(val_delta_rows)
    test_delta_rows = [all_delta[i] for i, t in enumerate(cell_targets) if t in set(test_t)]
    test_targets_list = [cell_targets[i] for i, t in enumerate(cell_targets) if t in set(test_t)]
    test_delta = np.stack(test_delta_rows)

    val_mean_deltas = compute_mean_deltas(val_delta, val_targets_list, val_t)
    test_mean_deltas = compute_mean_deltas(test_delta, test_targets_list, test_t)

    # Training matrix: rows=training genes, cols=gene expression (1500)
    train_gene_list = sorted(train_mean_deltas.keys())
    D_train = np.stack([train_mean_deltas[g] for g in train_gene_list])  # (n_train_genes, n_expr_genes)
    print(f"Training perturbation matrix: {D_train.shape}")

    # Load Gene2Vec
    print("Loading Gene2Vec ...")
    gene2vec_table = load_gene2vec_table(Path(args.gene2vec_path))

    mean_std_table = {
        gene: np.array([control_mean[i], X_ctrl[:, i].std()], dtype=np.float32)
        for i, gene in enumerate(top_gene_names)
    }
    gene2vec_lookup = {g: gene2vec_table[g] for g in top_gene_names if g in gene2vec_table}

    response_role_dim = min(30, len(train_gene_list), D_train.shape[1])
    response_role_table = fit_response_role_descriptors(
        train_targets_list, train_delta, top_gene_names, n_components=response_role_dim, seed=args.seed,
    )
    print(f"Response-role descriptor: {response_role_dim} components")

    # ------------------------------------------------------------------ #
    # Evaluation helper: predict mean delta per gene from a descriptor
    # ------------------------------------------------------------------ #
    def evaluate_descriptor(name, table, dim, svd_rank=None):
        """Predict test-gene mean deltas from a gene descriptor table.

        Two approaches:
        1. Direct ridge: concatenate descriptor, regress -> mean delta (1500-dim)
        2. SVD latent: regress -> K codes, reconstruct from SVD basis
        """
        X_tr = np.array([table.get(g, np.zeros(dim, dtype=np.float32)) for g in train_gene_list])
        X_va = np.array([table.get(g, np.zeros(dim, dtype=np.float32)) for g in val_t])
        X_te = np.array([table.get(g, np.zeros(dim, dtype=np.float32)) for g in test_t])
        Y_tr = D_train  # (n_train_genes, n_output_genes)

        results = {}

        # Direct ridge: regress features -> full 1500-dim mean delta
        for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
            m = Ridge(alpha=alpha)
            m.fit(X_tr, Y_tr)
            val_preds = {g: m.predict(X_va[[i]])[0] for i, g in enumerate(val_t)}
            val_r, _, _ = pearson_all_genes(val_preds, val_mean_deltas)
            if "direct_ridge" not in results or val_r > results["direct_ridge"]["val_pearson"]:
                results["direct_ridge"] = {
                    "best_alpha": alpha, "val_pearson": val_r,
                    "model": m, "X_te": X_te,
                }

        best_ridge = results["direct_ridge"]
        m = best_ridge["model"]
        test_preds = {g: m.predict(X_te[[i]])[0] for i, g in enumerate(test_t)}
        test_r, test_r_med, _ = pearson_all_genes(test_preds, test_mean_deltas)
        test_mae_val = float(np.mean([
            np.mean(np.abs(test_preds[g] - test_mean_deltas[g])) for g in test_t
        ]))
        results["direct_ridge"] = {
            "best_alpha": best_ridge["best_alpha"],
            "val_pearson": best_ridge["val_pearson"],
            "test_pearson_mean": test_r,
            "test_pearson_median": test_r_med,
            "test_mae": test_mae_val,
        }
        print(f"  {name}+direct_ridge: test Pearson={test_r:.4f}, MAE={test_mae_val:.4f}")

        # SVD latent factor approach: factorize D_train, regress -> codes, reconstruct
        for k in (10, 20, 30, 50):
            if k >= min(D_train.shape):
                continue
            svd = TruncatedSVD(n_components=k, random_state=args.seed)
            codes_train = svd.fit_transform(D_train)  # (n_train_genes, k)
            # Regress features -> codes
            best_svd_val = -np.inf
            best_svd_model = None
            for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
                m = Ridge(alpha=alpha)
                m.fit(X_tr, codes_train)
                val_codes = m.predict(X_va)
                val_preds_svd = {g: svd.inverse_transform(val_codes[[i]])[0] for i, g in enumerate(val_t)}
                val_r_svd, _, _ = pearson_all_genes(val_preds_svd, val_mean_deltas)
                if val_r_svd > best_svd_val:
                    best_svd_val, best_svd_model = val_r_svd, (m, alpha)
            m_best, best_alpha = best_svd_model
            test_codes = m_best.predict(X_te)
            test_preds_svd = {g: svd.inverse_transform(test_codes[[i]])[0] for i, g in enumerate(test_t)}
            test_r_svd, test_r_svd_med, _ = pearson_all_genes(test_preds_svd, test_mean_deltas)
            test_mae_svd = float(np.mean([
                np.mean(np.abs(test_preds_svd[g] - test_mean_deltas[g])) for g in test_t
            ]))
            key = f"svd_k{k}"
            results[key] = {
                "k": k, "best_alpha": best_alpha,
                "val_pearson": best_svd_val,
                "test_pearson_mean": test_r_svd,
                "test_pearson_median": test_r_svd_med,
                "test_mae": test_mae_svd,
            }
            print(f"  {name}+SVD_k={k}: test Pearson={test_r_svd:.4f}, MAE={test_mae_svd:.4f}")

        return {k: {kk: vv for kk, vv in v.items() if kk != "model" and kk != "X_te"}
                for k, v in results.items()}

    # ------------------------------------------------------------------ #
    # Baselines first
    # ------------------------------------------------------------------ #
    print("Computing baselines ...")
    # No-effect: predict zeros
    zero_preds = {g: np.zeros(args.n_top_genes, dtype=np.float32) for g in test_t}
    no_effect_r, _, _ = pearson_all_genes(zero_preds, test_mean_deltas)
    no_effect_mae_val = float(np.mean([
        np.mean(np.abs(test_mean_deltas[g])) for g in test_t
    ]))

    # Global mean: predict the average training perturbation
    global_mean = D_train.mean(axis=0)
    global_mean_preds = {g: global_mean for g in test_t}
    global_r, global_r_med, _ = pearson_all_genes(global_mean_preds, test_mean_deltas)
    global_mae_val = float(np.mean([
        np.mean(np.abs(global_mean - test_mean_deltas[g])) for g in test_t
    ]))

    print(f"No-effect baseline: Pearson={no_effect_r:.4f}, MAE={no_effect_mae_val:.4f}")
    print(f"Global-mean baseline: Pearson={global_r:.4f}, MAE={global_mae_val:.4f}")

    # ------------------------------------------------------------------ #
    # Descriptor-based prediction
    # ------------------------------------------------------------------ #
    print("\nEvaluating mean_std descriptor ...")
    mean_std_results = evaluate_descriptor("mean_std", mean_std_table, 2)

    print("\nEvaluating Gene2Vec descriptor ...")
    gene2vec_results = evaluate_descriptor("gene2vec", gene2vec_lookup, 200)

    print("\nEvaluating response_role descriptor ...")
    rr_results = evaluate_descriptor("response_role", response_role_table, response_role_dim)

    summary = {
        "dataset": "Replogle et al. 2022 K562 essential RAW SINGLE-CELL",
        "evaluation": "gene-level mean-delta Pearson correlation (standard GEARS metric)",
        "n_distinct_targets": len(unique_targets),
        "n_train_targets": len(train_t), "n_val_targets": len(val_t), "n_test_targets": len(test_t),
        "fetch_time_seconds": fetch_time,
        "baselines": {
            "no_effect": {"test_pearson_mean": no_effect_r, "test_mae": no_effect_mae_val},
            "global_mean": {"test_pearson_mean": global_r, "test_pearson_median": global_r_med, "test_mae": global_mae_val},
        },
        "mean_std": mean_std_results,
        "gene2vec": gene2vec_results,
        "response_role": rr_results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("mean_std", "gene2vec", "response_role")}, indent=2))


if __name__ == "__main__":
    main()

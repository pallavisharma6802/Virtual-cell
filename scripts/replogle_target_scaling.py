"""Milestone 6: does having ~15x more distinct perturbation targets fix
the architecture-doesn't-help problem seen on Arc (150 targets, ~90 train)?

Uses the Replogle et al. 2022 (Cell) K562 "essential" genome-scale
Perturb-seq dataset, pseudobulk release (CC BY-NC-SA 4.0, non-commercial
research use), downloaded directly from its Figshare+ deposit
(doi 10.25452/figshare.plus.20029387, file K562_essential_raw_bulk_01.h5ad,
~2176 distinct perturbed genes + 109 non-targeting control replicates,
8563 genes profiled, 80MB pseudobulk file -- no EC2 needed).

This is a genuinely different regime from the Arc pilots: one row per
perturbation (pseudobulk mean across that perturbation's cells), not
per-cell. So there is no per-cell "control profile" input varying across
examples -- every example shares the same global control pseudobulk
profile. This isolates exactly the question we want answered: with ~15x
more distinct targets to learn from, does a target-descriptor -> delta
regression generalize better to held-out targets than it did with only
150 targets (90 train)?

Descriptors compared, both already implemented in
virtualcell.data.target_descriptors and reused unchanged here:
  - mean_std: each target gene's own baseline (control) mean/std
    expression, read directly from this file's `var` statistics.
  - gene2vec: the same pretrained Gene2Vec embedding table used in
    Milestone 5, looked up by gene symbol.

Models compared (same baseline-ladder code as the Arc pilots):
  - no-effect
  - PCA+ridge (constant "control" input, so this effectively degrades to
    ridge directly on the target descriptor -- kept for consistency)
  - small MLP
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.model_selection import train_test_split

from virtualcell.evaluation.metrics import mae
from virtualcell.models.baselines import NoEffectBaseline, PCARidgeBaseline, SmallMLPBaseline

TARGET_RE = re.compile(r"^\d+_(.+?)_P\d(?:P\d)?_ENSG\d+$")


def parse_target_gene(obs_index: str) -> str | None:
    """Extract the target gene symbol from this file's obs index format,
    e.g. '10023_ZC3H18_P1P2_ENSG00000158545' -> 'ZC3H18'. Non-targeting
    control rows ('..._non-targeting_non-targeting_non-targeting') return
    None."""
    if "non-targeting" in obs_index:
        return None
    m = TARGET_RE.match(obs_index)
    return m.group(1) if m else None


def load_gene2vec_lookup(path: Path) -> dict[str, np.ndarray]:
    from virtualcell.data.target_descriptors import load_gene2vec_table

    return load_gene2vec_table(path)


def build_descriptors(
    var: "ad.AnnData.var",
    gene_names: list[str],
    gene2vec_table: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float]:
    """mean_std descriptor (per gene_name, from this file's own control
    statistics) and gene2vec descriptor (external, by gene symbol)."""
    mean_std = {}
    for name, mean_v, std_v in zip(var["gene_name"], var["clean_mean"], var["clean_std"]):
        mean_std[name] = np.array([mean_v, std_v], dtype=np.float32)

    gene2vec = {g: gene2vec_table[g] for g in gene_names if g in gene2vec_table}
    coverage = len(gene2vec) / len(gene_names) if gene_names else 0.0
    return mean_std, gene2vec, coverage


def fit_response_role_descriptors(
    train_targets: list[str], train_delta: np.ndarray, gene_names: list[str], n_components: int = 30, seed: int = 0,
) -> dict[str, np.ndarray]:
    """Milestone 7: a target descriptor built from REAL causal perturbation
    outcomes, not co-expression (unlike Gene2Vec/the in-dataset
    co-expression descriptor). Rationale: Milestones 5 and 6 both failed
    using co-expression-based signal (how genes correlate in baseline
    expression) -- a fundamentally different, correlational signal from
    what this task needs (how perturbing gene X causally changes gene Y).

    Idea: fit PCA on the TRAIN set's own observed perturbation-delta
    matrix (train_targets x all_profiled_genes). This factors the data
    into latent "response modules" -- patterns of genes that tend to
    move together AS A CONSEQUENCE of various perturbations (not as a
    consequence of merely being correlated at baseline). PCA's
    components_ matrix is (n_components x n_profiled_genes): column j is
    gene j's own loading vector, i.e. "this gene's role/participation
    profile across the observed causal response modules."

    Every profiled gene gets such a loading vector, INCLUDING genes never
    used as a target in train (any gene can be a downstream/output gene
    in this matrix regardless of whether it was itself perturbed). So a
    held-out target gene's descriptor is "how does this gene behave when
    OTHER genes are perturbed" -- never information from its own
    perturbation's outcome (that would leak the thing being predicted).
    This is the non-leaking trick that makes it usable for genuinely
    unseen targets, same requirement as every other descriptor here.

    `train_delta` must contain ONLY train-split perturbation rows.
    """
    n_components = min(n_components, train_delta.shape[0], train_delta.shape[1])
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(train_delta)
    # components_: (n_components, n_profiled_genes) -> transpose for per-gene lookup
    loadings = pca.components_.T  # (n_profiled_genes, n_components)
    return {gene: loadings[i].astype(np.float32) for i, gene in enumerate(gene_names)}


def descriptor_matrix(targets: list[str], table: dict[str, np.ndarray], dim: int) -> np.ndarray:
    out = np.zeros((len(targets), dim), dtype=np.float32)
    for i, t in enumerate(targets):
        if t in table:
            out[i] = table[t]
    return out


def run_ladder(
    name: str,
    train_desc: np.ndarray, train_delta: np.ndarray, train_ctrl: np.ndarray,
    val_desc: np.ndarray, val_delta: np.ndarray, val_ctrl: np.ndarray,
    test_desc: np.ndarray, test_delta: np.ndarray, test_ctrl: np.ndarray,
) -> dict:
    no_effect = NoEffectBaseline(n_genes=test_delta.shape[1])
    no_effect_mae = mae(test_delta, no_effect.predict_delta(len(test_delta)))

    best_ridge = None
    best_ridge_val = np.inf
    for alpha in (0.1, 1.0, 10.0, 100.0):
        n_comp = min(10, train_ctrl.shape[0], train_ctrl.shape[1])
        m = PCARidgeBaseline(n_components=n_comp, alpha=alpha)
        m.fit(train_ctrl, train_desc, train_delta)
        v = mae(val_delta, m.predict_delta(val_ctrl, val_desc))
        if v < best_ridge_val:
            best_ridge_val, best_ridge = v, (m, alpha)
    ridge_model, ridge_alpha = best_ridge
    ridge_test_mae = mae(test_delta, ridge_model.predict_delta(test_ctrl, test_desc))

    best_mlp = None
    best_mlp_val = np.inf
    for hidden_dim in (32, 64, 128, 256):
        m = SmallMLPBaseline(hidden_dim=hidden_dim, seed=0, max_iter=300)
        m.fit(train_ctrl, train_desc, train_delta)
        v = mae(val_delta, m.predict_delta(val_ctrl, val_desc))
        if v < best_mlp_val:
            best_mlp_val, best_mlp = v, (m, hidden_dim)
    mlp_model, mlp_hidden = best_mlp
    mlp_test_mae = mae(test_delta, mlp_model.predict_delta(test_ctrl, test_desc))

    return {
        "descriptor": name,
        "no_effect_mae": float(no_effect_mae),
        "pca_ridge": {"best_alpha": ridge_alpha, "val_mae": float(best_ridge_val), "test_mae": float(ridge_test_mae)},
        "small_mlp": {"best_hidden_dim": mlp_hidden, "val_mae": float(best_mlp_val), "test_mae": float(mlp_test_mae)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad-path", default="data/replogle_k562_essential/K562_essential_raw_bulk_01.h5ad")
    parser.add_argument("--gene2vec-path", default="data/gene2vec/gene2vec_dim_200_iter_9.txt")
    parser.add_argument("--output", default="outputs/replogle_target_scaling/metrics.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading {args.h5ad_path} ...")
    a = ad.read_h5ad(args.h5ad_path)
    X = np.asarray(a.X, dtype=np.float32)
    obs_index = a.obs.index.tolist()
    gene_names = a.var["gene_name"].tolist()

    targets = [parse_target_gene(i) for i in obs_index]
    ctrl_rows = [i for i, t in enumerate(targets) if t is None]
    pert_rows = [i for i, t in enumerate(targets) if t is not None]
    print(f"{len(pert_rows)} perturbation rows, {len(ctrl_rows)} control (non-targeting) rows, "
          f"{len(set(targets[i] for i in pert_rows))} distinct target genes, {X.shape[1]} genes profiled")

    control_mean = X[ctrl_rows].mean(axis=0)
    pert_targets = [targets[i] for i in pert_rows]
    pert_delta = X[pert_rows] - control_mean[None, :]

    print("Loading Gene2Vec table ...")
    gene2vec_table = load_gene2vec_lookup(Path(args.gene2vec_path))
    mean_std_table, gene2vec_lookup, gene2vec_coverage = build_descriptors(a.var, gene_names, gene2vec_table)
    print(f"Gene2Vec coverage of this dataset's {len(gene_names)} genes: {gene2vec_coverage:.1%}")

    unique_targets = sorted(set(pert_targets))
    train_t, test_t = train_test_split(unique_targets, test_size=0.3, random_state=args.seed)
    val_t, test_t = train_test_split(test_t, test_size=0.5, random_state=args.seed)
    print(f"Held-out-target split: {len(train_t)} train / {len(val_t)} val / {len(test_t)} test targets "
          f"(vs. Arc pilots' 90 train / 30 val / 30 test)")

    def rows_for(target_set):
        idx = [i for i, t in enumerate(pert_targets) if t in target_set]
        return idx

    train_idx, val_idx, test_idx = rows_for(set(train_t)), rows_for(set(val_t)), rows_for(set(test_t))

    def targets_at(idx):
        return [pert_targets[i] for i in idx]

    control_const = {
        "train": np.tile(control_mean, (len(train_idx), 1)),
        "val": np.tile(control_mean, (len(val_idx), 1)),
        "test": np.tile(control_mean, (len(test_idx), 1)),
    }

    print("Fitting response-role descriptor on TRAIN perturbation deltas only ...")
    response_role_dim = min(30, len(train_idx), pert_delta.shape[1])
    response_role_table = fit_response_role_descriptors(
        targets_at(train_idx), pert_delta[train_idx], gene_names, n_components=response_role_dim, seed=args.seed,
    )
    response_role_coverage = sum(1 for t in unique_targets if t in response_role_table) / len(unique_targets)
    print(f"Response-role descriptor coverage: {response_role_coverage:.1%} of {len(unique_targets)} targets "
          f"({response_role_dim} components)")

    results = {}
    for desc_name, table, dim in (
        ("mean_std", mean_std_table, 2),
        ("gene2vec", gene2vec_lookup, 200),
        ("response_role", response_role_table, response_role_dim),
    ):
        train_desc = descriptor_matrix(targets_at(train_idx), table, dim)
        val_desc = descriptor_matrix(targets_at(val_idx), table, dim)
        test_desc = descriptor_matrix(targets_at(test_idx), table, dim)

        results[desc_name] = run_ladder(
            desc_name,
            train_desc, pert_delta[train_idx], control_const["train"],
            val_desc, pert_delta[val_idx], control_const["val"],
            test_desc, pert_delta[test_idx], control_const["test"],
        )
        print(json.dumps(results[desc_name], indent=2))

    summary = {
        "dataset": "Replogle et al. 2022 Cell, K562 essential Perturb-seq pseudobulk "
                   "(CC BY-NC-SA 4.0, https://doi.org/10.25452/figshare.plus.20029387)",
        "n_genes_profiled": int(X.shape[1]),
        "n_distinct_targets": len(unique_targets),
        "n_train_targets": len(train_t), "n_val_targets": len(val_t), "n_test_targets": len(test_t),
        "gene2vec_coverage": gene2vec_coverage,
        "response_role_coverage": response_role_coverage,
        "response_role_dim": response_role_dim,
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

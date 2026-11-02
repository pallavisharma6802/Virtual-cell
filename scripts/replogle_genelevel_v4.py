"""Milestone 12: hierarchical averaging (per-gem_group → average groups) + all cells.

The key insight from Milestones 10-11: using all cells with global mean HURTS
(0.213 vs 0.218) because more cells per gene means more gem_group diversity
per gene, amplifying batch-to-batch variation. Batch-matched controls hurt more
(0.201) because per-batch control means are too noisy (223 cells/group).

The proper fix: hierarchical averaging.
  1. For each target gene t and gem_group g: compute mean expression of
     t-cells in gem_group g, subtract control mean of gem_group g.
     -> per-group mean delta delta_t_g
  2. Average delta_t_g across all gem_groups that have cells for gene t.
     -> delta_t = mean_g(delta_t_g)

This gives each gem_group equal weight regardless of cell count, removing
the batch composition bias while keeping batch-matched control subtraction.
Both steps use ALL available cells.
"""
from __future__ import annotations
import json, time
from pathlib import Path
from collections import defaultdict
import anndata as ad
import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import sys; sys.path.insert(0, "/home/ec2-user/virtualcell")
from virtualcell.data.target_descriptors import fit_response_role_descriptors, load_gene2vec_table

MIN_CELLS = 5

def load_rows_top_genes(adata, row_positions, top_gene_idx, chunk_size=8000):
    chunks = []
    for start in range(0, len(row_positions), chunk_size):
        chunk_rows = row_positions[start: start + chunk_size]
        X_chunk = adata.X[chunk_rows]
        X_sub = X_chunk[:, top_gene_idx]
        chunks.append(np.asarray(X_sub.todense() if hasattr(X_sub, "todense") else X_sub, dtype=np.float32))
    return np.vstack(chunks)

def pearson_per_gene(pred_dict, true_dict):
    genes = sorted(set(pred_dict) & set(true_dict))
    vals = []
    for g in genes:
        p, t = pred_dict[g], true_dict[g]
        if np.std(t) < 1e-6 or np.std(p) < 1e-6: vals.append(0.0)
        else: vals.append(float(np.clip(pearsonr(p, t)[0], -1, 1)))
    return float(np.mean(vals)), float(np.median(vals))

t0 = time.time()
adata = ad.read_h5ad("data/replogle/K562_essential_raw_singlecell_01.h5ad", backed="r")
gene_col = adata.obs["gene"].astype(str).tolist()
gem_col = adata.obs["gem_group"].astype(str).tolist()
gene_names = adata.var["gene_name"].tolist() if "gene_name" in adata.var.columns else adata.var_names.tolist()
print(f"Full: {adata.shape[0]} cells x {adata.shape[1]} genes")
rng = np.random.default_rng(0)
ctrl_positions = [i for i, g in enumerate(gene_col) if g == "non-targeting"]
target_by_gene = defaultdict(list)
for i, g in enumerate(gene_col):
    if g != "non-targeting": target_by_gene[g].append(i)
all_genes = sorted(k for k, v in target_by_gene.items() if len(v) >= MIN_CELLS)
all_target_pos = sorted(p for g in all_genes for p in target_by_gene[g])
print(f"Controls: {len(ctrl_positions)}, targets: {len(all_target_pos)} across {len(all_genes)} genes")
# Top genes from sampled controls
ctrl_sample = rng.choice(ctrl_positions, size=min(8000, len(ctrl_positions)), replace=False)
X_ctrl_full = adata.X[sorted(ctrl_sample.tolist())]
X_ctrl_full = np.asarray(X_ctrl_full.todense() if hasattr(X_ctrl_full, "todense") else X_ctrl_full, dtype=np.float32)
ctrl_var = X_ctrl_full.var(axis=0)
top_gene_idx = np.sort(np.argsort(ctrl_var)[::-1][:1500])
top_gene_names = [gene_names[i] for i in top_gene_idx]
X_ctrl_sample = X_ctrl_full[:, top_gene_idx]
del X_ctrl_full
# Per-gem_group control mean using ALL control cells
X_ctrl_all = load_rows_top_genes(adata, sorted(ctrl_positions), top_gene_idx)
ctrl_gemgroups = [gem_col[p] for p in sorted(ctrl_positions)]
ctrl_mean_by_gg = {}
for gg in sorted(set(ctrl_gemgroups)):
    idx = [i for i, g in enumerate(ctrl_gemgroups) if g == gg]
    ctrl_mean_by_gg[gg] = X_ctrl_all[idx].mean(axis=0)
global_ctrl_mean = X_ctrl_all.mean(axis=0)
del X_ctrl_all
print(f"{len(ctrl_mean_by_gg)} gem_groups with control cells")
# Load all target cells chunked
print("Loading all target cells...")
X_target = load_rows_top_genes(adata, all_target_pos, top_gene_idx)
target_genes_flat = [gene_col[p] for p in all_target_pos]
target_gg_flat = [gem_col[p] for p in all_target_pos]
fetch_time = time.time() - t0
print(f"Loaded in {fetch_time:.1f}s. {X_target.nbytes/1e9:.2f}GB")
# Hierarchical averaging: per-gene, per-gem_group mean delta, then average across groups
all_mean_deltas = {}
for g in all_genes:
    # cells for this gene
    cell_idx = [i for i, t in enumerate(target_genes_flat) if t == g]
    # group by gem_group
    group_means = []
    for gg in sorted(set(target_gg_flat[i] for i in cell_idx)):
        gg_idx = [i for i in cell_idx if target_gg_flat[i] == gg]
        if gg_idx:
            ctrl_m = ctrl_mean_by_gg.get(gg, global_ctrl_mean)
            group_means.append(X_target[gg_idx].mean(axis=0) - ctrl_m)
    if group_means:
        all_mean_deltas[g] = np.stack(group_means).mean(axis=0)
print(f"Computed hierarchical mean deltas for {len(all_mean_deltas)} genes")
X_ctrl_sample_col = X_ctrl_sample  # for std computation in mean_std descriptor
# Split
train_t, test_t_full = train_test_split(list(all_mean_deltas.keys()), test_size=0.3, random_state=0)
val_t, test_t = train_test_split(test_t_full, test_size=0.5, random_state=0)
print(f"Split: {len(train_t)} train / {len(val_t)} val / {len(test_t)} test")
train_mean_d = {g: all_mean_deltas[g] for g in train_t}
val_mean_d = {g: all_mean_deltas[g] for g in val_t}
test_mean_d = {g: all_mean_deltas[g] for g in test_t}
train_gene_list = sorted(train_mean_d)
D_train = np.stack([train_mean_d[g] for g in train_gene_list])
print(f"D_train: {D_train.shape}")
# Descriptors
gene2vec_table = load_gene2vec_table(Path("data/gene2vec/gene2vec_dim_200_iter_9.txt"))
gene2vec_lookup = {g: gene2vec_table[g] for g in top_gene_names if g in gene2vec_table}
mean_std_table = {gene: np.array([global_ctrl_mean[i], X_ctrl_sample_col[:, i].std()], dtype=np.float32) for i, gene in enumerate(top_gene_names)}
train_set = set(train_t)
train_idx = [i for i, t in enumerate(target_genes_flat) if t in train_set]
train_targets_list = [target_genes_flat[i] for i in train_idx]
train_cell_delta = X_target[train_idx] - global_ctrl_mean
rr_dim = min(30, len(train_gene_list))
response_role_table = fit_response_role_descriptors(train_targets_list, train_cell_delta, top_gene_names, n_components=rr_dim, seed=0)
del X_target, train_cell_delta
# Baselines
gm = D_train.mean(axis=0)
no_m, no_med = pearson_per_gene({g: np.zeros(1500, np.float32) for g in test_t}, test_mean_d)
gm_m, gm_med = pearson_per_gene({g: gm for g in test_t}, test_mean_d)
print(f"No-effect: {no_m:.4f}; Global mean: {gm_m:.4f} (median {gm_med:.4f})")
def evaluate(name, table, dim):
    X_tr = np.array([table.get(g, np.zeros(dim, np.float32)) for g in train_gene_list])
    X_va = np.array([table.get(g, np.zeros(dim, np.float32)) for g in val_t])
    X_te = np.array([table.get(g, np.zeros(dim, np.float32)) for g in test_t])
    out = {}
    best_dr = None
    for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
        m = Ridge(alpha=alpha); m.fit(X_tr, D_train)
        va_preds = {g: m.predict(X_va[[i]])[0] for i, g in enumerate(val_t)}
        vr, _ = pearson_per_gene(va_preds, val_mean_d)
        if best_dr is None or vr > best_dr[0]: best_dr = (vr, alpha, m)
    _, alpha, m = best_dr
    te_preds = {g: m.predict(X_te[[i]])[0] for i, g in enumerate(test_t)}
    tr_m, tr_med = pearson_per_gene(te_preds, test_mean_d)
    out["direct_ridge"] = {"alpha": alpha, "test_pearson_mean": tr_m, "test_pearson_median": tr_med}
    print(f"  {name}+ridge: {tr_m:.4f} (median {tr_med:.4f})")
    for k in (10, 20, 30, 50, 100):
        if k >= min(D_train.shape): continue
        svd = TruncatedSVD(n_components=k, random_state=0); codes_tr = svd.fit_transform(D_train)
        best_svd = None
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
            m = Ridge(alpha=alpha); m.fit(X_tr, codes_tr)
            va_codes = m.predict(X_va)
            va_preds = {g: svd.inverse_transform(va_codes[[i]])[0] for i, g in enumerate(val_t)}
            vr, _ = pearson_per_gene(va_preds, val_mean_d)
            if best_svd is None or vr > best_svd[0]: best_svd = (vr, alpha, m)
        _, alpha, m = best_svd; te_codes = m.predict(X_te)
        te_preds = {g: svd.inverse_transform(te_codes[[i]])[0] for i, g in enumerate(test_t)}
        tr_m, tr_med = pearson_per_gene(te_preds, test_mean_d)
        out[f"svd_k{k}"] = {"k": k, "alpha": alpha, "test_pearson_mean": tr_m, "test_pearson_median": tr_med}
        print(f"  {name}+SVD_k={k}: {tr_m:.4f} (median {tr_med:.4f})")
    return out
print("\nmean_std:"); ms = evaluate("mean_std", mean_std_table, 2)
print("\ngene2vec:"); g2v = evaluate("gene2vec", gene2vec_lookup, 200)
print("\nresponse_role:"); rr = evaluate("response_role", response_role_table, rr_dim)
out = {"control_matching": "hierarchical (per-gem_group mean delta, then average across groups)",
       "cells_per_target_cap": "none",
       "n_target_cells": len(all_target_pos),
       "baselines": {"no_effect": {"test_pearson_mean": no_m, "test_pearson_median": no_med},
                     "global_mean": {"test_pearson_mean": gm_m, "test_pearson_median": gm_med}},
       "mean_std": ms, "gene2vec": g2v, "response_role": rr}
Path("outputs/replogle_genelevel_v4").mkdir(parents=True, exist_ok=True)
Path("outputs/replogle_genelevel_v4/metrics.json").write_text(json.dumps(out, indent=2))
print("Wrote outputs/replogle_genelevel_v4/metrics.json")

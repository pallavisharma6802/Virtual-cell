"""Milestone 13: log1p-normalize counts before computing mean delta.

The dataset file is RAW single-cell counts (K562_essential_raw_singlecell_01.h5ad).
All prior milestones worked on raw counts. GEARS and published baselines work on
log-normalized data (log1p after library-size normalization to 10k). This is
almost certainly the main source of the gap between our 0.198 and literature 0.267
for the trivial global-mean baseline.

Normalization: X_norm = log1p(X / sum(X) * 10000) per cell.
Then: control mean and deltas are in log1p space.
Top-variable genes are recomputed in normalized space.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import anndata as ad
import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import sys; sys.path.insert(0, "/home/ec2-user/virtualcell")
from virtualcell.data.target_descriptors import fit_response_role_descriptors, load_gene2vec_table

MIN_CELLS = 5
CELLS_CAP = 80

def log1p_normalize(X, target_sum=10000.0):
    """Library-size normalize then log1p. X: dense float32 array (n_cells, n_genes)."""
    lib_sizes = X.sum(axis=1, keepdims=True)
    lib_sizes = np.where(lib_sizes == 0, 1, lib_sizes)
    return np.log1p(X / lib_sizes * target_sum).astype(np.float32)

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
gene_names = adata.var["gene_name"].tolist() if "gene_name" in adata.var.columns else adata.var_names.tolist()
print(f"Full: {adata.shape[0]} cells x {adata.shape[1]} genes")
rng = np.random.default_rng(0)

ctrl_positions = [i for i, g in enumerate(gene_col) if g == "non-targeting"]
by_target = {}
for i, g in enumerate(gene_col):
    if g != "non-targeting": by_target.setdefault(g, []).append(i)

# Sample cells (80 cap) for consistent comparison with v2
ctrl_sample = sorted(rng.choice(ctrl_positions, size=min(4000, len(ctrl_positions)), replace=False).tolist())
target_positions, kept_targets = [], []
for t, positions in by_target.items():
    if len(positions) < MIN_CELLS: continue
    chosen = rng.choice(positions, size=min(CELLS_CAP, len(positions)), replace=False)
    target_positions.extend(chosen.tolist())
    kept_targets.append(t)
target_positions = sorted(target_positions)
cell_targets = [gene_col[p] for p in target_positions]
unique_targets = sorted(set(cell_targets))
print(f"Controls: {len(ctrl_sample)}, target cells: {len(target_positions)} across {len(unique_targets)} genes")

# Load control cells FULL matrix then normalize then find top-variable genes
print("Loading control cells (full gene matrix) for normalization + top-gene selection...")
X_ctrl_raw = adata.X[ctrl_sample]
X_ctrl_raw = np.asarray(X_ctrl_raw.todense() if hasattr(X_ctrl_raw, "todense") else X_ctrl_raw, dtype=np.float32)
X_ctrl_norm = log1p_normalize(X_ctrl_raw)
del X_ctrl_raw
ctrl_var = X_ctrl_norm.var(axis=0)
top_gene_idx = np.sort(np.argsort(ctrl_var)[::-1][:1500])
top_gene_names = [gene_names[i] for i in top_gene_idx]
X_ctrl = X_ctrl_norm[:, top_gene_idx]
control_mean = X_ctrl.mean(axis=0)
del X_ctrl_norm

# Load target cells sparse-select top genes, then normalize
print("Loading target cells...")
X_target_sparse = adata.X[target_positions]
X_target_sparse_top = X_target_sparse[:, top_gene_idx]
# But we need full counts for library-size normalization! Need full row sums.
# Compute row sums from full sparse matrix
print("Computing library sizes from full matrix...")
lib_sizes_target = np.asarray(X_target_sparse.sum(axis=1)).ravel().astype(np.float32)
lib_sizes_target = np.where(lib_sizes_target == 0, 1, lib_sizes_target)
# Densify top-gene columns only
X_target_top = np.asarray(X_target_sparse_top.todense() if hasattr(X_target_sparse_top, "todense") else X_target_sparse_top, dtype=np.float32)
del X_target_sparse, X_target_sparse_top
# Normalize: X_norm_i = log1p(X_i / libsize_i * 10000)
X_target = np.log1p(X_target_top / lib_sizes_target[:, None] * 10000.0).astype(np.float32)
del X_target_top
fetch_time = time.time() - t0
print(f"Loaded+normalized in {fetch_time:.1f}s. {X_target.nbytes/1e9:.2f}GB")

# Recompute control mean in log1p space with correct library sizes
# (redo with all 4000 control cells)
X_ctrl_raw2 = adata.X[ctrl_sample]
X_ctrl_raw2 = np.asarray(X_ctrl_raw2.todense() if hasattr(X_ctrl_raw2, "todense") else X_ctrl_raw2, dtype=np.float32)
X_ctrl_norm2 = log1p_normalize(X_ctrl_raw2)
X_ctrl2 = X_ctrl_norm2[:, top_gene_idx]
control_mean = X_ctrl2.mean(axis=0)
del X_ctrl_raw2, X_ctrl_norm2, X_ctrl2

# Split
train_t, test_t_full = train_test_split(unique_targets, test_size=0.3, random_state=0)
val_t, test_t = train_test_split(test_t_full, test_size=0.5, random_state=0)
print(f"Split: {len(train_t)} train / {len(val_t)} val / {len(test_t)} test")

def delta_for_split(t_set):
    idx = [i for i, t in enumerate(cell_targets) if t in t_set]
    split_targets = [cell_targets[i] for i in idx]
    split_X = X_target[idx]
    mean_deltas = {}
    for g in t_set:
        mask = [j for j, t in enumerate(split_targets) if t == g]
        if mask: mean_deltas[g] = split_X[mask].mean(axis=0) - control_mean
    return mean_deltas

train_md = delta_for_split(set(train_t))
val_md = delta_for_split(set(val_t))
test_md = delta_for_split(set(test_t))
train_gene_list = sorted(train_md)
D_train = np.stack([train_md[g] for g in train_gene_list])
print(f"D_train: {D_train.shape}")

gene2vec_table = load_gene2vec_table(Path("data/gene2vec/gene2vec_dim_200_iter_9.txt"))
gene2vec_lookup = {g: gene2vec_table[g] for g in top_gene_names if g in gene2vec_table}
mean_std_table = {gene: np.array([control_mean[i], X_ctrl[:, i].std() if hasattr(X_ctrl, "shape") else 0.0], dtype=np.float32) for i, gene in enumerate(top_gene_names)}
# Recompute X_ctrl for std
X_ctrl_raw3 = adata.X[ctrl_sample]
X_ctrl_raw3 = np.asarray(X_ctrl_raw3.todense() if hasattr(X_ctrl_raw3, "todense") else X_ctrl_raw3, dtype=np.float32)
X_ctrl_norm3 = log1p_normalize(X_ctrl_raw3)
X_ctrl3 = X_ctrl_norm3[:, top_gene_idx]
mean_std_table = {gene: np.array([control_mean[i], X_ctrl3[:, i].std()], dtype=np.float32) for i, gene in enumerate(top_gene_names)}
del X_ctrl_raw3, X_ctrl_norm3, X_ctrl3

train_set = set(train_t)
train_idx = [i for i, t in enumerate(cell_targets) if t in train_set]
train_targets_list = [cell_targets[i] for i in train_idx]
train_cell_delta = X_target[train_idx] - control_mean
rr_dim = min(30, len(train_gene_list))
response_role_table = fit_response_role_descriptors(train_targets_list, train_cell_delta, top_gene_names, n_components=rr_dim, seed=0)

gm = D_train.mean(axis=0)
no_m, no_med = pearson_per_gene({g: np.zeros(1500, np.float32) for g in test_t}, test_md)
gm_m, gm_med = pearson_per_gene({g: gm for g in test_t}, test_md)
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
        vr, _ = pearson_per_gene(va_preds, val_md)
        if best_dr is None or vr > best_dr[0]: best_dr = (vr, alpha, m)
    _, alpha, m = best_dr
    te_preds = {g: m.predict(X_te[[i]])[0] for i, g in enumerate(test_t)}
    tr_m, tr_med = pearson_per_gene(te_preds, test_md)
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
            vr, _ = pearson_per_gene(va_preds, val_md)
            if best_svd is None or vr > best_svd[0]: best_svd = (vr, alpha, m)
        _, alpha, m = best_svd; te_codes = m.predict(X_te)
        te_preds = {g: svd.inverse_transform(te_codes[[i]])[0] for i, g in enumerate(test_t)}
        tr_m, tr_med = pearson_per_gene(te_preds, test_md)
        out[f"svd_k{k}"] = {"k": k, "alpha": alpha, "test_pearson_mean": tr_m, "test_pearson_median": tr_med}
        print(f"  {name}+SVD_k={k}: {tr_m:.4f} (median {tr_med:.4f})")
    return out

print("\nmean_std:"); ms = evaluate("mean_std", mean_std_table, 2)
print("\ngene2vec:"); g2v = evaluate("gene2vec", gene2vec_lookup, 200)
print("\nresponse_role:"); rr = evaluate("response_role", response_role_table, rr_dim)

out = {"normalization": "log1p (library-size to 10k, then log1p)", "cells_per_target_cap": CELLS_CAP,
       "baselines": {"no_effect": {"test_pearson_mean": no_m, "test_pearson_median": no_med},
                     "global_mean": {"test_pearson_mean": gm_m, "test_pearson_median": gm_med}},
       "mean_std": ms, "gene2vec": g2v, "response_role": rr}
Path("outputs/replogle_genelevel_v5").mkdir(parents=True, exist_ok=True)
Path("outputs/replogle_genelevel_v5/metrics.json").write_text(json.dumps(out, indent=2))
print("Wrote outputs/replogle_genelevel_v5/metrics.json")

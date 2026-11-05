"""Milestone 21: Dispersion-normalized HVG selection + investigate global mean gap.

The global mean = 0.240 vs literature 0.267 is an unexplained 0.027 gap.
Hypothesis: our variance-based HVG selection from control cells only differs from
the literature standard (scanpy highly_variable_genes, seurat method), which uses
mean-variance normalization across all cells.

This script tries:
  A) Scanpy-style dispersion-normalized HVG selection (from sampled all cells).
  B) Same MLP ensemble pipeline as v13 on new HVGs.
  C) Record whether the global mean closes toward 0.267 as a diagnostic.

Scanpy seurat HVG selection:
  1. Compute mean and variance of each gene across a large cell sample (ctrl + target).
  2. Fit mean-variance trend (by binning genes by expression decile).
  3. Select top N genes by normalized dispersion (variance / expected_variance_at_that_mean).

This avoids selecting genes that are variable only because they are highly expressed
(mean-variance coupling artifact in count data).
"""
from __future__ import annotations

import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from virtualcell.data.target_descriptors import (
    fit_response_role_descriptors,
    load_gene2vec_table,
)

MIN_CELLS = 5
TARGET_SUM = 10_000.0
N_HVG = 5000
CHUNK_SIZE = 4000
OUT_DIR = Path("outputs/replogle_genelevel_v14")
CACHE_PATH = OUT_DIR / "mean_delta_cache_v14.npz"
GENES_PATH = OUT_DIR / "mean_delta_cache_v14_genes.json"


def lognorm_chunk(X_sparse, top_gene_idx):
    lib = np.asarray(X_sparse.sum(axis=1)).ravel().astype(np.float64)
    lib = np.where(lib == 0, 1.0, lib)
    X_top = X_sparse[:, top_gene_idx]
    X_dense = np.asarray(
        X_top.todense() if hasattr(X_top, "todense") else X_top,
        dtype=np.float32,
    )
    X_dense /= lib[:, None].astype(np.float32)
    X_dense *= TARGET_SUM
    return np.log1p(X_dense)


def pearson_per_gene(pred_dict, true_dict):
    genes = sorted(set(pred_dict) & set(true_dict))
    vals = []
    for g in genes:
        p, t = pred_dict[g], true_dict[g]
        if np.std(t) < 1e-6 or np.std(p) < 1e-6:
            vals.append(0.0)
        else:
            vals.append(float(np.clip(pearsonr(p, t)[0], -1, 1)))
    return float(np.mean(vals)), float(np.median(vals))


def select_hvgs_dispersion(adata, gene_names, n_hvg, n_sample=15000, seed=0):
    """Scanpy-style dispersion-normalized HVG selection.

    Samples n_sample cells from all cells (control + target), computes
    mean and CV (std/mean) per gene, bins by mean, normalizes dispersion
    within each bin, returns top n_hvg gene indices.
    """
    rng = np.random.default_rng(seed)
    n_cells = adata.shape[0]
    sample_idx = sorted(rng.choice(n_cells, size=min(n_sample, n_cells), replace=False).tolist())
    X = adata.X[sample_idx]
    X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float32)
    # Library-size normalize to 10k counts, log1p
    lib = X.sum(axis=1, keepdims=True)
    lib = np.where(lib == 0, 1.0, lib)
    X_norm = np.log1p(X / lib * TARGET_SUM)
    del X
    gene_mean = X_norm.mean(axis=0)   # (n_genes,)
    gene_var  = X_norm.var(axis=0)    # (n_genes,)
    del X_norm

    # Avoid division by zero
    gene_mean_safe = np.where(gene_mean == 0, 1e-9, gene_mean)
    gene_dispersion = gene_var / gene_mean_safe  # CV^2 (Fano factor)

    # Normalize dispersion within expression bins (10 bins by mean)
    bins = np.percentile(gene_mean, np.linspace(0, 100, 11))
    bins[0] -= 1e-9
    bin_idx = np.digitize(gene_mean, bins) - 1
    bin_idx = np.clip(bin_idx, 0, 9)

    norm_dispersion = np.zeros_like(gene_dispersion)
    for b in range(10):
        mask = bin_idx == b
        if mask.sum() < 2:
            continue
        mean_disp = gene_dispersion[mask].mean()
        std_disp  = gene_dispersion[mask].std()
        if std_disp < 1e-9:
            norm_dispersion[mask] = 0.0
        else:
            norm_dispersion[mask] = (gene_dispersion[mask] - mean_disp) / std_disp

    top_gene_idx = np.sort(np.argsort(norm_dispersion)[::-1][:n_hvg])
    top_gene_names = [gene_names[i] for i in top_gene_idx]
    print(f"  HVG dispersion selection: sampled {len(sample_idx)} cells, "
          f"top {n_hvg} of {adata.shape[1]} genes.")
    return top_gene_idx, top_gene_names


class MLP(nn.Module):
    def __init__(self, in_d, hidden, out_d, n_layers=2, dropout=0.3):
        super().__init__()
        layers = []
        d = in_d
        for _ in range(n_layers):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.Linear(d, out_d))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(X_tr, y_tr, X_va, val_t, val_md, svd, hidden=512, epochs=800, lr=1e-3, wd=1e-4, seed=42):
    torch.manual_seed(seed)
    model = MLP(X_tr.shape[1], hidden, y_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    best_vr, best_state = -999, None
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = nn.MSELoss()(model(X_tr_t), y_tr_t)
        loss.backward()
        opt.step()
        sched.step()
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                va_codes = model(X_va_t).numpy()
            va_preds = {g: svd.inverse_transform(va_codes[[i]])[0] for i, g in enumerate(val_t)}
            vr, _ = pearson_per_gene(va_preds, val_md)
            if vr > best_vr:
                best_vr = vr
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_vr


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Open adata ---
    print("Opening adata ...")
    adata = ad.read_h5ad("data/replogle/K562_essential_raw_singlecell_01.h5ad", backed="r")
    gene_col = adata.obs["gene"].astype(str).tolist()
    gene_names = (
        adata.var["gene_name"].tolist() if "gene_name" in adata.var.columns
        else adata.var_names.tolist()
    )
    print(f"  {adata.shape[0]} cells x {adata.shape[1]} genes")

    ctrl_positions = sorted([i for i, g in enumerate(gene_col) if g == "non-targeting"])
    target_by_gene: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(gene_col):
        if g != "non-targeting":
            target_by_gene[g].append(i)
    all_genes = sorted(k for k, v in target_by_gene.items() if len(v) >= MIN_CELLS)
    all_target_pos = sorted(p for g in all_genes for p in target_by_gene[g])
    print(f"  Controls: {len(ctrl_positions)}, target cells: {len(all_target_pos)} across {len(all_genes)} genes")

    # --- Dispersion-normalized HVG selection (from all cell types) ---
    print("Selecting HVGs (dispersion-normalized, all cells sample) ...")
    top_gene_idx, top_gene_names = select_hvgs_dispersion(adata, gene_names, N_HVG, n_sample=15000, seed=0)

    # --- Control mean (all control cells) ---
    print(f"Computing control mean over {len(ctrl_positions)} control cells ...")
    ctrl_sum = np.zeros(N_HVG, dtype=np.float64)
    n_ctrl = 0
    for start in range(0, len(ctrl_positions), CHUNK_SIZE):
        chunk = ctrl_positions[start: start + CHUNK_SIZE]
        X_c = lognorm_chunk(adata.X[chunk], top_gene_idx).astype(np.float64)
        ctrl_sum += X_c.sum(axis=0)
        n_ctrl += len(chunk)
    control_mean = (ctrl_sum / n_ctrl).astype(np.float32)
    del ctrl_sum
    print(f"  control_mean computed over {n_ctrl} cells. Time: {time.time()-t0:.1f}s")

    # --- Single-pass vectorized mean delta accumulation ---
    print(f"Computing mean deltas (single pass, {len(all_target_pos)} target cells) ...")
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    gene_sum_arr = np.zeros((len(all_genes), N_HVG), dtype=np.float32)
    gene_count_arr = np.zeros(len(all_genes), dtype=np.int32)
    n_all = len(all_target_pos)
    for start in range(0, n_all, CHUNK_SIZE):
        chunk_rows = all_target_pos[start: start + CHUNK_SIZE]
        X_t = lognorm_chunk(adata.X[chunk_rows], top_gene_idx)
        chunk_idxs = np.array([gene_to_idx.get(gene_col[p], -1) for p in chunk_rows], dtype=np.int32)
        valid = chunk_idxs >= 0
        np.add.at(gene_sum_arr, chunk_idxs[valid], X_t[valid])
        np.add.at(gene_count_arr, chunk_idxs[valid], 1)
        del X_t, chunk_idxs, valid
        if (start // CHUNK_SIZE) % 10 == 0:
            pct = min(100, 100 * (start + CHUNK_SIZE) // n_all)
            print(f"  {pct}%", flush=True)
    print(f"  Done. Time: {time.time()-t0:.1f}s")

    valid_mask = gene_count_arr >= MIN_CELLS
    valid_genes = [g for g, ok in zip(all_genes, valid_mask) if ok]
    valid_sums = gene_sum_arr[valid_mask]
    valid_counts = gene_count_arr[valid_mask]

    # Save cache and free adata page cache
    np.savez(str(CACHE_PATH), valid_sums=valid_sums, valid_counts=valid_counts, control_mean=control_mean)
    with open(GENES_PATH, "w") as f:
        json.dump(valid_genes, f)
    print(f"Saved cache: {len(valid_genes)} genes x {N_HVG} HVGs. Time: {time.time()-t0:.1f}s")

    del gene_sum_arr, gene_count_arr, valid_sums, valid_counts, adata
    gc.collect()
    print(f"Freed adata. Available memory before models ...")

    # Reload from cache
    cache = np.load(str(CACHE_PATH))
    all_mean_deltas = {
        g: (cache["valid_sums"][i] / cache["valid_counts"][i]).astype(np.float32) - control_mean
        for i, g in enumerate(valid_genes)
    }
    del cache
    gc.collect()
    print(f"  {len(all_mean_deltas)} gene mean deltas ready.")

    # --- Split ---
    train_t, test_t_full = train_test_split(list(all_mean_deltas.keys()), test_size=0.3, random_state=0)
    val_t, test_t = train_test_split(test_t_full, test_size=0.5, random_state=0)
    print(f"Split: {len(train_t)} train / {len(val_t)} val / {len(test_t)} test")

    train_md = {g: all_mean_deltas[g] for g in train_t}
    val_md   = {g: all_mean_deltas[g] for g in val_t}
    test_md  = {g: all_mean_deltas[g] for g in test_t}

    train_gene_list = sorted(train_md)
    D_train = np.stack([train_md[g] for g in train_gene_list])
    n_out = D_train.shape[1]

    gm = D_train.mean(axis=0)
    gm_m, gm_med = pearson_per_gene({g: gm for g in test_t}, test_md)
    print(f"\nGlobal mean (dispersion HVGs): {gm_m:.4f} (median {gm_med:.4f})")
    print(f"  (v7-v13 global mean was 0.240; literature = 0.267)")

    # --- Descriptors ---
    gene2vec_table = load_gene2vec_table(Path("data/gene2vec/gene2vec_dim_200_iter_9.txt"))
    gene2vec_lookup = {}
    for g in top_gene_names:
        if g in gene2vec_table:
            gene2vec_lookup[g] = gene2vec_table[g]
    for g in all_mean_deltas:
        if g not in gene2vec_lookup and g in gene2vec_table:
            gene2vec_lookup[g] = gene2vec_table[g]

    rr_dim = min(50, len(train_gene_list))
    train_cell_delta_approx = np.stack([train_md[g] for g in train_t])
    response_role_table = fit_response_role_descriptors(
        train_t, train_cell_delta_approx, top_gene_names, n_components=rr_dim, seed=0,
    )
    concat_dim = 200 + rr_dim
    concat_table = {
        g: np.concatenate([
            gene2vec_lookup.get(g, np.zeros(200, np.float32)),
            response_role_table.get(g, np.zeros(rr_dim, np.float32))
        ])
        for g in all_mean_deltas
    }

    results = {}
    svd = TruncatedSVD(n_components=100, random_state=0)
    codes_tr = svd.fit_transform(D_train)

    for feat_name, feat_table, feat_dim in [
        ("gene2vec", gene2vec_lookup, 200),
        ("concat",   concat_table,   concat_dim),
    ]:
        X_tr = np.array([feat_table.get(g, np.zeros(feat_dim, np.float32)) for g in train_gene_list])
        X_va = np.array([feat_table.get(g, np.zeros(feat_dim, np.float32)) for g in val_t])
        X_te = np.array([feat_table.get(g, np.zeros(feat_dim, np.float32)) for g in test_t])

        # Ridge baseline
        best_ridge = None
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
            m = Ridge(alpha=alpha)
            m.fit(X_tr, codes_tr)
            va_codes = m.predict(X_va)
            va_preds = {g: svd.inverse_transform(va_codes[[i]])[0] for i, g in enumerate(val_t)}
            vr, _ = pearson_per_gene(va_preds, val_md)
            if best_ridge is None or vr > best_ridge[0]:
                best_ridge = (vr, alpha, m)
        _, alpha_r, m_r = best_ridge
        te_codes = m_r.predict(X_te)
        te_preds = {g: svd.inverse_transform(te_codes[[i]])[0] for i, g in enumerate(test_t)}
        tr_m, tr_med = pearson_per_gene(te_preds, test_md)
        results[f"{feat_name}_ridge_svd100"] = {"test_pearson_mean": tr_m, "test_pearson_median": tr_med}
        print(f"  {feat_name}+ridge+svd100: {tr_m:.4f}")

        # MLP ensemble (5 seeds)
        ensemble_codes = []
        seed_scores = []
        for seed in range(5):
            model, best_vr = train_mlp(
                X_tr, codes_tr, X_va, val_t, val_md, svd,
                hidden=512, epochs=800, lr=1e-3, wd=1e-4, seed=seed
            )
            model.eval()
            with torch.no_grad():
                te_codes_s = model(torch.tensor(X_te, dtype=torch.float32)).numpy()
            ensemble_codes.append(te_codes_s)
            te_p = {g: svd.inverse_transform(te_codes_s[[i]])[0] for i, g in enumerate(test_t)}
            sm, smed = pearson_per_gene(te_p, test_md)
            seed_scores.append(sm)
            print(f"    seed {seed}: val={best_vr:.4f} test={sm:.4f}")

        mean_codes = np.mean(ensemble_codes, axis=0)
        ens_preds = {g: svd.inverse_transform(mean_codes[[i]])[0] for i, g in enumerate(test_t)}
        ens_m, ens_med = pearson_per_gene(ens_preds, test_md)
        results[f"{feat_name}_mlp_ens_svd100"] = {
            "test_pearson_mean": ens_m, "test_pearson_median": ens_med, "per_seed": seed_scores
        }
        print(f"  {feat_name}+MLP ensemble+svd100: {ens_m:.4f} (median {ens_med:.4f})")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    summary = {
        "dataset": "Replogle et al. 2022 K562 essential RAW SINGLE-CELL",
        "note": "v14: dispersion-normalized HVG selection from all cells (scanpy-style)",
        "hvg_method": "dispersion-normalized (seurat-style), 15k sampled cells",
        "n_hvg": N_HVG,
        "n_distinct_targets": len(all_mean_deltas),
        "n_train_targets": len(train_t),
        "n_val_targets": len(val_t),
        "n_test_targets": len(test_t),
        "elapsed_seconds": elapsed,
        "literature_reference_points": {
            "mean_training_delta_baseline": 0.267,
            "oracle_nn_retrieval": 0.592,
            "our_oracle_nn_k5_v12": 0.4739,
        },
        "baselines": {
            "global_mean_dispersion_hvgs": {"test_pearson_mean": gm_m, "test_pearson_median": gm_med},
            "v13_best_global_mean": 0.2405,
            "note": "If global_mean improves, dispersion HVG hypothesis confirmed",
        },
        "experiments": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_DIR}/metrics.json")


if __name__ == "__main__":
    main()

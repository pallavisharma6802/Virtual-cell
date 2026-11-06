"""Milestone 22: Cosine loss + wider MLP + larger SVD.

Key insight: our eval metric is Pearson correlation, which is equivalent to
cosine similarity on mean-centered vectors. Using MSE on SVD codes may not
optimize for the right objective.

Changes from v13 (best = 0.2714):
  A) CosineEmbeddingLoss instead of MSE on SVD codes — directly optimizes
     for the cosine similarity that Pearson correlation measures.
  B) Wider MLP: hidden=1024, 3 layers (up from 512, 2 layers).
  C) Larger SVD k=200 (up from 100) — more expressive low-rank target space.
  D) Direct delta prediction head: skip SVD, predict 5000-dim output directly
     with cosine loss on the full delta vector. This avoids information loss
     from truncated SVD.

Loads from v10 cache (variance-based HVGs, same as v13 baseline).
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

from virtualcell.data.target_descriptors import (
    fit_response_role_descriptors,
    load_gene2vec_table,
)

TARGET_SUM = 10_000.0
N_HVG = 5000
CHUNK_SIZE = 4000
OUT_DIR = Path("outputs/replogle_genelevel_v15")
# Load from v10 cache (same as v11-v13)
V10_CACHE = Path("outputs/replogle_genelevel_v10/mean_delta_cache.npz")
V10_GENES = Path("outputs/replogle_genelevel_v10/mean_delta_cache_genes.json")


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


class MLP(nn.Module):
    def __init__(self, in_d, hidden, out_d, n_layers=2, dropout=0.3):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_d
        for _ in range(n_layers):
            layers.extend([nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)])
            d = hidden
        layers.append(nn.Linear(d, out_d))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def cosine_loss(pred, target):
    """Mean cosine distance (1 - cosine_similarity) over a batch."""
    return 1.0 - F.cosine_similarity(pred, target, dim=1).mean()


def train_mlp_cosine(X_tr, y_tr, X_va, val_t, val_md, svd,
                     hidden=512, n_layers=2, epochs=800, lr=1e-3, wd=1e-4, seed=42):
    """Train MLP with cosine loss on SVD codes."""
    torch.manual_seed(seed)
    model = MLP(X_tr.shape[1], hidden, y_tr.shape[1], n_layers=n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    best_vr, best_state = -999, None
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(X_tr_t)
        loss = cosine_loss(pred, y_tr_t)
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


def train_mlp_direct(X_tr, y_tr_full, X_va, val_t, val_md,
                     hidden=512, n_layers=2, epochs=600, lr=1e-3, wd=1e-4, seed=42):
    """Train MLP to directly predict 5000-dim delta with cosine loss (no SVD)."""
    torch.manual_seed(seed)
    out_d = y_tr_full.shape[1]
    model = MLP(X_tr.shape[1], hidden, out_d, n_layers=n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr_full, dtype=torch.float32)
    best_vr, best_state = -999, None
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(X_tr_t)
        loss = cosine_loss(pred, y_tr_t)
        loss.backward()
        opt.step()
        sched.step()
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                va_preds_arr = model(X_va_t).numpy()
            va_preds = {g: va_preds_arr[i] for i, g in enumerate(val_t)}
            vr, _ = pearson_per_gene(va_preds, val_md)
            if vr > best_vr:
                best_vr = vr
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_vr


def run_ensemble(train_fn, n_seeds, X_tr, y_tr, X_va, val_t, val_md, X_te, test_t, test_md,
                 decode_fn, **kwargs):
    """Run n_seeds ensemble, return (ens_mean_pearson, ens_median, per_seed_scores)."""
    all_preds = []
    seed_scores = []
    for seed in range(n_seeds):
        model, best_vr = train_fn(X_tr, y_tr, X_va, val_t, val_md, seed=seed, **kwargs)
        model.eval()
        with torch.no_grad():
            raw = model(torch.tensor(X_te, dtype=torch.float32)).numpy()
        preds = decode_fn(raw, test_t)
        sm, _ = pearson_per_gene(preds, test_md)
        all_preds.append(raw)
        seed_scores.append(sm)
        print(f"    seed {seed}: val={best_vr:.4f} test={sm:.4f}")
    # Ensemble by averaging raw outputs then decoding
    mean_raw = np.mean(all_preds, axis=0)
    ens_preds = decode_fn(mean_raw, test_t)
    ens_m, ens_med = pearson_per_gene(ens_preds, test_md)
    return ens_m, ens_med, seed_scores


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


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load v10 cache ---
    print("Loading v10 mean-delta cache ...")
    with open(V10_GENES) as f:
        raw = json.load(f)
        valid_genes = raw if isinstance(raw, list) else raw["genes"]
    cache = np.load(str(V10_CACHE))
    # Need control_mean — recompute from adata (control cells only, cheap)
    print("Opening adata for control_mean + top_gene_names ...")
    adata = ad.read_h5ad("data/replogle/K562_essential_raw_singlecell_01.h5ad", backed="r")
    gene_col = adata.obs["gene"].astype(str).tolist()
    gene_names_all = (
        adata.var["gene_name"].tolist() if "gene_name" in adata.var.columns
        else adata.var_names.tolist()
    )
    ctrl_positions = sorted([i for i, g in enumerate(gene_col) if g == "non-targeting"])
    n_ctrl = len(ctrl_positions)

    # HVG selection (variance of 8000 control cells, seed=0 — same as v11-v13)
    rng = np.random.default_rng(0)
    ctrl_sample = sorted(rng.choice(ctrl_positions, size=min(8000, n_ctrl), replace=False).tolist())
    X_cs = adata.X[ctrl_sample]
    X_cs_dense = np.asarray(X_cs.todense() if hasattr(X_cs, "todense") else X_cs, dtype=np.float32)
    lib = X_cs_dense.sum(axis=1, keepdims=True)
    lib = np.where(lib == 0, 1.0, lib)
    X_cs_norm = np.log1p(X_cs_dense / lib * TARGET_SUM)
    del X_cs, X_cs_dense
    gene_var = X_cs_norm.var(axis=0)
    top_gene_idx = np.sort(np.argsort(gene_var)[::-1][:N_HVG])
    top_gene_names = [gene_names_all[i] for i in top_gene_idx]
    del X_cs_norm, gene_var

    # Control mean over all control cells
    ctrl_sum = np.zeros(N_HVG, dtype=np.float64)
    for start in range(0, n_ctrl, CHUNK_SIZE):
        chunk = ctrl_positions[start: start + CHUNK_SIZE]
        X_c = lognorm_chunk(adata.X[chunk], top_gene_idx).astype(np.float64)
        ctrl_sum += X_c.sum(axis=0)
    control_mean = (ctrl_sum / n_ctrl).astype(np.float32)
    del ctrl_sum, adata
    gc.collect()
    print(f"  control_mean ready. Time: {time.time()-t0:.1f}s")

    all_mean_deltas = {
        g: (cache["valid_sums"][i] / cache["valid_counts"][i]).astype(np.float32) - control_mean
        for i, g in enumerate(valid_genes)
    }
    del cache
    gc.collect()
    print(f"  {len(all_mean_deltas)} gene mean deltas. Time: {time.time()-t0:.1f}s")

    # --- Split (same seed as v11-v13 for fair comparison) ---
    train_t, test_t_full = train_test_split(list(all_mean_deltas.keys()), test_size=0.3, random_state=0)
    val_t, test_t = train_test_split(test_t_full, test_size=0.5, random_state=0)
    print(f"Split: {len(train_t)} train / {len(val_t)} val / {len(test_t)} test")

    train_md = {g: all_mean_deltas[g] for g in train_t}
    val_md   = {g: all_mean_deltas[g] for g in val_t}
    test_md  = {g: all_mean_deltas[g] for g in test_t}
    train_gene_list = sorted(train_md)
    D_train = np.stack([train_md[g] for g in train_gene_list])
    D_val   = np.stack([val_md[g] for g in val_t])

    gm = D_train.mean(axis=0)
    gm_m, gm_med = pearson_per_gene({g: gm for g in test_t}, test_md)
    print(f"\nGlobal mean: {gm_m:.4f} (median {gm_med:.4f})")

    # --- Descriptors (same as v13) ---
    gene2vec_table = load_gene2vec_table(Path("data/gene2vec/gene2vec_dim_200_iter_9.txt"))
    gene2vec_lookup: dict[str, np.ndarray] = {}
    for g in top_gene_names:
        if g in gene2vec_table:
            gene2vec_lookup[g] = gene2vec_table[g]
    for g in all_mean_deltas:
        if g not in gene2vec_lookup and g in gene2vec_table:
            gene2vec_lookup[g] = gene2vec_table[g]

    rr_dim = 50
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
    print(f"  Descriptors ready. Time: {time.time()-t0:.1f}s")

    results: dict[str, dict] = {}

    # Build feature arrays
    feat_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    for feat_name, feat_table, feat_dim in [
        ("gene2vec", gene2vec_lookup, 200),
        ("concat",   concat_table,   concat_dim),
    ]:
        X_tr = np.array([feat_table.get(g, np.zeros(feat_dim, np.float32)) for g in train_gene_list])
        X_va = np.array([feat_table.get(g, np.zeros(feat_dim, np.float32)) for g in val_t])
        X_te = np.array([feat_table.get(g, np.zeros(feat_dim, np.float32)) for g in test_t])
        feat_arrays[feat_name] = (X_tr, X_va, X_te, feat_dim)

    # Experiment A: SVD k=200 with cosine loss (best combo from v13 feature)
    print("\n--- Experiment A: SVD k=200 + cosine loss ---")
    svd200 = TruncatedSVD(n_components=200, random_state=0)
    codes200_tr = svd200.fit_transform(D_train)

    for feat_name in ["gene2vec", "concat"]:
        X_tr, X_va, X_te, feat_dim = feat_arrays[feat_name]

        def decode_svd200(raw, gene_list):
            return {g: svd200.inverse_transform(raw[[i]])[0] for i, g in enumerate(gene_list)}

        # Cosine loss, hidden=512, 2 layers (same as v13 but with cosine loss)
        print(f"  {feat_name} + cosine_loss + SVD200 + h512 ens:")
        ens_m, ens_med, seed_sc = run_ensemble(
            train_mlp_cosine, 5, X_tr, codes200_tr, X_va, val_t, val_md,
            X_te, test_t, test_md, decode_svd200,
            svd=svd200, hidden=512, n_layers=2, epochs=800, lr=1e-3, wd=1e-4,
        )
        key = f"{feat_name}_cosine_svd200_h512"
        results[key] = {"test_pearson_mean": ens_m, "test_pearson_median": ens_med, "per_seed": seed_sc}
        print(f"  → {ens_m:.4f} (median {ens_med:.4f})")

        # Wider: hidden=1024, 3 layers, cosine loss, SVD200
        print(f"  {feat_name} + cosine_loss + SVD200 + h1024-3L ens:")
        ens_m2, ens_med2, seed_sc2 = run_ensemble(
            train_mlp_cosine, 5, X_tr, codes200_tr, X_va, val_t, val_md,
            X_te, test_t, test_md, decode_svd200,
            svd=svd200, hidden=1024, n_layers=3, epochs=800, lr=5e-4, wd=1e-4,
        )
        key2 = f"{feat_name}_cosine_svd200_h1024_3L"
        results[key2] = {"test_pearson_mean": ens_m2, "test_pearson_median": ens_med2, "per_seed": seed_sc2}
        print(f"  → {ens_m2:.4f} (median {ens_med2:.4f})")

    # Experiment B: Direct delta prediction (no SVD, cosine loss on 5000-dim output)
    print("\n--- Experiment B: Direct 5000-dim delta prediction, cosine loss ---")
    for feat_name in ["gene2vec", "concat"]:
        X_tr, X_va, X_te, feat_dim = feat_arrays[feat_name]

        def decode_direct(raw, gene_list):
            return {g: raw[i] for i, g in enumerate(gene_list)}

        print(f"  {feat_name} + direct_cosine + h512 ens:")
        ens_m3, ens_med3, seed_sc3 = run_ensemble(
            train_mlp_direct, 5, X_tr, D_train, X_va, val_t, val_md,
            X_te, test_t, test_md, decode_direct,
            hidden=512, n_layers=2, epochs=600, lr=1e-3, wd=1e-4,
        )
        key3 = f"{feat_name}_direct_cosine_h512"
        results[key3] = {"test_pearson_mean": ens_m3, "test_pearson_median": ens_med3, "per_seed": seed_sc3}
        print(f"  → {ens_m3:.4f} (median {ens_med3:.4f})")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    summary = {
        "dataset": "Replogle et al. 2022 K562 essential RAW SINGLE-CELL",
        "note": "v15: cosine loss + wider MLP + SVD k=200 + direct delta prediction",
        "hvg_method": "variance of 8000 control cells (same as v11-v13)",
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
            "global_mean": {"test_pearson_mean": gm_m, "test_pearson_median": gm_med},
            "v13_best": 0.2714,
        },
        "experiments": results,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_DIR}/metrics.json")


if __name__ == "__main__":
    main()

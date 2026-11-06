"""Milestone 24: Contrastive metric learning + retrieval prediction.

Motivation: The oracle achieves 0.4739 by using TRUE test-gene deltas to find nearest
training neighbors. Our direct-prediction models peak at 0.2728. The gap (0.2011) is
large. A key insight: the oracle does RETRIEVAL, not regression.

Hypothesis: We can close much of this gap by training a metric-learning model that
learns to embed genes such that similar-perturbation-effect genes cluster together in
embedding space, then using k-NN retrieval (weighted mean of training deltas) at
inference time.

Approach:
  1. For each pair of training genes (gi, gj), compute target similarity:
       s_ij = Pearson(delta_gi, delta_gj)
  2. Train a Siamese MLP: gene_features → embedding (d=64),
       loss = MSE(cosine_sim(e_i, e_j), s_ij) for sampled pairs
  3. At test time: embed test gene, find top-k training genes by cosine similarity,
       return weighted mean of their deltas (weights = similarity^2 to sharpen).

Compare against:
  - Direct cosine prediction (best from v15: 0.2728)
  - Oracle NN (ceiling: 0.4739)

Features used: g2v + STRING_net + response_role (450d) = best available.
"""
from __future__ import annotations

import gc
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

from virtualcell.data.target_descriptors import (
    fit_response_role_descriptors,
    load_gene2vec_table,
)

TARGET_SUM = 10_000.0
N_HVG = 5000
OUT_DIR = Path("outputs/replogle_genelevel_v17")
V10_CACHE = Path("outputs/replogle_genelevel_v10/mean_delta_cache.npz")
V10_GENES = Path("outputs/replogle_genelevel_v10/mean_delta_cache_genes.json")
SC_H5AD   = Path("data/replogle/K562_essential_raw_singlecell_01.h5ad")
STRING_DIR = Path("data/string")
STRING_LINKS = STRING_DIR / "9606.protein.links.v12.0.txt.gz"
STRING_INFO  = STRING_DIR / "9606.protein.info.v12.0.txt.gz"
MIN_STRING_SCORE = 400
CHUNK_SIZE = 4000


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


# ---- Metric Learning ----

class MetricMLP(nn.Module):
    """Maps gene features → L2-normalized embedding."""
    def __init__(self, in_d: int, hidden: int = 256, emb_d: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, emb_d),
        )

    def forward(self, x):
        e = self.net(x)
        return F.normalize(e, dim=1)


def train_metric_mlp(
    X_tr: np.ndarray,          # (N_train, feat_d)
    delta_tr: np.ndarray,      # (N_train, D)  — training gene deltas
    X_va: np.ndarray,          # (N_val, feat_d)
    val_t: list[str],
    val_md: dict,
    train_gene_list: list[str],
    hidden: int = 256,
    emb_d: int = 64,
    n_pairs_per_epoch: int = 4096,
    epochs: int = 400,
    lr: float = 1e-3,
    wd: float = 1e-4,
    seed: int = 42,
    k_retrieval: int = 5,
):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_tr = len(X_tr)

    model = MetricMLP(X_tr.shape[1], hidden=hidden, emb_d=emb_d)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)

    # Precompute all pairwise Pearson correlations among training genes
    # (N_train × N_train matrix; for 1439 genes this is ~1439² × 4B ≈ 8MB, fine)
    print(f"    Precomputing {n_tr}x{n_tr} pairwise Pearson matrix ...")
    D_norm = delta_tr - delta_tr.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(D_norm, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    D_unit = (D_norm / norms).astype(np.float32)
    sim_matrix = torch.tensor(D_unit @ D_unit.T, dtype=torch.float32)  # (N,N) in [-1,1]

    best_vr, best_state = -999, None

    for epoch in range(epochs):
        model.train()
        # Sample random pairs
        idx_i = rng.integers(0, n_tr, size=n_pairs_per_epoch)
        idx_j = rng.integers(0, n_tr, size=n_pairs_per_epoch)
        same = idx_i == idx_j
        idx_j[same] = (idx_j[same] + 1) % n_tr

        xi = X_tr_t[idx_i]
        xj = X_tr_t[idx_j]
        target_sim = sim_matrix[idx_i, idx_j]

        ei = model(xi)
        ej = model(xj)
        pred_sim = (ei * ej).sum(dim=1)  # cosine sim (already L2-normalized)

        loss = F.mse_loss(pred_sim, target_sim)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if epoch % 40 == 0:
            model.eval()
            with torch.no_grad():
                emb_tr = model(X_tr_t).numpy()   # (N_tr, emb_d)
                emb_va = model(X_va_t).numpy()   # (N_va, emb_d)

            # Retrieve: for each val gene, find top-k training genes by cosine sim
            # emb_tr and emb_va are already L2-normalized
            sim_va_tr = emb_va @ emb_tr.T        # (N_va, N_tr)
            va_preds = {}
            for i, g in enumerate(val_t):
                sims = sim_va_tr[i]              # (N_tr,)
                topk = np.argsort(sims)[-k_retrieval:]
                weights = sims[topk]
                weights = np.maximum(weights, 0)
                w_sum = weights.sum()
                if w_sum < 1e-9:
                    pred = delta_tr[topk].mean(axis=0)
                else:
                    pred = (weights[:, None] * delta_tr[topk]).sum(axis=0) / w_sum
                va_preds[g] = pred

            vr, _ = pearson_per_gene(va_preds, val_md)
            if vr > best_vr:
                best_vr = vr
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_vr


def run_metric_ensemble(
    n_seeds, X_tr, D_train, X_va, val_t, val_md, X_te, test_t, test_md,
    train_gene_list, k_retrieval=5, **kw
):
    # Precompute Pearson similarities once (shared across seeds)
    D_norm = D_train - D_train.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(D_norm, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    D_unit = (D_norm / norms).astype(np.float32)

    all_pred_vecs, seed_sc = [], []
    for seed in range(n_seeds):
        m, bvr = train_metric_mlp(
            X_tr, D_train, X_va, val_t, val_md, train_gene_list,
            seed=seed, k_retrieval=k_retrieval, **kw
        )
        m.eval()
        with torch.no_grad():
            emb_tr = m(torch.tensor(X_tr, dtype=torch.float32)).numpy()
            emb_te = m(torch.tensor(X_te, dtype=torch.float32)).numpy()

        sim_te_tr = emb_te @ emb_tr.T  # (N_te, N_tr)
        preds = {}
        for i, g in enumerate(test_t):
            sims = sim_te_tr[i]
            topk = np.argsort(sims)[-k_retrieval:]
            weights = np.maximum(sims[topk], 0)
            w_sum = weights.sum()
            if w_sum < 1e-9:
                pred = D_train[topk].mean(axis=0)
            else:
                pred = (weights[:, None] * D_train[topk]).sum(axis=0) / w_sum
            preds[g] = pred

        sm, _ = pearson_per_gene(preds, test_md)
        all_pred_vecs.append(np.stack([preds[g] for g in test_t]))
        seed_sc.append(sm)
        print(f"    seed {seed}: val={bvr:.4f} test={sm:.4f}")

    # Ensemble: average raw prediction vectors, re-evaluate
    ens_stack = np.mean(all_pred_vecs, axis=0)
    ens_preds = {g: ens_stack[i] for i, g in enumerate(test_t)}
    em, emed = pearson_per_gene(ens_preds, test_md)
    return em, emed, seed_sc


# ---- Direct cosine MLP (from v15/v16, as baseline) ----

class DirectMLP(nn.Module):
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


def train_direct_mlp(X_tr, y_tr, X_va, val_t, val_md,
                     hidden=512, epochs=600, lr=1e-3, wd=1e-4, seed=42):
    torch.manual_seed(seed)
    model = DirectMLP(X_tr.shape[1], hidden, y_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    best_vr, best_state = -999, None
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = 1.0 - F.cosine_similarity(model(X_tr_t), y_tr_t, dim=1).mean()
        loss.backward()
        opt.step()
        sched.step()
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                va_arr = model(X_va_t).numpy()
            va_preds = {g: va_arr[i] for i, g in enumerate(val_t)}
            vr, _ = pearson_per_gene(va_preds, val_md)
            if vr > best_vr:
                best_vr = vr
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_vr


def run_direct_ensemble(n_seeds, X_tr, D_train, X_va, val_t, val_md, X_te, test_t, test_md, **kw):
    all_preds, seed_sc = [], []
    for seed in range(n_seeds):
        m, bvr = train_direct_mlp(X_tr, D_train, X_va, val_t, val_md, seed=seed, **kw)
        m.eval()
        with torch.no_grad():
            raw = m(torch.tensor(X_te, dtype=torch.float32)).numpy()
        preds = {g: raw[i] for i, g in enumerate(test_t)}
        sm, _ = pearson_per_gene(preds, test_md)
        all_preds.append(raw)
        seed_sc.append(sm)
        print(f"    seed {seed}: val={bvr:.4f} test={sm:.4f}")
    ens = {g: np.mean(all_preds, axis=0)[i] for i, g in enumerate(test_t)}
    em, emed = pearson_per_gene(ens, test_md)
    return em, emed, seed_sc


# ---- STRING loading ----

def load_string_network(gene2vec_lookup: dict) -> dict:
    print("Loading STRING protein info ...")
    string_id_to_gene: dict[str, str] = {}
    with gzip.open(STRING_INFO, "rt") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            string_id_to_gene[parts[0]] = parts[1]

    all_genes_of_interest = set(gene2vec_lookup.keys())
    print(f"Loading STRING links (score >= {MIN_STRING_SCORE}) ...")
    neighbors: dict[str, list] = defaultdict(list)
    n_loaded = 0
    with gzip.open(STRING_LINKS, "rt") as f:
        f.readline()
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            score = int(parts[2])
            if score < MIN_STRING_SCORE:
                continue
            g1 = string_id_to_gene.get(parts[0])
            g2 = string_id_to_gene.get(parts[1])
            if g1 is None or g2 is None:
                continue
            if g1 in all_genes_of_interest or g2 in all_genes_of_interest:
                neighbors[g1].append((g2, score))
                neighbors[g2].append((g1, score))
            n_loaded += 1
    print(f"  Loaded {n_loaded} edges. Coverage: "
          f"{sum(1 for g in all_genes_of_interest if g in neighbors)}/{len(all_genes_of_interest)}")

    zero_emb = np.zeros(200, dtype=np.float32)
    net_emb: dict[str, np.ndarray] = {}
    for gene in all_genes_of_interest:
        neigh = neighbors.get(gene, [])
        if not neigh:
            net_emb[gene] = zero_emb
            continue
        total_w, weighted_sum = 0.0, np.zeros(200, np.float64)
        for nbr, sc in neigh:
            g2v = gene2vec_lookup.get(nbr)
            if g2v is not None:
                w = sc / 1000.0
                weighted_sum += w * g2v
                total_w += w
        net_emb[gene] = (weighted_sum / total_w).astype(np.float32) if total_w > 0 else zero_emb
    return net_emb


def lognorm_chunk(X_sparse, top_gene_idx):
    lib = np.asarray(X_sparse.sum(axis=1)).ravel().astype(np.float64)
    lib = np.where(lib == 0, 1.0, lib)
    X_top = X_sparse[:, top_gene_idx]
    X_dense = np.asarray(X_top.todense() if hasattr(X_top, "todense") else X_top, dtype=np.float32)
    X_dense /= lib[:, None].astype(np.float32)
    X_dense *= TARGET_SUM
    return np.log1p(X_dense)


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load v10 mean-delta cache ---
    print("Loading v10 mean-delta cache ...")
    with open(V10_GENES) as f:
        raw = json.load(f)
        valid_genes = raw if isinstance(raw, list) else raw["genes"]
    cache = np.load(str(V10_CACHE))

    # Compute control_mean + top_gene_names
    print("Opening sc adata for control_mean + top_gene_names ...")
    import anndata as ad
    adata = ad.read_h5ad(str(SC_H5AD), backed="r")
    gene_col = adata.obs["gene"].astype(str).tolist()
    gene_names_all = (
        adata.var["gene_name"].tolist() if "gene_name" in adata.var.columns
        else adata.var_names.tolist()
    )
    ctrl_positions = sorted([i for i, g in enumerate(gene_col) if g == "non-targeting"])
    n_ctrl = len(ctrl_positions)
    rng = np.random.default_rng(0)
    ctrl_sample = sorted(rng.choice(ctrl_positions, size=min(8000, n_ctrl), replace=False).tolist())
    X_cs = adata.X[ctrl_sample]
    X_cs_dense = np.asarray(X_cs.todense() if hasattr(X_cs, "todense") else X_cs, dtype=np.float32)
    lib = X_cs_dense.sum(axis=1, keepdims=True)
    lib = np.where(lib == 0, 1.0, lib)
    gene_var = np.log1p(X_cs_dense / lib * TARGET_SUM).var(axis=0)
    top_gene_idx = np.sort(np.argsort(gene_var)[::-1][:N_HVG])
    top_gene_names = [gene_names_all[i] for i in top_gene_idx]
    del X_cs, X_cs_dense, gene_var

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

    # --- Gene2Vec ---
    gene2vec_table = load_gene2vec_table(Path("data/gene2vec/gene2vec_dim_200_iter_9.txt"))
    gene2vec_lookup: dict[str, np.ndarray] = {}
    for g in top_gene_names:
        if g in gene2vec_table: gene2vec_lookup[g] = gene2vec_table[g]
    for g in all_mean_deltas:
        if g not in gene2vec_lookup and g in gene2vec_table: gene2vec_lookup[g] = gene2vec_table[g]

    # --- STRING network propagation ---
    net_emb = load_string_network(gene2vec_lookup)
    print(f"  STRING network embeddings ready. Time: {time.time()-t0:.1f}s")

    # --- Split ---
    train_t, test_t_full = train_test_split(list(all_mean_deltas.keys()), test_size=0.3, random_state=0)
    val_t, test_t = train_test_split(test_t_full, test_size=0.5, random_state=0)
    print(f"Split: {len(train_t)} train / {len(val_t)} val / {len(test_t)} test")

    train_md = {g: all_mean_deltas[g] for g in train_t}
    val_md   = {g: all_mean_deltas[g] for g in val_t}
    test_md  = {g: all_mean_deltas[g] for g in test_t}
    train_gene_list = sorted(train_md)
    D_train = np.stack([train_md[g] for g in train_gene_list])

    gm = D_train.mean(axis=0)
    gm_m, _ = pearson_per_gene({g: gm for g in test_t}, test_md)
    print(f"\nGlobal mean: {gm_m:.4f}")

    # Response role (50d NMF on training deltas)
    rr_dim = 50
    train_cell_delta_approx = np.stack([train_md[g] for g in train_t])
    response_role_table = fit_response_role_descriptors(
        train_t, train_cell_delta_approx, top_gene_names, n_components=rr_dim, seed=0,
    )
    print(f"  Descriptors ready. Time: {time.time()-t0:.1f}s")

    def get_feat_g2v_net_rr(genes):
        rows = []
        for g in genes:
            g2v = gene2vec_lookup.get(g, np.zeros(200, np.float32))
            rr  = response_role_table.get(g, np.zeros(rr_dim, np.float32))
            net = net_emb.get(g, np.zeros(200, np.float32))
            rows.append(np.concatenate([g2v, net, rr]))
        return np.array(rows)

    X_tr = get_feat_g2v_net_rr(train_gene_list)
    X_va = get_feat_g2v_net_rr(val_t)
    X_te = get_feat_g2v_net_rr(test_t)
    feat_dim = X_tr.shape[1]
    print(f"\nFeature dim: {feat_dim}")

    results: dict[str, dict] = {}

    # ---- Experiment A: Direct cosine prediction (reproduce v15/v16 best) ----
    print("\n--- Experiment A: Direct cosine MLP (g2v+net+rr, h512) ---")
    em_a, emed_a, sc_a = run_direct_ensemble(
        5, X_tr, D_train, X_va, val_t, val_md, X_te, test_t, test_md,
        hidden=512, epochs=600, lr=1e-3, wd=1e-4,
    )
    results["direct_cosine_g2v_net_rr"] = {
        "test_pearson_mean": em_a, "test_pearson_median": emed_a, "per_seed": sc_a
    }
    print(f"  → {em_a:.4f} (median {emed_a:.4f})")

    # ---- Experiment B: Contrastive metric learning + k-NN retrieval ----
    print("\n--- Experiment B: Contrastive metric learning + k=5 retrieval ---")
    em_b, emed_b, sc_b = run_metric_ensemble(
        5, X_tr, D_train, X_va, val_t, val_md, X_te, test_t, test_md,
        train_gene_list=train_gene_list,
        hidden=256, emb_d=64, n_pairs_per_epoch=4096, epochs=400,
        lr=1e-3, wd=1e-4, k_retrieval=5,
    )
    results["contrastive_knn5_g2v_net_rr"] = {
        "test_pearson_mean": em_b, "test_pearson_median": emed_b, "per_seed": sc_b
    }
    print(f"  → {em_b:.4f} (median {emed_b:.4f})")

    # ---- Experiment C: Contrastive + k=10 retrieval ----
    print("\n--- Experiment C: Contrastive metric learning + k=10 retrieval ---")
    em_c, emed_c, sc_c = run_metric_ensemble(
        5, X_tr, D_train, X_va, val_t, val_md, X_te, test_t, test_md,
        train_gene_list=train_gene_list,
        hidden=256, emb_d=64, n_pairs_per_epoch=4096, epochs=400,
        lr=1e-3, wd=1e-4, k_retrieval=10,
    )
    results["contrastive_knn10_g2v_net_rr"] = {
        "test_pearson_mean": em_c, "test_pearson_median": emed_c, "per_seed": sc_c
    }
    print(f"  → {em_c:.4f} (median {emed_c:.4f})")

    # ---- Experiment D: Feature-only k-NN (no metric learning, raw cosine similarity) ----
    print("\n--- Experiment D: Raw feature cosine k-NN (no metric learning, k=5) ---")
    # Normalize feature vectors
    X_tr_norm = X_tr / (np.linalg.norm(X_tr, axis=1, keepdims=True) + 1e-9)
    X_te_norm = X_te / (np.linalg.norm(X_te, axis=1, keepdims=True) + 1e-9)
    sim_te_tr = X_te_norm @ X_tr_norm.T  # (N_te, N_tr)
    raw_knn_preds = {}
    for i, g in enumerate(test_t):
        sims = sim_te_tr[i]
        topk = np.argsort(sims)[-5:]
        weights = np.maximum(sims[topk], 0)
        w_sum = weights.sum()
        if w_sum < 1e-9:
            raw_knn_preds[g] = D_train[topk].mean(axis=0)
        else:
            raw_knn_preds[g] = (weights[:, None] * D_train[topk]).sum(axis=0) / w_sum
    em_d, emed_d = pearson_per_gene(raw_knn_preds, test_md)
    results["raw_feature_knn5"] = {"test_pearson_mean": em_d, "test_pearson_median": emed_d}
    print(f"  → {em_d:.4f} (median {emed_d:.4f})")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    summary = {
        "dataset": "Replogle et al. 2022 K562 essential RAW SINGLE-CELL",
        "note": "v17: Contrastive metric learning + k-NN retrieval vs direct prediction",
        "feature_set": "g2v(200) + STRING_net(200) + response_role(50) = 450d",
        "n_distinct_targets": len(all_mean_deltas),
        "n_train_targets": len(train_t),
        "n_val_targets": len(val_t),
        "n_test_targets": len(test_t),
        "elapsed_seconds": elapsed,
        "reference_points": {
            "global_mean_baseline": gm_m,
            "literature_mean_delta": 0.267,
            "our_oracle_nn_k5_v12": 0.4739,
            "v15_best_direct_cosine": 0.2728,
        },
        "experiments": results,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_DIR}/metrics.json")


if __name__ == "__main__":
    main()

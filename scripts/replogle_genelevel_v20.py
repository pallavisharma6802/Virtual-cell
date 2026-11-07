"""Milestone 27: Meta-learning — Prototypical Network + GP kernel regression.

The core insight from v17 (contrastive learning): the gap is feature quality.
Meta-learning won't fix bad features, but it might better exploit the structure
in what we have by learning how to generalize from few examples.

Approaches:
  A) Prototypical Network (episode training):
     - Each episode: pick N support genes → embed → compute prototype
     - Query gene: embed → distance to prototypes → predict as weighted avg of support deltas
     - Train embedding net with cosine loss on prediction quality

  B) Gaussian Process with STRING kernel:
     - K(g1, g2) = exp(-||h(g1) - h(g2)||^2 / (2*l^2)) where h = learned embedding
     - GP posterior mean = best prediction for held-out genes
     - Very few parameters, interpretable uncertainty

  C) Task-specific fine-tuning baseline:
     - Train global model, then fine-tune per-test gene using 1-NN support set
     - Cheap approximation to MAML without 2nd-order gradients

This script tests A and C (GP-B is expensive at N=1439).
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

from virtualcell.data.target_descriptors import load_gene2vec_table

TARGET_SUM = 10_000.0
N_HVG = 5000
CHUNK_SIZE = 4000
OUT_DIR = Path("outputs/replogle_genelevel_v20")
V10_CACHE = Path("outputs/replogle_genelevel_v10/mean_delta_cache.npz")
V10_GENES = Path("outputs/replogle_genelevel_v10/mean_delta_cache_genes.json")
SC_H5AD   = Path("data/replogle/K562_essential_raw_singlecell_01.h5ad")
STRING_LINKS = Path("data/string/9606.protein.links.v12.0.txt.gz")
STRING_INFO  = Path("data/string/9606.protein.info.v12.0.txt.gz")
MIN_STRING_SCORE = 400
N_EPISODES = 8000
N_SUPPORT = 10
N_QUERY = 15


def pearson_per_gene(pred_dict, true_dict):
    genes = sorted(set(pred_dict) & set(true_dict))
    vals = []
    for g in genes:
        p, t = pred_dict[g], true_dict[g]
        if np.std(t) < 1e-6 or np.std(p) < 1e-6: vals.append(0.0)
        else: vals.append(float(np.clip(pearsonr(p, t)[0], -1, 1)))
    return float(np.mean(vals)), float(np.median(vals))


def lognorm_chunk(X_sparse, top_gene_idx):
    lib = np.asarray(X_sparse.sum(axis=1)).ravel().astype(np.float64)
    lib = np.where(lib == 0, 1.0, lib)
    X_top = X_sparse[:, top_gene_idx]
    X_dense = np.asarray(X_top.todense() if hasattr(X_top, "todense") else X_top, dtype=np.float32)
    X_dense /= lib[:, None].astype(np.float32); X_dense *= TARGET_SUM
    return np.log1p(X_dense)


def load_string_net_feats(gene2vec_lookup, all_gene_names, dim=200):
    """Load 1-hop STRING propagated features (same as v16 baseline)."""
    print("Loading STRING ...")
    string_id_to_gene: dict[str, str] = {}
    with gzip.open(STRING_INFO, "rt") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                string_id_to_gene[parts[0]] = parts[1]

    all_set = set(all_gene_names)
    neighbors: dict[str, list] = defaultdict(list)
    with gzip.open(STRING_LINKS, "rt") as f:
        f.readline()
        for line in f:
            parts = line.split()
            if len(parts) < 3: continue
            score = int(parts[2])
            if score < MIN_STRING_SCORE: continue
            g1 = string_id_to_gene.get(parts[0])
            g2 = string_id_to_gene.get(parts[1])
            if g1 and g2 and (g1 in all_set or g2 in all_set):
                neighbors[g1].append((g2, score / 1000.0))
                neighbors[g2].append((g1, score / 1000.0))

    zero = np.zeros(dim, np.float32)
    feat_dict = {}
    for g in all_gene_names:
        g2v = gene2vec_lookup.get(g, zero)
        nbrs = neighbors.get(g, [])
        if nbrs:
            total_w, ws = 0.0, np.zeros(dim, np.float64)
            for nbr, w in nbrs:
                v = gene2vec_lookup.get(nbr)
                if v is not None:
                    ws += w * v; total_w += w
            net = (ws / total_w).astype(np.float32) if total_w > 0 else zero
        else:
            net = zero
        feat_dict[g] = np.concatenate([g2v, net])  # 400d
    print(f"  STRING net feats ready, {len(feat_dict)} genes, 400d")
    return feat_dict


class EmbedNet(nn.Module):
    """Shared embedding network for prototypical / meta-learning."""
    def __init__(self, in_d: int, embed_d: int = 128, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, embed_d),
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class ProtoDecoder(nn.Module):
    """Decode weighted combination of support deltas to predict query delta."""
    def __init__(self, delta_d: int, embed_d: int = 128, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(delta_d + embed_d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, delta_d),
        )
    def forward(self, context_delta, query_emb):
        return self.net(torch.cat([context_delta, query_emb], dim=-1))


class ProtoNet(nn.Module):
    def __init__(self, feat_d: int, delta_d: int, embed_d: int = 128):
        super().__init__()
        self.encoder = EmbedNet(feat_d, embed_d)
        self.decoder = ProtoDecoder(delta_d, embed_d)

    def forward(self, support_feats, support_deltas, query_feats):
        """
        support_feats:  (n_support, feat_d)
        support_deltas: (n_support, delta_d)
        query_feats:    (n_query, feat_d)
        Returns:        (n_query, delta_d) predictions
        """
        s_emb = self.encoder(support_feats)       # (S, E)
        q_emb = self.encoder(query_feats)          # (Q, E)

        # Soft-attention over support set for each query
        sim = torch.mm(q_emb, s_emb.t())           # (Q, S)
        attn = torch.softmax(sim * 10.0, dim=-1)   # (Q, S) — temperature=10

        # Weighted mean of support deltas
        context = torch.mm(attn, support_deltas)   # (Q, delta_d)

        return self.decoder(context, q_emb)        # (Q, delta_d)


def sample_episode(genes, feat_dict, delta_dict, n_support, n_query, rng):
    """Sample one episode: S support + Q query genes (disjoint)."""
    chosen = rng.choice(genes, size=n_support + n_query, replace=False).tolist()
    sup, qry = chosen[:n_support], chosen[n_support:]
    s_feats = torch.tensor([feat_dict[g] for g in sup], dtype=torch.float32)
    s_del   = torch.tensor([delta_dict[g] for g in sup], dtype=torch.float32)
    q_feats = torch.tensor([feat_dict[g] for g in qry], dtype=torch.float32)
    q_del   = torch.tensor([delta_dict[g] for g in qry], dtype=torch.float32)
    return s_feats, s_del, q_feats, q_del


def train_protonet(train_genes, feat_dict, delta_dict, feat_d, delta_d,
                   n_episodes=N_EPISODES, seed=42):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = ProtoNet(feat_d, delta_d)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_episodes)

    best_loss, best_state = 1e9, None
    for ep in range(n_episodes):
        s_f, s_d, q_f, q_d = sample_episode(train_genes, feat_dict, delta_dict,
                                             N_SUPPORT, N_QUERY, rng)
        model.train(); opt.zero_grad()
        pred = model(s_f, s_d, q_f)
        loss = 1.0 - F.cosine_similarity(pred, q_d, dim=1).mean()
        loss.backward(); opt.step(); sched.step()

        if ep % 400 == 0:
            lval = loss.item()
            if lval < best_loss: best_loss = lval; best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if ep % 2000 == 0: print(f"    ep {ep}/{n_episodes} loss={lval:.4f}")

    model.load_state_dict(best_state)
    return model


def predict_protonet(model, test_genes, support_genes, feat_dict, delta_dict):
    """Predict for each test gene using all support (training) genes."""
    s_feats = torch.tensor([feat_dict[g] for g in support_genes], dtype=torch.float32)
    s_dels  = torch.tensor([delta_dict[g] for g in support_genes], dtype=torch.float32)
    q_feats = torch.tensor([feat_dict[g] for g in test_genes], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        preds = model(s_feats, s_dels, q_feats)
    return {g: preds[i].numpy() for i, g in enumerate(test_genes)}


def run_direct_mlp(feat_dict, feat_d, train_genes, train_md, val_t, val_md, test_t, test_md,
                   hidden=512, epochs=600, n_seeds=5):
    """Direct cosine MLP baseline (same as v16) for comparison."""
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(feat_d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(hidden, list(train_md.values())[0].shape[0]),
            )
        def forward(self, x): return self.net(x)

    D_train = np.stack([train_md[g] for g in sorted(train_genes)])
    train_gene_list = sorted(train_genes)
    X_tr = np.array([feat_dict.get(g, np.zeros(feat_d, np.float32)) for g in train_gene_list])
    X_va = np.array([feat_dict.get(g, np.zeros(feat_d, np.float32)) for g in val_t])
    X_te = np.array([feat_dict.get(g, np.zeros(feat_d, np.float32)) for g in test_t])
    preds_all = []
    for seed in range(n_seeds):
        torch.manual_seed(seed); m = MLP()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        Xtt = torch.tensor(X_tr, dtype=torch.float32)
        ytt = torch.tensor(D_train, dtype=torch.float32)
        Xvt = torch.tensor(X_va, dtype=torch.float32)
        best_vr, best_state = -999, None
        for epoch in range(epochs):
            m.train(); opt.zero_grad()
            loss = 1.0 - F.cosine_similarity(m(Xtt), ytt, dim=1).mean()
            loss.backward(); opt.step(); sched.step()
            if epoch % 20 == 0:
                m.eval()
                with torch.no_grad(): va = m(Xvt).numpy()
                vr, _ = pearson_per_gene({g: va[i] for i, g in enumerate(val_t)}, val_md)
                if vr > best_vr: best_vr = vr; best_state = {k: v.clone() for k, v in m.state_dict().items()}
        m.load_state_dict(best_state); m.eval()
        with torch.no_grad(): raw = m(torch.tensor(X_te, dtype=torch.float32)).numpy()
        preds_all.append(raw)
        sm, _ = pearson_per_gene({g: raw[i] for i, g in enumerate(test_t)}, test_md)
        print(f"    seed {seed}: val={best_vr:.4f} test={sm:.4f}")
    ens = {g: np.mean(preds_all, axis=0)[i] for i, g in enumerate(test_t)}
    em, emed = pearson_per_gene(ens, test_md)
    return em, emed


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading v10 mean-delta cache ...")
    with open(V10_GENES) as f:
        raw = json.load(f)
        valid_genes = raw if isinstance(raw, list) else raw["genes"]
    cache = np.load(str(V10_CACHE))

    print("Opening sc adata for control_mean ...")
    import anndata as ad
    adata = ad.read_h5ad(str(SC_H5AD), backed="r")
    gene_col = adata.obs["gene"].astype(str).tolist()
    ctrl_positions = sorted([i for i, g in enumerate(gene_col) if g == "non-targeting"])
    n_ctrl = len(ctrl_positions)
    rng = np.random.default_rng(0)
    ctrl_sample = sorted(rng.choice(ctrl_positions, size=min(8000, n_ctrl), replace=False).tolist())
    X_cs = adata.X[ctrl_sample]
    X_cs_dense = np.asarray(X_cs.todense() if hasattr(X_cs, "todense") else X_cs, dtype=np.float32)
    lib = X_cs_dense.sum(axis=1, keepdims=True); lib = np.where(lib == 0, 1.0, lib)
    gene_var = np.log1p(X_cs_dense / lib * TARGET_SUM).var(axis=0)
    top_gene_idx = np.sort(np.argsort(gene_var)[::-1][:N_HVG])
    del X_cs, X_cs_dense, gene_var
    ctrl_sum = np.zeros(N_HVG, dtype=np.float64)
    for start in range(0, n_ctrl, CHUNK_SIZE):
        chunk = ctrl_positions[start: start + CHUNK_SIZE]
        ctrl_sum += lognorm_chunk(adata.X[chunk], top_gene_idx).astype(np.float64).sum(axis=0)
    control_mean = (ctrl_sum / n_ctrl).astype(np.float32)
    del ctrl_sum, adata; gc.collect()
    print(f"  control_mean ready. Time: {time.time()-t0:.1f}s")

    all_mean_deltas = {
        g: (cache["valid_sums"][i] / cache["valid_counts"][i]).astype(np.float32) - control_mean
        for i, g in enumerate(valid_genes)
    }
    del cache; gc.collect()

    gene2vec_table = load_gene2vec_table(Path("data/gene2vec/gene2vec_dim_200_iter_9.txt"))
    gene2vec_lookup = {g: gene2vec_table[g] for g in list(all_mean_deltas.keys()) if g in gene2vec_table}
    all_gene_names = list(all_mean_deltas.keys())

    feat_dict = load_string_net_feats(gene2vec_lookup, all_gene_names)
    feat_d = 400

    train_t, test_t_full = train_test_split(all_gene_names, test_size=0.3, random_state=0)
    val_t, test_t = train_test_split(test_t_full, test_size=0.5, random_state=0)
    print(f"Split: {len(train_t)} train / {len(val_t)} val / {len(test_t)} test")

    train_md = {g: all_mean_deltas[g] for g in train_t}
    val_md   = {g: all_mean_deltas[g] for g in val_t}
    test_md  = {g: all_mean_deltas[g] for g in test_t}
    delta_d = list(all_mean_deltas.values())[0].shape[0]

    gm = np.stack(list(train_md.values())).mean(axis=0)
    gm_m, _ = pearson_per_gene({g: gm for g in test_t}, test_md)
    print(f"\nGlobal mean: {gm_m:.4f}")

    results = {}

    # --- Experiment A: Direct cosine MLP baseline (g2v_net = v16 best) ---
    print("\n--- Experiment A: Direct cosine MLP (g2v_net 400d, baseline) ---")
    em_a, emed_a = run_direct_mlp(feat_dict, feat_d, train_t, train_md, val_t, val_md, test_t, test_md)
    results["direct_mlp_g2v_net"] = {"test_pearson_mean": em_a, "test_pearson_median": emed_a}
    print(f"  → {em_a:.4f} (median {emed_a:.4f})")

    # --- Experiment B: Prototypical Network (5-seed ensemble) ---
    print("\n--- Experiment B: Prototypical Network (episodic meta-learning) ---")
    proto_preds_all = []
    proto_seed_sc = []
    for seed in range(5):
        print(f"  Training seed {seed} ...")
        model = train_protonet(train_t, feat_dict, train_md, feat_d, delta_d,
                               n_episodes=N_EPISODES, seed=seed)
        preds = predict_protonet(model, test_t, train_t, feat_dict, train_md)
        raw = np.array([preds[g] for g in test_t])
        sm, _ = pearson_per_gene(preds, test_md)
        proto_preds_all.append(raw)
        proto_seed_sc.append(sm)
        print(f"    seed {seed}: test={sm:.4f}")

    ens = {g: np.mean(proto_preds_all, axis=0)[i] for i, g in enumerate(test_t)}
    em_b, emed_b = pearson_per_gene(ens, test_md)
    results["protonet_episodic"] = {
        "test_pearson_mean": em_b, "test_pearson_median": emed_b, "per_seed": proto_seed_sc
    }
    print(f"  → {em_b:.4f} (median {emed_b:.4f})")

    # --- Experiment C: 1-NN retrieval (soft analog to meta-learning test-time adapt) ---
    print("\n--- Experiment C: Soft kNN retrieval (k=5, STRING-weighted cosine) ---")
    X_tr_arr = np.array([feat_dict[g] for g in train_t])
    X_te_arr = np.array([feat_dict[g] for g in test_t])
    # Normalize for cosine
    X_tr_norm = X_tr_arr / (np.linalg.norm(X_tr_arr, axis=1, keepdims=True) + 1e-8)
    X_te_norm = X_te_arr / (np.linalg.norm(X_te_arr, axis=1, keepdims=True) + 1e-8)
    sims = X_te_norm @ X_tr_norm.T  # (n_test, n_train)
    D_train_arr = np.array([train_md[g] for g in train_t])

    knn_preds_dict = {}
    for k in [1, 5, 10]:
        preds_k = {}
        for i, g in enumerate(test_t):
            top_k = np.argsort(sims[i])[::-1][:k]
            w = sims[i][top_k]
            w = np.maximum(w, 0); w = w / (w.sum() + 1e-9)
            preds_k[g] = (w[:, None] * D_train_arr[top_k]).sum(axis=0)
        em_k, _ = pearson_per_gene(preds_k, test_md)
        knn_preds_dict[k] = preds_k
        results[f"knn_cosine_k{k}"] = {"test_pearson_mean": em_k}
        print(f"  k={k}: {em_k:.4f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    summary = {
        "note": "v20: Meta-learning — ProtoNet + kNN retrieval",
        "n_train": len(train_t), "n_val": len(val_t), "n_test": len(test_t),
        "elapsed_seconds": elapsed,
        "reference": {"global_mean": gm_m, "v16_best": 0.3103, "oracle": 0.4739},
        "experiments": results,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_DIR}/metrics.json")


if __name__ == "__main__":
    main()

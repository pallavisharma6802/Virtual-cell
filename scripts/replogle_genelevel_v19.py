"""Milestone 26: Multi-hop GNN on STRING — learned aggregation.

v16 did 1-hop weighted average of Gene2Vec (fixed, hand-engineered).
This script implements a 2-layer GraphSAGE-style GNN directly in PyTorch
(no external library) where the aggregation weights are LEARNED end-to-end.

Architecture:
  Layer 0: node feature = Gene2Vec(g) ∈ R^200
  Layer 1: h1(g) = ReLU(W1 · concat[g2v(g), mean_pool(g2v(nbr))])  → R^256
  Layer 2: h2(g) = ReLU(W2 · concat[h1(g), mean_pool(h1(nbr))])    → R^128
  Prediction: MLP(h2(g)) → R^5000  (direct cosine loss)

Key difference from v16:
  - v16: fixed weighted avg with STRING scores as weights
  - v19: learned linear transform at each hop, uniform or score-weighted aggregation
  - Two hops capture 2-hop neighborhood (e.g., kinase → substrate → downstream target)

Also tests fixed 2-hop (no learning in GNN layers) as ablation.
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
OUT_DIR = Path("outputs/replogle_genelevel_v19")
V10_CACHE = Path("outputs/replogle_genelevel_v10/mean_delta_cache.npz")
V10_GENES = Path("outputs/replogle_genelevel_v10/mean_delta_cache_genes.json")
SC_H5AD   = Path("data/replogle/K562_essential_raw_singlecell_01.h5ad")
STRING_LINKS = Path("data/string/9606.protein.links.v12.0.txt.gz")
STRING_INFO  = Path("data/string/9606.protein.info.v12.0.txt.gz")
MIN_STRING_SCORE = 400


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


def load_string_neighbors(all_genes: set[str]) -> dict[str, list[tuple[str, float]]]:
    """Load STRING neighbors for all genes of interest (bidirectional)."""
    print("Loading STRING ...")
    string_id_to_gene: dict[str, str] = {}
    with gzip.open(STRING_INFO, "rt") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                string_id_to_gene[parts[0]] = parts[1]

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
            if g1 and g2 and (g1 in all_genes or g2 in all_genes):
                neighbors[g1].append((g2, score / 1000.0))
                neighbors[g2].append((g1, score / 1000.0))

    print(f"  STRING: {sum(1 for g in all_genes if g in neighbors)}/{len(all_genes)} genes have neighbors")
    return dict(neighbors)


def compute_multihop_embeddings(
    gene2vec_lookup: dict[str, np.ndarray],
    neighbors: dict[str, list[tuple[str, float]]],
    all_genes: list[str],
    n_hops: int = 2,
    use_score_weights: bool = True,
) -> dict[str, np.ndarray]:
    """Compute fixed multi-hop aggregation (no learning). Returns R^(200*(1+n_hops)) per gene."""
    dim = 200
    zero = np.zeros(dim, np.float32)
    # H[0] = Gene2Vec
    H = [{g: gene2vec_lookup.get(g, zero) for g in all_genes}]

    for hop in range(n_hops):
        H_prev = H[-1]
        H_new: dict[str, np.ndarray] = {}
        for g in all_genes:
            nbrs = neighbors.get(g, [])
            if not nbrs:
                H_new[g] = zero
                continue
            if use_score_weights:
                total_w, ws = 0.0, np.zeros(dim, np.float64)
                for nbr, w in nbrs:
                    v = H_prev.get(nbr)
                    if v is not None:
                        ws += w * v; total_w += w
                H_new[g] = (ws / total_w).astype(np.float32) if total_w > 0 else zero
            else:
                vecs = [H_prev[nbr] for nbr, _ in nbrs if nbr in H_prev]
                H_new[g] = np.mean(vecs, axis=0).astype(np.float32) if vecs else zero
        H.append(H_new)

    # Concatenate all hops: [H0, H1, H2, ...] = 200*(1+n_hops) per gene
    combined: dict[str, np.ndarray] = {}
    for g in all_genes:
        combined[g] = np.concatenate([H[hop][g] for hop in range(len(H))])
    return combined


class GNNEncoder(nn.Module):
    """2-layer GraphSAGE-style encoder: learns aggregation weights."""
    def __init__(self, in_d: int = 200, h1: int = 256, h2: int = 128, dropout: float = 0.2):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_d * 2, h1), nn.LayerNorm(h1), nn.GELU(), nn.Dropout(dropout)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(h1 * 2, h2), nn.LayerNorm(h2), nn.GELU(), nn.Dropout(dropout)
        )
        self.h1 = h1
        self.h2 = h2

    def forward(self, node_feats: torch.Tensor,
                adj_indices: list[list[int]],
                adj_weights: list[list[float]]) -> torch.Tensor:
        """
        node_feats: (N, in_d)
        adj_indices: list of neighbor index lists (one per node)
        adj_weights: list of neighbor weight lists (one per node)
        Returns: (N, h2)
        """
        N = node_feats.shape[0]

        # Layer 1: aggregate Gene2Vec, concat with self
        agg1 = torch.zeros(N, node_feats.shape[1], device=node_feats.device)
        for i, (nbr_idx, nbr_w) in enumerate(zip(adj_indices, adj_weights)):
            if nbr_idx:
                w = torch.tensor(nbr_w, dtype=torch.float32, device=node_feats.device)
                w = w / (w.sum() + 1e-9)
                agg1[i] = (w.unsqueeze(1) * node_feats[nbr_idx]).sum(0)

        h1 = self.layer1(torch.cat([node_feats, agg1], dim=1))  # (N, h1)

        # Layer 2: aggregate h1, concat with self
        agg2 = torch.zeros(N, self.h1, device=node_feats.device)
        for i, (nbr_idx, nbr_w) in enumerate(zip(adj_indices, adj_weights)):
            if nbr_idx:
                w = torch.tensor(nbr_w, dtype=torch.float32, device=node_feats.device)
                w = w / (w.sum() + 1e-9)
                agg2[i] = (w.unsqueeze(1) * h1[nbr_idx]).sum(0)

        h2 = self.layer2(torch.cat([h1, agg2], dim=1))  # (N, h2)
        return h2


class GNNPredictor(nn.Module):
    """GNN encoder + MLP decoder for direct delta prediction."""
    def __init__(self, in_d: int, out_d: int, gnn_h1: int = 256, gnn_h2: int = 128,
                 mlp_hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.encoder = GNNEncoder(in_d=in_d, h1=gnn_h1, h2=gnn_h2, dropout=dropout)
        self.decoder = nn.Sequential(
            nn.Linear(gnn_h2, mlp_hidden), nn.LayerNorm(mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, out_d),
        )

    def forward(self, node_feats, adj_indices, adj_weights):
        emb = self.encoder(node_feats, adj_indices, adj_weights)
        return self.decoder(emb)


def build_adj(gene_list: list[str], neighbors: dict) -> tuple[list[list[int]], list[list[float]]]:
    """Build adjacency lists (index-based) for gene_list nodes."""
    idx = {g: i for i, g in enumerate(gene_list)}
    adj_idx, adj_w = [], []
    for g in gene_list:
        nbrs = neighbors.get(g, [])
        ii, ww = [], []
        for nbr, w in nbrs:
            if nbr in idx:
                ii.append(idx[nbr]); ww.append(w)
        adj_idx.append(ii); adj_w.append(ww)
    return adj_idx, adj_w


def train_gnn(gene_feats: np.ndarray, D_train: np.ndarray,
              adj_idx: list, adj_w: list,
              val_gene_feats: np.ndarray, val_adj_idx: list, val_adj_w: list,
              val_t: list, val_md: dict,
              epochs: int = 400, lr: float = 1e-3, wd: float = 1e-4, seed: int = 42):
    torch.manual_seed(seed)
    model = GNNPredictor(gene_feats.shape[1], D_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X_tr_t = torch.tensor(gene_feats, dtype=torch.float32)
    y_tr_t = torch.tensor(D_train, dtype=torch.float32)
    X_va_t = torch.tensor(val_gene_feats, dtype=torch.float32)
    best_vr, best_state = -999, None

    for epoch in range(epochs):
        model.train(); opt.zero_grad()
        pred = model(X_tr_t, adj_idx, adj_w)
        loss = 1.0 - F.cosine_similarity(pred, y_tr_t, dim=1).mean()
        loss.backward(); opt.step(); sched.step()

        if epoch % 40 == 0:
            model.eval()
            with torch.no_grad():
                va_pred = model(X_va_t, val_adj_idx, val_adj_w).numpy()
            vr, _ = pearson_per_gene({g: va_pred[i] for i, g in enumerate(val_t)}, val_md)
            if vr > best_vr: best_vr = vr; best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_vr


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load cache ---
    print("Loading v10 mean-delta cache ...")
    with open(V10_GENES) as f:
        raw = json.load(f)
        valid_genes = raw if isinstance(raw, list) else raw["genes"]
    cache = np.load(str(V10_CACHE))

    print("Opening sc adata for control_mean ...")
    import anndata as ad
    adata = ad.read_h5ad(str(SC_H5AD), backed="r")
    gene_col = adata.obs["gene"].astype(str).tolist()
    gene_names_all = adata.var["gene_name"].tolist() if "gene_name" in adata.var.columns else adata.var_names.tolist()
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
    print(f"  {len(all_mean_deltas)} gene mean deltas. Time: {time.time()-t0:.1f}s")

    gene2vec_table = load_gene2vec_table(Path("data/gene2vec/gene2vec_dim_200_iter_9.txt"))
    gene2vec_lookup: dict[str, np.ndarray] = {
        g: gene2vec_table[g] for g in list(all_mean_deltas.keys()) if g in gene2vec_table
    }

    all_gene_names = list(all_mean_deltas.keys())
    neighbors = load_string_neighbors(set(all_gene_names))

    # --- Split ---
    train_t, test_t_full = train_test_split(all_gene_names, test_size=0.3, random_state=0)
    val_t, test_t = train_test_split(test_t_full, test_size=0.5, random_state=0)
    print(f"Split: {len(train_t)} train / {len(val_t)} val / {len(test_t)} test")
    train_gene_list = sorted(train_t)
    train_md = {g: all_mean_deltas[g] for g in train_t}
    val_md   = {g: all_mean_deltas[g] for g in val_t}
    test_md  = {g: all_mean_deltas[g] for g in test_t}
    D_train = np.stack([train_md[g] for g in train_gene_list])

    gm = D_train.mean(axis=0)
    gm_m, _ = pearson_per_gene({g: gm for g in test_t}, test_md)
    print(f"\nGlobal mean: {gm_m:.4f}")

    zero = np.zeros(200, np.float32)

    results: dict[str, dict] = {}

    # ---- Experiment A: Fixed 2-hop STRING (ablation, compare with v16 1-hop) ----
    print("\n--- Experiment A: Fixed 2-hop STRING aggregation ---")
    combined_2hop = compute_multihop_embeddings(gene2vec_lookup, neighbors, all_gene_names, n_hops=2)
    feat_dim_2hop = 200 * 3  # g2v + hop1 + hop2 = 600d
    print(f"  Feature dim: {feat_dim_2hop}")

    class SimpleMLP(nn.Module):
        def __init__(self, in_d, hidden, out_d, dropout=0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, out_d),
            )
        def forward(self, x): return self.net(x)

    def run_simple_ensemble(feat_dict, feat_dim, n_seeds=5):
        X_tr = np.array([feat_dict.get(g, np.zeros(feat_dim, np.float32)) for g in train_gene_list])
        X_va = np.array([feat_dict.get(g, np.zeros(feat_dim, np.float32)) for g in val_t])
        X_te = np.array([feat_dict.get(g, np.zeros(feat_dim, np.float32)) for g in test_t])
        preds_all, seed_sc = [], []
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            m = SimpleMLP(feat_dim, 512, D_train.shape[1])
            opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=600)
            Xtt = torch.tensor(X_tr, dtype=torch.float32)
            Xvt = torch.tensor(X_va, dtype=torch.float32)
            ytt = torch.tensor(D_train, dtype=torch.float32)
            best_vr, best_state = -999, None
            for epoch in range(600):
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
            preds = {g: raw[i] for i, g in enumerate(test_t)}
            sm, _ = pearson_per_gene(preds, test_md)
            preds_all.append(raw); seed_sc.append(sm)
            print(f"    seed {seed}: val={best_vr:.4f} test={sm:.4f}")
        ens = {g: np.mean(preds_all, axis=0)[i] for i, g in enumerate(test_t)}
        em, emed = pearson_per_gene(ens, test_md)
        return em, emed, seed_sc

    em_a, emed_a, sc_a = run_simple_ensemble(combined_2hop, feat_dim_2hop)
    results["fixed_2hop_600d"] = {"test_pearson_mean": em_a, "test_pearson_median": emed_a, "per_seed": sc_a}
    print(f"  → {em_a:.4f} (median {emed_a:.4f})")

    # ---- Experiment B: Learned GNN (2-layer GraphSAGE) ----
    print("\n--- Experiment B: Learned 2-layer GraphSAGE GNN ---")
    tr_node_feats = np.array([gene2vec_lookup.get(g, zero) for g in train_gene_list])
    va_node_feats = np.array([gene2vec_lookup.get(g, zero) for g in val_t])
    te_node_feats = np.array([gene2vec_lookup.get(g, zero) for g in test_t])
    tr_adj_idx, tr_adj_w = build_adj(train_gene_list, neighbors)
    va_adj_idx, va_adj_w = build_adj(val_t, neighbors)
    te_adj_idx, te_adj_w = build_adj(test_t, neighbors)

    gnn_preds_all, gnn_seed_sc = [], []
    for seed in range(5):
        m, bvr = train_gnn(tr_node_feats, D_train, tr_adj_idx, tr_adj_w,
                           va_node_feats, va_adj_idx, va_adj_w, val_t, val_md,
                           epochs=400, seed=seed)
        m.eval()
        with torch.no_grad():
            raw = m(torch.tensor(te_node_feats, dtype=torch.float32), te_adj_idx, te_adj_w).numpy()
        preds = {g: raw[i] for i, g in enumerate(test_t)}
        sm, _ = pearson_per_gene(preds, test_md)
        gnn_preds_all.append(raw); gnn_seed_sc.append(sm)
        print(f"    seed {seed}: val={bvr:.4f} test={sm:.4f}")
    ens = {g: np.mean(gnn_preds_all, axis=0)[i] for i, g in enumerate(test_t)}
    em_b, emed_b = pearson_per_gene(ens, test_md)
    results["learned_gnn_2layer"] = {"test_pearson_mean": em_b, "test_pearson_median": emed_b, "per_seed": gnn_seed_sc}
    print(f"  → {em_b:.4f} (median {emed_b:.4f})")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    summary = {
        "note": "v19: Multi-hop GNN — fixed 2-hop vs learned GraphSAGE",
        "n_train": len(train_t), "n_val": len(val_t), "n_test": len(test_t),
        "elapsed_seconds": elapsed,
        "reference": {"global_mean": gm_m, "v16_best": 0.3103, "oracle": 0.4739},
        "experiments": results,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_DIR}/metrics.json")


if __name__ == "__main__":
    main()

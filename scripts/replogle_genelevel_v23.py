"""Milestone 30: Stronger pre-training (SimCLR NT-Xent) + deeper architecture.

v22 finding: combo (1040d) + bulk pre-train (MSE contrastive) = 0.3432.

This script pushes further with:
  1. SimCLR NT-Xent loss for pre-training (harder negatives, temperature scaling)
     vs v22's MSE on pairwise cosine similarity
  2. Deeper encoder: 1024→1024→512→256 vs v22's 512→512→256
  3. More pre-training epochs: 500 vs 300
  4. Combined cosine+MSE fine-tuning loss: alpha * cosine_loss + (1-alpha) * mse_loss
     (cosine alone ignores scale; adding MSE may help predict magnitude)

Architecture:
  Encoder: 1040 → 1024 → LayerNorm → GELU → Dropout(0.2)
                → 1024 → LayerNorm → GELU → Dropout(0.2)
                → 512  → LayerNorm → GELU → Dropout(0.2)
                → 256
  Head: LayerNorm(256) → GELU → Dropout(0.3) → 512 → LayerNorm → GELU → Dropout(0.3) → 5000

NT-Xent loss (SimCLR):
  - For each gene g, create two views: original features f(g) and
    STRING-neighbor-augmented features f_aug(g) = f(g) + 0.1 * mean(f(nbr))
  - Within batch, treat (g, g_aug) as positive pairs, all others as negatives
  - Loss: -log(exp(sim(z_i, z_j)/tau) / sum_k exp(sim(z_i, z_k)/tau))
  - Temperature tau=0.1

Experiments:
  A) combo + SimCLR pre-train (deeper encoder, tau=0.1)
  B) combo + SimCLR + combined loss (cosine + MSE fine-tune)
  C) combo + MSE pre-train (v22 baseline for comparison) with deeper encoder
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
OUT_DIR = Path("outputs/replogle_genelevel_v23")
V10_CACHE = Path("outputs/replogle_genelevel_v10/mean_delta_cache.npz")
V10_GENES = Path("outputs/replogle_genelevel_v10/mean_delta_cache_genes.json")
SC_H5AD   = Path("data/replogle/K562_essential_raw_singlecell_01.h5ad")
BULK_H5AD = Path("data/replogle_k562_essential/K562_essential_raw_bulk_01.h5ad")
ESM2_CACHE = Path("data/esm2/esm2_embeddings.npz")
STRING_LINKS = Path("data/string/9606.protein.links.v12.0.txt.gz")
STRING_INFO  = Path("data/string/9606.protein.info.v12.0.txt.gz")
MIN_STRING_SCORE = 400
PRETRAIN_EPOCHS = 500
FINETUNE_EPOCHS = 600
EMBED_D = 256
SIMCLR_TAU = 0.1
SIMCLR_BATCH = 256


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


def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, tau: float = SIMCLR_TAU) -> torch.Tensor:
    """NT-Xent (SimCLR) contrastive loss. z_i, z_j: (N, D) L2-normalized embeddings."""
    N = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)  # (2N, D)
    sim = torch.mm(z, z.t()) / tau    # (2N, 2N)
    # Remove self-similarity diagonal
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -1e9)
    # Positives: (i, N+i) and (N+i, i)
    pos_idx = torch.arange(N, device=z.device)
    pos_i = sim[pos_idx, pos_idx + N]          # (N,)
    pos_j = sim[pos_idx + N, pos_idx]          # (N,)
    loss = -torch.cat([pos_i, pos_j]).mean() + torch.logsumexp(sim, dim=1).mean()
    return loss


def load_esm2_embeddings() -> dict[str, np.ndarray]:
    data = np.load(str(ESM2_CACHE))
    genes = data["genes"].tolist()
    embs = data["embeddings"]
    print(f"  Loaded {len(genes)} ESM2 embeddings (dim={embs.shape[1]})")
    return {g: embs[i] for i, g in enumerate(genes)}


def load_string_net(gene2vec_lookup, esm2_lookup, all_gene_names) -> tuple:
    print("Loading STRING ...")
    string_id_to_gene: dict[str, str] = {}
    with gzip.open(STRING_INFO, "rt") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2: string_id_to_gene[parts[0]] = parts[1]

    all_set = set(all_gene_names)
    neighbors: dict[str, list] = defaultdict(list)
    string_scores: dict = {}
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
                string_scores[(min(g1, g2), max(g1, g2))] = score / 1000.0

    zero_g2v = np.zeros(200, np.float32)
    zero_esm2 = np.zeros(320, np.float32)
    combo_dict = {}
    for g in all_gene_names:
        g2v = gene2vec_lookup.get(g, zero_g2v)
        esm2 = esm2_lookup.get(g, zero_esm2)
        nbrs = neighbors.get(g, [])
        if nbrs:
            total_w, ws_g2v, ws_esm2 = 0.0, np.zeros(200, np.float64), np.zeros(320, np.float64)
            for nbr, w in nbrs:
                vg = gene2vec_lookup.get(nbr)
                ve = esm2_lookup.get(nbr)
                if vg is not None: ws_g2v += w * vg; total_w += w
                if ve is not None: ws_esm2 += w * ve
            denom = total_w if total_w > 0 else 1.0
            net_g2v = (ws_g2v / denom).astype(np.float32)
            net_esm2 = (ws_esm2 / denom).astype(np.float32)
        else:
            net_g2v, net_esm2 = zero_g2v, zero_esm2
        combo_dict[g] = np.concatenate([esm2, net_esm2, g2v, net_g2v])  # 1040d

    print(f"  STRING done. combo=1040d, {len(string_scores)} pairs")
    return combo_dict, string_scores, neighbors


def load_bulk_signatures(gene2vec_genes: set) -> dict[str, np.ndarray]:
    print("Loading bulk h5ad ...")
    import anndata as ad
    adata = ad.read_h5ad(str(BULK_H5AD))

    def parse_gene(idx):
        parts = str(idx).split("_")
        return parts[1] if len(parts) >= 2 else str(idx)

    obs_gene = [parse_gene(idx) for idx in adata.obs_names]
    var_gene = adata.var["gene_name"].tolist() if "gene_name" in adata.var.columns else adata.var_names.tolist()
    shared_genes = sorted(set(var_gene) & gene2vec_genes)[:N_HVG]
    shared_idx = [var_gene.index(g) for g in shared_genes]
    ctrl_mask = np.array(["non-targeting" in str(idx).lower() or "nontarget" in parse_gene(idx).lower()
                           for idx in adata.obs_names])

    X = adata.X
    if hasattr(X, "toarray"): X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    lib = X.sum(axis=1, keepdims=True); lib = np.where(lib == 0, 1.0, lib)
    X_norm = np.log1p(X / lib * TARGET_SUM)
    ctrl_mean = X_norm[ctrl_mask][:, shared_idx].mean(axis=0) if ctrl_mask.sum() > 0 else np.zeros(len(shared_idx), np.float32)

    deltas: dict[str, list] = defaultdict(list)
    for i, g in enumerate(obs_gene):
        if ctrl_mask[i]: continue
        if g in gene2vec_genes: deltas[g].append(X_norm[i, shared_idx] - ctrl_mean)

    signatures = {g: np.mean(v, axis=0).astype(np.float32) for g, v in deltas.items()}
    del adata, X, X_norm; gc.collect()
    print(f"  {len(signatures)} bulk signatures")
    return signatures


class DeepEncoder(nn.Module):
    """Deeper encoder: 4-layer 1040→1024→1024→512→256."""
    def __init__(self, in_d: int = 1040, embed_d: int = EMBED_D, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, 512),  nn.LayerNorm(512),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, embed_d),
        )
    def forward(self, x): return self.net(x)


class DeepModel(nn.Module):
    def __init__(self, in_d: int, out_d: int, embed_d: int = EMBED_D, dropout: float = 0.3):
        super().__init__()
        self.encoder = DeepEncoder(in_d, embed_d)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_d, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, out_d),
        )
    def forward(self, x): return self.head(F.normalize(self.encoder(x), dim=-1))


def build_augmented_views(feat_dict, all_genes, neighbors, aug_strength=0.1):
    """Build augmented view for SimCLR: f_aug(g) = f(g) + aug_strength * mean(f(nbr))."""
    dim = list(feat_dict.values())[0].shape[0]
    aug_dict = {}
    for g in all_genes:
        orig = feat_dict[g]
        nbrs = [nbr for nbr, _ in neighbors.get(g, []) if nbr in feat_dict]
        if nbrs:
            nbr_mean = np.mean([feat_dict[n] for n in nbrs[:20]], axis=0)
            aug_dict[g] = orig + aug_strength * nbr_mean
        else:
            aug_dict[g] = orig + np.random.randn(dim).astype(np.float32) * aug_strength * 0.1
    return aug_dict


def pretrain_simclr(encoder, feat_dict, aug_dict, genes, pretrain_epochs=PRETRAIN_EPOCHS,
                    batch_size=SIMCLR_BATCH, tau=SIMCLR_TAU, seed=42):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=pretrain_epochs)
    best_loss, best_state = 1e9, None

    for epoch in range(pretrain_epochs):
        batch_genes = rng.choice(genes, size=min(batch_size, len(genes)), replace=False).tolist()
        f_orig = torch.tensor(np.array([feat_dict[g] for g in batch_genes]), dtype=torch.float32)
        f_aug  = torch.tensor(np.array([aug_dict[g]  for g in batch_genes]), dtype=torch.float32)
        encoder.train(); opt.zero_grad()
        z_i = F.normalize(encoder(f_orig), dim=-1)
        z_j = F.normalize(encoder(f_aug),  dim=-1)
        loss = nt_xent_loss(z_i, z_j, tau=tau)
        loss.backward(); opt.step(); sched.step()

        if loss.item() < best_loss: best_loss = loss.item(); best_state = {k: v.clone() for k, v in encoder.state_dict().items()}
        if epoch % 100 == 0: print(f"    simclr epoch {epoch}: loss={loss.item():.4f} (best={best_loss:.4f})")

    encoder.load_state_dict(best_state)
    return encoder


def pretrain_mse(encoder, feat_dict, bulk_sigs, string_scores, pretrain_epochs=PRETRAIN_EPOCHS, seed=42):
    """MSE contrastive (v22-style) for ablation comparison."""
    torch.manual_seed(seed)
    genes = [g for g in bulk_sigs if g in feat_dict]
    pair_genes = [(k[0], k[1], v) for k, v in string_scores.items()
                  if k[0] in feat_dict and k[1] in feat_dict
                  and k[0] in bulk_sigs and k[1] in bulk_sigs]
    if len(pair_genes) > 30000: pair_genes = pair_genes[:30000]

    all_sigs = {g: bulk_sigs[g] for g in genes if g in bulk_sigs}
    sig_genes = list(all_sigs.keys())
    sig_idx = {g: i for i, g in enumerate(sig_genes)}
    sig_arr = np.array(list(all_sigs.values()), dtype=np.float32)
    sig_norm = sig_arr / (np.linalg.norm(sig_arr, axis=1, keepdims=True) + 1e-8)

    opt = torch.optim.Adam(encoder.parameters(), lr=5e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    for epoch in range(pretrain_epochs):
        batch_idx = rng.choice(len(pair_genes), size=min(512, len(pair_genes)), replace=False)
        batch = [pair_genes[i] for i in batch_idx]
        g1_list, g2_list, str_sims = zip(*batch)
        target_sims = []
        for g1, g2, ss in batch:
            if g1 in sig_idx and g2 in sig_idx:
                bc = float(sig_norm[sig_idx[g1]] @ sig_norm[sig_idx[g2]])
                target_sims.append(0.5 * ss + 0.5 * max(bc, 0.0))
            else: target_sims.append(ss * 0.5)

        f1 = torch.tensor(np.array([feat_dict[g] for g in g1_list]), dtype=torch.float32)
        f2 = torch.tensor(np.array([feat_dict[g] for g in g2_list]), dtype=torch.float32)
        t  = torch.tensor(target_sims, dtype=torch.float32)
        encoder.train(); opt.zero_grad()
        e1 = F.normalize(encoder(f1), dim=-1); e2 = F.normalize(encoder(f2), dim=-1)
        loss = F.mse_loss((e1 * e2).sum(dim=-1), t)
        loss.backward(); opt.step()
        if epoch % 100 == 0: print(f"    mse pretrain epoch {epoch}: loss={loss.item():.4f}")
    return encoder


def finetune(model, feat_dict, train_genes, train_md, val_t, val_md, test_t, test_md,
             loss_mode="cosine", freeze_epochs=100, epochs=FINETUNE_EPOCHS, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    feat_d = list(feat_dict.values())[0].shape[0]
    train_gene_list = sorted(train_genes)
    D_train = np.stack([train_md[g] for g in train_gene_list])
    X_tr = np.array([feat_dict.get(g, np.zeros(feat_d, np.float32)) for g in train_gene_list])
    X_va = np.array([feat_dict.get(g, np.zeros(feat_d, np.float32)) for g in val_t])
    X_te = np.array([feat_dict.get(g, np.zeros(feat_d, np.float32)) for g in test_t])
    Xtt = torch.tensor(X_tr, dtype=torch.float32)
    ytt = torch.tensor(D_train, dtype=torch.float32)
    Xvt = torch.tensor(X_va, dtype=torch.float32)

    for p in model.encoder.parameters(): p.requires_grad = False
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_vr, best_state = -999, None

    for epoch in range(epochs):
        if epoch == freeze_epochs:
            for p in model.encoder.parameters(): p.requires_grad = True
            opt = torch.optim.Adam(model.parameters(), lr=lr * 0.1, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - freeze_epochs)

        model.train(); opt.zero_grad()
        pred = model(Xtt)
        if loss_mode == "cosine":
            loss = 1.0 - F.cosine_similarity(pred, ytt, dim=1).mean()
        elif loss_mode == "combined":
            cos_loss = 1.0 - F.cosine_similarity(pred, ytt, dim=1).mean()
            mse_loss = F.mse_loss(pred, ytt)
            # Normalize MSE to similar scale as cosine loss
            loss = 0.7 * cos_loss + 0.3 * (mse_loss / (mse_loss.detach() + 1e-6))
        loss.backward(); opt.step(); sched.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad(): va = model(Xvt).numpy()
            vr, _ = pearson_per_gene({g: va[i] for i, g in enumerate(val_t)}, val_md)
            if vr > best_vr: best_vr = vr; best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): raw = model(torch.tensor(X_te, dtype=torch.float32)).numpy()
    preds = {g: raw[i] for i, g in enumerate(test_t)}
    em, emed = pearson_per_gene(preds, test_md)
    return em, emed, best_vr


def run_ensemble(name, feat_dict, feat_d, train_genes, train_md, val_t, val_md, test_t, test_md,
                 pretrain_fn=None, loss_mode="cosine", n_seeds=5):
    delta_d = list(train_md.values())[0].shape[0]
    preds_all, seed_sc = [], []
    X_te = np.array([feat_dict.get(g, np.zeros(feat_d, np.float32)) for g in test_t])

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        m = DeepModel(feat_d, delta_d)
        if pretrain_fn is not None:
            enc = DeepEncoder(feat_d, EMBED_D)
            enc = pretrain_fn(enc, seed=seed)
            m.encoder.load_state_dict(enc.state_dict())
        em_s, emed_s, bvr = finetune(m, feat_dict, train_genes, train_md, val_t, val_md,
                                      test_t, test_md, loss_mode=loss_mode, seed=seed)
        m.eval()
        with torch.no_grad(): raw = m(torch.tensor(X_te, dtype=torch.float32)).numpy()
        preds_all.append(raw); seed_sc.append(em_s)
        print(f"    seed {seed}: val={bvr:.4f} test={em_s:.4f}")

    ens = {g: np.mean(preds_all, axis=0)[i] for i, g in enumerate(test_t)}
    em, emed = pearson_per_gene(ens, test_md)
    print(f"  → {em:.4f} (median {emed:.4f})")
    return em, emed, seed_sc


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading v10 mean-delta cache ...")
    with open(V10_GENES) as f:
        raw = json.load(f); valid_genes = raw if isinstance(raw, list) else raw["genes"]
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
    all_gene_names = list(all_mean_deltas.keys())

    gene2vec_table = load_gene2vec_table(Path("data/gene2vec/gene2vec_dim_200_iter_9.txt"))
    gene2vec_lookup = {g: gene2vec_table[g] for g in all_gene_names if g in gene2vec_table}
    esm2_lookup = load_esm2_embeddings()

    combo_dict, string_scores, neighbors = load_string_net(gene2vec_lookup, esm2_lookup, all_gene_names)
    feat_d = 1040
    bulk_sigs = load_bulk_signatures(set(gene2vec_lookup.keys()))

    # Build augmented views for SimCLR
    print("Building SimCLR augmented views ...")
    aug_dict = build_augmented_views(combo_dict, all_gene_names, neighbors)
    simclr_genes = [g for g in all_gene_names if g in combo_dict]

    train_t, test_t_full = train_test_split(all_gene_names, test_size=0.3, random_state=0)
    val_t, test_t = train_test_split(test_t_full, test_size=0.5, random_state=0)
    print(f"Split: {len(train_t)} train / {len(val_t)} val / {len(test_t)} test")
    train_md = {g: all_mean_deltas[g] for g in train_t}
    val_md   = {g: all_mean_deltas[g] for g in val_t}
    test_md  = {g: all_mean_deltas[g] for g in test_t}
    gm = np.stack(list(train_md.values())).mean(axis=0)
    gm_m, _ = pearson_per_gene({g: gm for g in test_t}, test_md)
    print(f"\nGlobal mean: {gm_m:.4f}")

    results = {}

    # A: combo + MSE pre-train + deeper encoder (v22 ablation with deeper arch)
    print("\n--- Experiment A: combo + MSE pre-train (deeper encoder, v22 ablation) ---")
    em_a, emed_a, sc_a = run_ensemble(
        "combo_mse_deep", combo_dict, feat_d, train_t, train_md, val_t, val_md, test_t, test_md,
        pretrain_fn=lambda enc, seed: pretrain_mse(enc, combo_dict, bulk_sigs, string_scores,
                                                    pretrain_epochs=PRETRAIN_EPOCHS, seed=seed),
    )
    results["combo_mse_pretrain_deep"] = {"test_pearson_mean": em_a, "test_pearson_median": emed_a, "per_seed": sc_a}

    # B: combo + SimCLR pre-train + cosine fine-tune
    print("\n--- Experiment B: combo + SimCLR pre-train + cosine fine-tune ---")
    em_b, emed_b, sc_b = run_ensemble(
        "combo_simclr_cosine", combo_dict, feat_d, train_t, train_md, val_t, val_md, test_t, test_md,
        pretrain_fn=lambda enc, seed: pretrain_simclr(enc, combo_dict, aug_dict, simclr_genes,
                                                       pretrain_epochs=PRETRAIN_EPOCHS, seed=seed),
        loss_mode="cosine",
    )
    results["combo_simclr_cosine"] = {"test_pearson_mean": em_b, "test_pearson_median": emed_b, "per_seed": sc_b}

    # C: combo + SimCLR pre-train + combined loss fine-tune
    print("\n--- Experiment C: combo + SimCLR + combined loss (cosine+MSE) fine-tune ---")
    em_c, emed_c, sc_c = run_ensemble(
        "combo_simclr_combined", combo_dict, feat_d, train_t, train_md, val_t, val_md, test_t, test_md,
        pretrain_fn=lambda enc, seed: pretrain_simclr(enc, combo_dict, aug_dict, simclr_genes,
                                                       pretrain_epochs=PRETRAIN_EPOCHS, seed=seed),
        loss_mode="combined",
    )
    results["combo_simclr_combined"] = {"test_pearson_mean": em_c, "test_pearson_median": emed_c, "per_seed": sc_c}

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    summary = {
        "note": "v23: SimCLR NT-Xent pre-training + deeper encoder",
        "n_train": len(train_t), "n_val": len(val_t), "n_test": len(test_t),
        "elapsed_seconds": elapsed,
        "reference": {"global_mean": gm_m, "v22_best": 0.3432, "oracle": 0.4739},
        "experiments": results,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_DIR}/metrics.json")


if __name__ == "__main__":
    main()

"""Milestone 35: Genome-wide pre-training (K562 gwps bulk, 9474 genes).

Hypothesis: the architecture ceiling at 0.3557 is a DATA bottleneck, not an
embedding or architecture bottleneck. The K562 essential screen has only 2,025
bulk gene signatures for pre-training. The genome-wide K562 screen (gwps) has
9,474 bulk signatures (4.7× more). More diverse pre-training pairs → better
encoder generalisation → higher downstream Pearson.

Strategy: keep fine-tuning/test identical to best v25 setup (essential K562
single-cell, 1439/309/309 split, seed=0). Only change the pre-training data
source from essential-bulk (2,025 genes) to gwps-bulk (9,474 genes).

Experiments:
  A) 400d (g2v+net_g2v only) + GWPS pre-train
     - Works for all 9,474 GWPS genes with gene2vec coverage
     - Feature dim: 400d
  B) 1040d combo (esm2+net_esm2+g2v+net_g2v) + GWPS pre-train (baseline: essential)
     - Same feature set as v25 best (essential genes only have ESM2)
     - Feature dim: 1040d, but uses GWPS bulk for pre-training signal

Comparison with prior best (v25): same test metric, same test genes.
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
OUT_DIR = Path("outputs/replogle_genelevel_v28")
V10_CACHE = Path("outputs/replogle_genelevel_v10/mean_delta_cache.npz")
V10_GENES = Path("outputs/replogle_genelevel_v10/mean_delta_cache_genes.json")
SC_H5AD   = Path("data/replogle/K562_essential_raw_singlecell_01.h5ad")
ESS_BULK  = Path("data/replogle_k562_essential/K562_essential_raw_bulk_01.h5ad")
GWPS_BULK = Path("data/replogle_k562_gwps/K562_gwps_raw_bulk_01.h5ad")
ESM2_CACHE = Path("data/esm2/esm2_embeddings.npz")
STRING_LINKS = Path("data/string/9606.protein.links.v12.0.txt.gz")
STRING_INFO  = Path("data/string/9606.protein.info.v12.0.txt.gz")
MIN_STRING_SCORE = 400
PRETRAIN_EPOCHS = 800
FINETUNE_EPOCHS = 600
EMBED_D = 256


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


def load_esm2_embeddings(cache_path: Path) -> dict[str, np.ndarray]:
    data = np.load(str(cache_path))
    genes = data["genes"].tolist()
    embs = data["embeddings"]
    print(f"  Loaded {len(genes)} ESM2 embeddings (dim={embs.shape[1]})")
    return {g: embs[i] for i, g in enumerate(genes)}


def load_string_net(gene2vec_lookup, esm2_lookup, all_gene_names, use_esm2=True):
    esm2_dim = 320 if use_esm2 else 0
    feat_d = esm2_dim * 2 + 200 * 2 if use_esm2 else 200 * 2
    print(f"Loading STRING (use_esm2={use_esm2}, feat_d={feat_d}) ...")
    string_id_to_gene = {}
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
    zero_g2v  = np.zeros(200, np.float32)
    zero_esm2 = np.zeros(320, np.float32) if use_esm2 else None
    combo_dict = {}
    for g in all_gene_names:
        raw_g2v  = gene2vec_lookup.get(g, zero_g2v)
        nbrs = neighbors.get(g, [])
        if nbrs:
            total_w = 0.0
            ws_g2v  = np.zeros(200, np.float64)
            ws_esm2 = np.zeros(320, np.float64) if use_esm2 else None
            for nbr, w in nbrs:
                vg = gene2vec_lookup.get(nbr)
                if vg is not None: ws_g2v += w * vg; total_w += w
                if use_esm2 and esm2_lookup:
                    ve = esm2_lookup.get(nbr)
                    if ve is not None: ws_esm2 += w * ve
            denom = total_w if total_w > 0 else 1.0
            net_g2v = (ws_g2v / denom).astype(np.float32)
            net_esm2 = (ws_esm2 / denom).astype(np.float32) if use_esm2 else None
        else:
            net_g2v  = zero_g2v
            net_esm2 = zero_esm2 if use_esm2 else None
        if use_esm2:
            raw_esm2 = esm2_lookup.get(g, zero_esm2) if esm2_lookup else zero_esm2
            combo_dict[g] = np.concatenate([raw_esm2, net_esm2, raw_g2v, net_g2v])
        else:
            combo_dict[g] = np.concatenate([raw_g2v, net_g2v])
    all_pairs = list(string_scores.items())
    print(f"  STRING done. feat_d={feat_d}, {len(all_pairs)} total pairs")
    return combo_dict, feat_d, all_pairs


def load_bulk_signatures(h5ad_path: Path, gene2vec_genes: set, label: str):
    """Load bulk perturbation signatures from an h5ad file."""
    print(f"Loading {label} from {h5ad_path.name} ...")
    import anndata as ad
    adata = ad.read_h5ad(str(h5ad_path))

    def parse_gene(idx):
        parts = str(idx).split("_")
        return parts[1] if len(parts) >= 2 else str(idx)

    # Get output gene names (may be ENSEMBL for GWPS, gene symbols for essential)
    if "gene_name" in adata.var.columns:
        var_gene = adata.var["gene_name"].tolist()
    else:
        var_gene = adata.var_names.tolist()

    obs_gene = [parse_gene(idx) for idx in adata.obs_names]
    ctrl_mask = np.array(["non-targeting" in str(idx).lower()
                           for idx in adata.obs_names])

    shared_genes = sorted(set(var_gene) & gene2vec_genes)[:N_HVG]
    shared_idx = [var_gene.index(g) for g in shared_genes]

    X = adata.X
    if hasattr(X, "toarray"): X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    lib = X.sum(axis=1, keepdims=True); lib = np.where(lib == 0, 1.0, lib)
    X_norm = np.log1p(X / lib * TARGET_SUM)
    ctrl_mean = X_norm[ctrl_mask][:, shared_idx].mean(axis=0) if ctrl_mask.sum() > 0 \
                else np.zeros(len(shared_idx), np.float32)
    deltas: dict[str, list] = defaultdict(list)
    for i, g in enumerate(obs_gene):
        if ctrl_mask[i]: continue
        if g in gene2vec_genes: deltas[g].append(X_norm[i, shared_idx] - ctrl_mean)
    signatures = {g: np.mean(v, axis=0).astype(np.float32) for g, v in deltas.items()}
    sig_genes = list(signatures.keys())
    sig_arr = np.array(list(signatures.values()), dtype=np.float32)
    sig_norm = sig_arr / (np.linalg.norm(sig_arr, axis=1, keepdims=True) + 1e-8)
    sig_idx = {g: i for i, g in enumerate(sig_genes)}
    del adata, X, X_norm; gc.collect()
    print(f"  {len(signatures)} {label} signatures")
    return sig_norm, sig_idx


class WideEncoder(nn.Module):
    def __init__(self, in_d: int, embed_d: int = EMBED_D, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, 2048), nn.LayerNorm(2048), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2048, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, 512),  nn.LayerNorm(512),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, embed_d),
        )
    def forward(self, x): return self.net(x)


class WideModel(nn.Module):
    def __init__(self, in_d: int, out_d: int, embed_d: int = EMBED_D, dropout: float = 0.3):
        super().__init__()
        self.encoder = WideEncoder(in_d, embed_d)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_d, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, out_d),
        )
    def forward(self, x): return self.head(F.normalize(self.encoder(x), dim=-1))


def pretrain(encoder, feat_dict, all_pairs, sig_norm, sig_idx,
             pretrain_epochs=PRETRAIN_EPOCHS, batch_size=512, seed=42):
    """Pre-train using STRING + bulk cosine similarity target."""
    torch.manual_seed(seed)
    valid_pairs = [(k, v) for k, v in all_pairs
                   if k[0] in feat_dict and k[1] in feat_dict
                   and k[0] in sig_idx and k[1] in sig_idx]
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(encoder.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=200, T_mult=1)
    best_loss, best_state = 1e9, None
    for epoch in range(pretrain_epochs):
        batch_idx = rng.choice(len(valid_pairs), size=min(batch_size, len(valid_pairs)), replace=False)
        batch = [valid_pairs[i] for i in batch_idx]
        target_sims = [0.5 * ss + 0.5 * max(float(sig_norm[sig_idx[g1]] @ sig_norm[sig_idx[g2]]), 0.0)
                       for (g1, g2), ss in batch]
        f1 = torch.tensor(np.array([feat_dict[b[0][0]] for b in batch]), dtype=torch.float32)
        f2 = torch.tensor(np.array([feat_dict[b[0][1]] for b in batch]), dtype=torch.float32)
        t  = torch.tensor(target_sims, dtype=torch.float32)
        encoder.train(); opt.zero_grad()
        e1 = F.normalize(encoder(f1), dim=-1); e2 = F.normalize(encoder(f2), dim=-1)
        loss = F.mse_loss((e1 * e2).sum(dim=-1), t)
        loss.backward(); opt.step(); sched.step()
        if loss.item() < best_loss: best_loss = loss.item(); best_state = {k: v.clone() for k, v in encoder.state_dict().items()}
        if epoch % 200 == 0: print(f"    pretrain epoch {epoch}: loss={loss.item():.4f} (best={best_loss:.4f})")
    encoder.load_state_dict(best_state)
    return encoder


def finetune(model, feat_dict, train_genes, train_md, val_t, val_md, test_t, test_md,
             freeze_epochs=100, epochs=FINETUNE_EPOCHS, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    feat_d = list(feat_dict.values())[0].shape[0]
    train_gene_list = sorted(train_genes)
    D_train = np.stack([train_md[g] for g in train_gene_list])
    X_tr = np.array([feat_dict.get(g, np.zeros(feat_d, np.float32)) for g in train_gene_list])
    X_va = np.array([feat_dict.get(g, np.zeros(feat_d, np.float32)) for g in val_t])
    X_te = np.array([feat_dict.get(g, np.zeros(feat_d, np.float32)) for g in test_t])
    Xtt = torch.tensor(X_tr, dtype=torch.float32); ytt = torch.tensor(D_train, dtype=torch.float32)
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
        loss = 1.0 - F.cosine_similarity(model(Xtt), ytt, dim=1).mean()
        loss.backward(); opt.step(); sched.step()
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad(): va = model(Xvt).numpy()
            vr, _ = pearson_per_gene({g: va[i] for i, g in enumerate(val_t)}, val_md)
            if vr > best_vr: best_vr = vr; best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): raw = model(torch.tensor(X_te, dtype=torch.float32)).numpy()
    return pearson_per_gene({g: raw[i] for i, g in enumerate(test_t)}, test_md) + (best_vr,)


def run_ensemble(feat_dict, feat_d, train_genes, train_md, val_t, val_md, test_t, test_md,
                 all_pairs, sig_norm, sig_idx, label, n_seeds=5):
    delta_d = list(train_md.values())[0].shape[0]
    preds_all, seed_sc = [], []
    X_te = np.array([feat_dict.get(g, np.zeros(feat_d, np.float32)) for g in test_t])
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        m = WideModel(feat_d, delta_d)
        enc = WideEncoder(feat_d, EMBED_D)
        enc = pretrain(enc, feat_dict, all_pairs, sig_norm, sig_idx, seed=seed)
        m.encoder.load_state_dict(enc.state_dict())
        em_s, emed_s, bvr = finetune(m, feat_dict, train_genes, train_md, val_t, val_md, test_t, test_md, seed=seed)
        m.eval()
        with torch.no_grad(): raw = m(torch.tensor(X_te, dtype=torch.float32)).numpy()
        preds_all.append(raw); seed_sc.append(em_s)
        print(f"    [{label}] seed {seed}: val={bvr:.4f} test={em_s:.4f}")
    ens = {g: np.mean(preds_all, axis=0)[i] for i, g in enumerate(test_t)}
    em, emed = pearson_per_gene(ens, test_md)
    print(f"  [{label}] → {em:.4f} (median {emed:.4f})")
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
    print(f"  control_mean ready. {time.time()-t0:.1f}s")

    all_mean_deltas = {
        g: (cache["valid_sums"][i] / cache["valid_counts"][i]).astype(np.float32) - control_mean
        for i, g in enumerate(valid_genes)
    }
    del cache; gc.collect()
    all_gene_names = list(all_mean_deltas.keys())  # essential genes (for train/val/test)

    gene2vec_table = load_gene2vec_table(Path("data/gene2vec/gene2vec_dim_200_iter_9.txt"))
    gene2vec_lookup = {g: gene2vec_table[g] for g in gene2vec_table}  # all g2v genes

    esm2_lookup = load_esm2_embeddings(ESM2_CACHE)

    # Build feature dicts for two variants (using essential test gene universe)
    combo_1040, feat_d_1040, all_pairs = load_string_net(
        gene2vec_lookup, esm2_lookup, all_gene_names, use_esm2=True)
    combo_400, feat_d_400, _ = load_string_net(
        gene2vec_lookup, None, all_gene_names, use_esm2=False)

    # Load pre-training targets from GWPS (the main hypothesis)
    gwps_sig_norm, gwps_sig_idx = load_bulk_signatures(
        GWPS_BULK, set(gene2vec_lookup.keys()), "GWPS bulk")
    # Load essential bulk too (for replication control)
    ess_sig_norm, ess_sig_idx = load_bulk_signatures(
        ESS_BULK, set(gene2vec_lookup.keys()), "Essential bulk")

    # Train/val/test split on essential genes (same as v22-v27)
    train_t, test_t_full = train_test_split(all_gene_names, test_size=0.3, random_state=0)
    val_t, test_t = train_test_split(test_t_full, test_size=0.5, random_state=0)
    print(f"Split: {len(train_t)} train / {len(val_t)} val / {len(test_t)} test")
    train_md = {g: all_mean_deltas[g] for g in train_t}
    val_md   = {g: all_mean_deltas[g] for g in val_t}
    test_md  = {g: all_mean_deltas[g] for g in test_t}
    gm = np.stack(list(train_md.values())).mean(axis=0)
    gm_m, _ = pearson_per_gene({g: gm for g in test_t}, test_md)
    print(f"\nGlobal mean: {gm_m:.4f}")

    n_gwps_pairs = sum(1 for (g1, g2), _ in all_pairs
                       if g1 in gwps_sig_idx and g2 in gwps_sig_idx)
    n_ess_pairs  = sum(1 for (g1, g2), _ in all_pairs
                       if g1 in ess_sig_idx  and g2 in ess_sig_idx)
    print(f"GWPS-covered STRING pairs: {n_gwps_pairs:,} vs essential: {n_ess_pairs:,}")

    results = {}

    # Exp C: 1040d combo + GWPS pre-train (main hypothesis: more pre-training data)
    print(f"\n--- Exp C: 1040d combo + GWPS pre-train ({len(gwps_sig_idx)} genes) ---")
    em_c, emed_c, sc_c = run_ensemble(combo_1040, feat_d_1040, train_t, train_md,
                                       val_t, val_md, test_t, test_md,
                                       all_pairs, gwps_sig_norm, gwps_sig_idx, "1040d-GWPS")
    results["combo_1040d_gwps_pretrain"] = {
        "test_pearson_mean": em_c, "test_pearson_median": emed_c, "per_seed": sc_c}

    # Exp B (replication): 1040d combo + essential pre-train
    print(f"\n--- Exp B: 1040d combo + essential pre-train ({len(ess_sig_idx)} genes, replication) ---")
    em_b, emed_b, sc_b = run_ensemble(combo_1040, feat_d_1040, train_t, train_md,
                                       val_t, val_md, test_t, test_md,
                                       all_pairs, ess_sig_norm, ess_sig_idx, "1040d-ess")
    results["combo_1040d_ess_pretrain"] = {
        "test_pearson_mean": em_b, "test_pearson_median": emed_b, "per_seed": sc_b}

    # Exp A: 400d (g2v only) + GWPS pre-train (checks if features or data dominate)
    print(f"\n--- Exp A: 400d g2v-only + GWPS pre-train ---")
    em_a, emed_a, sc_a = run_ensemble(combo_400, feat_d_400, train_t, train_md,
                                       val_t, val_md, test_t, test_md,
                                       all_pairs, gwps_sig_norm, gwps_sig_idx, "400d-GWPS")
    results["combo_400d_gwps_pretrain"] = {
        "test_pearson_mean": em_a, "test_pearson_median": emed_a, "per_seed": sc_a}

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    summary = {
        "note": "v28: GWPS bulk pre-training (9474 genes) vs essential bulk (2025 genes)",
        "n_train": len(train_t), "n_val": len(val_t), "n_test": len(test_t),
        "n_gwps_genes": len(gwps_sig_idx), "n_ess_genes": len(ess_sig_idx),
        "n_gwps_string_pairs": n_gwps_pairs, "n_ess_string_pairs": n_ess_pairs,
        "elapsed_seconds": elapsed,
        "reference": {"global_mean": gm_m, "v25_v27_best": 0.3557, "oracle": 0.4739},
        "experiments": results,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_DIR}/metrics.json")


if __name__ == "__main__":
    main()

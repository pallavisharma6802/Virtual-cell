"""Milestone 32: Expanded pre-training — all STRING pairs, not just bulk-covered.

v24 finding: wider encoder (0.3538) gives +0.001 over v23 (0.3526). Architecture
gains are diminishing. Pre-training is essential (wide no-pretrain = 0.3182).

Key gap in previous pre-training: only STRING pairs where BOTH genes have bulk
signatures are used. The STRING graph has ~331K pairs total; bulk_sigs covers ~2,025
genes, leaving most pairs unused.

This script expands pre-training to use ALL ~331K STRING pairs:
  - If both genes have bulk sigs: target = 0.5 * STRING_score + 0.5 * bulk_cosine
  - Otherwise: target = 0.5 * STRING_score (structural signal only)

This gives ~10-50× more pre-training pairs without any new data download.

Uses best architecture from v24: WideEncoder (1040→2048→1024→512→256).

Experiments:
  A) all-STRING pre-train (new) vs bulk-only pre-train (v24 replication)
     — isolates the effect of expanded pairs
  B) all-STRING pre-train, 800 epochs (main)
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
OUT_DIR = Path("outputs/replogle_genelevel_v29")
V10_CACHE = Path("outputs/replogle_genelevel_v10/mean_delta_cache.npz")
V10_GENES = Path("outputs/replogle_genelevel_v10/mean_delta_cache_genes.json")
SC_H5AD   = Path("data/replogle/K562_essential_raw_singlecell_01.h5ad")
BULK_H5AD = Path("data/replogle_k562_essential/K562_essential_raw_bulk_01.h5ad")
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
                string_scores[(min(g1, g2), max(g1, g2))] = score / 1000.0

    zero_g2v = np.zeros(200, np.float32)
    zero_esm2 = np.zeros(320, np.float32)
    combo_dict = {}
    neighbors: dict[str, list] = defaultdict(list)
    for (g1, g2), sc in string_scores.items():
        neighbors[g1].append((g2, sc))
        neighbors[g2].append((g1, sc))

    for g in all_gene_names:
        g2v = gene2vec_lookup.get(g, zero_g2v)
        esm2 = esm2_lookup.get(g, zero_esm2)
        nbrs = neighbors.get(g, [])
        if nbrs:
            tw_g2v, tw_esm2 = 0.0, 0.0
            ws_g2v, ws_esm2 = np.zeros(200, np.float64), np.zeros(320, np.float64)
            for nbr, w in nbrs:
                vg = gene2vec_lookup.get(nbr)
                ve = esm2_lookup.get(nbr)
                if vg is not None: ws_g2v += w * vg; tw_g2v += w
                if ve is not None: ws_esm2 += w * ve; tw_esm2 += w
            net_g2v = (ws_g2v / (tw_g2v if tw_g2v > 0 else 1.0)).astype(np.float32)
            net_esm2 = (ws_esm2 / (tw_esm2 if tw_esm2 > 0 else 1.0)).astype(np.float32)
        else:
            net_g2v, net_esm2 = zero_g2v, zero_esm2
        combo_dict[g] = np.concatenate([esm2, net_esm2, g2v, net_g2v])  # 1040d

    all_pairs = list(string_scores.items())  # [(g1,g2), score]
    print(f"  STRING: {len(all_pairs)} total pairs, combo=1040d")
    return combo_dict, all_pairs


def load_bulk_signatures(gene2vec_genes: set, train_genes: set | None = None) -> dict[str, np.ndarray]:
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

    # Exclude val/test genes from pre-training supervision to prevent leakage
    eligible = gene2vec_genes if train_genes is None else (gene2vec_genes & train_genes)
    deltas: dict[str, list] = defaultdict(list)
    for i, g in enumerate(obs_gene):
        if ctrl_mask[i]: continue
        if g in eligible: deltas[g].append(X_norm[i, shared_idx] - ctrl_mean)

    signatures = {g: np.mean(v, axis=0).astype(np.float32) for g, v in deltas.items()}
    # Normalize for cosine computation
    sig_arr = np.array(list(signatures.values()), dtype=np.float32)
    sig_genes = list(signatures.keys())
    sig_norm = sig_arr / (np.linalg.norm(sig_arr, axis=1, keepdims=True) + 1e-8)
    sig_idx = {g: i for i, g in enumerate(sig_genes)}

    del adata, X, X_norm; gc.collect()
    print(f"  {len(signatures)} bulk signatures")
    return sig_norm, sig_idx


class WideEncoder(nn.Module):
    def __init__(self, in_d: int = 1040, embed_d: int = EMBED_D, dropout: float = 0.2):
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


def pretrain_all_string(encoder, feat_dict, all_pairs, sig_norm, sig_idx,
                         bulk_only=False, pretrain_epochs=PRETRAIN_EPOCHS,
                         batch_size=512, seed=42):
    """Pre-train on STRING pairs.
    bulk_only=True: only pairs where BOTH genes have bulk sigs (v24-style, blended target).
    bulk_only=False: ALL pairs where both in feat_dict; bulk-covered get blended, rest get STRING-only target.
    """
    torch.manual_seed(seed)
    if bulk_only:
        valid_pairs = [(k, v) for k, v in all_pairs
                       if k[0] in feat_dict and k[1] in feat_dict
                       and k[0] in sig_idx and k[1] in sig_idx]
    else:
        valid_pairs = [(k, v) for k, v in all_pairs if k[0] in feat_dict and k[1] in feat_dict]
    rng = np.random.default_rng(seed)

    opt = torch.optim.Adam(encoder.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=200, T_mult=1)
    best_loss, best_state = 1e9, None

    for epoch in range(pretrain_epochs):
        batch_idx = rng.choice(len(valid_pairs), size=min(batch_size, len(valid_pairs)), replace=False)
        batch = [valid_pairs[i] for i in batch_idx]

        target_sims = []
        for (g1, g2), ss in batch:
            if g1 in sig_idx and g2 in sig_idx:
                bc = float(sig_norm[sig_idx[g1]] @ sig_norm[sig_idx[g2]])
                target_sims.append(0.5 * ss + 0.5 * max(bc, 0.0))
            else:
                # Structural signal only: STRING score scaled to [0, 0.5]
                target_sims.append(ss * 0.5)

        f1 = torch.tensor(np.array([feat_dict[b[0][0]] for b in batch]), dtype=torch.float32)
        f2 = torch.tensor(np.array([feat_dict[b[0][1]] for b in batch]), dtype=torch.float32)
        t  = torch.tensor(target_sims, dtype=torch.float32)

        encoder.train(); opt.zero_grad()
        e1 = F.normalize(encoder(f1), dim=-1)
        e2 = F.normalize(encoder(f2), dim=-1)
        loss = F.mse_loss((e1 * e2).sum(dim=-1), t)
        loss.backward(); opt.step(); sched.step()
        if loss.item() < best_loss: best_loss = loss.item(); best_state = {k: v.clone() for k, v in encoder.state_dict().items()}
        if epoch % 200 == 0: print(f"    pretrain epoch {epoch}: loss={loss.item():.4f} (best={best_loss:.4f}), pairs={len(valid_pairs)}")

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
        loss = 1.0 - F.cosine_similarity(pred, ytt, dim=1).mean()
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
                 pretrain_fn=None, n_seeds=5, save_dir: Path | None = None):
    delta_d = list(train_md.values())[0].shape[0]
    preds_all, seed_sc = [], []
    X_te = np.array([feat_dict.get(g, np.zeros(feat_d, np.float32)) for g in test_t])

    ckpt_dir = None
    if save_dir is not None:
        ckpt_dir = save_dir / name / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        m = WideModel(feat_d, delta_d)
        if pretrain_fn is not None:
            enc = WideEncoder(feat_d, EMBED_D)
            enc = pretrain_fn(enc, seed=seed)
            m.encoder.load_state_dict(enc.state_dict())
        em_s, emed_s, bvr = finetune(m, feat_dict, train_genes, train_md, val_t, val_md,
                                      test_t, test_md, seed=seed)
        m.eval()
        with torch.no_grad(): raw = m(torch.tensor(X_te, dtype=torch.float32)).numpy()
        preds_all.append(raw); seed_sc.append(em_s)
        print(f"    seed {seed}: val={bvr:.4f} test={em_s:.4f}")
        if ckpt_dir is not None:
            torch.save(m.state_dict(), ckpt_dir / f"seed{seed}.pt")

    ens_preds = np.mean(preds_all, axis=0)
    ens = {g: ens_preds[i] for i, g in enumerate(test_t)}
    em, emed = pearson_per_gene(ens, test_md)
    print(f"  → {em:.4f} (median {emed:.4f})")

    if save_dir is not None:
        pred_dir = save_dir / name
        pred_dir.mkdir(parents=True, exist_ok=True)
        # Per-target ensemble predictions as npz
        np.savez(
            pred_dir / "predictions.npz",
            genes=np.array(test_t),
            preds=ens_preds,
        )
        # Split IDs for auditing
        import json as _json
        (pred_dir / "split_ids.json").write_text(_json.dumps(
            {"train": sorted(train_genes), "val": sorted(val_t), "test": sorted(test_t)}, indent=2
        ))

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

    combo_dict, all_pairs = load_string_net(gene2vec_lookup, esm2_lookup, all_gene_names)
    feat_d = 1040

    # Split BEFORE loading bulk signatures so test/val genes are excluded from pre-training
    train_t, test_t_full = train_test_split(all_gene_names, test_size=0.3, random_state=0)
    val_t, test_t = train_test_split(test_t_full, test_size=0.5, random_state=0)
    print(f"Split: {len(train_t)} train / {len(val_t)} val / {len(test_t)} test")

    sig_norm, sig_idx = load_bulk_signatures(set(gene2vec_lookup.keys()), train_genes=set(train_t))
    n_bulk_pairs = sum(1 for (g1, g2), _ in all_pairs if g1 in sig_idx and g2 in sig_idx)
    print(f"  All STRING pairs: {len(all_pairs)}, bulk-covered pairs (train only): {n_bulk_pairs}")
    train_md = {g: all_mean_deltas[g] for g in train_t}
    val_md   = {g: all_mean_deltas[g] for g in val_t}
    test_md  = {g: all_mean_deltas[g] for g in test_t}
    gm = np.stack(list(train_md.values())).mean(axis=0)
    gm_m, _ = pearson_per_gene({g: gm for g in test_t}, test_md)
    print(f"\nGlobal mean: {gm_m:.4f}")

    results = {}

    # A: bulk-only pairs pre-train (v24 replication for fair comparison)
    print("\n--- Experiment A: bulk-only pairs pre-train (v24 replication) ---")
    em_a, emed_a, sc_a = run_ensemble(
        "bulk_pairs_pretrain", combo_dict, feat_d, train_t, train_md, val_t, val_md, test_t, test_md,
        save_dir=OUT_DIR,
        pretrain_fn=lambda enc, seed: pretrain_all_string(
            enc, combo_dict, all_pairs, sig_norm, sig_idx,
            bulk_only=True, pretrain_epochs=PRETRAIN_EPOCHS, seed=seed),
    )
    results["bulk_pairs_pretrain"] = {"test_pearson_mean": em_a, "test_pearson_median": emed_a, "per_seed": sc_a}

    # B: all STRING pairs (expanded), blended target where available
    print("\n--- Experiment B: all STRING pairs (expanded pre-training) ---")
    em_b, emed_b, sc_b = run_ensemble(
        "all_string_pretrain", combo_dict, feat_d, train_t, train_md, val_t, val_md, test_t, test_md,
        save_dir=OUT_DIR,
        pretrain_fn=lambda enc, seed: pretrain_all_string(
            enc, combo_dict, all_pairs, sig_norm, sig_idx,
            bulk_only=False, pretrain_epochs=PRETRAIN_EPOCHS, seed=seed),
    )
    results["all_string_pretrain"] = {"test_pearson_mean": em_b, "test_pearson_median": emed_b, "per_seed": sc_b}

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    summary = {
        "note": "v29: Leak-free rerun of v25 — bulk signatures filtered to train genes only before pre-training",
        "n_train": len(train_t), "n_val": len(val_t), "n_test": len(test_t),
        "n_all_string_pairs": len(all_pairs),
        "n_bulk_covered_pairs": n_bulk_pairs,
        "elapsed_seconds": elapsed,
        "reference": {"global_mean": gm_m, "v25_with_leakage": 0.3557, "v24_best": 0.3538, "oracle": 0.4739},
        "experiments": results,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_DIR}/metrics.json")


if __name__ == "__main__":
    main()

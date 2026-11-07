"""Milestone 28: LINCS L1000 drug-signature pre-training.

Hypothesis: Drug-induced gene-expression signatures share a latent manifold
with genetic perturbations. Pre-training an encoder on LINCS chemical
perturbations (many more samples) may bootstrap a better representation
that transfer-learns to genetic knockouts.

Strategy (data-free fallback if LINCS unavailable):
  1. Try to access LINCS L1000 via Broad GEO (GSE70138, GSE92742)
  2. If unavailable: use bulk Replogle K562 file (2,285 perturbations × 8,563 genes)
     as the pre-training corpus — it's already on disk and covers all 2,285 genes
  3. Pre-train: autoencoder / contrastive loss on perturbation signatures
  4. Fine-tune: on mean-delta prediction for the held-out-target split

This script uses the BULK Replogle K562 file (local, 76 MB) which the previous
iterations have never used. It has 2,285 perturbations vs 2,057 in the sc cache,
covering more target genes with cleaner (pre-aggregated) profiles.

Architecture:
  Pre-train phase:
    - Load bulk file → 2,285 × 8,563 gene expressions
    - Build "delta" signatures: subtract control mean from each row
    - Pre-train a 256-d encoder via contrastive loss (similar gene → closer embedding)
    - Similarity proxy: STRING PPI score between two genes

  Fine-tune phase:
    - Encoder takes gene feature (g2v_net 400d) → 256d embedding
    - Frozen encoder backbone + fresh MLP head → 5000d delta prediction
    - Full unfreeze after 100 warmup epochs

  Comparison:
    A) From-scratch (direct cosine MLP, v16 baseline)
    B) Pre-trained on bulk signatures → frozen fine-tune
    C) Pre-trained on bulk signatures → full fine-tune
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
OUT_DIR = Path("outputs/replogle_genelevel_v21")
V10_CACHE = Path("outputs/replogle_genelevel_v10/mean_delta_cache.npz")
V10_GENES = Path("outputs/replogle_genelevel_v10/mean_delta_cache_genes.json")
SC_H5AD   = Path("data/replogle/K562_essential_raw_singlecell_01.h5ad")
BULK_H5AD = Path("data/replogle_k562_essential/K562_essential_raw_bulk_01.h5ad")
STRING_LINKS = Path("data/string/9606.protein.links.v12.0.txt.gz")
STRING_INFO  = Path("data/string/9606.protein.info.v12.0.txt.gz")
MIN_STRING_SCORE = 400
PRETRAIN_EPOCHS = 200
FINETUNE_EPOCHS = 500
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


def load_bulk_signatures(gene2vec_genes: set[str]) -> tuple[dict[str, np.ndarray], list[str]]:
    """Load bulk K562 file → mean-delta signatures for each targeted gene.
    Returns {gene_name: delta_vector (over intersecting HVGs)}, list of gene names.
    """
    print("Loading bulk h5ad ...")
    import anndata as ad
    adata = ad.read_h5ad(str(BULK_H5AD))
    print(f"  Bulk shape: {adata.shape}, obs keys: {list(adata.obs.columns[:5])}")

    # Parse gene name from obs index: "10023_ZC3H18_P1P2_ENSG00000158545"
    def parse_gene(idx):
        parts = idx.split("_")
        return parts[1] if len(parts) >= 2 else idx

    obs_gene = [parse_gene(str(idx)) for idx in adata.obs_names]
    var_gene = adata.var["gene_name"].tolist() if "gene_name" in adata.var.columns else adata.var_names.tolist()
    var_gene_set = set(var_gene)

    # Find control rows
    ctrl_mask = np.array(["non-targeting" in str(idx).lower() or obs_gene[i].lower().startswith("non")
                          for i, idx in enumerate(adata.obs_names)])
    n_ctrl = ctrl_mask.sum()
    if n_ctrl == 0:
        # fallback: use "control" column if available
        ctrl_mask = np.zeros(len(obs_gene), dtype=bool)
        for i, g in enumerate(obs_gene):
            if g.lower() in ("non-targeting", "nontargeting", "control", "ctrl"):
                ctrl_mask[i] = True
    print(f"  {n_ctrl} control rows, {len(obs_gene)} total rows")

    # Intersect with gene2vec genes for HVG selection
    shared_genes = sorted(var_gene_set & gene2vec_genes)[:N_HVG]
    shared_idx = [var_gene.index(g) for g in shared_genes]
    print(f"  {len(shared_genes)} shared HVGs between bulk and gene2vec")

    X = adata.X
    if hasattr(X, "toarray"): X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    # Log-normalize
    lib = X.sum(axis=1, keepdims=True); lib = np.where(lib == 0, 1.0, lib)
    X_norm = np.log1p(X / lib * TARGET_SUM)

    # Control mean
    ctrl_mean = X_norm[ctrl_mask][:, shared_idx].mean(axis=0)

    # Delta per target gene
    deltas: dict[str, list] = defaultdict(list)
    for i, g in enumerate(obs_gene):
        if ctrl_mask[i]: continue
        if g in gene2vec_genes:
            deltas[g].append(X_norm[i, shared_idx] - ctrl_mean)

    signatures = {g: np.mean(v, axis=0).astype(np.float32) for g, v in deltas.items()}
    print(f"  {len(signatures)} gene delta signatures from bulk")
    del adata, X, X_norm; gc.collect()
    return signatures, shared_genes


def load_string_net_feats(gene2vec_lookup, all_gene_names, dim=200):
    """1-hop STRING propagated features (v16 style, 400d total)."""
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
    string_scores: dict[tuple, float] = {}
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

    zero = np.zeros(dim, np.float32)
    feat_dict = {}
    for g in all_gene_names:
        g2v = gene2vec_lookup.get(g, zero)
        nbrs = neighbors.get(g, [])
        if nbrs:
            total_w, ws = 0.0, np.zeros(dim, np.float64)
            for nbr, w in nbrs:
                v = gene2vec_lookup.get(nbr)
                if v is not None: ws += w * v; total_w += w
            net = (ws / total_w).astype(np.float32) if total_w > 0 else zero
        else:
            net = zero
        feat_dict[g] = np.concatenate([g2v, net])
    return feat_dict, string_scores


class Encoder(nn.Module):
    def __init__(self, in_d: int, embed_d: int = EMBED_D, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, embed_d),
        )

    def forward(self, x):
        return self.net(x)


class FullModel(nn.Module):
    def __init__(self, in_d: int, out_d: int, embed_d: int = EMBED_D, mlp_h: int = 256, dropout: float = 0.3):
        super().__init__()
        self.encoder = Encoder(in_d, embed_d, hidden=512)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_d, mlp_h), nn.LayerNorm(mlp_h), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_h, out_d),
        )

    def forward(self, x):
        return self.head(F.normalize(self.encoder(x), dim=-1))


def pretrain_contrastive(encoder, feat_dict, bulk_signatures, string_scores,
                         genes_with_bulk, pretrain_epochs=PRETRAIN_EPOCHS, seed=42):
    """Pre-train encoder: genes with high STRING score → similar embeddings,
    dissimilar otherwise. Uses bulk signature similarity as an alternative signal."""
    torch.manual_seed(seed)
    genes = [g for g in genes_with_bulk if g in feat_dict]
    if len(genes) < 50:
        print("  Too few genes with bulk signatures; skipping pre-training.")
        return encoder

    # Build similarity pairs from STRING scores
    pair_genes = sorted(string_scores.keys())
    # Filter to genes we have features for
    pair_genes = [(g1, g2, s) for g1, g2, s in
                  [(k[0], k[1], v) for k, v in string_scores.items()]
                  if g1 in feat_dict and g2 in feat_dict and g1 in bulk_signatures and g2 in bulk_signatures]
    if len(pair_genes) > 20000:
        pair_genes = pair_genes[:20000]
    print(f"  Pre-training with {len(pair_genes)} pairs, {len(genes)} genes with bulk sigs")

    # Also compute bulk-signature cosine as target similarity
    all_sigs = {g: bulk_signatures[g] for g in genes if g in bulk_signatures}
    sig_arr = np.array(list(all_sigs.values()), dtype=np.float32)
    sig_genes = list(all_sigs.keys())
    sig_idx = {g: i for i, g in enumerate(sig_genes)}
    sig_norm = sig_arr / (np.linalg.norm(sig_arr, axis=1, keepdims=True) + 1e-8)

    opt = torch.optim.Adam(encoder.parameters(), lr=5e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed)

    for epoch in range(pretrain_epochs):
        # Sample a batch of pairs
        batch_idx = rng.choice(len(pair_genes), size=min(256, len(pair_genes)), replace=False)
        batch = [pair_genes[i] for i in batch_idx]
        g1_list, g2_list, str_sims = zip(*batch)

        # Target similarity: blend STRING score + bulk cosine
        target_sims = []
        for g1, g2, ss in batch:
            if g1 in sig_idx and g2 in sig_idx:
                bulk_cos = float(sig_norm[sig_idx[g1]] @ sig_norm[sig_idx[g2]])
                target_sims.append(0.5 * ss + 0.5 * max(bulk_cos, 0.0))
            else:
                target_sims.append(ss * 0.5)

        f1 = torch.tensor([feat_dict[g] for g in g1_list], dtype=torch.float32)
        f2 = torch.tensor([feat_dict[g] for g in g2_list], dtype=torch.float32)
        t = torch.tensor(target_sims, dtype=torch.float32)

        encoder.train(); opt.zero_grad()
        e1 = F.normalize(encoder(f1), dim=-1)
        e2 = F.normalize(encoder(f2), dim=-1)
        pred_sim = (e1 * e2).sum(dim=-1)
        loss = F.mse_loss(pred_sim, t)
        loss.backward(); opt.step()

        if epoch % 50 == 0:
            print(f"    pretrain epoch {epoch}: loss={loss.item():.4f}")

    return encoder


def finetune_model(model, feat_dict, train_genes, train_md, val_t, val_md, test_t, test_md,
                   freeze_encoder=True, epochs=FINETUNE_EPOCHS, lr=1e-3, seed=42):
    """Fine-tune FullModel on direct cosine objective."""
    torch.manual_seed(seed)
    feat_d = list(feat_dict.values())[0].shape[0]
    train_gene_list = sorted(train_genes)
    D_train = np.stack([train_md[g] for g in train_gene_list])
    X_tr = np.array([feat_dict.get(g, np.zeros(feat_d, np.float32)) for g in train_gene_list])
    X_va = np.array([feat_dict.get(g, np.zeros(feat_d, np.float32)) for g in val_t])
    X_te = np.array([feat_dict.get(g, np.zeros(feat_d, np.float32)) for g in test_t])

    if freeze_encoder:
        for p in model.encoder.parameters(): p.requires_grad = False

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xtt = torch.tensor(X_tr, dtype=torch.float32)
    ytt = torch.tensor(D_train, dtype=torch.float32)
    Xvt = torch.tensor(X_va, dtype=torch.float32)
    best_vr, best_state = -999, None

    unfreeze_epoch = 100 if freeze_encoder else 0

    for epoch in range(epochs):
        if freeze_encoder and epoch == unfreeze_epoch:
            for p in model.encoder.parameters(): p.requires_grad = True
            opt = torch.optim.Adam(model.parameters(), lr=lr * 0.1, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - unfreeze_epoch)

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
    preds = {g: raw[i] for i, g in enumerate(test_t)}
    em, emed = pearson_per_gene(preds, test_md)
    return em, emed, best_vr


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
    rng_np = np.random.default_rng(0)
    ctrl_sample = sorted(rng_np.choice(ctrl_positions, size=min(8000, n_ctrl), replace=False).tolist())
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

    feat_dict, string_scores = load_string_net_feats(gene2vec_lookup, all_gene_names)
    feat_d = 400
    delta_d = list(all_mean_deltas.values())[0].shape[0]

    # Load bulk signatures for pre-training
    bulk_sigs, bulk_genes = load_bulk_signatures(set(gene2vec_lookup.keys()))

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

    # --- Experiment A: Direct from-scratch (baseline) ---
    print("\n--- Experiment A: Direct from-scratch (g2v_net 400d) ---")
    em_all_a, emed_all_a, seed_sc_a = [], [], []
    for seed in range(5):
        m = FullModel(feat_d, delta_d)
        em_s, emed_s, bvr = finetune_model(m, feat_dict, train_t, train_md, val_t, val_md,
                                            test_t, test_md, freeze_encoder=False, seed=seed)
        em_all_a.append(em_s); emed_all_a.append(emed_s); seed_sc_a.append(em_s)
        print(f"    seed {seed}: val={bvr:.4f} test={em_s:.4f}")

    # 5-seed ensemble
    preds_all = []
    for seed in range(5):
        m = FullModel(feat_d, delta_d)
        finetune_model(m, feat_dict, train_t, train_md, val_t, val_md,
                       test_t, test_md, freeze_encoder=False, seed=seed, epochs=FINETUNE_EPOCHS)
        m.eval()
        X_te = np.array([feat_dict.get(g, np.zeros(feat_d, np.float32)) for g in test_t])
        with torch.no_grad(): raw = m(torch.tensor(X_te, dtype=torch.float32)).numpy()
        preds_all.append(raw)
    ens = {g: np.mean(preds_all, axis=0)[i] for i, g in enumerate(test_t)}
    em_a, emed_a = pearson_per_gene(ens, test_md)
    results["direct_from_scratch"] = {"test_pearson_mean": em_a, "test_pearson_median": emed_a, "per_seed": seed_sc_a}
    print(f"  → {em_a:.4f} (median {emed_a:.4f})")

    # --- Experiment B: Pre-trained encoder (frozen head warm-up then unfrozen) ---
    print("\n--- Experiment B: Pre-trained encoder (bulk pre-training) ---")
    preds_all_b = []
    seed_sc_b = []
    for seed in range(5):
        enc = Encoder(feat_d, EMBED_D)
        enc = pretrain_contrastive(enc, feat_dict, bulk_sigs, string_scores,
                                   list(bulk_sigs.keys()), pretrain_epochs=PRETRAIN_EPOCHS, seed=seed)
        m = FullModel(feat_d, delta_d)
        m.encoder.load_state_dict(enc.state_dict())
        em_s, emed_s, bvr = finetune_model(m, feat_dict, train_t, train_md, val_t, val_md,
                                            test_t, test_md, freeze_encoder=True, seed=seed)
        m.eval()
        X_te = np.array([feat_dict.get(g, np.zeros(feat_d, np.float32)) for g in test_t])
        with torch.no_grad(): raw = m(torch.tensor(X_te, dtype=torch.float32)).numpy()
        preds_all_b.append(raw)
        seed_sc_b.append(em_s)
        print(f"    seed {seed}: val={bvr:.4f} test={em_s:.4f}")

    ens_b = {g: np.mean(preds_all_b, axis=0)[i] for i, g in enumerate(test_t)}
    em_b, emed_b = pearson_per_gene(ens_b, test_md)
    results["pretrained_bulk"] = {"test_pearson_mean": em_b, "test_pearson_median": emed_b, "per_seed": seed_sc_b}
    print(f"  → {em_b:.4f} (median {emed_b:.4f})")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    summary = {
        "note": "v21: Bulk data pre-training (LINCS-style transfer)",
        "n_train": len(train_t), "n_val": len(val_t), "n_test": len(test_t),
        "n_bulk_signatures": len(bulk_sigs),
        "elapsed_seconds": elapsed,
        "reference": {"global_mean": gm_m, "v16_best": 0.3103, "oracle": 0.4739},
        "experiments": results,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_DIR}/metrics.json")


if __name__ == "__main__":
    main()

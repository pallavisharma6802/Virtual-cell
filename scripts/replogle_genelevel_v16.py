"""Milestone 23: STRING network-propagated Gene2Vec embeddings.

Hypothesis: genes that interact in the protein-protein interaction (PPI) network
tend to have correlated perturbation effects. STRING encodes the K562 biology
indirectly through known interactions.

Feature engineering:
  For each target gene g, compute a "network neighborhood" embedding:
    net_emb(g) = weighted mean of Gene2Vec(neighbor) for all STRING neighbors
                 weighted by combined_score / 1000

  This propagates Gene2Vec information through the PPI graph.
  Concatenate [Gene2Vec(g), net_emb(g), response_role(g)] as features.

Downloads:
  - https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz
  - https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz
"""
from __future__ import annotations

import gc
import gzip
import json
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import anndata as ad
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
CHUNK_SIZE = 4000
OUT_DIR = Path("outputs/replogle_genelevel_v16")
V10_CACHE = Path("outputs/replogle_genelevel_v10/mean_delta_cache.npz")
V10_GENES = Path("outputs/replogle_genelevel_v10/mean_delta_cache_genes.json")
SC_H5AD   = Path("data/replogle/K562_essential_raw_singlecell_01.h5ad")
STRING_DIR = Path("data/string")
STRING_LINKS = STRING_DIR / "9606.protein.links.v12.0.txt.gz"
STRING_INFO  = STRING_DIR / "9606.protein.info.v12.0.txt.gz"
MIN_STRING_SCORE = 400  # combined score threshold (0-1000)


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
    return 1.0 - F.cosine_similarity(pred, target, dim=1).mean()


def train_mlp_direct(X_tr, y_tr, X_va, val_t, val_md,
                     hidden=512, n_layers=2, epochs=600, lr=1e-3, wd=1e-4, seed=42):
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
        loss = cosine_loss(model(X_tr_t), y_tr_t)
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


def run_ensemble(n_seeds, X_tr, y_tr, X_va, val_t, val_md, X_te, test_t, test_md, **kw):
    all_preds, seed_sc = [], []
    for seed in range(n_seeds):
        m, bvr = train_mlp_direct(X_tr, y_tr, X_va, val_t, val_md, seed=seed, **kw)
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


def download_string():
    STRING_DIR.mkdir(parents=True, exist_ok=True)
    for url, dest in [
        ("https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz", STRING_LINKS),
        ("https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz",  STRING_INFO),
    ]:
        if dest.exists():
            print(f"  {dest.name} already exists, skipping download.")
            continue
        print(f"  Downloading {dest.name} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
        print(f"  Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def load_string_network(gene2vec_lookup: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Load STRING PPI network, return network-propagated Gene2Vec for each gene."""
    # Load gene symbol mapping: STRING protein ID → preferred gene name
    print("Loading STRING protein info ...")
    string_id_to_gene: dict[str, str] = {}
    with gzip.open(STRING_INFO, "rt") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            string_id = parts[0]            # e.g. "9606.ENSP00000000233"
            preferred_name = parts[1]       # gene symbol
            string_id_to_gene[string_id] = preferred_name

    # Genes we care about
    all_genes_of_interest = set(gene2vec_lookup.keys())

    # Load interaction links
    print(f"Loading STRING links (score >= {MIN_STRING_SCORE}) ...")
    neighbors: dict[str, list[tuple[str, float]]] = defaultdict(list)
    n_loaded = 0
    with gzip.open(STRING_LINKS, "rt") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            p1, p2, score_str = parts[0], parts[1], parts[2]
            score = int(score_str)
            if score < MIN_STRING_SCORE:
                continue
            g1 = string_id_to_gene.get(p1)
            g2 = string_id_to_gene.get(p2)
            if g1 is None or g2 is None:
                continue
            if g1 in all_genes_of_interest or g2 in all_genes_of_interest:
                neighbors[g1].append((g2, score))
                neighbors[g2].append((g1, score))
            n_loaded += 1
    print(f"  Loaded {n_loaded} edges (score>={MIN_STRING_SCORE}). "
          f"Coverage: {sum(1 for g in all_genes_of_interest if g in neighbors)} / {len(all_genes_of_interest)} genes")

    # Compute network-propagated embeddings: weighted mean of neighbor Gene2Vec
    zero_emb = np.zeros(200, dtype=np.float32)
    net_emb: dict[str, np.ndarray] = {}
    for gene in all_genes_of_interest:
        neigh = neighbors.get(gene, [])
        if not neigh:
            net_emb[gene] = zero_emb
            continue
        total_w = 0.0
        weighted_sum = np.zeros(200, dtype=np.float64)
        for nbr, sc in neigh:
            g2v = gene2vec_lookup.get(nbr)
            if g2v is not None:
                w = sc / 1000.0
                weighted_sum += w * g2v
                total_w += w
        if total_w > 0:
            net_emb[gene] = (weighted_sum / total_w).astype(np.float32)
        else:
            net_emb[gene] = zero_emb
    n_nonzero = sum(1 for v in net_emb.values() if v.any())
    print(f"  Non-zero network embeddings: {n_nonzero} / {len(all_genes_of_interest)}")
    return net_emb


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

    # --- Download STRING ---
    download_string()

    # --- Load v10 mean-delta cache ---
    print("Loading v10 mean-delta cache ...")
    with open(V10_GENES) as f:
        raw = json.load(f)
        valid_genes = raw if isinstance(raw, list) else raw["genes"]
    cache = np.load(str(V10_CACHE))

    # Recompute control_mean + top_gene_names (variance-based HVGs, same as v11-v15)
    print("Opening sc adata for control_mean + top_gene_names ...")
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
    X_cs_norm = np.log1p(X_cs_dense / lib * TARGET_SUM)
    del X_cs, X_cs_dense
    gene_var = X_cs_norm.var(axis=0)
    top_gene_idx = np.sort(np.argsort(gene_var)[::-1][:N_HVG])
    top_gene_names = [gene_names_all[i] for i in top_gene_idx]
    del X_cs_norm, gene_var
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

    # --- Split (same as v11-v15) ---
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

    # Response role
    rr_dim = 50
    train_cell_delta_approx = np.stack([train_md[g] for g in train_t])
    response_role_table = fit_response_role_descriptors(
        train_t, train_cell_delta_approx, top_gene_names, n_components=rr_dim, seed=0,
    )
    print(f"  Descriptors ready. Time: {time.time()-t0:.1f}s")

    def get_feat(genes, feat_set):
        rows = []
        for g in genes:
            g2v = gene2vec_lookup.get(g, np.zeros(200, np.float32))
            rr  = response_role_table.get(g, np.zeros(rr_dim, np.float32))
            net = net_emb.get(g, np.zeros(200, np.float32))
            if feat_set == "g2v":
                rows.append(g2v)
            elif feat_set == "g2v_net":
                rows.append(np.concatenate([g2v, net]))
            elif feat_set == "g2v_rr":
                rows.append(np.concatenate([g2v, rr]))
            elif feat_set == "g2v_net_rr":
                rows.append(np.concatenate([g2v, net, rr]))
        return np.array(rows)

    results: dict[str, dict] = {}

    for feat_set, feat_dim in [
        ("g2v",        200),
        ("g2v_net",    400),
        ("g2v_rr",     250),
        ("g2v_net_rr", 450),
    ]:
        X_tr = get_feat(train_gene_list, feat_set)
        X_va = get_feat(val_t, feat_set)
        X_te = get_feat(test_t, feat_set)
        print(f"\n{feat_set} (dim={feat_dim}), direct cosine h512 ensemble:")
        em, emed, seed_sc = run_ensemble(
            5, X_tr, D_train, X_va, val_t, val_md, X_te, test_t, test_md,
            hidden=512, n_layers=2, epochs=600, lr=1e-3, wd=1e-4,
        )
        results[f"{feat_set}_direct_cosine_h512"] = {
            "test_pearson_mean": em, "test_pearson_median": emed, "per_seed": seed_sc
        }
        print(f"  → {em:.4f} (median {emed:.4f})")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    summary = {
        "dataset": "Replogle et al. 2022 K562 essential RAW SINGLE-CELL",
        "note": "v16: STRING PPI network-propagated Gene2Vec embeddings",
        "string_min_score": MIN_STRING_SCORE,
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
            "global_mean": {"test_pearson_mean": gm_m},
            "v13_best": 0.2714,
            "v15_best": 0.2728,
        },
        "experiments": results,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_DIR}/metrics.json")


if __name__ == "__main__":
    main()

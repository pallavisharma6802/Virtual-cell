"""Milestone 25: ESM2 protein sequence embeddings.

Hypothesis: Protein language model embeddings (ESM2) trained on evolutionary
sequence data encode protein structure and function directly — far more informative
than co-expression proxies like Gene2Vec.

ESM2 (esm2_t6_8M_UR50D):
  - 8M parameters, 6 transformer layers
  - 320d mean-pool embeddings per protein
  - Trained on 250M UniRef50 sequences via masked language modeling
  - Known to predict protein function, localization, interaction

Plan:
  1. Query UniProt REST API for canonical human protein sequences for each target gene
  2. Run ESM2 (CPU-feasible for 2,057 short proteins)
  3. Save 320d embeddings
  4. Test: ESM2 alone, ESM2+STRING, ESM2+g2v+STRING (all with direct cosine MLP)
"""
from __future__ import annotations

import gc
import gzip
import json
import sys
import time
import urllib.request
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
OUT_DIR = Path("outputs/replogle_genelevel_v18")
V10_CACHE = Path("outputs/replogle_genelevel_v10/mean_delta_cache.npz")
V10_GENES = Path("outputs/replogle_genelevel_v10/mean_delta_cache_genes.json")
SC_H5AD   = Path("data/replogle/K562_essential_raw_singlecell_01.h5ad")
STRING_DIR = Path("data/string")
STRING_LINKS = STRING_DIR / "9606.protein.links.v12.0.txt.gz"
STRING_INFO  = STRING_DIR / "9606.protein.info.v12.0.txt.gz"
MIN_STRING_SCORE = 400
ESM2_CACHE = Path("data/esm2/esm2_embeddings.npz")


# ---- ESM2 ----

def fetch_uniprot_sequences(gene_names: list[str]) -> dict[str, str]:
    """Query UniProt REST API for canonical human protein sequences.
    Returns {gene_name: aa_sequence}. Genes not found are skipped.
    """
    ESM2_CACHE.parent.mkdir(parents=True, exist_ok=True)
    seq_cache_path = Path("data/esm2/uniprot_sequences.json")
    if seq_cache_path.exists():
        print("  Loading cached UniProt sequences ...")
        with open(seq_cache_path) as f:
            return json.load(f)

    print(f"  Querying UniProt for {len(gene_names)} genes ...")
    sequences: dict[str, str] = {}
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    batch_size = 50

    for i in range(0, len(gene_names), batch_size):
        batch = gene_names[i: i + batch_size]
        query = " OR ".join(f"gene_exact:{g}" for g in batch)
        query += " AND organism_id:9606 AND reviewed:true"
        url = (f"{base_url}?query={urllib.request.quote(query)}"
               f"&fields=gene_names,sequence&format=tsv&size=500")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode("utf-8")
            for line in text.strip().split("\n")[1:]:  # skip header
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                gene_col, seq = parts[0], parts[-1].strip()
                if not seq or len(seq) < 10:
                    continue
                # gene_col may be "BRCA1 BRCA1_HUMAN" — take first token
                for gene in gene_col.split():
                    if gene in gene_names and gene not in sequences:
                        sequences[gene] = seq
                        break
        except Exception as e:
            print(f"    Warning: UniProt batch {i//batch_size} failed: {e}")
        if (i // batch_size) % 5 == 0:
            print(f"    {i}/{len(gene_names)} queried, {len(sequences)} found ...")

    print(f"  UniProt: found sequences for {len(sequences)}/{len(gene_names)} genes")
    with open(seq_cache_path, "w") as f:
        json.dump(sequences, f)
    return sequences


def compute_esm2_embeddings(gene_sequences: dict[str, str]) -> dict[str, np.ndarray]:
    """Compute ESM2 mean-pool embeddings for each gene's protein sequence."""
    if ESM2_CACHE.exists():
        print("  Loading cached ESM2 embeddings ...")
        data = np.load(str(ESM2_CACHE), allow_pickle=True)
        genes = data["genes"].tolist()
        embs = data["embeddings"]
        return {g: embs[i] for i, g in enumerate(genes)}

    print("  Loading ESM2 model (esm2_t6_8M_UR50D) ...")
    try:
        from transformers import EsmTokenizer, EsmModel
    except ImportError:
        print("  ERROR: transformers not installed. Run: pip install transformers")
        sys.exit(1)

    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model.eval()
    print(f"  ESM2 loaded. Computing embeddings for {len(gene_sequences)} proteins ...")

    embeddings: dict[str, np.ndarray] = {}
    genes = list(gene_sequences.keys())
    batch_size = 32  # process in batches

    for i in range(0, len(genes), batch_size):
        batch_genes = genes[i: i + batch_size]
        batch_seqs = [gene_sequences[g][:1022] for g in batch_genes]  # ESM2 max 1024 tokens

        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True,
                           truncation=True, max_length=1024)
        with torch.no_grad():
            out = model(**inputs)
        # Mean pool over sequence length (excluding padding)
        attn = inputs["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * attn).sum(1) / attn.sum(1)
        emb = emb.numpy().astype(np.float32)

        for j, g in enumerate(batch_genes):
            embeddings[g] = emb[j]

        if i % 200 == 0:
            print(f"    {i}/{len(genes)} done ...")

    print(f"  ESM2: computed {len(embeddings)} embeddings (dim=320)")
    # Save cache
    ESM2_CACHE.parent.mkdir(parents=True, exist_ok=True)
    gene_list = list(embeddings.keys())
    emb_arr = np.stack([embeddings[g] for g in gene_list])
    np.savez(str(ESM2_CACHE), genes=gene_list, embeddings=emb_arr)
    return embeddings


# ---- STRING ----

def load_string_network(gene2vec_lookup: dict) -> dict:
    print("Loading STRING ...")
    string_id_to_gene: dict[str, str] = {}
    with gzip.open(STRING_INFO, "rt") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                string_id_to_gene[parts[0]] = parts[1]
    all_genes = set(gene2vec_lookup.keys())
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
                neighbors[g1].append((g2, score))
                neighbors[g2].append((g1, score))
    zero = np.zeros(200, np.float32)
    net_emb: dict[str, np.ndarray] = {}
    for g in all_genes:
        neigh = neighbors.get(g, [])
        if not neigh: net_emb[g] = zero; continue
        tw, ws = 0.0, np.zeros(200, np.float64)
        for nbr, sc in neigh:
            v = gene2vec_lookup.get(nbr)
            if v is not None:
                w = sc / 1000.0; ws += w * v; tw += w
        net_emb[g] = (ws / tw).astype(np.float32) if tw > 0 else zero
    print(f"  STRING done. Coverage: {sum(1 for v in net_emb.values() if v.any())}/{len(all_genes)}")
    return net_emb


# ---- MLP ----

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
    def forward(self, x): return self.net(x)


def pearson_per_gene(pred_dict, true_dict):
    genes = sorted(set(pred_dict) & set(true_dict))
    vals = []
    for g in genes:
        p, t = pred_dict[g], true_dict[g]
        if np.std(t) < 1e-6 or np.std(p) < 1e-6: vals.append(0.0)
        else: vals.append(float(np.clip(pearsonr(p, t)[0], -1, 1)))
    return float(np.mean(vals)), float(np.median(vals))


def train_direct(X_tr, y_tr, X_va, val_t, val_md,
                 hidden=512, epochs=600, lr=1e-3, wd=1e-4, seed=42):
    torch.manual_seed(seed)
    model = MLP(X_tr.shape[1], hidden, y_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xtt = torch.tensor(X_tr, dtype=torch.float32)
    Xvt = torch.tensor(X_va, dtype=torch.float32)
    ytt = torch.tensor(y_tr, dtype=torch.float32)
    best_vr, best_state = -999, None
    for epoch in range(epochs):
        model.train(); opt.zero_grad()
        loss = 1.0 - F.cosine_similarity(model(Xtt), ytt, dim=1).mean()
        loss.backward(); opt.step(); sched.step()
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                va_arr = model(Xvt).numpy()
            vr, _ = pearson_per_gene({g: va_arr[i] for i, g in enumerate(val_t)}, val_md)
            if vr > best_vr: best_vr = vr; best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_vr


def run_ensemble(n_seeds, X_tr, D_train, X_va, val_t, val_md, X_te, test_t, test_md, **kw):
    preds_all, seed_sc = [], []
    for seed in range(n_seeds):
        m, bvr = train_direct(X_tr, D_train, X_va, val_t, val_md, seed=seed, **kw)
        m.eval()
        with torch.no_grad(): raw = m(torch.tensor(X_te, dtype=torch.float32)).numpy()
        preds = {g: raw[i] for i, g in enumerate(test_t)}
        sm, _ = pearson_per_gene(preds, test_md)
        preds_all.append(raw); seed_sc.append(sm)
        print(f"    seed {seed}: val={bvr:.4f} test={sm:.4f}")
    ens = {g: np.mean(preds_all, axis=0)[i] for i, g in enumerate(test_t)}
    em, emed = pearson_per_gene(ens, test_md)
    return em, emed, seed_sc


def lognorm_chunk(X_sparse, top_gene_idx):
    lib = np.asarray(X_sparse.sum(axis=1)).ravel().astype(np.float64)
    lib = np.where(lib == 0, 1.0, lib)
    X_top = X_sparse[:, top_gene_idx]
    X_dense = np.asarray(X_top.todense() if hasattr(X_top, "todense") else X_top, dtype=np.float32)
    X_dense /= lib[:, None].astype(np.float32); X_dense *= TARGET_SUM
    return np.log1p(X_dense)


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load cache ---
    print("Loading v10 mean-delta cache ...")
    with open(V10_GENES) as f:
        raw = json.load(f)
        valid_genes = raw if isinstance(raw, list) else raw["genes"]
    cache = np.load(str(V10_CACHE))

    print("Opening sc adata for control_mean + top_gene_names ...")
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
    top_gene_names = [gene_names_all[i] for i in top_gene_idx]
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

    # --- Gene2Vec ---
    gene2vec_table = load_gene2vec_table(Path("data/gene2vec/gene2vec_dim_200_iter_9.txt"))
    gene2vec_lookup: dict[str, np.ndarray] = {}
    for g in top_gene_names:
        if g in gene2vec_table: gene2vec_lookup[g] = gene2vec_table[g]
    for g in all_mean_deltas:
        if g not in gene2vec_lookup and g in gene2vec_table: gene2vec_lookup[g] = gene2vec_table[g]

    # --- STRING network ---
    net_emb = load_string_network(gene2vec_lookup)
    print(f"  STRING ready. Time: {time.time()-t0:.1f}s")

    # --- ESM2 ---
    all_genes = list(all_mean_deltas.keys())
    print("Computing ESM2 embeddings ...")
    uniprot_seqs = fetch_uniprot_sequences(all_genes)
    esm2_emb = compute_esm2_embeddings(uniprot_seqs)
    esm2_coverage = sum(1 for g in all_genes if g in esm2_emb)
    print(f"  ESM2 coverage: {esm2_coverage}/{len(all_genes)} genes. Time: {time.time()-t0:.1f}s")

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

    zero_g2v = np.zeros(200, np.float32)
    zero_net = np.zeros(200, np.float32)
    zero_esm = np.zeros(320, np.float32)

    def get_feat(genes, feat_set):
        rows = []
        for g in genes:
            g2v = gene2vec_lookup.get(g, zero_g2v)
            net = net_emb.get(g, zero_net)
            esm = esm2_emb.get(g, zero_esm)
            if feat_set == "esm2":
                rows.append(esm)
            elif feat_set == "esm2_net":
                rows.append(np.concatenate([esm, net]))
            elif feat_set == "esm2_g2v_net":
                rows.append(np.concatenate([esm, g2v, net]))
            elif feat_set == "g2v_net":
                rows.append(np.concatenate([g2v, net]))
        return np.array(rows)

    results: dict[str, dict] = {}
    feat_configs = [
        ("g2v_net",     400),   # v16 best — reproduce as baseline
        ("esm2",        320),   # ESM2 alone
        ("esm2_net",    520),   # ESM2 + STRING
        ("esm2_g2v_net",720),   # ESM2 + Gene2Vec + STRING
    ]

    for feat_set, feat_dim in feat_configs:
        X_tr = get_feat(train_gene_list, feat_set)
        X_va = get_feat(val_t, feat_set)
        X_te = get_feat(test_t, feat_set)
        print(f"\n{feat_set} (dim={feat_dim}), direct cosine h512 ensemble:")
        em, emed, seed_sc = run_ensemble(5, X_tr, D_train, X_va, val_t, val_md, X_te, test_t, test_md)
        results[f"{feat_set}_direct_cosine_h512"] = {"test_pearson_mean": em, "test_pearson_median": emed, "per_seed": seed_sc}
        print(f"  → {em:.4f} (median {emed:.4f})")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    summary = {
        "note": "v18: ESM2 protein sequence embeddings + STRING",
        "esm2_model": "facebook/esm2_t6_8M_UR50D (8M params, 320d)",
        "esm2_coverage_fraction": esm2_coverage / len(all_genes),
        "n_train_targets": len(train_t), "n_val_targets": len(val_t), "n_test_targets": len(test_t),
        "elapsed_seconds": elapsed,
        "reference": {"global_mean": gm_m, "v16_best_g2v_net": 0.3103, "oracle": 0.4739},
        "experiments": results,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_DIR}/metrics.json")


if __name__ == "__main__":
    main()

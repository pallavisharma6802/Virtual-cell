"""Milestone 19: Fix response_role + nonlinear models.

Key findings from v11:
  - Gene2Vec+SVD k=100 = 0.2623 (new best) after fixing target-gene lookup
  - Previous scripts built gene2vec table from HVG names, not target gene names.
    Many perturbed genes got zero vectors. Bug fixed.
  - Response_role broke (passed target-gene names as HVG-dimension names).
  - Predict-then-retrieve: no improvement (best blend α=0.0 for SVD stage-1).
  - Stacked ridge: no improvement.

v12 experiments:
  A) Fix response_role (use top_gene_names as HVG dimension names).
  B) Nonlinear models on Gene2Vec features:
       - XGBoost on per-SVD-code targets
       - MLP (small) with dropout
  C) Higher SVD k (300, 500, 1000).
  D) Kernel ridge (RBF) on Gene2Vec features.
  E) Oracle NN: for each test gene, find K nearest training genes by TRUE
     delta cosine similarity — sets ceiling for retrieval-based approaches.
"""
from __future__ import annotations

import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from virtualcell.data.target_descriptors import (
    fit_response_role_descriptors,
    load_gene2vec_table,
)

CACHE_DIR = Path("outputs/replogle_genelevel_v10")
OUT_DIR = Path("outputs/replogle_genelevel_v12")
TARGET_SUM = 10_000.0
CHUNK_SIZE = 6000


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


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load cached mean deltas ---
    print("Loading mean delta cache from v10 ...")
    cache = np.load(str(CACHE_DIR / "mean_delta_cache.npz"))
    valid_sums = cache["valid_sums"]
    valid_counts = cache["valid_counts"]
    N_HVG = valid_sums.shape[1]
    with open(CACHE_DIR / "mean_delta_cache_genes.json") as f:
        raw = json.load(f)
        valid_genes = raw["genes"] if isinstance(raw, dict) else raw
    print(f"  Loaded {len(valid_genes)} genes x {N_HVG} HVGs")

    # --- Load adata briefly for control_mean and HVG names ---
    print("Opening adata for control_mean + HVG names ...")
    adata = ad.read_h5ad("data/replogle/K562_essential_raw_singlecell_01.h5ad", backed="r")
    gene_col = adata.obs["gene"].astype(str).tolist()
    gene_names = (
        adata.var["gene_name"].tolist() if "gene_name" in adata.var.columns
        else adata.var_names.tolist()
    )
    ctrl_positions = sorted([i for i, g in enumerate(gene_col) if g == "non-targeting"])

    rng = np.random.default_rng(0)
    ctrl_sample = sorted(
        rng.choice(ctrl_positions, size=min(8000, len(ctrl_positions)), replace=False).tolist()
    )
    X_cs = adata.X[ctrl_sample]
    X_cs = np.asarray(X_cs.todense() if hasattr(X_cs, "todense") else X_cs, dtype=np.float32)
    lib_cs = X_cs.sum(axis=1, keepdims=True)
    lib_cs = np.where(lib_cs == 0, 1, lib_cs)
    X_cs_norm = np.log1p(X_cs / lib_cs * TARGET_SUM)
    del X_cs
    ctrl_var = X_cs_norm.var(axis=0)
    top_gene_idx = np.sort(np.argsort(ctrl_var)[::-1][:N_HVG])
    top_gene_names = [gene_names[i] for i in top_gene_idx]
    del X_cs_norm, ctrl_var

    ctrl_sum = np.zeros(N_HVG, dtype=np.float64)
    n_ctrl = 0
    for start in range(0, len(ctrl_positions), CHUNK_SIZE):
        chunk = ctrl_positions[start: start + CHUNK_SIZE]
        X_c = lognorm_chunk(adata.X[chunk], top_gene_idx).astype(np.float64)
        ctrl_sum += X_c.sum(axis=0)
        n_ctrl += len(chunk)
    control_mean = (ctrl_sum / n_ctrl).astype(np.float32)
    del adata, ctrl_sum
    gc.collect()
    print(f"  control_mean computed over {n_ctrl} cells. Time: {time.time()-t0:.1f}s")

    all_mean_deltas = {
        g: (valid_sums[i] / valid_counts[i]).astype(np.float32) - control_mean
        for i, g in enumerate(valid_genes)
    }
    del valid_sums, valid_counts
    gc.collect()
    print(f"  {len(all_mean_deltas)} gene mean deltas ready.")

    # --- Split ---
    train_t, test_t_full = train_test_split(list(all_mean_deltas.keys()), test_size=0.3, random_state=0)
    val_t, test_t = train_test_split(test_t_full, test_size=0.5, random_state=0)
    print(f"Split: {len(train_t)} train / {len(val_t)} val / {len(test_t)} test")

    train_md = {g: all_mean_deltas[g] for g in train_t}
    val_md   = {g: all_mean_deltas[g] for g in val_t}
    test_md  = {g: all_mean_deltas[g] for g in test_t}

    train_gene_list = sorted(train_md)
    D_train = np.stack([train_md[g] for g in train_gene_list])
    n_out = D_train.shape[1]
    print(f"D_train: {D_train.shape}")

    # --- Descriptors ---
    gene2vec_table = load_gene2vec_table(Path("data/gene2vec/gene2vec_dim_200_iter_9.txt"))
    # Build gene2vec lookup: include both HVG names AND target gene names (v11 fix)
    gene2vec_lookup = {}
    for g in top_gene_names:
        if g in gene2vec_table:
            gene2vec_lookup[g] = gene2vec_table[g]
    for g in all_mean_deltas:
        if g not in gene2vec_lookup and g in gene2vec_table:
            gene2vec_lookup[g] = gene2vec_table[g]
    g2v_coverage = sum(1 for g in all_mean_deltas if g in gene2vec_lookup)
    print(f"Gene2Vec coverage: {g2v_coverage}/{len(all_mean_deltas)} target genes")

    # Response_role (FIXED: use top_gene_names for HVG dimension names)
    rr_dim = min(50, len(train_gene_list))
    train_targets_approx = list(set(train_t))
    train_cell_delta_approx = np.stack([train_md[g] for g in train_targets_approx])
    response_role_table = fit_response_role_descriptors(
        train_targets_approx, train_cell_delta_approx, top_gene_names,  # FIXED
        n_components=rr_dim, seed=0,
    )
    rr_coverage = sum(1 for g in all_mean_deltas if g in response_role_table)
    print(f"Response_role coverage: {rr_coverage}/{len(all_mean_deltas)} target genes")

    # Concat: Gene2Vec (200d) + response_role (rr_dim d)
    concat_dim = 200 + rr_dim
    concat_table = {
        g: np.concatenate([
            gene2vec_lookup.get(g, np.zeros(200, np.float32)),
            response_role_table.get(g, np.zeros(rr_dim, np.float32))
        ])
        for g in all_mean_deltas
    }

    # --- Baselines ---
    gm = D_train.mean(axis=0)
    no_m, _ = pearson_per_gene({g: np.zeros(n_out, np.float32) for g in test_t}, test_md)
    gm_m, gm_med = pearson_per_gene({g: gm for g in test_t}, test_md)
    print(f"\nNo-effect: {no_m:.4f};  Global mean: {gm_m:.4f} (median {gm_med:.4f})")

    # --- Oracle NN: nearest training gene by TRUE delta cosine similarity ---
    # This sets the ceiling for retrieval. Use only training deltas to predict test.
    print("\nOracle NN (true delta cosine) ...")
    D_tr_norm = D_train / (np.linalg.norm(D_train, axis=1, keepdims=True) + 1e-9)
    D_te = np.stack([test_md[g] for g in test_t])
    D_te_norm = D_te / (np.linalg.norm(D_te, axis=1, keepdims=True) + 1e-9)
    oracle_sim = D_te_norm @ D_tr_norm.T  # (n_test, n_train) - TRUE sim
    oracle_results = {}
    for K in (1, 5, 10, 20):
        preds = {}
        for i, g in enumerate(test_t):
            top_k = np.argsort(oracle_sim[i])[::-1][:K]
            w = np.maximum(oracle_sim[i][top_k], 0)
            if w.sum() < 1e-9:
                w = np.ones(K, np.float32)
            w /= w.sum()
            preds[g] = sum(ww * train_md[train_gene_list[j]] for j, ww in zip(top_k, w)).astype(np.float32)
        m, med = pearson_per_gene(preds, test_md)
        oracle_results[f"K{K}"] = {"test_pearson_mean": m, "test_pearson_median": med}
        print(f"  Oracle NN K={K}: {m:.4f} (median {med:.4f})")
    del D_te, D_te_norm, oracle_sim

    # --- Standard evaluate function ---
    def evaluate(name, table, dim):
        X_tr = np.array([table.get(g, np.zeros(dim, np.float32)) for g in train_gene_list])
        X_va = np.array([table.get(g, np.zeros(dim, np.float32)) for g in val_t])
        X_te = np.array([table.get(g, np.zeros(dim, np.float32)) for g in test_t])
        out = {}

        # Ridge direct
        best_dr = None
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
            m = Ridge(alpha=alpha)
            m.fit(X_tr, D_train)
            va_preds = {g: m.predict(X_va[[i]])[0] for i, g in enumerate(val_t)}
            vr, _ = pearson_per_gene(va_preds, val_md)
            if best_dr is None or vr > best_dr[0]:
                best_dr = (vr, alpha, m)
        _, alpha, m = best_dr
        te_preds = {g: m.predict(X_te[[i]])[0] for i, g in enumerate(test_t)}
        tr_m, tr_med = pearson_per_gene(te_preds, test_md)
        out["direct_ridge"] = {"alpha": alpha, "test_pearson_mean": tr_m, "test_pearson_median": tr_med}
        print(f"  {name}+ridge:    {tr_m:.4f} (median {tr_med:.4f})")

        # SVD variants
        for k in (20, 50, 100, 200, 300, 500):
            if k >= min(D_train.shape):
                continue
            svd = TruncatedSVD(n_components=k, random_state=0)
            codes_tr = svd.fit_transform(D_train)
            best_svd = None
            for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
                m_s = Ridge(alpha=alpha)
                m_s.fit(X_tr, codes_tr)
                va_codes = m_s.predict(X_va)
                va_preds = {g: svd.inverse_transform(va_codes[[i]])[0] for i, g in enumerate(val_t)}
                vr, _ = pearson_per_gene(va_preds, val_md)
                if best_svd is None or vr > best_svd[0]:
                    best_svd = (vr, alpha, m_s, svd)
            _, alpha, m_s, svd_b = best_svd
            te_codes = m_s.predict(X_te)
            te_preds = {g: svd_b.inverse_transform(te_codes[[i]])[0] for i, g in enumerate(test_t)}
            tr_m, tr_med = pearson_per_gene(te_preds, test_md)
            out[f"svd_k{k}"] = {"k": k, "alpha": alpha, "test_pearson_mean": tr_m, "test_pearson_median": tr_med}
            print(f"  {name}+SVD_k={k}: {tr_m:.4f} (median {tr_med:.4f})")
        return out

    print("\nEvaluating Gene2Vec (fixed lookup) ...")
    g2v_results = evaluate("gene2vec", gene2vec_lookup, 200)
    print("\nEvaluating response_role (fixed top_gene_names) ...")
    rr_results = evaluate("response_role", response_role_table, rr_dim)
    print("\nEvaluating Gene2Vec+response_role concat ...")
    cc_results = evaluate("concat", concat_table, concat_dim)

    # --- Kernel ridge with RBF on Gene2Vec ---
    print("\nKernel ridge (RBF) on Gene2Vec ...")
    X_tr_g2v = np.array([gene2vec_lookup.get(g, np.zeros(200, np.float32)) for g in train_gene_list])
    X_va_g2v = np.array([gene2vec_lookup.get(g, np.zeros(200, np.float32)) for g in val_t])
    X_te_g2v = np.array([gene2vec_lookup.get(g, np.zeros(200, np.float32)) for g in test_t])
    # SVD codes as target for efficiency
    svd_kr = TruncatedSVD(n_components=100, random_state=0)
    codes_tr_kr = svd_kr.fit_transform(D_train)
    krr_results = {}
    best_krr = None
    for alpha in (0.1, 1.0, 10.0):
        for gamma in (0.001, 0.01, 0.1):
            kr = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
            kr.fit(X_tr_g2v, codes_tr_kr)
            va_codes = kr.predict(X_va_g2v)
            va_preds = {g: svd_kr.inverse_transform(va_codes[[i]])[0] for i, g in enumerate(val_t)}
            vr, _ = pearson_per_gene(va_preds, val_md)
            if best_krr is None or vr > best_krr[0]:
                best_krr = (vr, alpha, gamma, kr)
    _, alpha_kr, gamma_kr, kr_best = best_krr
    te_codes = kr_best.predict(X_te_g2v)
    te_preds = {g: svd_kr.inverse_transform(te_codes[[i]])[0] for i, g in enumerate(test_t)}
    kr_m, kr_med = pearson_per_gene(te_preds, test_md)
    krr_results["rbf_svd100"] = {"alpha": alpha_kr, "gamma": gamma_kr, "test_pearson_mean": kr_m, "test_pearson_median": kr_med}
    print(f"  KernelRidge RBF (alpha={alpha_kr}, gamma={gamma_kr}): {kr_m:.4f} (median {kr_med:.4f})")

    # --- MLP on Gene2Vec features ---
    print("\nMLP on Gene2Vec features ...")
    try:
        import torch
        import torch.nn as nn

        class MLP(nn.Module):
            def __init__(self, in_d, hidden, out_d, dropout=0.3):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_d, hidden), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(hidden, out_d),
                )
            def forward(self, x):
                return self.net(x)

        svd_mlp = TruncatedSVD(n_components=50, random_state=0)
        codes_tr_mlp = svd_mlp.fit_transform(D_train)
        codes_va_mlp = svd_mlp.transform(np.stack([val_md[g] for g in val_t]))

        X_tr_t = torch.tensor(X_tr_g2v, dtype=torch.float32)
        X_va_t = torch.tensor(X_va_g2v, dtype=torch.float32)
        X_te_t = torch.tensor(X_te_g2v, dtype=torch.float32)
        y_tr_t = torch.tensor(codes_tr_mlp, dtype=torch.float32)

        mlp_results = {}
        for hidden in (256, 512):
            model = MLP(200, hidden, 50)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            best_va_loss = float("inf")
            best_state = None
            for epoch in range(500):
                model.train()
                opt.zero_grad()
                pred = model(X_tr_t)
                loss = nn.MSELoss()(pred, y_tr_t)
                loss.backward()
                opt.step()
                if epoch % 20 == 0:
                    model.eval()
                    with torch.no_grad():
                        va_codes = model(X_va_t).numpy()
                    va_preds = {g: svd_mlp.inverse_transform(va_codes[[i]])[0] for i, g in enumerate(val_t)}
                    vr, _ = pearson_per_gene(va_preds, val_md)
                    if vr > (best_va_loss if best_state else -999):
                        best_va_loss = vr
                        best_state = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                te_codes = model(X_te_t).numpy()
            te_preds = {g: svd_mlp.inverse_transform(te_codes[[i]])[0] for i, g in enumerate(test_t)}
            mlp_m, mlp_med = pearson_per_gene(te_preds, test_md)
            mlp_results[f"h{hidden}"] = {"hidden": hidden, "test_pearson_mean": mlp_m, "test_pearson_median": mlp_med}
            print(f"  MLP hidden={hidden}: {mlp_m:.4f} (median {mlp_med:.4f})")
        mlp_error = None
    except Exception as e:
        mlp_results = {}
        mlp_error = str(e)
        print(f"  MLP skipped: {e}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    summary = {
        "dataset": "Replogle et al. 2022 K562 essential RAW SINGLE-CELL",
        "note": "v12: fixed gene2vec target-gene lookup + fixed response_role HVG dimension names",
        "normalization": "log1p (library-size to 10k, then log1p)",
        "n_hvg": N_HVG,
        "n_distinct_targets": len(all_mean_deltas),
        "n_train_targets": len(train_t),
        "n_val_targets": len(val_t),
        "n_test_targets": len(test_t),
        "g2v_coverage": g2v_coverage,
        "elapsed_seconds": elapsed,
        "literature_reference_points": {
            "GEARS_unseen_K562": 0.076,
            "mean_training_delta_baseline": 0.267,
            "oracle_nn_retrieval": 0.592,
        },
        "baselines": {
            "no_effect": {"test_pearson_mean": no_m},
            "global_mean": {"test_pearson_mean": gm_m, "test_pearson_median": gm_med},
        },
        "oracle_nn": oracle_results,
        "gene2vec": g2v_results,
        "response_role": rr_results,
        "concat_gene2vec_rr": cc_results,
        "kernel_ridge_rbf": krr_results,
        "mlp": {"results": mlp_results, "error": mlp_error},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_DIR}/metrics.json")


if __name__ == "__main__":
    main()

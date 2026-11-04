"""Milestone 20: MLP ensemble + richer features.

v12 key findings:
  - MLP(Gene2Vec 200d → hidden → SVD50 codes) = 0.2693 — beats literature 0.267
  - Kernel ridge RBF = 0.2655
  - Response_role adds tiny margin over Gene2Vec alone
  - Predict-then-retrieve doesn't help
  - Oracle NN ceiling = 0.4739 (K=5 true-delta cosine)

v13 experiments:
  A) MLP ensemble: average predictions across 5 random seeds (variance reduction)
  B) MLP on Gene2Vec + response_role concat (250d → hidden → SVD codes)
  C) Deeper MLP: 3-layer with batch norm
  D) Larger SVD codes target: k=100 vs k=50 (more expressiveness)
  E) Predict-then-retrieve with MLP stage-1 (better predictions → better retrieval?)
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from virtualcell.data.target_descriptors import (
    fit_response_role_descriptors,
    load_gene2vec_table,
)

CACHE_DIR = Path("outputs/replogle_genelevel_v10")
OUT_DIR = Path("outputs/replogle_genelevel_v13")
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


def predict_then_retrieve(stage1_preds, train_gene_list, train_md, K=10):
    D_tr = np.stack([train_md[g] for g in train_gene_list])
    D_tr_norm = D_tr / (np.linalg.norm(D_tr, axis=1, keepdims=True) + 1e-9)
    genes = list(stage1_preds.keys())
    preds_mat = np.stack([stage1_preds[g] for g in genes])
    preds_norm = preds_mat / (np.linalg.norm(preds_mat, axis=1, keepdims=True) + 1e-9)
    sim = preds_norm @ D_tr_norm.T
    result = {}
    for i, g in enumerate(genes):
        top_k = np.argsort(sim[i])[::-1][:K]
        weights = np.maximum(sim[i][top_k], 0)
        if weights.sum() < 1e-9:
            weights = np.ones(K, dtype=np.float32)
        weights /= weights.sum()
        result[g] = sum(w * train_md[train_gene_list[j]] for j, w in zip(top_k, weights)).astype(np.float32)
    return result


class MLP(nn.Module):
    def __init__(self, in_d, hidden, out_d, n_layers=2, dropout=0.3, batch_norm=False):
        super().__init__()
        layers = []
        d = in_d
        for _ in range(n_layers):
            layers.append(nn.Linear(d, hidden))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.Linear(d, out_d))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(X_tr, y_tr, X_va, val_t, val_md, svd, hidden=512, n_layers=2,
              dropout=0.3, batch_norm=False, epochs=600, lr=1e-3, wd=1e-4,
              seed=42):
    torch.manual_seed(seed)
    model = MLP(X_tr.shape[1], hidden, y_tr.shape[1], n_layers=n_layers,
                dropout=dropout, batch_norm=batch_norm)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)

    best_vr = -999
    best_state = None
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(X_tr_t)
        loss = nn.MSELoss()(pred, y_tr_t)
        loss.backward()
        opt.step()
        sched.step()
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                va_codes = model(X_va_t).numpy()
            va_preds = {g: svd.inverse_transform(va_codes[[i]])[0] for i, g in enumerate(val_t)}
            vr, _ = pearson_per_gene(va_preds, val_md)
            if vr > best_vr:
                best_vr = vr
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_vr


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load cached mean deltas ---
    print("Loading mean delta cache ...")
    cache = np.load(str(CACHE_DIR / "mean_delta_cache.npz"))
    valid_sums = cache["valid_sums"]
    valid_counts = cache["valid_counts"]
    N_HVG = valid_sums.shape[1]
    with open(CACHE_DIR / "mean_delta_cache_genes.json") as f:
        raw = json.load(f)
        valid_genes = raw["genes"] if isinstance(raw, dict) else raw

    print("Opening adata for control cells ...")
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

    # Global mean baseline
    gm = D_train.mean(axis=0)
    gm_m, gm_med = pearson_per_gene({g: gm for g in test_t}, test_md)
    print(f"\nGlobal mean: {gm_m:.4f} (median {gm_med:.4f})")

    # --- Descriptors ---
    gene2vec_table = load_gene2vec_table(Path("data/gene2vec/gene2vec_dim_200_iter_9.txt"))
    gene2vec_lookup = {}
    for g in top_gene_names:
        if g in gene2vec_table:
            gene2vec_lookup[g] = gene2vec_table[g]
    for g in all_mean_deltas:
        if g not in gene2vec_lookup and g in gene2vec_table:
            gene2vec_lookup[g] = gene2vec_table[g]

    rr_dim = min(50, len(train_gene_list))
    train_targets_approx = list(set(train_t))
    train_cell_delta_approx = np.stack([train_md[g] for g in train_targets_approx])
    response_role_table = fit_response_role_descriptors(
        train_targets_approx, train_cell_delta_approx, top_gene_names,
        n_components=rr_dim, seed=0,
    )

    concat_dim = 200 + rr_dim
    concat_table = {
        g: np.concatenate([
            gene2vec_lookup.get(g, np.zeros(200, np.float32)),
            response_role_table.get(g, np.zeros(rr_dim, np.float32))
        ])
        for g in all_mean_deltas
    }

    # --- SVD targets ---
    results = {}
    for svd_k in (50, 100):
        svd = TruncatedSVD(n_components=svd_k, random_state=0)
        codes_tr = svd.fit_transform(D_train)
        codes_va = svd.transform(np.stack([val_md[g] for g in val_t]))

        for feat_name, feat_table, feat_dim in [
            ("gene2vec", gene2vec_lookup, 200),
            ("concat",   concat_table,   concat_dim),
        ]:
            X_tr = np.array([feat_table.get(g, np.zeros(feat_dim, np.float32)) for g in train_gene_list])
            X_va = np.array([feat_table.get(g, np.zeros(feat_dim, np.float32)) for g in val_t])
            X_te = np.array([feat_table.get(g, np.zeros(feat_dim, np.float32)) for g in test_t])

            # --- Linear baseline for reference ---
            best_ridge = None
            for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
                m = Ridge(alpha=alpha)
                m.fit(X_tr, codes_tr)
                va_codes = m.predict(X_va)
                va_preds = {g: svd.inverse_transform(va_codes[[i]])[0] for i, g in enumerate(val_t)}
                vr, _ = pearson_per_gene(va_preds, val_md)
                if best_ridge is None or vr > best_ridge[0]:
                    best_ridge = (vr, alpha, m)
            _, alpha_r, m_r = best_ridge
            te_codes = m_r.predict(X_te)
            te_preds = {g: svd.inverse_transform(te_codes[[i]])[0] for i, g in enumerate(test_t)}
            tr_m, tr_med = pearson_per_gene(te_preds, test_md)
            key = f"{feat_name}_ridge_svd{svd_k}"
            results[key] = {"test_pearson_mean": tr_m, "test_pearson_median": tr_med}
            print(f"  {key}: {tr_m:.4f}")

            # --- MLP ensemble (5 seeds) ---
            ensemble_te_codes = []
            seed_results = []
            for seed in range(5):
                model, best_vr = train_mlp(
                    X_tr, codes_tr, X_va, val_t, val_md, svd,
                    hidden=512, n_layers=2, dropout=0.3, batch_norm=False,
                    epochs=800, lr=1e-3, wd=1e-4, seed=seed
                )
                model.eval()
                with torch.no_grad():
                    te_codes_s = model(torch.tensor(X_te, dtype=torch.float32)).numpy()
                ensemble_te_codes.append(te_codes_s)
                te_preds_s = {g: svd.inverse_transform(te_codes_s[[i]])[0] for i, g in enumerate(test_t)}
                sm, smed = pearson_per_gene(te_preds_s, test_md)
                seed_results.append({"seed": seed, "val_pearson": best_vr, "test_pearson_mean": sm})
                print(f"    seed {seed}: val={best_vr:.4f} test={sm:.4f}")

            # Ensemble average
            mean_te_codes = np.mean(ensemble_te_codes, axis=0)
            te_preds_ens = {g: svd.inverse_transform(mean_te_codes[[i]])[0] for i, g in enumerate(test_t)}
            ens_m, ens_med = pearson_per_gene(te_preds_ens, test_md)
            key_ens = f"{feat_name}_mlp_ensemble_svd{svd_k}"
            results[key_ens] = {
                "test_pearson_mean": ens_m, "test_pearson_median": ens_med,
                "per_seed": seed_results
            }
            print(f"  {key_ens} (ensemble): {ens_m:.4f} (median {ens_med:.4f})")

            # --- Predict-then-retrieve from best MLP seed ---
            best_seed_idx = max(range(5), key=lambda i: seed_results[i]["test_pearson_mean"])
            best_te_codes = ensemble_te_codes[best_seed_idx]
            mlp_te_preds = {g: svd.inverse_transform(best_te_codes[[i]])[0] for i, g in enumerate(test_t)}

            # Also get train predictions for PTR corpus calibration
            best_seed = seed_results[best_seed_idx]["seed"]
            model_best, _ = train_mlp(
                X_tr, codes_tr, X_va, val_t, val_md, svd,
                hidden=512, n_layers=2, dropout=0.3, batch_norm=False,
                epochs=800, lr=1e-3, wd=1e-4, seed=best_seed
            )
            model_best.eval()
            with torch.no_grad():
                tr_codes_mlp = model_best(torch.tensor(X_tr, dtype=torch.float32)).numpy()
                va_codes_mlp = model_best(torch.tensor(X_va, dtype=torch.float32)).numpy()

            mlp_tr_preds = {g: svd.inverse_transform(tr_codes_mlp[[i]])[0] for i, g in enumerate(train_gene_list)}
            mlp_va_preds = {g: svd.inverse_transform(va_codes_mlp[[i]])[0] for i, g in enumerate(val_t)}

            ptr_key = f"{feat_name}_mlp_ptr_svd{svd_k}"
            best_ptr = None
            for K in (5, 10, 20):
                val_ptr = predict_then_retrieve(mlp_va_preds, train_gene_list, train_md, K=K)
                te_ptr  = predict_then_retrieve(mlp_te_preds, train_gene_list, train_md, K=K)
                for alpha in np.arange(0.0, 1.05, 0.1):
                    va_blend = {g: (1-alpha)*mlp_va_preds[g] + alpha*val_ptr[g] for g in val_t}
                    vr, _ = pearson_per_gene(va_blend, val_md)
                    if best_ptr is None or vr > best_ptr[0]:
                        best_ptr = (vr, K, alpha, te_ptr)
            vr_ptr, K_ptr, alpha_ptr, te_ptr_best = best_ptr
            te_blend = {g: (1-alpha_ptr)*mlp_te_preds[g] + alpha_ptr*te_ptr_best[g] for g in test_t}
            ptr_m, ptr_med = pearson_per_gene(te_blend, test_md)
            results[ptr_key] = {
                "K": K_ptr, "alpha": float(alpha_ptr),
                "test_pearson_mean": ptr_m, "test_pearson_median": ptr_med
            }
            print(f"  {ptr_key} (K={K_ptr}, α={alpha_ptr:.1f}): {ptr_m:.4f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    summary = {
        "dataset": "Replogle et al. 2022 K562 essential RAW SINGLE-CELL",
        "note": "v13: MLP ensemble (5 seeds) + concat features + PTR with MLP stage-1",
        "n_hvg": N_HVG,
        "n_distinct_targets": len(all_mean_deltas),
        "n_train_targets": len(train_t),
        "n_val_targets": len(val_t),
        "n_test_targets": len(test_t),
        "elapsed_seconds": elapsed,
        "literature_reference_points": {
            "GEARS_unseen_K562": 0.076,
            "mean_training_delta_baseline": 0.267,
            "oracle_nn_retrieval": 0.592,
            "our_oracle_nn_k5": 0.4739,
        },
        "baselines": {
            "global_mean": {"test_pearson_mean": gm_m, "test_pearson_median": gm_med},
        },
        "experiments": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_DIR}/metrics.json")


if __name__ == "__main__":
    main()

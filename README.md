# VirtualCell

A research project trying to predict what happens to gene expression in a cell
when you turn off a specific gene using CRISPR.

---

## What this project does

When scientists knock out a gene in a cell (using CRISPRi), the whole cell
reacts — thousands of other genes change their expression levels. This project
tries to build a model that can **predict that reaction for genes it has never
seen knocked out before**.

The harder version of this problem (which is what we work on here) is:
> "Given only the name/sequence of a gene, can you predict what the cell will
> look like after you knock it out?"

This is hard because you have to generalize to *unseen* target genes, not just
memorize training examples.

---

## Dataset

We use the **Replogle et al. 2022 K562 essential gene screen**
([figshare](https://doi.org/10.25452/figshare.plus.20029387)):

- Cell line: K562 (human leukemia)
- Perturbation method: CRISPRi (knockdown, not knockout)
- ~2,057 target genes, ~10 GB of single-cell RNA-seq data
- We predict the **mean expression delta** (perturbed − control) per gene

The evaluation split holds out 309 target genes (15%) that the model never
sees during training. Metric: **Pearson correlation** between predicted and
actual mean delta, averaged across those 309 test genes.

---

## Results summary

| Approach | Test Pearson | Notes |
|---|---|---|
| Global mean baseline | 0.240 | Predict the same average response for every gene |
| Literature GEARS | 0.267 | Published baseline from Roohani et al. 2023 |
| **This project (leak-free best)** | **0.3292** | v29: same as v25, bulk pre-training restricted to train genes |
| This project (prior best, leaked) | ~~0.3557~~ | v25: test-gene bulk responses leaked into pre-training |
| Oracle upper bound | 0.474 | Train and test on the same gene (not generalizable) |

Our leak-free best beats GEARS by **23%**. An earlier number (0.3557) was overstated
because bulk perturbation signatures from held-out test genes were visible during
pre-training. The corrected number uses identical architecture and split, with
test/val gene signatures excluded from pre-training supervision.

---

## How the model works (plain English)

1. **Represent each gene** as a fixed vector using:
   - Protein sequence embedding (ESM2, 320 dimensions)
   - Gene2Vec co-expression embedding (200 dimensions)
   - Network-propagated versions of both (via STRING PPI network)
   - Total: 1040-dimensional gene fingerprint

2. **Pre-train** a small neural network to predict functional similarity
   between gene pairs (using STRING interaction scores + bulk expression
   similarity as supervision). This teaches the network what "genes that do
   similar things" looks like.

3. **Fine-tune** the same network to predict mean expression delta from the
   gene fingerprint. Only 1,439 training genes are available.

4. **Evaluate** on 309 held-out test genes the model has never seen.

The encoder is a 4-layer MLP (1040→2048→1024→512→256) with LayerNorm and
GELU activations, followed by a 2-layer prediction head (256→512→output).
We train 5 random seeds and report the ensemble mean.

**Evaluation note (v29):** v25 had a pre-training data leak: bulk perturbation
signatures were loaded before the train/test split, so test-gene bulk responses
contributed to pre-training signal. v29 fixes this by filtering pre-training
to train genes only. The honest leak-free result is **0.3292** (v25 reported
0.3557, which is now retired). The score drop also reflects a simultaneous
STRING normalization fix, so contributions cannot be separated without
further ablations.

---

## Full experiment history (35 milestones)

### Phase 1 — Architecture search with Arc data (Milestones 1–9)

We started on a different dataset (Arc Institute Perturb-seq) with a
per-cell prediction formulation before realizing the fundamental setup was wrong.

| What we tried | Result |
|---|---|
| PCA + ridge regression baseline | Pearson ~0.05 — essentially random |
| Small MLP on control-cell state + target gene descriptor | No better than ridge |
| Co-expression profile as target descriptor | No gain |
| Gene2Vec pretrained embeddings | No gain |
| Causal descriptor (effect direction from literature) | No gain |
| GEARS-lite (GO-similarity effect retrieval) | No gain |
| Scaling up to 100k+ cells | Did not change results |
| Many distinct target genes (more training variety) | No gain |

**Key pivot:** Per-cell prediction (predict individual cell state) is a much
harder task. We switched to predicting the **mean** expression delta per gene —
the average response across all perturbed cells. This is what the GEARS paper
evaluates on.

### Phase 2 — Gene-level prediction, finding the right setup (Milestones 10–17)

Switched to the Replogle K562 essential dataset and gene-level mean-delta
prediction. Spent several milestones getting the evaluation metric right.

| What we tried | Result |
|---|---|
| Mean delta prediction, wrong normalization | Low numbers |
| log1p normalization (like GEARS) | **Big jump** — this was the missing piece |
| 5000 HVGs instead of all genes | No change |
| Concatenating multiple descriptor types | No synergy |
| NN retrieval (find similar training genes, use their response) | Weak |
| Per-cell response_role (secondary PCA representation) | Small gain |

**Checkpoint:** Got Pearson ~0.27, matching GEARS literature baseline.

### Phase 3 — Feature engineering, hitting the ceiling (Milestones 18–35)

| What we tried | Pearson | vs best |
|---|---|---|
| ESM2-8M protein embeddings (v18) | 0.323 | — new best |
| Multi-hop STRING GNN (v19) | 0.318 | worse |
| Episodic meta-learning (v20) | 0.316 | worse |
| Bulk pre-training on STRING pairs (v21) | 0.329 | new best |
| ESM2 + Gene2Vec + STRING + bulk combo (v22) | 0.343 | new best |
| Deeper encoder + SimCLR NT-Xent loss (v23) | 0.353 | new best |
| Wider encoder + longer pre-training (v24) | 0.354 | new best |
| **Expanded STRING pre-training — v25** | **0.3557** | **best ever** |
| 2-hop STRING propagation (v26) | 0.3557 | no gain |
| ESM2-650M 1280d embeddings (v27) | 0.346 | **worse −0.009** |
| GWPS genome-wide bulk pre-training (v28) | 0.348 | **worse −0.008** |
| **Leak-free rerun of v25 (v29)** | **0.3292** | honest number; test-gene bulk excluded from pre-training |

**Things that didn't move the needle at all once we hit v25:**
- Deeper or wider networks
- Longer pre-training (200 → 800 epochs)
- Different loss functions (cosine, combo, NT-Xent)
- Larger protein embeddings (ESM2-650M)
- 2× more STRING graph hops
- 4.7× more pre-training genes (GWPS bulk, 9,474 genes)

**Hypothesis:** The ceiling at **~0.3557** may be caused by having only 1,439
fine-tuning genes. No embedding or architecture improvement has moved it, which
is consistent with a data-size limit, but this is not definitively established —
a qualitatively different architecture or objective could also break through.
The gap to oracle (0.474) likely requires either a larger perturbation screen or
a fundamentally different approach.

---

## Project layout

```
scripts/
  replogle_genelevel_v*.py   — versioned experiment scripts (v2–v28)
  embed_esm2_650M.py         — re-embed genes with ESM2-650M
docs/
  experiment_log.md          — full notes on every milestone (1–35)
  results.md                 — summary results table
  dataset_card.md            — data provenance and preprocessing
data/                        — gitignored; large files live on EC2
  replogle/                  — K562 essential single-cell h5ad
  replogle_k562_gwps/        — K562 genome-wide bulk h5ad
  esm2/                      — ESM2 embedding caches (npz)
  gene2vec/                  — Gene2Vec embeddings
  string/                    — STRING PPI network
outputs/                     — gitignored; metrics.json per experiment
src/virtualcell/             — original library scaffold (Milestones 1–9)
```

---

## Reproducing the best result (v29)

**Requirements:** Python 3.11+, ~12 GB RAM, the data files downloaded to
`data/` (see `docs/dataset_card.md`).

v29 depends on a preprocessed cache from v10. Run these in order:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
# Step 1: build the v10 mean-delta cache (required by v29)
python -u scripts/replogle_genelevel_v10.py
# → outputs/replogle_genelevel_v10/mean_delta_cache.npz
# → outputs/replogle_genelevel_v10/mean_delta_cache_genes.json
# Step 2: run v29 (leak-free)
python -u scripts/replogle_genelevel_v29.py
# → outputs/replogle_genelevel_v29/metrics.json
```

The scripts download nothing — all data must be pre-staged. v10 takes ~25
minutes and v29 takes ~40 minutes on a CPU-only machine (m7i-flex.large or
equivalent).

---

## Interactive visualization

A live 3D K562 cell with CRISPRi knockdown simulation:
**[https://claude.ai/artifact/Nr1d5BcaGWY7yen7CSYyRX](https://claude.ai/artifact/Nr1d5BcaGWY7yen7CSYyRX)**

Hover a gene node (green dots) to see its name. Click a node to simulate
a knockdown — blue nodes go up-regulated, orange nodes go down-regulated.
The cell auto-rotates; drag to orbit. Built with Three.js (WebGL).

---

## Infrastructure

Experiments were run on AWS EC2 (`m7i-flex.large`, 8 vCPU / 8 GB RAM).
The instance is stopped between runs. No GPU was used.

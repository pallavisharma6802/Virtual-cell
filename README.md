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
| **This project (best)** | **0.3557** | v25: ESM2-8M + Gene2Vec + STRING pre-training |
| Oracle upper bound | 0.474 | Train and test on the same gene (not generalizable) |

Our best model beats GEARS by **33%** without any deep GEARS-style graph neural
network — just gene embeddings + a small MLP with pre-training on protein
interaction networks.

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

The architecture is a 4-layer MLP (1040→512→256→128→64→output) with
LayerNorm, GELU activations, and dropout. We train 5 random seeds and report
the ensemble mean.

---

## What we tried and what happened

Over 35 milestones, every major improvement avenue was tested:

| What we tried | Result |
|---|---|
| Deeper / wider networks | No gain past v14 |
| Longer pre-training (200→800 epochs) | Marginal gain, plateau at 800 |
| Different loss functions (cosine, combo) | No gain |
| ESM2-650M (1280d) instead of ESM2-8M (320d) | **Worse** (−0.009) |
| 2-hop STRING propagation | **Worse** (−0.009) |
| Expanding STRING to all pairs (not just bulk-covered) | Slight gain |
| GWPS bulk pre-training (9,474 genes vs 2,025) | **Worse** (−0.008) |

**Conclusion:** The ceiling at ~0.3557 is caused by having only 1,439
fine-tuning genes. No embedding or architecture improvement can substitute
for more direct training data. The gap to oracle (0.474) requires either a
larger perturbation screen or a fundamentally different approach.

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

## Reproducing the best result (v25)

**Requirements:** Python 3.11+, ~12 GB RAM, the data files downloaded to
`data/` (see `docs/dataset_card.md`).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python -u scripts/replogle_genelevel_v25.py
# → outputs/replogle_genelevel_v25/metrics.json
```

The script downloads nothing — all data must be pre-staged. It takes ~40
minutes on a CPU-only machine (m7i-flex.large or equivalent).

---

## Infrastructure

Experiments were run on AWS EC2 (`m7i-flex.large`, 8 vCPU / 8 GB RAM).
The instance is stopped between runs. No GPU was used.

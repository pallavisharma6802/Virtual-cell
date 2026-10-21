# Results — Milestone 4 (synthetic fixture, research-grade evaluation)

**Scope:** All results below are on the deterministic synthetic fixture
(`make_synthetic_dataset(seed=0)`, 540 cells, 200 genes, 24 perturbation
targets, held-out-target split: 14 train / 5 val / 5 test targets). **No
real biological dataset has been trained or evaluated on yet.** This is a
pipeline/sanity-check result, not a scientific finding about real gene
perturbation response.

- **Run IDs:** training run `outputs/conditional_mlp/full` (Milestone 3,
  commit `c552bc9`); evaluation run `outputs/evaluation_milestone4/`
  (this milestone).
- **Model:** `ConditionalDeltaModel`, full variant (control context +
  control-mean/std target descriptor), seed 0, checkpoint reloaded from
  disk for this evaluation (not re-trained).
- **Test set:** 100 cells across 5 held-out targets (20 cells/target),
  genuinely disjoint from the 14 training targets (verified by
  `tests/test_splits.py`).

## Per-target metrics (held-out test targets)

| target | n cells | MAE (scaled) | RMSE (scaled) | Pearson | Spearman |
|---|---|---|---|---|---|
| gene_0032 | 20 | 0.349 | 0.619 | 0.502 | 0.659 |
| gene_0105 | 20 | 0.355 | 0.526 | 0.314 | 0.398 |
| gene_0109 | 20 | 0.342 | 0.556 | 0.396 | 0.561 |
| gene_0150 | 20 | 0.286 | 0.504 | 0.419 | 0.548 |
| gene_0171 | 20 | 0.358 | 0.550 | 0.335 | 0.459 |

**Grouped bootstrap CI on per-target MAE** (resampling over the 5
targets, 1000 resamples, 95% CI): mean 0.338, CI [0.311, 0.355]. With only
5 held-out targets this CI is necessarily wide/under-powered — reported
honestly as a limitation, not smoothed over.

Note: these per-target MAE values (computed on each target's *mean*
delta profile) are lower than the pooled per-cell MAE reported in the
Milestone 3 log (0.880) because averaging over 20 replicate cells per
target cancels cell-level noise — both numbers are correct, they answer
different questions (plan section 5: "per-target metrics... not just
pooled averages").

## Per-batch breakdown (context/confounding check)

| batch | n cells | MAE | RMSE |
|---|---|---|---|
| batch_0 | 35 | 0.879 | 1.122 |
| batch_1 | 32 | 0.868 | 1.112 |
| batch_2 | 33 | 0.892 | 1.137 |

MAE is close across batches (0.868–0.892), i.e. no strong batch
confounding detected on this fixture — expected, since the synthetic
generator applies batch shift symmetrically and the control-sampling
procedure is batch-matched.

## Worst 3 targets (failure-case review)

1. **gene_0171** — MAE 0.358, Pearson 0.335 (lowest correlation of the 5).
2. **gene_0105** — MAE 0.355, Pearson 0.314 (lowest of all 5 targets).
3. **gene_0032** — MAE 0.349, but *highest* Pearson (0.502) of the group —
   high absolute error with comparatively well-ranked direction, i.e. a
   systematic scale/magnitude miss rather than a directionally wrong
   prediction. See `outputs/evaluation_milestone4/observed_vs_predicted_worst_target.png`
   for the scatter plot of true vs. predicted mean delta for gene_0171
   (the single worst target by MAE).

No obvious pattern links these three to a specific batch or a low
replicate count (all three have the full 20 cells); on this fixture the
per-target effect vectors are independently drawn random sparse vectors,
so some targets having weaker, less-structured true effects (closer to
noise) is the most likely explanation — not a flaw specific to these
gene names, which carry no real biological meaning here.

## Differentially-expressed gene overlap (top-20 predicted vs. t-test DE set)

DE genes defined by Welch's t-test (perturbed-target cells vs. the pooled
train-control cells) with Benjamini-Hochberg FDR correction at α=0.05.

| target | n DE genes (BH, α=0.05) | precision@20 | recall@20 |
|---|---|---|---|
| gene_0032 | 19 | 0.05 | 0.053 |
| gene_0105 | 19 | 0.05 | 0.053 |
| gene_0109 | 21 | 0.05 | 0.048 |
| gene_0150 | 10 | 0.00 | 0.000 |
| gene_0171 | 15 | 0.00 | 0.000 |

**This is a clear negative/weak result, reported as such:** the model's
top-20-by-|predicted delta| genes barely overlap with the statistically
defined DE gene set (precision/recall ≈ 0.00–0.05, far below what even a
modest ranking signal would show). Combined with the Milestone 3 finding
that the `ablation_no_target` variant performed comparably to the full
model, this suggests the model is capturing *some* aggregate
magnitude/direction signal (positive MAE improvement over no-effect,
positive Pearson correlations) without correctly identifying *which*
individual genes are most affected. That is a meaningful limitation of
this specific weak target descriptor + small MLP, on this fixture — not
evidence about real perturbation data, and not something to paper over.

## Conclusions (synthetic fixture only)

- **H1 (conditional model beats no-effect baseline on held-out targets):**
  weakly supported on this fixture — 18.9% pooled MAE improvement
  (Milestone 3), positive but modest per-target Pearson/Spearman
  correlations (0.31–0.50).
- **H2 (control-conditioning improves over perturbation-only input):**
  supported — `ablation_no_control` collapsed to ~no-effect-baseline
  performance (Milestone 3), confirming control context materially
  contributes.
- **H3 (persists across a harder context split):** not yet tested — no
  secondary batch/donor split has been built or evaluated.
- **Gene-level identification of DE genes:** not supported by this
  model/descriptor on this fixture (near-chance precision/recall@20).

## Limitations

- Entirely synthetic-fixture based; no real single-cell data used.
- Only 5 held-out test targets — any CI or per-target claim here is
  necessarily low-powered; this is a pipeline correctness check, not a
  statistically powered scientific result.
- Target descriptor is a deliberately weak 2-scalar control mean/std, not
  a real gene-representation model — the DE-overlap failure likely
  reflects this descriptor's limited information content, not a
  fundamental property of the conditional-modeling approach.
- No H3 (context/batch-holdout) split has been run.
- No real dataset (Arc Virtual Cell Challenge or scPerturb) has been
  trained or evaluated on. Milestone 2 verified access/schema and built a
  working adapter; integrating it into training/evaluation is future
  work, not claimed here.

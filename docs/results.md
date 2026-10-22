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

## Full baseline ladder (added after owner request — plan section 4A order)

The plan specifies building baselines in order before the conditional
model: no-effect → seen-target mean → PCA+ridge → small MLP. The first
pass of this project only built the first two and jumped straight to the
conditional model, skipping PCA+ridge and a standalone small-MLP
baseline — an antipattern the plan itself warns against (section 10:
"build a giant transformer before baselines are reproducible"). This was
corrected: `scripts/compare_baseline_ladder.py` runs all five models on
the identical held-out-target split, scaler, target descriptor, and
matched-control sampling.

| model | test MAE (scaled) | improvement vs no-effect |
|---|---|---|
| 1. No-effect | 1.084 | 0.0% |
| 2. Seen-target mean | 1.084 | 0.0% (falls back to zero for every held-out target, by design) |
| 3. PCA + ridge (20 components) | 0.921 | +15.1% |
| 4. **Small MLP** (1 hidden layer, same inputs) | **0.874** | **+19.4%** |
| 5. Conditional model (control + target encoders + fusion) | 0.880 | +18.9% |

**Honest finding: the plain small MLP baseline slightly outperforms the
conditional model** on this synthetic fixture (0.874 vs. 0.880 MAE — a
small gap, within noise, but the conditional model does NOT clearly beat
the simpler baseline it was meant to improve on). PCA+ridge is weaker
than both MLP variants but still clearly ahead of the no-effect/seen-mean
baselines.

**What this means:** on this fixture, the conditional model's extra
structure (separate control/target encoders + fusion block) is not
earning its added complexity over a single-hidden-layer MLP given the
same two input blocks (control profile, 2-scalar target descriptor). This
is a legitimate, reportable negative result for the architecture choice,
not a flaw in the baselines — and it is consistent with Milestone 4's
finding that the `ablation_no_target` variant performed similarly to the
full conditional model: the weak 2-scalar target descriptor may simply
not carry enough information for the extra encoder/fusion machinery to
exploit, regardless of which model wraps around it. A richer target
descriptor (plan section 4A's suggested pretrained gene representation,
not yet implemented) is the more likely lever for separating these models
meaningfully, rather than further architecture changes to the conditional
model on top of this same weak descriptor.

## Richer target descriptor: co-expression profile (added after owner request)

Per the plan's suggestion (section 4A) that a richer, generalizable
target descriptor is the likely lever for improving held-out-target
prediction, a second descriptor was added:
`fit_coexpression_descriptors` — each gene's Pearson correlation with
every other gene across TRAIN control cells, PCA-reduced to 20
components. This is still hand-built (not a pretrained embedding), but
captures real network/co-expression structure rather than two scalars.
Fit strictly on train controls only, so it is non-leaking, same as the
weak descriptor.

| model | weak descriptor (mean/std, 2-dim) MAE | improvement | co-expression descriptor (20-dim) MAE | improvement |
|---|---|---|---|---|
| PCA + ridge | 0.921 | +15.1% | **1.070** | **+1.3%** |
| Small MLP | 0.874 | +19.4% | **0.918** | **+15.3%** |
| Conditional model | 0.880 | +18.9% | 0.883 | +18.6% |

**Honest, somewhat surprising finding: the richer descriptor made things
*worse* for the simpler baselines, and essentially flat for the
conditional model.** The most plausible explanation, not papered over:
this fixture only has 14 training targets. A 20-dimensional descriptor
gives PCA+ridge and the small MLP far more parameters to fit against far
fewer effective training examples (grouped by target) than the 2-scalar
descriptor did, so they likely overfit the correlation-structure noise
rather than learning a genuinely better signal. The conditional model's
separate target encoder (which compresses the 20-dim input through its
own small hidden layer before fusion) appears mildly more robust to this,
but shows no real net improvement over its own weak-descriptor result
(0.880 -> 0.883, within noise) — i.e. **this richer descriptor did not
demonstrably help on this fixture**, which is itself a negative,
reportable result, not evidence that richer descriptors are a dead end
in general. This fixture's small target count (24 total) is likely the
binding constraint, not the descriptor concept — a real dataset with
hundreds of training targets (like the actual Arc data, which has 150
targets total) would be a fairer test of whether this kind of descriptor
helps, and was not re-tested here.

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
- The conditional model does not clearly beat a plain small MLP baseline
  given the same inputs (see "Full baseline ladder" above) — the
  architecture's value is unproven on this fixture with this descriptor.
- No H3 (context/batch-holdout) split has been run.
- No real dataset (Arc Virtual Cell Challenge or scPerturb) has been
  trained or evaluated on. Milestone 2 verified access/schema and built a
  working adapter; integrating it into training/evaluation is future
  work, not claimed here.

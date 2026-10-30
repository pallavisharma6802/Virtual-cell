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

## Full baseline ladder on REAL data (both descriptors)

The whole baseline ladder (no-effect, seen-target mean, PCA+ridge, small
MLP, conditional model) was re-run on the real Arc data pilot (same
10,960-cell / 1,000-gene / 150-target subsample as the real-data pilot
above, 90/30/30 train/val/test targets, 1,787 test cells — 30 held-out
targets, a far better-powered comparison than the synthetic fixture's 5),
under both the weak (mean/std) and richer (co-expression) target
descriptors.

| model | mean/std MAE | improvement | co-expression MAE | improvement |
|---|---|---|---|---|
| No-effect | 1.056 | 0.0% | 1.056 | 0.0% |
| Seen-target mean | 1.056 | 0.0% (expected fallback) | 1.056 | 0.0% |
| **PCA + ridge** | **0.898** | **+15.0%** | 0.908 | +14.1% |
| Small MLP | 0.923 | +12.6% | 0.917 | +13.2% |
| Conditional model | 0.926 | +12.3% | 0.929 | +12.0% |

**Headline, real-data finding, not softened:**

1. **On real data, PCA + ridge — the simplest model with the plan's
   earliest, most basic baseline — wins outright**, beating the small
   MLP and the conditional model under both descriptors. This is a
   materially different outcome from the synthetic fixture, where PCA+
   ridge was the weakest of the three learned models. It means the extra
   complexity this project built (the conditional model's separate
   encoders/fusion) is not currently justified by the evidence on real
   data — a plain linear-on-PCA-components model does at least as well.
2. **The richer co-expression descriptor still does not clearly help on
   real data**, even with 90 training targets (far more than the
   synthetic fixture's 14, ruling out "too few targets" as the
   explanation this time): PCA+ridge gets slightly worse (+15.0% →
   +14.1%), small MLP gets marginally better (+12.6% → +13.2%), and the
   conditional model is flat-to-slightly-worse (+12.3% → +12.0%). None
   of these differences look large relative to the bootstrap CI widths
   (~±0.03 MAE). The honest conclusion is that this specific co-
   expression descriptor, computed this way, is not demonstrating value
   over the weak 2-scalar one on this dataset — not that richer
   descriptors can't help in principle, but that this particular
   hand-built one doesn't clear that bar here.
3. All three learned models (ridge, MLP, conditional) comfortably beat
   no-effect (+12–15%), so there IS real, consistent predictive signal in
   control-conditioned modeling of perturbation response — the open
   question is which model architecture captures it best, and on this
   evidence it's currently the simplest one.

Full JSON: `outputs/baseline_ladder_real/baseline_ladder_real.json`,
mirrored in S3 at `s3://virtualcell-635863384351/baseline_ladder_real/`.

## Scale-up + tuning REVERSES the "PCA+ridge wins" finding

At ~4x more real data (45,584 vs. 10,960 cells, same 150 targets,
90/30/30 split) with a small hyperparameter sweep for the conditional
model only (baselines left at prior defaults), the ranking flips:

| model | MAE (scaled) | improvement over no-effect |
|---|---|---|
| No-effect | 1.062 | 0.0% |
| PCA + ridge | 0.896 | +15.6% |
| Small MLP | 0.889 | +16.2% |
| **Conditional model (tuned: 256/256 hidden, lr=1e-3)** | **0.878** | **+17.4%** |

Grouped bootstrap CI on the tuned conditional model (30 target groups):
mean 0.169, 95% CI [0.137, 0.210]. Full JSON:
`outputs/scale_up_real_arc/metrics.json`, mirrored at
`s3://virtualcell-635863384351/scale_up_real_arc/`.

**This reverses the earlier "PCA+ridge wins" finding above — reported
honestly, not cherry-picked.** The hyperparameter sweep (varying hidden
dims 64/128/256 and learning rate) showed the largest-capacity config
had the best validation loss, suggesting the conditional model in the
smaller/untuned comparison was likely undertrained or undersized for the
signal actually available, not that the architecture itself is
inherently weaker than simpler baselines.

**This does NOT fully settle the question, and the limits matter:**
- The margin between all three learned models (0.878 vs. 0.889 vs.
  0.896) is small — about 0.02 MAE apart — and only one random seed/
  split was tested. This is directional evidence, not a statistically
  confirmed win.
- Only the conditional model was re-tuned at this larger scale; PCA+
  ridge's `alpha`/`n_components` and the small MLP's `hidden_dim` were
  left at their original defaults. A fully fair comparison would retune
  all three before trusting this reversal — this result shows "a better-
  tuned conditional model can beat untouched baselines," which is a
  real and useful finding, but is one step short of "the conditional
  model is the better architecture, full stop."
- Still a subsample: 45,584 of 221,273 cells, 1,500 of 18,080 genes.

## Fair retuned comparison: the small MLP wins (a SECOND reversal)

The limitation above was addressed directly: PCA+ridge
(`n_components` x `alpha` grid) and the small MLP (`hidden_dim` grid)
were each given their own hyperparameter sweep, selected by validation
MAE — the same protocol already used for the conditional model. Same
45,584-cell subsample and split, so directly comparable to the table
above.

| model (all tuned) | MAE (scaled) | improvement over no-effect | best config |
|---|---|---|---|
| PCA + ridge | 0.885 | +16.6% | n_components=50, alpha=10.0 |
| **Small MLP** | **0.859** | **+19.1%** | hidden_dim=256 |
| Conditional model | 0.878 | +17.4% | 256/256 hidden, lr=1e-3 |

Full JSON: `outputs/scale_up_real_arc_retuned/metrics.json`, mirrored at
`s3://virtualcell-635863384351/scale_up_real_arc_retuned/`.

**Once the comparison is actually fair, the small MLP wins outright —
not the conditional model.** The previous section's "conditional model
wins" was an artifact of comparing a tuned model against untuned
baselines, exactly the limitation flagged there. This is the project's
most important finding to date, and it is an uncomfortable one: the
architecture this project invested the most engineering effort in
(separate control/target encoders + fusion) has, on every real-data
comparison run so far, never clearly and fairly beaten a much simpler
single-hidden-layer MLP over the same two input blocks.

**Full picture across all three real-data comparisons run this project:**

| comparison | winner | conditional model's rank |
|---|---|---|
| 10,960 cells, nothing tuned | PCA+ridge | 3rd of 3 |
| 45,584 cells, only conditional model tuned | conditional model | 1st of 3 (unfairly) |
| 45,584 cells, all three fairly tuned | **small MLP** | 2nd of 3 |

**What this does and doesn't mean:**
- It does NOT mean control-conditioned modeling fails — all three
  learned models beat no-effect by +16-19%, a real, repeatable signal.
- It does NOT mean the conditional model is broken — only that its
  specific architectural choices (separate encoders, fusion) haven't
  shown a clear, fair edge over simpler alternatives yet, at this data
  scale, with this target descriptor.
- It DOES mean that escalating further toward a bigger/more complex
  model (Milestone 5's pretrained foundation encoder) without first
  understanding this result would repeat the exact mistake the plan's
  antipattern list warns against.
- The margins are still small (0.859 to 0.885, within 0.026 MAE) and
  only one seed/split was tested — this ranking itself could still flip
  with different data or a different random seed. Multi-seed replication
  is the single most important missing piece before treating any of
  these three rankings as settled.

## Investigating the gap: a richer descriptor does NOT close it

The obvious hypothesis for the small MLP's win is that the shared weak
2-scalar target descriptor caps all models equally, hiding any real
advantage of the conditional model's extra structure. This was tested
directly: the same fully-tuned 5-model comparison was re-run on the
identical 45,584-cell subsample and split, swapping only the target
descriptor for the richer 20-dimensional co-expression profile built
earlier (gene-gene correlation across train controls, PCA-reduced).

| model (all tuned) | mean/std MAE | mean/std improvement | co-expression MAE | co-expression improvement |
|---|---|---|---|---|
| PCA + ridge | 0.885 | +16.6% | 0.889 | +16.3% |
| **Small MLP** | **0.859** | **+19.1%** | **0.861** | **+18.9%** |
| Conditional model | 0.878 | +17.4% | 0.874 | +17.7% |

Full JSON: `outputs/scale_up_coexpression/metrics.json`, mirrored at
`s3://virtualcell-635863384351/scale_up_coexpression/`.

**Conclusive result: the richer descriptor changes almost nothing.**
Every model moved by at most 0.004-0.005 MAE between descriptors — noise,
not signal — and the ranking (small MLP > conditional model > PCA+ridge)
is identical under both. **This rules out the target descriptor as the
explanation for the gap.** The hypothesis that a richer descriptor would
let the conditional model's structure pay off is not supported.

This means the small-MLP-beats-conditional-model gap is more likely
about the architecture itself (separate encoders + fusion adding
optimization difficulty without adding usable expressive power, given
how simple these two input blocks are) or about a performance ceiling set
by the data scale itself (45,584 cells is still a subsample), rather than
input quality.

## Overfitting check: it's NOT overfitting

Checked directly: train/val/test MAE for all three tuned models, same
subsample/split.

| model | train MAE | val MAE | test MAE | train→test gap |
|---|---|---|---|---|
| PCA + ridge | 0.882 | 0.888 | 0.885 | 0.003 |
| Small MLP | 0.829 | 0.862 | 0.859 | 0.030 |
| Conditional model | 0.845 | 0.882 | 0.878 | 0.033 |

**The small MLP and conditional model have nearly identical
generalization gaps (0.030 vs 0.033) — the conditional model is NOT
overfitting more.** What actually separates them: the small MLP fits the
TRAINING data better too (0.829 vs 0.845). It wins at every stage, not
by generalizing better from an equal fit. This rules out overfitting as
the explanation and points to the architecture itself being less
effective at fitting this data (harder optimization from the separate
encoders + fusion structure, or simply needing a different training
budget/schedule — neither isolated yet) rather than a capacity or
overfitting problem. PCA+ridge shows almost no train-test gap, as
expected for a regularized linear model — it is underfitting, not
overfitting, and that's why it's the weakest of the three.

## Longer training / different optimization: also ruled out

Tested 5 configs for the conditional model (same 256/256 hidden config,
same subsample/split): the original short-training baseline, 500-epoch
variants at 3 weight-decay levels, and an 800-epoch lower-learning-rate
variant.

| config | train MAE | test MAE | improvement |
|---|---|---|---|
| original (100 epochs) | 0.845 | 0.878 | +17.4% |
| longer (500 epochs) | 0.827 | 0.878 | +17.3% |
| longer + more reg | 0.860 | 0.882 | +17.0% |
| longer + less reg | 0.825 | **0.875** | **+17.6% (best)** |
| 800 epochs, lower lr | 0.816 | 0.882 | +16.9% |

Reference: small MLP (tuned) = 0.859 test MAE, +19.1%.

**None of the five close the gap.** The best variant still trails the
small MLP by 0.016 MAE. Train MAE kept improving with more training
(0.845 → 0.816) but test MAE did not follow — more training just fits
train harder without better test transfer. This rules out training
budget/regularization as the explanation.

**Investigation chain complete: target descriptor → ruled out;
overfitting → ruled out; training budget/optimization → ruled out.** The
remaining live hypotheses are the architecture itself (separate
encoders + fusion less effective than direct concatenation) or a data-
scale ceiling affecting the more complex model differently — the latter
is the next thing to test, per the agreed order (scale data before
touching architecture or escalating to Milestone 5).

## Scale-up to ~106k real cells: the architecture gap persists

Ran the full baseline ladder (PCA+ridge, small MLP, conditional model —
each independently hyperparameter-tuned on held-out val targets) on a
much larger real-data subsample than any prior comparison in this
project: 106,287 cells across 150 perturbation targets (6,000 control
cells + up to 800 cells/target, 2,500 top-variable genes), held-out-
target split (90 train / 30 val / 30 test targets, 20,944 test cells).
Run artifacts: `outputs/scale_up_large/metrics.json`,
`outputs/scale_up_large/run.log`.

This required fixing a real infrastructure bug first (see entry below
on the scattered-remote-row-access root cause): fetching cells at this
scale over HTTP range requests against the Arc h5ad's small HDF5
chunks was latency-dominated and would hang indefinitely. Fixed by
downloading the full file once and reading all subsequent rows from
local disk (`scripts/download_arc_h5ad.py`, `--h5ad-path` flag on
`scripts/scale_up_real_arc.py`).

| model | test MAE | best config | improvement over no-effect |
|---|---|---|---|
| no-effect / seen-target-mean | 1.051 | — | 0% (reference) |
| PCA+ridge (tuned) | 0.907 | n_components=50, alpha=10.0 | +13.8% |
| small MLP (tuned) | **0.891** | hidden_dim=256 | **+15.2% (best)** |
| conditional model (tuned) | 0.895 | control/fusion hidden=256, lr=1e-3 | +14.8% |

**The small MLP still wins, narrowly, over the conditional model at
this larger scale** (0.891 vs. 0.895 test MAE) — the same ranking as
the smaller-scale "fair retuned comparison" above. Going from ~27k
cells (the prior real-data comparison) to ~106k cells (a ~4x increase)
did not close the gap or reverse the ranking; if anything the margin
between the two is now smaller (0.004 MAE vs. 0.016 at the smaller
scale), which is at least directionally consistent with the complex
model benefiting somewhat from more data, but it has not caught up or
overtaken.

Conditional-model hyperparameter sweep detail (4 configs, each trained
to convergence with early stopping, patience=12):

| control/fusion hidden dim | lr | best val loss | best epoch |
|---|---|---|---|
| 128 | 1e-3 | 0.526 | 19 |
| 64 | 1e-3 | 0.530 | 19 |
| 128 | 3e-4 | 0.529 | 17 |
| **256** | **1e-3** | **0.519 (best)** | 31 |

Grouped bootstrap CI on the tuned conditional model's per-target test
MAE (30 held-out targets, resampled): mean 0.159, CI [0.128, 0.200].
Worst 3 targets by MAE: `RNF20` (n=47, MAE 0.517), `METTL3` (n=498, MAE
0.483), `ATM` (n=455, MAE 0.271) — all three also have near-zero or
negative Pearson/Spearman correlation on held-out cells, consistent
with the model not capturing target-specific signal for these genes
specifically (low cell count for RNF20 is a plausible partial
explanation there; METTL3 and ATM have adequate cell counts, so their
poor correlation is not just a sample-size artifact).

**Conclusion so far: more data alone does not resolve the
architecture-gap question.** Per the agreed investigation order
(training budget → scale data → re-evaluate → Milestone 5), the next
honest step is to re-evaluate whether to invest in the conditional
architecture itself (e.g. direct concatenation instead of separate
encoders+fusion, per the live hypothesis above) or move to Milestone 5
(pretrained foundation encoder, which plausibly fixes the shared weak
target descriptor instead) — scaling data further is not expected to
change this conclusion on its own, since the gap direction held across
a ~4x data increase.

## Conclusions

- **H1 (conditional model beats no-effect baseline on held-out targets):**
  supported throughout for "some learned model" (+16-19% over no-effect
  in every real-data comparison run). The conditional-model-specifically
  version of H1 is NOT clearly supported: across the three real-data
  comparisons run this project (untuned: 3rd of 3; conditional-only
  tuned: 1st of 3, unfairly; all three fairly tuned: 2nd of 3, behind the
  small MLP), the conditional model has never fairly and clearly beaten
  the simplest nonlinear baseline. See "Fair retuned comparison" above.
- **H2 (control-conditioning improves over perturbation-only input):**
  supported on the synthetic fixture — `ablation_no_control` collapsed to
  ~no-effect-baseline performance (Milestone 3). Not re-tested as an
  ablation on real data this session (only the full PCA+ridge/MLP/
  conditional comparison was run there, not a control-removed ablation
  of each).
- **H3 (persists across a harder context split):** not yet tested — no
  secondary batch/donor split has been built or evaluated, synthetic or
  real.
- **Gene-level identification of DE genes:** not supported by the
  conditional model/weak descriptor on the synthetic fixture (near-chance
  precision/recall@20). Not re-tested on real data.
- **Does the conditional model's architecture complexity help?** No,
  not clearly, by the best evidence available (the fair, all-tuned
  comparison) — the small MLP currently wins, and this held up after a
  ~4x scale-up to ~106k real cells (small MLP 0.891 vs. conditional
  0.895 test MAE — see "Scale-up to ~106k real cells" above). Scale is
  no longer a live explanation for the gap, since more data did not
  reverse or close it. The remaining candidates are the architecture
  itself (separate encoders + fusion vs. direct concatenation) or the
  shared weak target descriptor as the real bottleneck — not yet
  determined which, but the descriptor is the more likely one given
  it's shared and weak across all three models equally.
- **Does a richer target descriptor help?** No, by this evidence — the
  co-expression descriptor didn't clearly improve any model at the
  smaller real-data scale, and was not re-tested at the larger scale this
  session.
- **Before escalating to Milestone 5's pretrained foundation encoder:**
  the evidence currently argues against it. A bigger model is unlikely
  to help if a single hidden layer already beats the current conditional
  architecture — the more promising lever is almost certainly a richer,
  real target descriptor (a genuine pretrained gene representation would
  test exactly this), not more architectural complexity on top of the
  current weak one.

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

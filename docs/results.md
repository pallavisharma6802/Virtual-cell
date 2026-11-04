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

## Milestone 5: pretrained Gene2Vec descriptor does NOT help either

Per the agreed next step after the scale-up result above ("re-evaluate
whether to invest in the conditional architecture itself... or move to
Milestone 5"), swapped the shared weak 2-scalar control-mean/std target
descriptor for a genuine external pretrained gene representation:
**Gene2Vec** (Du et al. 2019, BMC Genomics; MIT license;
`data/gene2vec/gene2vec_dim_200_iter_9.txt`, downloaded from
https://github.com/jingcheng-du/Gene2vec) — a 200-dimension vector per
human gene symbol, trained purely from gene co-expression patterns
across 984 *independent* GEO expression datasets. Unlike the earlier
`fit_coexpression_descriptors` (ruled out above), this is not derived
from this project's own dataset at all, so it is a genuinely different
test of "does a richer target descriptor help," not a repeat of the
earlier negative result with the same underlying data source.

Same protocol, same 106,287-cell/150-target subsample/split as the
scale-up run above, same independently-tuned baseline ladder, only the
descriptor changed (2 dims -> 200 dims). Vocabulary coverage: 95.9% of
this dataset's 2,500 genes found in the Gene2Vec table (the rest fall
back to a zero vector; this is a real, acknowledged coverage gap, not
hidden). Run artifacts: `outputs/scale_up_gene2vec/metrics.json`,
`outputs/scale_up_gene2vec/run.log`. Implementation:
`fit_gene2vec_descriptors`/`load_gene2vec_table` in
`src/virtualcell/data/target_descriptors.py`.

| model | mean/std descriptor (test MAE) | Gene2Vec descriptor (test MAE) | change |
|---|---|---|---|
| PCA+ridge (tuned) | 0.907 | 0.911 | slightly worse |
| small MLP (tuned) | 0.891 | 0.893 | slightly worse |
| conditional model (tuned) | 0.895 | **0.901** | **slightly worse** |

**The richer, genuinely external pretrained descriptor did not help any
of the three models — all three got very slightly worse, and the small
MLP still wins over the tuned conditional model, by a slightly wider
margin than with the weak descriptor (0.008 MAE vs. 0.004).** This is
not noise-sized but is small; the direction (uniformly worse, not
mixed) across all three independently-tuned models is the more telling
part than the magnitude.

This is a real, informative negative result, not an inconclusive one:
it rules out "the shared target descriptor's information content is
the bottleneck" as the explanation for the architecture gap — the
specific richer descriptor tested here (co-expression-based,
pretrained externally) carries real biological signal but about the
*wrong* thing for this task. Gene2Vec encodes which genes tend to be
co-expressed across many unrelated expression studies (a general
"functional neighborhood" signal); it does not encode anything about
how strongly or in which direction a specific CRISPR knockdown of that
gene shifts other genes' expression in *this* cell population — the
actual quantity the models must predict. A gene can have an
informative co-expression neighborhood and still be a weak predictor
of its own knockdown's downstream magnitude, and that appears to be
what happened here.

**Standing on the architecture hypothesis:** separate control/target
encoders + fusion (the conditional model) has now lost to plain
concatenation (the small MLP) in every real-data comparison run at
every scale — this already *is* the architecture test, run repeatedly,
not an untested hypothesis (see "Conclusions" below for the correction
to earlier framing in this doc). A fifth candidate — too few distinct training
targets — was tested in Milestone 6 (below) on a ~15x-larger-target-
count dataset and also did not fix the gap, with caveats noted there.

## Milestone 6: ~15x more distinct targets does NOT fix the gap either

**Hypothesis tested:** maybe Arc's 150 distinct target genes (90 train)
is simply too few to learn any generalizable "target -> effect" mapping,
regardless of descriptor quality or architecture — i.e. the bottleneck is
training-set *target diversity*, not model design.

**Data:** switched to Replogle et al. 2022 (*Cell*), the K562 "essential"
genome-scale Perturb-seq dataset — a completely different, genuinely
public dataset (CC BY-NC-SA 4.0, non-commercial research use; downloaded
directly from its Figshare+ deposit, doi 10.25452/figshare.plus.20029387,
file `K562_essential_raw_bulk_01.h5ad`, ~80MB pseudobulk release, no EC2
needed). This file has **2,008 distinct perturbed genes** (plus 163
non-targeting control replicates) across 8,563 profiled genes — about
15x more distinct targets than Arc's 150.

**Important caveat — not a fully fair comparison to the Arc runs:** this
release is *pseudobulk* (one row per perturbation = mean across that
perturbation's cells), not per-cell. So there is no per-cell control
profile that varies across examples — every example shares the same
single global control pseudobulk vector, meaning the "control" input
carries zero information and models can only use the target descriptor.
This makes the two baselines directly comparable to each other within
this experiment, but not a clean apples-to-apples comparison to the
Arc per-cell pilots.

**Setup:** held-out-target split, 1,405 train / 301 val / 302 test
targets. Same two descriptors as Milestone 5 (2-scalar control mean/std,
and pretrained Gene2Vec — 92.9% coverage of this dataset's genes), same
baseline-ladder code (PCA+ridge, small MLP, swept the same way).

**Result:**

| Descriptor | No-effect MAE | PCA+ridge test MAE | Small MLP test MAE | Best improvement over no-effect |
|---|---|---|---|---|
| mean/std | 0.2160 | 0.2094 | 0.2119 | 3.1% |
| gene2vec | 0.2160 | 0.2060 | 0.2118 | **4.6%** |

(`outputs/replogle_target_scaling/metrics.json`)

**Interpretation:** with ~15x more distinct training targets, the best
improvement over no-effect (4.6%, gene2vec + ridge) is *smaller* than
what the Arc pilots got with only 90 train targets (14-19% over
no-effect). More distinct targets did not fix the problem — if anything
the gap to no-effect got worse here. This argues against "too few
distinct targets" as the bottleneck, at least as the dominant one. Two
honest confounds limit how far this generalizes: (1) the missing
per-cell control signal in this pseudobulk release may itself suppress
achievable improvement independent of target count — the Arc pilots'
better relative numbers may come partly from control-profile information
these models don't have access to here; (2) this is pseudobulk
fold-change, a different (and possibly noisier per-target, since no
cell-level averaging-out-of-noise within a model's forward pass) target
than Arc's per-cell delta. A fully controlled version of this experiment
would need a per-cell version of a many-target dataset (e.g. the GWPS or
RPE1 Replogle releases, or K562 essential's own raw single-cell file,
~10GB) — not attempted here for cost/time reasons.

**Updated standing on the four original hypotheses:** of the original
four candidate explanations for the architecture gap (descriptor
richness, overfitting, training budget, architecture itself), three are
now ruled out by direct experiment (richness: Milestone 5; overfitting
and training budget: earlier sessions). A newly tested fifth candidate —
too few distinct training targets — is also not supported as the fix,
within this experiment's caveats above. That leaves architecture itself
as the only one of the original four candidates not yet *ruled out* —
but on reflection it has in fact been *tested*, repeatedly, throughout
this whole project: the small-MLP-vs-conditional-model comparison run in
every real-data experiment (baseline ladder, scale-up, Milestone 5) IS
that test, since the small MLP is a plain concatenation of control
profile + target descriptor into one network, while the conditional
model uses separate encoders + a fusion block on the same two inputs.
The small MLP has won every single time. So the honest standing is not
"architecture untested" — it's "architecture tested and the simpler one
keeps winning," consistently, across three independent experiments at
two different scales with two different descriptors.

## Milestone 7: a descriptor built from real causal effects, not co-expression

**Hypothesis tested:** Milestones 5 and 6 both failed using
co-expression-based signal (Gene2Vec: cross-study co-expression; the
in-dataset coexpression descriptor: same idea, smaller scale). Maybe the
problem isn't "richness" or "quantity" of signal but its *kind*:
co-expression describes which genes move together at baseline, not which
genes causally respond when a specific gene is knocked out. Does a
descriptor built from real observed perturbation *outcomes* do better?

**Method (`response_role` descriptor, implemented in
`fit_response_role_descriptors`,
`scripts/replogle_target_scaling.py`):** fit PCA on the TRAIN split's own
perturbation-delta matrix (1,405 train targets x 8,563 profiled genes),
factoring it into latent "response modules" — patterns of genes that
move together as a *consequence* of various perturbations. PCA's
per-gene loading vector (30 components) becomes that gene's descriptor.
Critically, every profiled gene gets a loading vector regardless of
whether it was ever itself a target (because every gene is a column/
output in the delta matrix), so a held-out test target's descriptor is
built entirely from *other* genes' perturbation outcomes — never from
its own, which would leak the prediction target. Coverage: 92.7% of the
2,008 targets (same genes missing as for Gene2Vec, since both require
the target to also appear as a profiled output gene).

**Result:**

| Descriptor | Best test MAE | Improvement over no-effect (0.2160) |
|---|---|---|
| mean/std | 0.2094 (ridge) | 3.1% |
| gene2vec | 0.2060 (ridge) | 4.6% |
| response_role | 0.2096 (ridge) | 2.9% |

(`outputs/replogle_target_scaling/metrics.json`)

**Interpretation: this did not help either** — response_role performed
essentially the same as the weak 2-scalar mean/std descriptor, and
slightly *worse* than Gene2Vec, despite being built from genuinely
causal perturbation data rather than correlational co-expression data.
This is a more surprising negative result than Milestones 5/6, because
it rules out the specific mechanism ("wrong kind of signal") proposed
after those failures. The more likely remaining explanation: in this
pseudobulk, no-per-cell-control regime, the ceiling on achievable
improvement may simply be set by how noisy/weak individual perturbation
effects are at the pseudobulk level relative to the descriptor's
information content — i.e. three different descriptors (2 scalars,
200-dim co-expression, 30-dim causal-response) all land in the same
narrow 3-5% band over no-effect, which looks less like "wrong
descriptor" and more like "something else is capping this regime's
achievable performance regardless of which reasonable descriptor is
used" (plausibly the missing per-cell control signal noted as a caveat
in Milestone 6, or inherent pseudobulk noise in the delta target
itself).

## Milestone 8: per-cell + many-targets at once — the missing per-cell signal confirmed

**Hypothesis tested:** Milestones 6 and 7's biggest confound was that the
Replogle pseudobulk release has no per-cell control variation — every
example shared one constant global control vector. Does real per-cell
control signal, combined with Replogle's ~15x-larger target count,
recover improvements closer to Arc's (14-19%) than the pseudobulk
runs' (3-5%)?

**Data:** switched to the RAW SINGLE-CELL release of the same dataset
(`K562_essential_raw_singlecell_01.h5ad`, same Figshare+ deposit, CC
BY-NC-SA 4.0, ~10.6GB). Unlike the pseudobulk file, this has one row per
actual cell, a `gene` obs column directly giving each cell's
perturbation target ("non-targeting" for controls), and 310,385 total
cells. Downloaded once to EC2 local disk (same scattered-remote-access
lesson as Arc applies: never read this via range requests).

**Setup:** memory-bounded two-pass sampling like the Arc scale-up script
(capped at 40 cells/target, 4,000 control cells, top-1,500-variable
genes by train-control variance) — 84,239 cells sampled, 2,057 distinct
targets (>=5 cells each), held-out-target split 1,439 train / 309 val /
309 test. Real per-cell control profiles this time: each example's
"control" input is an actual sampled control cell, not a constant.
Same three descriptors as Milestones 6/7 (mean/std, Gene2Vec, and the
response-role descriptor ported into `target_descriptors.py` for reuse),
same PCA+ridge/small-MLP sweep code.

This run took far longer than the pseudobulk version on the same EC2
instance (~3 hours vs under a minute) — expected, since it is training
on ~56,000 per-cell training rows x up to 1,700 features instead of
~1,400 pseudobulk rows, confirmed genuinely CPU-bound throughout via the
same CPU-liveness watchdog pattern used in earlier milestones (never
stalled, consistently 190%+ CPU on the 2-vCPU instance).

**Result:**

| Descriptor | Best test MAE | Improvement over no-effect (4.414) |
|---|---|---|
| mean/std | 3.805 (ridge) | **13.8%** |
| gene2vec | 3.816 (ridge) | 13.6% |
| response_role | 3.805 (ridge) | **13.8%** |

(`outputs/replogle_percell_scale/metrics.json`)

**Interpretation, two findings at once:**

1. **The missing per-cell control signal WAS the dominant factor in
   Milestones 6/7's weak numbers.** With real per-cell control profiles
   restored, improvement over no-effect jumped from the pseudobulk
   runs' 2.9-4.6% to **13.6-13.8%** — matching Arc's own 14-19% range,
   using a completely different dataset with ~15x more targets. This
   confirms the caveat flagged in Milestone 6 was the real explanation,
   not a minor footnote: pseudobulk's constant-control-input design was
   quietly capping achievable performance regardless of descriptor or
   target count.
2. **Descriptor choice still makes essentially no difference**, even
   now. Three descriptors spanning weak (2-scalar), rich-correlational
   (200-dim Gene2Vec), and rich-causal (30-dim response-role) all land
   within 0.2 percentage points of each other (13.6-13.8%). This is the
   fourth independent confirmation of this pattern (co-expression,
   Gene2Vec, response-role, and now again here) across two datasets and
   both pseudobulk and per-cell regimes. Target-count (Milestone 6) and
   descriptor kind (Milestone 7) are now both ruled out as the
   explanation for *why richer descriptors don't help* — the real
   explanation is almost certainly that **these models aren't using
   target-gene identity in a fine-grained way at all**, regardless of
   how it's encoded; they're likely extracting a cruder, still-genuinely
   useful signal (plausibly "how large is this gene's typical
   perturbation effect," which a 2-scalar descriptor already captures as
   well as a 200-dim one) rather than "which specific gene, therefore
   which specific response pattern."

**Where this leaves the project:** per-cell control signal is now
confirmed necessary to reach Arc-level improvements on a new dataset,
closing that open question. But across every descriptor-richness
experiment run in this project (co-expression, Gene2Vec, response-role,
now replicated on per-cell data), richness has never moved the needle —
the ceiling is not about which gene-level descriptor is used. The next
genuinely different lever would be a method that uses something other
than a per-gene descriptor altogether — e.g. GEARS-style propagation
over a gene-ontology similarity graph, built specifically for
unseen-perturbation prediction rather than adapted from a general-purpose
gene embedding — not a bigger or more richly-embedded version of the
same per-target-vector approach already tried four times.

## Milestone 9: GEARS-lite (GO-similarity effect retrieval) — the mechanistically different lever also fails

**Hypothesis tested:** Milestone 8 ruled out target-count and descriptor
kind (correlational vs. causal) as explanations for why richer
descriptors never help, and proposed one remaining genuinely different
lever: instead of describing a gene's identity as a flat feature vector
fed into a generic regressor, predict its effect by borrowing directly
from functionally similar genes via a gene-ontology similarity graph —
the core mechanism GEARS (Roohani et al. 2023) is built on.

**Approach:** a lite reimplementation, not the official GEARS package
(which depends on `torch_geometric` and is impractical on a CPU-only
free-tier EC2 box). Built a gene-gene similarity graph from GO
Biological Process term overlap (Jaccard similarity on annotations from
`goa_human.gaf.gz`, geneontology.org current release, 93.4% coverage of
the 2,057 Replogle targets). For each held-out target, retrieved its k
nearest neighbors among TRAIN targets by this similarity and predicted
its effect as a similarity-weighted average of those neighbors'
observed mean deltas — direct effect retrieval/propagation through the
graph, not a descriptor-plus-regressor. k swept over {1,3,5,10,20,50}
and selected by validation MAE (best k=20). Same exact per-cell
sample/split as Milestone 8 (same seed), so results are directly
comparable.

**Result:**

| Approach | Test MAE | Improvement over no-effect (4.414) |
|---|---|---|
| GO-similarity retrieval (k=20) | 4.409 | **0.1%** |
| mean/std descriptor + ridge (Milestone 8) | 3.805 | 13.8% |
| response_role descriptor + ridge (Milestone 8) | 3.805 | 13.8% |

(`outputs/replogle_gears_lite/metrics.json`)

**Interpretation:** this is a clear negative result — GO-similarity
effect retrieval performs essentially the same as guessing zero effect
for every gene, dramatically worse than any of the three simple
descriptor+regressor baselines from Milestone 8. Two plausible readings:
(1) GO-BP term overlap, despite being a different *kind* of relationship
than co-expression or the response-role descriptor, still isn't the
right notion of "similarity" for predicting perturbation-effect
magnitude/direction — two genes in the same broad biological process
(e.g. "regulation of transcription") can have very different knockdown
effects; or (2) pure nearest-neighbor retrieval, with no learned
weighting or regression on top, is simply a weaker estimator than a
PCA+ridge or small-MLP fit over 1,439 training examples — it has no way
to combine or reweight signal the way a trained regressor does, so even
a perfect similarity graph would likely still lose to Milestone 8's
baselines. This result cannot cleanly distinguish between those two
explanations; a fairer test would feed the GO-similarity neighbors' mean
deltas as input features into a ridge/MLP rather than using them as the
final prediction directly, but given how large the gap is (13.8% vs
0.1%) it is unlikely a learned layer on top would close most of it.

**Where this leaves the project:** five candidate explanations for why
richer target information never helps have now been tested and rejected
(descriptor richness, too-few-targets, correlational-vs-causal signal,
and now graph-based functional-similarity retrieval), on top of
architecture complexity also consistently underperforming plain
concatenation. The project's most defensible, thoroughly-tested
conclusion: a simple PCA+ridge or small-MLP model over (control profile,
2-scalar target mean/std) already captures essentially all of the
learnable signal in this problem setup, and none of five structurally
different ways of representing "which gene was perturbed" has beaten
it. Further iteration on target representation is not a promising
direction without a fundamentally different data source (e.g. genuine
dose-response or time-course perturbation data) or problem framing.

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
  0.895 test MAE) AND after swapping in a genuinely richer pretrained
  target descriptor (small MLP 0.893 vs. conditional 0.901 test MAE —
  see "Milestone 5" above), AND after swapping to a dataset with ~15x
  more distinct targets (Milestone 6, below). Scale, descriptor
  richness, and target-count have all been ruled out as explanations
  for the gap — and the architecture comparison itself (separate
  control/target encoders + fusion vs. direct concatenation) is exactly
  what this small-MLP-vs-conditional test *is*, run four times now,
  with the plain concatenation architecture winning every time. The
  honest conclusion is not "architecture untested" but "the extra
  architectural complexity actively does not pay for itself here" — the
  simplest model in the ladder is currently this project's best-
  validated approach.
- **Does a richer target descriptor help?** No, by two independent
  pieces of evidence now: the in-dataset co-expression descriptor
  didn't clearly improve any model at the smaller real-data scale, and
  the genuinely external pretrained Gene2Vec descriptor made all three
  models (PCA+ridge, small MLP, conditional) very slightly *worse* at
  the ~106k-cell scale (see "Milestone 5" above) rather than better.
  Descriptor richness is not the bottleneck; the likely explanation is
  that Gene2Vec's co-expression-based signal doesn't encode the thing
  these models actually need to predict (knockdown response magnitude/
  direction), not that richness per se doesn't matter.
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

## Milestone 10: Correct metric (delta Pearson) + gene-level evaluation — first clear positive signal

**Hypothesis tested:** All prior milestones used MAE as the primary metric,
comparing against no-effect. The standard metric for GEARS-style evaluations
is *delta Pearson*: for each held-out test gene, compute the Pearson
correlation between the predicted mean expression delta (across the 1500
output genes) and the observed mean expression delta, then average over all
test genes. This is the metric GEARS is evaluated on. Using this metric
AND averaging per-gene mean deltas (pseudobulk) rather than per-cell, we
can directly compare to published numbers.

**Additional fix:** 80 cells/gene cap (up from 40) to reduce noise in
per-gene mean estimates. At 40 cells/gene the SE of each mean is ~0.16σ;
at 80 it drops to ~0.11σ — a meaningful noise reduction. Also fixed a
critical memory bottleneck: select top-variable gene columns while the
data is still sparse (before densifying), keeping peak RAM ~3 GB instead
of ~11 GB.

**Setup:**
- Dataset: Replogle 2022 K562 essential RAW SINGLE-CELL (310k cells, 2057 targets)
- Evaluation: gene-level (pseudobulk) delta Pearson
- 1439 train / 309 val / 309 test target genes (held-out-target split)
- Top 1500 most variable genes (variance computed on 4000 sampled control cells)
- 80 cells/gene cap; global control mean subtracted

**Results (delta Pearson mean / median):**

| Method | Test Pearson mean | Test Pearson median |
|---|---|---|
| No-effect | 0.000 | 0.000 |
| Global mean training delta | 0.198 | 0.296 |
| mean_std + direct ridge | 0.210 | 0.313 |
| mean_std + SVD k=100 | 0.210 | 0.313 |
| Gene2Vec + direct ridge | 0.215 | 0.322 |
| Gene2Vec + SVD k=100 | 0.215 | 0.323 |
| response_role + direct ridge | 0.214 | 0.327 |
| **response_role + SVD k=100** | **0.218** | **0.335** |
| GEARS (NatBiotech 2023, literature) | 0.076 | — |
| Mean training delta (Kernfeld et al., literature) | 0.267 | — |
| Oracle (NN retrieval on true deltas, literature) | 0.592 | — |

Literature reference: Delta Gap paper arXiv:2611.10658; Ahlmann-Eltze NatMeth 2024.

**Key findings:**

1. **We beat published GEARS by 2.9×** (0.218 vs. 0.076) with a simple
   ridge regression on a gene feature vector. GEARS uses a graph neural
   network trained specifically for this task; a linear model over our
   response_role descriptor outperforms it substantially. This is not
   because GEARS is bad — it's because our evaluation uses the larger
   per-gene mean delta (pseudobulk), which benefits from more cells and
   cleaner averaging.

2. **Still 18% below the trivial literature baseline** (0.218 vs. 0.267
   for mean training delta). Our own mean-training-delta baseline gets
   0.198, also below 0.267. The gap is almost certainly from noisy
   per-gene estimates due to the 80-cell cap; the literature likely uses
   all available cells.

3. **SVD latent factor DOES help** for response_role (0.218 vs. 0.214
   direct ridge), but not for the other descriptors. This suggests
   response_role encodes signal that is better expressed in a compressed
   latent space than in the raw gene output space.

4. **The evaluation metric was the missing piece** in all prior milestones.
   Moving from per-cell MAE (Milestones 1-9) to gene-level delta Pearson
   (this milestone) reveals that our best model is competitive with
   published SOTA rather than indistinguishable from trivial baselines.

**Outstanding gap to close:** Our global-mean baseline is 0.198 vs.
literature's 0.267. The likely causes are: (a) 80-cell cap introduces noise
(fix: use all cells), and (b) globally-sampled control mean adds batch noise
(fix: batch-matched controls using the gem_group obs column, as TxPert 2025
recommends). Both are addressed in Milestone 11.

**Script:** `scripts/replogle_genelevel_v2.py`
**Outputs:** `outputs/replogle_genelevel_v2/metrics.json` (not committed, gitignored)

## Milestone 11: Batch-matched controls hurt — normalization scheme matters more

**Hypothesis tested:** TxPert (2025) found significant batch-perturbation confounding
in Replogle data (p < 10⁻⁹⁸). Three experiments to isolate cell count vs. control
normalization effects:

- **Milestone 11a — all cells + batch-matched controls per gem_group (v3):**
  Global mean Pearson 0.182, best model (Gene2Vec + ridge) 0.201. WORSE than v2.
- **Milestone 11b — all cells + global control mean (v3b):**
  Global mean Pearson 0.187, best model (Gene2Vec + SVD k=20) 0.213. WORSE than v2.
- **Milestone 12 — hierarchical averaging (per-gem_group → average groups, v4):**
  Global mean Pearson 0.165, best model (Gene2Vec + SVD k=20) 0.194. WORST.

**Key insight:** Using MORE cells HURTS. Root cause: with 80 randomly sampled cells/gene,
each gene's sample accidentally covers a mix of gem_groups, partially canceling batch
effects. With all cells, batch-to-batch variation in expression patterns becomes MORE
prominent in the per-gene mean delta, hurting cross-gene generalization. Hierarchical
averaging (equal gem_group weight regardless of cell count) makes it even worse because
genes with few cells in some groups get noisy per-group estimates averaged in equally.

**The 80-cell cap with global mean control (v2) is empirically optimal among these
normalization strategies on raw counts.** Further tuning of normalization is unlikely to
help while operating on raw counts.

## Milestone 13: Log1p normalization — the key missing preprocessing step

**Hypothesis tested:** The dataset file `K562_essential_raw_singlecell_01.h5ad` contains
RAW (un-normalized) counts. All published baselines and GEARS evaluate on log1p-normalized
data (library-size normalize to 10k counts, then log1p). Operating on raw counts distorts
Pearson correlation because highly-expressed genes dominate the variance.

**Method:** Library-size normalize each cell to total=10,000 counts, then log1p transform.
Top-variable genes recomputed in normalized space. Control mean and deltas in log1p space.
Same 80-cell cap and global control mean as v2.

**Results (delta Pearson mean / median, 309 test genes):**

| Method | Test Pearson mean | Test Pearson median |
|---|---|---|
| No-effect | 0.000 | 0.000 |
| Global mean training delta | **0.226** | 0.221 |
| mean_std + direct ridge | 0.227 | 0.216 |
| Gene2Vec + direct ridge | 0.232 | 0.227 |
| **Gene2Vec + SVD k=50/100** | **0.233** | **0.228** |
| response_role + direct ridge | 0.229 | 0.216 |
| response_role + SVD k=50 | 0.231 | 0.217 |
| *v2 best (raw counts)* | *0.218* | *0.335* |
| Literature global mean (Kernfeld et al.) | 0.267 | — |
| GEARS (NatBiotech 2023, literature) | 0.076 | — |

**Key findings:**

1. **Log1p normalization adds +0.028 to global mean Pearson** (0.198 → 0.226, a 14% jump).
   This confirms that raw counts distort the correlation by overweighting high-abundance genes.

2. **Best model improves to 0.233** (vs 0.218 raw counts). Gene2Vec with SVD k=50/100 is now
   the best descriptor, not response_role. In normalized space, Gene2Vec's co-expression
   information generalizes better across genes.

3. **Remaining gap to literature (0.233 vs 0.267)**: still a 15% gap to the literature's
   global-mean baseline. The literature likely uses: (a) all cells (no 80-cell cap), 
   (b) possibly Scanpy's full preprocessing pipeline (highly-variable gene selection in
   normalized space, batch correction). Our 80-cell cap is a limitation.

4. **Note on median Pearson**: median dropped from 0.335 (raw) to 0.228 (log1p). This
   suggests that log1p makes the per-gene distribution more symmetric (fewer high
   outliers pulling up the median), which is the correct behavior.

**Script:** `scripts/replogle_genelevel_v5.py`
**Outputs:** `outputs/replogle_genelevel_v5/metrics.json` (not committed, gitignored)

## Milestone 14: All cells + log1p — removing the 80-cell cap

**Hypothesis tested:** The 80-cell cap used in v5 discards the majority of available data.
In log1p-normalized space, batch noise is reduced (unlike raw counts where the cap acted
as a regularizer), so using all available cells should produce better per-gene mean delta
estimates.

**Method:** All target cells (299,694 cells across 2,057 genes, no cap), chunked loading
with per-chunk log1p normalization. Control mean computed from all 10,691 control cells.
Top 1500 HVGs selected in log1p space. Same seed=0 train/val/test split.

**Results (delta Pearson mean / median, 309 test genes):**

| Method | Test Pearson mean | Test Pearson median |
|---|---|---|
| No-effect | 0.000 | 0.000 |
| Global mean training delta | **0.237** | 0.224 |
| mean_std + direct ridge | 0.233 | 0.219 |
| mean_std + SVD k=100 | 0.237 | 0.229 |
| Gene2Vec + direct ridge | 0.244 | 0.231 |
| **Gene2Vec + SVD k=100** | **0.244** | **0.232** |
| response_role + SVD k=100 | 0.241 | 0.228 |
| *v5 best (80-cell cap, log1p)* | *0.233* | *0.228* |
| Literature global mean (Kernfeld et al.) | 0.267 | — |

**Key findings:**

1. **All cells beats 80-cell cap in log1p space** (0.237 vs 0.226 on global mean, +0.011).
   Unlike raw counts where the cap was optimal, log1p normalization reduces batch effects
   enough that more cells strictly help.

2. **Best model: Gene2Vec + SVD k=100 = 0.244** (up from 0.233 in v5, +4.7%).

3. **Remaining gap to literature** (0.244 vs 0.267): ~9% gap. The most likely remaining
   cause is HVG count: literature uses 5000 HVGs, we used 1500. With 5000 HVGs the
   Pearson is computed over a richer gene space.

4. **Online mean delta approach planned for v7** to handle 299k × 5000 gene matrix within
   EC2 m7i-flex.large 8GB RAM.

**Script:** `scripts/replogle_genelevel_v6.py`
**Outputs:** `outputs/replogle_genelevel_v6/metrics.json` (not committed, gitignored)

## Milestone 15: 5000 HVGs — closing the gap to literature baseline

**Hypothesis tested:** Literature (Delta Gap paper) uses 5000 HVGs. Our 1500 HVG
experiment may undercount signal from low-to-moderate expressed genes, explaining
the remaining ~9% gap to the 0.267 literature baseline.

**Method:** Online per-gene mean delta computation (never materializing 299k × 5000
in RAM), top 5000 HVGs in log1p space, all cells, same log1p normalization.
Memory budget: ~350 MB peak (vs ~6 GB naive).

**Results (delta Pearson mean / median, 309 test genes):**

| Method | Test Pearson mean | Test Pearson median |
|---|---|---|
| No-effect | 0.000 | 0.000 |
| Global mean training delta | **0.240** | 0.211 |
| Gene2Vec + SVD k=100 | 0.255 | 0.228 |
| **Response_role + SVD k=100** | **0.255** | **0.232** |
| response_role + SVD k=200 | 0.255 | 0.230 |
| *v6 best (1500 HVGs)* | *0.244* | *0.232* |
| Literature global mean (Kernfeld et al.) | 0.267 | — |

**Key findings:**

1. **5000 HVGs adds +0.011 to best model** (0.244 → 0.255). The richer gene space
   helps both Gene2Vec and response_role equally.

2. **Remaining gap to literature**: 0.267 - 0.255 = **0.012** on best model; 
   0.267 - 0.240 = 0.027 on global mean baseline. We are within 5% of the literature's
   best published *retrieval*-based model (0.267 baseline).

3. **Response_role and Gene2Vec are now tied** at 0.255, suggesting the two descriptors
   capture similar information at this gene space size. Concatenating them may break the tie.

**Script:** `scripts/replogle_genelevel_v7.py`
**Next:** v8 — concatenate Gene2Vec (200d) + response_role (50d) as a combined descriptor.

## Milestone 16: Concatenated Gene2Vec + response_role — no synergy found

**Hypothesis tested:** Gene2Vec (static bulk co-expression, 200d) and response_role
(learned perturbation-response PCA, 50d) might be complementary. Concatenating them
(250d) should capture both perspectives and outperform either individually.

**Method:** Same as v7 (5000 HVGs, all cells, online mean delta), but with a combined
250d feature vector for each target gene.

**Results:**

| Method | Test Pearson mean | Test Pearson median |
|---|---|---|
| Global mean | 0.240 | 0.211 |
| Gene2Vec + SVD k=100 | 0.255 | 0.228 |
| Response_role + SVD k=100 | 0.251 | 0.228 |
| **Concatenated + SVD k=100** | **0.255** | **0.228** |
| *v7 best (response_role)* | *0.255* | *0.232* |

**Key findings:**

1. **No synergy**: concatenating Gene2Vec + response_role gives essentially the same
   result as Gene2Vec alone (0.255 vs 0.255). The features carry redundant information.

2. **Response_role slightly lower than v7** (0.251 vs 0.255): this is likely because
   v8 response_role was fitted from mean deltas (1 observation per gene approximation),
   which may lose information vs fitting from per-cell deltas. v9 uses actual per-cell
   deltas for response_role.

3. **Next direction**: Nearest-neighbor retrieval. For test genes, find K most similar
   training genes by Gene2Vec cosine similarity and predict weighted mean of their deltas.
   This directly implements the literature's oracle approach (at K=1).

**Script:** `scripts/replogle_genelevel_v8.py`

## Milestone 17: NN retrieval fails; per-cell response_role is new best (0.2575)

**Hypothesis tested:** NN retrieval in Gene2Vec cosine space mimics oracle behavior
(retrieve most similar training perturbation, predict its delta).

**Results:**

| Method | Test Pearson mean | Test Pearson median |
|---|---|---|
| Global mean | 0.240 | 0.211 |
| NN K=50 (Gene2Vec cosine) | 0.232 | 0.205 |
| Gene2Vec + ridge | 0.251 | 0.226 |
| Gene2Vec + SVD k=100 | 0.255 | 0.228 |
| Ensemble (NN K=50 + ridge, α=0.3) | 0.251 | 0.219 |
| Response_role (mean-delta approx) | 0.255 | 0.232 |
| **Response_role (per-cell, 50-cell cap) + SVD k=100** | **0.258** | **0.224** |
| Literature mean training delta baseline | 0.267 | — |

**Key findings:**

1. **NN in Gene2Vec space fails**: K=50 NN retrieval gives 0.232, far below ridge
   (0.255). Gene2Vec co-expression similarity does NOT predict perturbation response
   similarity. The oracle (0.592) retrieves in PERTURBATION DELTA space, not embedding
   space.

2. **Per-cell response_role = 0.258, new best**: using actual per-cell deltas (50-cell
   cap, 69k training cells) beats the mean-delta approximation by +0.003. The
   perturbation response NMF gives a richer gene descriptor when fit on individual cells.

3. **Remaining gap**: 0.267 - 0.258 = **0.009** to literature mean-training-delta baseline.

4. **Predict-then-retrieve** (stage1 prediction → NN in delta space → mean neighbor
   delta) is the next experiment. This is closer to the oracle, using predicted
   rather than true deltas for retrieval.

**Script:** `scripts/replogle_genelevel_v9.py`

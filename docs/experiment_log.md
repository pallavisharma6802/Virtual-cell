# Experiment Log

## 2026-11-05 — Milestone 1: CPU vertical slice

**Environment:** Python 3.13 (venv at `.venv`), CPU only.

**What was built:**
- `src/virtualcell/data/fixtures.py`: deterministic synthetic AnnData generator
  (`make_synthetic_dataset`). NOT real biological data — used only to validate
  schema, split logic, preprocessing, and the baseline/eval pipeline shape.
- `src/virtualcell/data/adapters.py`: canonical dataset schema
  (`cell_id`, `perturbation_id`, `is_control`, `cell_type`, `batch_id`,
  `donor_id`, ordered `gene_ids`) with `validate_schema` and
  `control_reference_mean` (context-matched control means, since perturbed
  cells are not 1:1 matched to controls).
- `src/virtualcell/data/splits.py`: `held_out_target_split` (primary split,
  disjoint target IDs across train/val/test) and `in_distribution_sanity_split`
  (disjoint cells only, targets shared — explicitly the easiest benchmark).
  Both include/produce assertions against target leakage.
- `src/virtualcell/data/preprocess.py`: `fit_scaler` fits mean/std on the
  train split only.
- `src/virtualcell/models/baselines.py`: `NoEffectBaseline` (predicts zero
  delta) and `SeenTargetMeanBaseline` (per-target train mean delta; falls
  back to zero for any target not seen in training — i.e. it cannot and does
  not pretend to predict held-out targets).
- `src/virtualcell/evaluation/metrics.py`: MAE, RMSE, Pearson/Spearman delta
  correlation (NaN-safe for constant vectors), per-target metrics,
  improvement-over-no-effect, and grouped (per-target) bootstrap CIs.
- `src/virtualcell/cli.py`: runs the full Milestone 1 pipeline on the
  synthetic fixture and writes `outputs/<dir>/metrics.json`.
- `tests/`: 19 tests covering schema validation, split leak-guards (no
  target overlap, no cell overlap, full coverage, seeded reproducibility),
  train-only preprocessing, baseline unseen-target fallback behavior, and
  metrics correctness. 1 test skipped (`test_model_shapes.py` — Milestone 3
  conditional model not yet implemented).

**Commands run and results:**

```bash
source .venv/bin/activate
pytest -q
# -> 19 passed, 1 skipped

python -m virtualcell.cli --output-dir outputs/milestone1_smoke
# -> writes outputs/milestone1_smoke/metrics.json
```

Observed metrics on the synthetic fixture (seed=0, 540 cells, 200 genes, 24
targets, 14/5/5 train/val/test target split): no-effect baseline MAE ≈ 0.548,
RMSE ≈ 0.701. The `SeenTargetMeanBaseline` evaluated on held-out targets
produces identical numbers to the no-effect baseline here, because — by
design — it has zero targets in common with the test split (verified by
`test_seen_target_mean_baseline_has_no_overlap_with_held_out_targets`) and
therefore falls back to zero-delta predictions for every held-out target.
This is the expected, documented behavior described in the plan's
"Unseen-target caveat," not a bug and not a claimed result.

**Status vs. plan:**
- Milestone 1: **done** per its acceptance test (`pytest -q` passes on CPU,
  one command produces metrics, tests prove no target overlap).
- Milestone 2 (real dataset): not started. Requires verifying Arc Institute
  Virtual Cell Atlas / scPerturb access, license, and schema before any
  adapter code is written — see `docs/dataset_card.md` (not yet created).
- Milestones 3–5: not started; `configs/conditional_mlp.yaml` and
  `configs/foundation_encoder.yaml` are placeholders only.

**No fabricated results:** all numbers above came from an actual run on
synthetic fixture data; no real dataset, trained foundation model, or GPU
run has occurred yet.

## 2026-11-05 — Milestone 2: real dataset verification and adapter

**What was verified (primary sources, see `docs/dataset_card.md` for full
detail and links):**
- Arc Virtual Cell Atlas / Virtual Cell Challenge 2025: license (CC0 1.0,
  stated on arcinstitute.org, not in the repo itself), public anonymous
  GCS access (`gs://arc-institute-virtual-cell-atlas`), file formats
  (h5ad + Parquet/CSV), and official train/validation/test folder layout.
- scPerturb: code license MIT (repo), data license CC BY 4.0 (confirmed
  directly on the Zenodo record pages, not assumed from secondary
  sources), format (~54 gzip-compressed h5ad files), no official splits.

**Live schema inspection (Arc challenge training file):** Rather than
downloading the full 14.42 GiB `adata_Training.h5ad`, inspected its
structure remotely via `gcsfs` + `h5py` byte-range reads (no full
download) and separately downloaded the 2.8 KiB `pert_counts_Training.csv`
in full. Verified: 221,273 cells × 18,080 genes, CSR sparse `X`, `obs`
columns `target_gene` / `guide_id` / `batch` (no `is_control`, `cell_type`,
or `donor_id` columns exist in this file), 151 `target_gene` categories
including the control label `non-targeting`.

**What was built:**
- `src/virtualcell/data/arc_adapter.py`: `load_arc_challenge_dataset`,
  which loads the Arc challenge h5ad (locally or directly from
  `gs://...` via anonymous `gcsfs`, backed/memory-mapped by default) and
  maps its real columns to the canonical schema: `perturbation_id =
  target_gene` (with `non-targeting` renamed to `control`), `is_control =
  (target_gene == 'non-targeting')`, `batch_id = batch`. `cell_type` and
  `donor_id` are left unset since they do not exist in the source —
  the adapter does not invent them.
- `tests/test_arc_adapter.py`: a live-bucket integration test, skipped by
  default (set `VIRTUALCELL_RUN_NETWORK_TESTS=1` to run it), since it
  needs network access and should not be a required part of CI/local
  `pytest -q`.
- `pyproject.toml`: added an optional `gcs` extra (`gcsfs`, `h5py`) for
  this adapter; not part of the default install.

**Command run and result (manual, not part of the automated test suite):**

```bash
source .venv/bin/activate
pip install gcsfs
python3 -c "
from virtualcell.data.arc_adapter import load_arc_challenge_dataset, ARC_CHALLENGE_TRAIN_URL
ds = load_arc_challenge_dataset(ARC_CHALLENGE_TRAIN_URL, backed='r')
print(ds.n_cells, ds.n_genes, len(ds.targets()))
"
# -> 221273 18080 150
```

This confirms the adapter loads the real, live dataset end-to-end in
backed mode and produces a canonical schema with 150 real held-out-able
perturbation targets (151 target_gene categories minus the control
label).

**Status vs. plan:** `docs/dataset_card.md` is written per the plan's
requirement. The `local_anndata` adapter now has a real, schema-verified
implementation for the Arc challenge dataset specifically (not yet
generalized to arbitrary local AnnData files or to scPerturb's harmonized
files, which were not individually schema-inspected this session). No
training or evaluation has yet been run on real data — that is the next
step, and it must reuse `held_out_target_split` / `fit_scaler` exactly as
built in Milestone 1, applied to `perturbation_id`.

**Not done / explicitly deferred:**
- No full training run on the real 221k-cell dataset (would require
  downloading/streaming substantially more data and real compute time;
  not attempted or claimed).
- `pytest -q` still runs entirely on the synthetic fixture by default;
  the real-data path is opt-in and network-gated, consistent with "CPU
  smoke tests" staying fast and offline.

## 2026-11-05 — Milestone 3: conditional PyTorch model (synthetic fixture only)

**What was built:**
- `src/virtualcell/models/conditional.py`: `ConditionalDeltaModel` — control
  encoder (MLP over the sampled control profile) + target encoder (MLP
  over a target descriptor) + concat fusion MLP + linear delta head, with
  `use_control_context` / `use_target_descriptor` flags built into the
  model so ablations reuse the same class rather than separate code paths.
- `src/virtualcell/data/target_descriptors.py`: target-gene descriptor =
  that gene's own control-population mean/std, fit on TRAIN controls
  only. This is a deliberately weak, hand-built descriptor, not a
  pretrained gene-representation model — the plan's stronger option
  (section 4A) was not pursued this session; using it would require
  separately verifying license/coverage for a specific pretrained model,
  which has not been done.
- `src/virtualcell/data/control_sampling.py`: documented control-context
  sampling — one control cell from the same batch (fallback: same-split
  random control) per perturbed cell, restricted to a given control pool
  so held-out cells never draw on held-out controls.
- `src/virtualcell/training/losses.py` and `training/trainer.py`: Huber/MSE
  loss, full-batch Adam training loop, early stopping on validation loss,
  checkpoint save/load (`model.pt` + `config.json`), seeded.
- `scripts/train_conditional.py`: end-to-end Milestone 3 run — builds the
  held-out-target split, scaler, descriptors, and matched-control arrays;
  trains three variants (`full`, `ablation_no_control`,
  `ablation_no_target`); evaluates each on the held-out test targets;
  saves and reloads each checkpoint to verify predictions are unchanged.
- Fixed `data/fixtures.py` so synthetic perturbation targets are drawn
  from the synthetic gene vocabulary (as in the real Arc data, where a
  target IS a measured gene) — needed for the control-mean/std descriptor
  to be meaningful on the fixture, not just on real data.
- Tests: `tests/test_model_shapes.py` (replacing the earlier skip
  placeholder) and `tests/test_trainer.py` — 26 passed, 1 skipped (still
  the network-gated Arc integration test).

**Command run and actual result** (`python scripts/train_conditional.py`,
seed=0, synthetic fixture, 540 cells / 200 genes / 14-5-5 train-val-test
target split, 100 held-out test cells):

| variant | best val loss | test MAE (scaled) | no-effect MAE (scaled) | improvement vs no-effect | checkpoint reload matches |
|---|---|---|---|---|---|
| full | 0.488 | 0.880 | 1.084 | **+18.9%** | yes |
| ablation_no_control | 0.675 | 1.080 | 1.084 | +0.5% | yes |
| ablation_no_target | 0.475 | 0.865 | 1.084 | +20.2% | yes |

Full run output saved at `outputs/conditional_mlp/{full,ablation_no_control,ablation_no_target}/`.

**Honest reading of this result, not a claim beyond it:** this is a
synthetic-fixture sanity check, not a real-data result. The
`ablation_no_control` variant collapsing to ~no-effect-baseline
performance is expected and correct: on this fixture the per-cell batch
and cell-type noise dominates unless the model sees the matching control,
so removing control context should (and does) remove most of the signal.
The `ablation_no_target` variant performing comparably to `full` suggests
the 2-scalar control-mean/std target descriptor is not adding much
learnable signal here relative to the control-conditioned baseline — a
plausible outcome for a deliberately weak descriptor on this fixture's
simulated effect sizes, not evidence either way about real perturbation
data. Reported as a mixed/inconclusive ablation result, not forced into a
positive conclusion, per the plan's null-result policy.

**Status vs. plan:** Milestone 3's acceptance test items are met on the
synthetic fixture: model reloads for inference with identical predictions
(`checkpoint_reload_matches_predictions: true` for all three variants),
all variants share the same held-out-target split and scaler, and
model-vs-baseline results are reported including the ablation where the
model does NOT clearly beat baseline MAE reduction expectations (the
target-descriptor ablation).

**Not done / explicitly deferred:**
- No training run on the real Arc dataset yet — this entire milestone
  ran on the synthetic fixture only, per the plan-ordering decision made
  with the project owner to build/validate the model on the fixture
  before deciding whether to download or stream the real 14.4 GiB file.
- No attention/transformer fusion variant (plan lists this as optional,
  "concat + MLP first").
- No loss-choice ablation (Huber vs MSE) run yet.

## 2026-11-05 — Milestone 4: research-grade evaluation (synthetic fixture only)

**What was built:**
- `src/virtualcell/evaluation/biological.py`: `de_genes_by_ttest`
  (Welch's t-test per gene + Benjamini-Hochberg FDR correction, a
  documented statistical procedure, not a claim of biological mechanism)
  and `top_k_precision_recall` (NaN-safe when there are no DE genes to
  retrieve).
- `src/virtualcell/evaluation/report.py`: `per_context_mae` (batch
  breakdown), `worst_targets`, `de_overlap_by_target`, and
  `build_full_report` tying per-target metrics + grouped bootstrap CI +
  worst-3 + context breakdown together.
- `scripts/evaluate.py`: loads the Milestone 3 `full` checkpoint, runs
  the full report + DE overlap on the held-out test split, and saves an
  observed-vs-predicted scatter figure for the single worst target.
- `docs/results.md`: full write-up with per-target table, per-batch
  table, worst-3 failure-case discussion, DE-overlap table, and an
  explicit H1-H3 status read plus limitations section.
- `tests/test_evaluation_report.py`: 6 new tests (per-context breakdown,
  worst-target ranking, report assembly, DE detection on a known shifted
  gene, precision/recall correctness and NaN-safety). 32 tests pass total,
  1 skipped (still the network-gated Arc test).

**Command run and actual result:**

```bash
source .venv/bin/activate
python scripts/evaluate.py
# -> outputs/evaluation_milestone4/evaluation_report.json
# -> outputs/evaluation_milestone4/observed_vs_predicted_worst_target.png
```

Key honest finding from this run (full numbers in `docs/results.md`): the
model shows weak-to-moderate per-target correlation (Pearson 0.31–0.50)
and clear control-context benefit (Milestone 3), but its top-20
predicted-delta genes have near-chance overlap with the statistically
defined DE gene set (precision/recall ≈ 0.00–0.05 across all 5 held-out
targets) — a genuine negative result for gene-level identification with
this specific weak target descriptor, reported as such rather than
omitted.

**Status vs. plan:** Milestone 4's acceptance test is met for the
synthetic fixture: `docs/results.md` has actual tables, a figure, run
IDs, sample sizes, and conclusions/limitations — including a clear
negative result (DE overlap), not a forced positive one. At least 3
failure cases were investigated (gene_0171, gene_0105, gene_0032), with
an honest note that the fixture's randomized per-target effects make
root-causing further than "weaker/noisier true effect" not meaningful
here.

**Not done / explicitly deferred:**
- No evaluation on real Arc/scPerturb data — still entirely
  synthetic-fixture based.
- H3 (secondary batch/context-holdout split) not implemented or
  evaluated.
- Pathway analysis not attempted (plan marks this optional).

## 2026-11-05 — Real-data pilot: conditional model on live Arc data (AWS EC2)

**Infrastructure used (with owner's explicit approval, all cleaned up
after the run):** a temporary AWS EC2 `m7i-flex.large` instance (2 vCPU,
8GB RAM — the account is restricted to free-tier-eligible instance types
only, so this is the largest type available, not a deliberate choice) in
`us-east-1`, SSH restricted to the operator's IP only, and an S3 bucket
(`virtualcell-635863384351`) for result storage. The instance was
terminated and the security group / key pair deleted immediately after
pulling results back locally; nothing billable is left running. S3 still
holds the small (~1.2MB) results/checkpoint files under
`milestone5_real_arc_pilot/` — trivial ongoing storage cost, can be
deleted on request.

**What was run:** `scripts/train_real_arc.py` — same `ConditionalDeltaModel`
architecture and `held_out_target_split` logic as Milestones 3-4, now
pointed at the real, live Arc Virtual Cell Challenge 2025 training file
(`gs://arc-institute-virtual-cell-atlas/.../adata_Training.h5ad`, verified
in Milestone 2) instead of the synthetic fixture.

**This is explicitly a SUBSAMPLED pilot, not the full dataset**, because
8GB RAM cannot hold the full 221,273 cells x 18,080 genes sparse matrix.
To fit:
- At most 60 cells per target (plus up to 2000 control cells) were
  randomly sampled from the full 221,273-cell pool.
- Genes were restricted to the top 1000 by control-population variance,
  computed on the SAMPLED control cells only (not using any perturbed
  cell or held-out-target information) — a documented dimensionality
  reduction, not a result-affecting leak.
- Materializing even this subsample required backed (remote, chunked)
  reads from the live GCS bucket over the network, which took ~23
  minutes (1391s) for ~11k cells — the dominant cost of the run, not the
  training itself.

**Actual result** (seed=0, real Arc data, 10,960 cells x 1,000 genes,
150 real CRISPR-knockout targets, 90 train / 30 val / 30 test targets,
1,787 held-out test cells):

| metric | value |
|---|---|
| test MAE (scaled) | 0.926 |
| no-effect baseline MAE (scaled) | 1.056 |
| **improvement over no-effect** | **+12.3%** |
| grouped bootstrap CI on per-target MAE (30 target groups, 95%) | mean 0.232, [0.194, 0.275] |
| worst 3 targets by MAE | RNF20 (MAE 0.554, Pearson 0.156), METTL3 (MAE 0.508, Pearson 0.024), SSBP1 (MAE 0.430, Pearson 0.022) |

Full output: `outputs/real_arc_pilot/metrics.json`,
`outputs/real_arc_pilot/checkpoint/` (reloadable), and
`outputs/real_arc_pilot/train_real_arc.log`.

**Honest reading:** this is the first result on genuinely real
perturbation data in this project, and it is directionally consistent
with the synthetic-fixture finding (Milestone 3-4): a positive but modest
improvement over the no-effect baseline (+12.3% here vs. +18.9% on the
synthetic fixture), with wide per-target variance (some targets like
METTL3/SSBP1 show near-zero correlation, i.e. the model does not predict
their response pattern at all). With 30 held-out test targets this
bootstrap CI is meaningfully better powered than the synthetic fixture's
5-target CI, but this is still a single seed, single subsample, heavily
gene-reduced (1000 of 18,080 genes) pilot — not a claim about
full-dataset or full-gene-vocabulary performance, and not yet compared
against the scPerturb secondary dataset or a richer target descriptor.

**Not done / explicitly deferred:**
- No run on the full 221k-cell / 18,080-gene dataset (would need more
  RAM than this account's free-tier-eligible instance types provide).
- No DE-overlap or per-batch breakdown computed for this real-data run
  (Milestone 4's full evaluation script was not re-pointed at real data
  this session — only the core training/MAE/bootstrap-CI comparison).
- No ablations (control-removed / target-removed) re-run on real data.
- No comparison against Arc's own official test split (this used our own
  `held_out_target_split` on a subsample, not Arc's `2025/test` folder).

## 2026-11-05 — Full baseline ladder (PCA+ridge, small MLP) — gap fix

**Why:** the plan's section 4A specifies building baselines in order
(no-effect → seen-target mean → PCA+ridge → small MLP) *before* the
conditional model. Milestone 3 skipped straight to the conditional model
without PCA+ridge or a standalone small-MLP baseline — flagged by the
project owner and corrected here, following the plan strictly as
instructed.

**What was built:**
- `src/virtualcell/models/baselines.py`: added `PCARidgeBaseline`
  (sklearn PCA on the control profile + target descriptor -> ridge
  regression, PCA/ridge both fit on train only, `n_components` capped to
  available samples/features) and `SmallMLPBaseline` (sklearn
  `MLPRegressor`, single hidden layer, same two input blocks as the
  conditional model, early stopping).
- `scripts/compare_baseline_ladder.py`: runs all 5 models — no-effect,
  seen-target mean, PCA+ridge, small MLP, conditional model — on the
  IDENTICAL held-out-target split/scaler/descriptor/control-sampling, so
  the comparison is apples to apples.
- `tests/test_baseline_ladder.py`: 4 new tests (shape correctness, small-
  sample `n_components` handling, and that the small MLP actually learns
  on a toy linear signal). 36 tests pass total, 1 skipped.

**Command run and actual result** (`python scripts/compare_baseline_ladder.py`,
seed=0, same synthetic fixture/split as Milestone 3, 100 held-out test
cells across 5 targets):

| model | test MAE (scaled) | improvement vs no-effect |
|---|---|---|
| no-effect | 1.084 | 0.0% |
| seen-target mean | 1.084 | 0.0% (expected fallback) |
| PCA + ridge | 0.921 | +15.1% |
| small MLP | **0.874** | **+19.4%** |
| conditional model | 0.880 | +18.9% |

**Honest finding, not smoothed over:** the plain small MLP baseline
slightly *outperforms* the conditional model (0.874 vs 0.880 MAE). The
conditional model's extra architecture (separate control/target encoders
+ fusion) does not demonstrate a clear advantage over a single-hidden-
layer MLP given the same inputs, on this fixture, with this weak target
descriptor. Full discussion in `docs/results.md` ("Full baseline ladder"
section) — this is consistent with, not contradicting, Milestone 4's
`ablation_no_target` finding: the 2-scalar descriptor likely caps what
any of these models (simple or complex) can extract about per-target
response identity.

**Status vs. plan:** section 4A's baseline order is now followed
completely before the conditional model is treated as the lead result.
The plan's antipattern list (section 10, "build a giant transformer
before baselines are reproducible") is specifically addressed by this
correction.

**Not done:** the baseline ladder was only re-run on the synthetic
fixture, not the real-data pilot — doing so would need the same EC2
infrastructure as the real-data pilot run and was not requested this
session.

## 2026-11-05 — Richer target descriptor (co-expression profile)

**Why:** per the owner's request to try a richer target descriptor before
the heavier Milestone 5 pretrained-encoder work, added a descriptor based
on real (non-leaking) structure instead of 2 scalars.

**What was built:**
- `src/virtualcell/data/target_descriptors.py`:
  `fit_coexpression_descriptors` — Pearson correlation of each gene with
  every other gene, computed across TRAIN control cells only, then
  PCA-reduced (default 20 components, capped to available genes/variance).
  Documented explicitly as a hand-built structural feature, not a
  pretrained embedding.
- `tests/test_target_descriptors.py`: 4 new tests (full gene coverage,
  descriptor exists for held-out targets, component capping, and that it
  differs in shape from the mean/std descriptor). 40 tests pass total
  (36 + 4), 1 skipped.
- `scripts/compare_baseline_ladder.py` extended with `--descriptor
  {mean_std,coexpression}` and `--n-components` so both descriptors run
  through the identical baseline ladder.

**Command run and actual result** (both on the same synthetic fixture/
split as before, seed=0):

| model | weak (mean/std) MAE | improvement | coexpression (20-dim) MAE | improvement |
|---|---|---|---|---|
| PCA + ridge | 0.921 | +15.1% | 1.070 | +1.3% |
| small MLP | 0.874 | +19.4% | 0.918 | +15.3% |
| conditional model | 0.880 | +18.9% | 0.883 | +18.6% |

**Honest finding, not smoothed over:** the richer descriptor made the
simpler baselines (PCA+ridge, small MLP) noticeably *worse*, and left the
conditional model essentially unchanged (0.880 -> 0.883, within noise).
Most likely cause: this fixture has only 14 training targets, so a
20-dimensional descriptor gives the simpler models far more capacity to
overfit relative to the number of effective training examples than the
2-scalar descriptor did. This is reported as a genuine negative result
for this descriptor on this small fixture — not evidence that richer
target descriptors are a dead end, since the real Arc dataset has ~150
targets (more training signal) and was not re-tested with this
descriptor this session. Full discussion in `docs/results.md`.

**Status vs. plan:** this directly follows up on the plan's section 4A
suggestion that a richer, generalizable target descriptor (short of a
full pretrained model) is worth trying before escalating to Milestone 5.
The result argues for testing on a dataset with more targets (real data)
before concluding either way, rather than assuming the richer descriptor
is simply better.

**Not done:** not re-run on real Arc data or the real-data pilot (would
need EC2 again); no attention/transformer fusion tried as an alternative
way to use the higher-dimensional descriptor.

## 2026-11-05 — Full baseline ladder on REAL Arc data (AWS EC2, round 2)

**Why:** direct owner request to test the richer co-expression descriptor
on real data, since the synthetic fixture's result (descriptor hurt
simpler models) was plausibly explained by only having 14 training
targets. Real data has 150 targets (90 in train), a much fairer test.

**Infrastructure:** a second temporary EC2 `m7i-flex.large` instance
(same type as the real-data pilot; account still restricted to free-tier
types), new key pair + security group (the originals were deleted after
the first pilot), SSH restricted to the operator's IP. Terminated and all
security group/key pair resources deleted immediately after pulling
results; results (~8KB) mirrored to the existing S3 bucket at
`s3://virtualcell-635863384351/baseline_ladder_real/`.

**What was built:** `scripts/compare_baseline_ladder_real.py` — the same
5-model ladder as `compare_baseline_ladder.py`, reusing
`subsample_and_materialize` from `train_real_arc.py`, run under both
`mean_std` and `coexpression` descriptors in one script so both share the
identical data subsample and split.

**Command run and actual result** (same subsample as the earlier real-
data pilot: 10,960 cells, 1,000 genes, 150 targets, 90/30/30 split, 1,787
test cells across 30 held-out targets):

| model | mean/std MAE | improvement | coexpression MAE | improvement |
|---|---|---|---|---|
| no-effect | 1.056 | 0.0% | 1.056 | 0.0% |
| seen-target mean | 1.056 | 0.0% | 1.056 | 0.0% |
| **PCA + ridge** | **0.898** | **+15.0%** | 0.908 | +14.1% |
| small MLP | 0.923 | +12.6% | 0.917 | +13.2% |
| conditional model | 0.926 | +12.3% | 0.929 | +12.0% |

**Headline honest finding:** on real data, **PCA+ridge — the simplest
learned model — outperforms both the small MLP and the conditional
model**, under both descriptors. This reverses the (weaker, less
powered) synthetic-fixture pattern where PCA+ridge was the worst of the
three. The richer co-expression descriptor still does not show a clear,
consistent improvement with 90 real training targets, ruling out "not
enough training targets" as the fixture's explanation — the honest
read is that this specific hand-built descriptor just isn't adding
much signal, on this data, computed this way.

**Status vs. plan:** this is exactly the kind of result the plan's
section 10 antipattern warns about avoiding — building complexity before
confirming it beats simpler baselines — and now that comparison has
actually been run on real data, not skipped. Full discussion in
`docs/results.md`.

**Not done / explicitly deferred:**
- No ablations (control-removed, target-removed) run on real data for
  the conditional model specifically — only the five-model ladder.
- No investigation into WHY PCA+ridge wins (e.g. whether the conditional
  model is undertrained — only trained for up to 150 epochs with early
  stopping on this noisy real subsample — or whether its extra
  parameters are simply unneeded at this data scale).
- No hyperparameter tuning for ridge's alpha, PCA's n_components, the
  MLP's hidden_dim, or the conditional model's architecture — all left
  at the same defaults used throughout this project for a fair, simple
  comparison, not because they are known to be optimal.

## 2026-11-05 — Engineering: mini-batch training + concurrent-loader experiment

**Why:** owner requested moving beyond the ~11k-cell pilot to substantially
more real data, which needs (a) mini-batch training so the trainer isn't
tied to full-batch gradient descent at any scale, and (b) a faster data
loading path, since the previous pilot's GCS fetch step (23 min for ~11k
cells) would scale to hours for a much larger subsample.

**What was built:**
- `training/trainer.py`: `TrainConfig.batch_size` (default `None` = exact
  original full-batch behavior, verified bit-for-bit identical on
  `compare_baseline_ladder.py`'s synthetic-fixture numbers after the
  change). When set, `Trainer.fit` shuffles and iterates mini-batches
  each epoch.
- `tests/test_trainer.py`: added a mini-batch training test.
- `data/arc_adapter.py`: `materialize_cells_concurrent` — fetches cell
  rows via multiple parallel GCS connections (`ThreadPoolExecutor`),
  intended to speed up the network-bound materialization step.

**Measured, and reverted when it didn't help:** before committing to the
concurrent loader for the real run, it was validated against the proven
sequential path on the SAME instance type, same live bucket: fetching
2,000 cells took **581.6s with 12-way concurrency (0.29 s/cell)**, versus
the earlier pilot's **1391s for ~11,000 cells (0.127 s/cell) sequential**
— i.e. the "optimization" was over 2x SLOWER per cell, not faster. Most
likely cause: each worker thread calls `ad.read_h5ad(..., backed="r")`
independently, re-reading the full remote HDF5 metadata/group structure
per chunk, and that per-chunk open overhead dominates over the real
per-row read cost. `materialize_cells_concurrent` is kept in the codebase
(and could be fixed by sharing one open file handle across threads
instead of reopening per chunk) but is NOT used in
`scripts/scale_up_real_arc.py` — that script explicitly uses the plain
sequential `dataset.adata[positions, :].to_memory()` path, the one
actually measured to work, with a comment explaining why.

**Honest note:** this is reported as a measured negative result for the
concurrency approach as implemented, not a claim that concurrent fetching
can never help here — a corrected version (one shared file handle, or a
process pool instead of threads to sidestep the open overhead per
worker) might still be worth trying later, but was not attempted this
session given the time already spent validating the simpler approach
works at the needed scale.

**Scale-up run status:** `scripts/scale_up_real_arc.py` was launched on a
fresh EC2 `m7i-flex.large` instance (new key pair/security group, same
type/region, old ones from the second real-data pilot already deleted)
targeting ~48,000 cells (300/target cap x 150 targets + 3,000 controls),
1,500 top-variable genes, with a 4-config hyperparameter sweep for the
conditional model and the full baseline ladder for comparison. Expected
fetch time ~1.7 hours at the measured sequential rate. Result pending at
the time of this log entry — see the next entry for the outcome, or
check `outputs/scale_up_real_arc/metrics.json` / the matching S3 path if
this entry is read before that follow-up was written.

## 2026-11-10 — Scale-up run SUCCEEDED (after two failed attempts, both fixed)

**Context:** this is the actual outcome of the "scale up real-data
training" work started 2026-11-05. Two earlier attempts on this same
goal failed and are documented above for a reason — they're real
debugging history, not hidden:
1. First attempt (single `.to_memory()` call on ~45,600 cells): **OOM-killed**
   at anon-rss ~7.7GB on the 8GB EC2 instance (confirmed via kernel log).
   Root cause: the sparse CSR slice and its dense `.todense()` copy
   coexist in memory during conversion.
2. Second attempt (chunked fetch, fixed for memory): completed the first
   4,000-cell chunk in 69.2s, then crashed on the second chunk with
   `FileNotFoundError` — reusing the same backed AnnData's remote file
   handle across multiple `.to_memory()` calls fails because the handle
   is consumed/closed after the first call.
3. **Third attempt: succeeded.** Fixed by re-opening a fresh backed
   handle (`load_arc_challenge_dataset`) for every chunk, not just once.

**Infrastructure:** a fourth temporary EC2 `m7i-flex.large` instance
(same type/region; prior instances from both failed attempts were
already terminated — the second one became unreachable via SSH and was
torn down without first confirming its outcome, which is why its result,
if any, could not be retrieved), new key pair/security group, both
deleted immediately after this run. Results (checkpoint + metrics + log,
~3.7MB) mirrored to the existing S3 bucket at
`s3://virtualcell-635863384351/scale_up_real_arc/`.

**What was run:** `scripts/scale_up_real_arc.py` — fetched 45,584 real
Arc cells (vs. 10,960 in the first real-data pilot) across 150 targets,
90/30/30 train/val/test split, 1,500 top-variable genes, using the fixed
memory-bounded chunked loader (4,000-cell chunks, fresh handle per
chunk). Fetch took 902.6s (~15 min) — notably FASTER per-cell than the
first pilot's 1391s for 11k cells (0.0198 s/cell here vs. 0.127 s/cell
before), plausibly because reopening a fresh handle per 4,000-cell chunk
let `gcsfs`'s internal prefetch/cache work more efficiently than one
giant single-shot `.to_memory()` call did — not a controlled comparison,
so treat this as an observation, not a verified causal claim.

Ran the full baseline ladder (no-effect, seen-target mean, PCA+ridge,
small MLP) plus a 4-config hyperparameter sweep for the conditional
model (varying `control_hidden_dim`/`fusion_hidden_dim` in {64,128,256}
and `lr` in {1e-3, 3e-4}), selecting the best by validation loss.

**Actual result** (seed=0, 45,584 cells, 1,500 genes, 150 targets, 8,510
held-out test cells across 30 test targets):

| model | MAE (scaled) | improvement over no-effect |
|---|---|---|
| no-effect | 1.062 | 0.0% |
| seen-target mean | 1.062 | 0.0% (expected fallback) |
| PCA + ridge | 0.896 | +15.6% |
| small MLP | 0.889 | +16.2% |
| **conditional model (tuned, 256/256 hidden, lr=1e-3)** | **0.878** | **+17.4%** |

Grouped bootstrap CI on the tuned conditional model's per-target MAE (30
target groups): mean 0.169, 95% CI [0.137, 0.210].

**Headline honest finding — this REVERSES the earlier (smaller-scale,
untuned) result:** at ~4x the real-data scale (45,584 vs. 10,960 cells)
with light hyperparameter tuning, **the conditional model now
outperforms both PCA+ridge and the small MLP**, where previously
(2026-11-05, 10,960 cells, untuned conditional model) PCA+ridge had won
outright (+15.0% vs. the untuned conditional model's +12.3%). The
hyperparameter sweep itself shows a clear, monotonic-ish trend: the
largest capacity config tried (256/256 hidden dim) had the best
validation loss (0.502) of the four configs, suggesting the conditional
model was likely undertrained/undersized relative to available signal in
the earlier, smaller/untuned comparison — consistent with the
`docs/results.md` real-data-ladder entry's own open question ("whether
the conditional model is undertrained").

**Honest limits on this conclusion:**
- Still a subsample (45,584 of 221,273 cells; 1,500 of 18,080 genes),
  not the full dataset.
- Only ONE random seed and ONE held-out-target split were used for this
  comparison; the margin between the top three models (0.878 vs 0.889 vs
  0.896 — all within ~0.02 MAE of each other) is not necessarily outside
  normal seed-to-seed variance. A multi-seed comparison was not run this
  session due to time/compute budget, so "the conditional model wins" is
  directionally supported but not statistically confirmed at this
  margin.
- The hyperparameter sweep only varied capacity and learning rate for
  the conditional model; PCA+ridge's `alpha`/`n_components` and the
  small MLP's `hidden_dim` were NOT re-tuned at this larger scale, so
  this is not a fully fair "best of each" comparison — it is a "better-
  tuned conditional model vs. still-default baselines" comparison, which
  somewhat favors the conditional model's win. Re-tuning the baselines
  too is the natural next step before trusting this reversal fully.

**Status vs. plan:** this is real progress toward the "improve and tune
the conditional PyTorch model" instruction — tuning on more real data
measurably changed the model-vs-baseline ranking, which is itself useful
information (model capacity/tuning matters more than the earlier,
smaller-scale comparison suggested). Multi-cell-type generalization
(the next planned phase) remains untouched this session, as agreed.

## 2026-11-10 — Fair retuned comparison: small MLP wins (second reversal)

**Why:** the previous entry's "conditional model beats baselines" result
only retuned the conditional model, leaving PCA+ridge and the small MLP
at their original defaults — an explicitly flagged limitation. Owner
asked to retune the baselines before trusting that result.

**What was built:** `hyperparameter_sweep_pca_ridge` (grid over
`n_components` in {10,20,50} x `alpha` in {0.1,1.0,10.0}, 9 configs) and
`hyperparameter_sweep_small_mlp` (grid over `hidden_dim` in
{32,64,128,256}, 4 configs) in `scripts/scale_up_real_arc.py`, both
selected by validation MAE — the same principle already used for the
conditional model's sweep. `tests` unaffected (41 pass, 1 skipped,
confirmed before deploying).

**Infrastructure:** a fifth temporary EC2 `m7i-flex.large` instance (same
type/region), new key pair/security group, both deleted immediately
after. Results (~3.7MB) mirrored to the existing S3 bucket at
`s3://virtualcell-635863384351/scale_up_real_arc_retuned/`. This time the
run was watched end-to-end via live log tailing until the result file
was explicitly confirmed present, before touching any infrastructure —
unlike the second scale-up attempt (2026-11-05/11-06), which was torn
down on an SSH disconnect without confirming its outcome.

**Command run and actual result** (same 45,584-cell / 1,500-gene / 150-
target subsample and 90/30/30 split as the previous entry, so directly
comparable):

| model | MAE (scaled) | improvement over no-effect | best config |
|---|---|---|---|
| no-effect | 1.062 | 0.0% | — |
| seen-target mean | 1.062 | 0.0% (expected fallback) | — |
| PCA + ridge (tuned) | 0.885 | +16.6% | n_components=50, alpha=10.0 |
| **small MLP (tuned)** | **0.859** | **+19.1%** | hidden_dim=256 |
| conditional model (tuned) | 0.878 | +17.4% | 256/256 hidden, lr=1e-3 |

**Headline, honest finding — a SECOND reversal, not smoothed over:**
once all three models get a comparable tuning budget, **the small MLP
baseline wins outright** (+19.1%), ahead of both the conditional model
(+17.4%) and PCA+ridge (+16.6%). The previous entry's "conditional model
wins" was an artifact of an unfair comparison (only it was retuned).
With fair tuning, the simplest nonlinear baseline — not the architecture
this project spent the most engineering effort on — is currently the
best performer on this subsample.

Pattern across all three real-data comparisons this project has run:

| comparison | winner |
|---|---|
| 10,960 cells, nothing tuned | PCA+ridge |
| 45,584 cells, only conditional model tuned | conditional model |
| 45,584 cells, all three tuned | **small MLP** |

**Honest reading:** more data and tuning clearly help (no-effect to
~+17-19% is a real, repeatable signal), but which architecture wins is
NOT stable across these three conditions — it has flipped every time the
comparison protocol changed. This is a meaningful, if humbling, result:
the conditional model's extra structure (separate control/target
encoders + fusion) has not yet demonstrated a clear, fair advantage over
much simpler alternatives on real data, at this scale, with this target
descriptor. It is not refuted either — only one seed/split has been
tested throughout, and the margins (0.859 vs 0.878 vs 0.885 — all within
0.026 MAE) are not necessarily outside normal variance.

**Status vs. plan:** this directly serves the plan's own antipattern
warning (section 10: "build a giant transformer before baselines are
reproducible") — the honest answer right now is that the baselines ARE
competitive with, and in this fair comparison slightly ahead of, the
conditional model. Continuing to add architecture complexity (Milestone
5's pretrained encoder) without first understanding why would repeat
exactly the mistake the plan warns against.

**Not done / explicitly deferred:**
- Multi-seed/multi-split replication to check if this ranking is stable
  or seed noise — the single biggest gap before trusting any of these
  three rankings as "the" answer.
- Investigating WHY the small MLP's single hidden layer over
  [control_profile, target_descriptor] concatenated directly outperforms
  the conditional model's separate-encoders architecture — plausible
  hypotheses (the fusion/separate-encoder structure adds capacity without
  adding information, given the weak 2-scalar target descriptor) are not
  yet tested.
- Full dataset / full gene vocabulary.
- Multi-cell-type generalization (remains the next explicitly agreed
  phase, still untouched).

## 2026-11-10 — Investigating the gap: richer descriptor does NOT close it

**Why:** owner asked to investigate why the small MLP beats the
conditional model before moving to multi-cell-type data. Hypothesis:
the shared 2-scalar target descriptor caps all models equally, so a
richer descriptor might let the conditional model's extra structure
(separate encoders + fusion) finally show an advantage.

**What was built:** `scripts/scale_up_real_arc.py` parameterized with
`--descriptor {mean_std,coexpression}` and `--n-components`, reusing the
already-built `fit_coexpression_descriptors` (gene correlation profile,
PCA-reduced) so the full fairly-tuned 5-model ladder can run under
either descriptor on real data, not just the synthetic fixture.

**Infrastructure:** a sixth temporary EC2 instance (same type/region),
terminated with security group/key pair deleted immediately after
pulling results. Results mirrored to
`s3://virtualcell-635863384351/scale_up_coexpression/`.

**Command run and actual result** (identical 45,584-cell subsample,
split, and hyperparameter grids as the mean_std comparison, only the
target descriptor changed):

| model (all tuned) | mean/std MAE | mean/std improvement | coexpression MAE | coexpression improvement |
|---|---|---|---|---|
| PCA + ridge | 0.885 | +16.6% | 0.889 | +16.3% |
| **Small MLP** | **0.859** | **+19.1%** | **0.861** | **+18.9%** |
| Conditional model | 0.878 | +17.4% | 0.874 | +17.7% |

**Conclusive, honest finding: the richer descriptor changes almost
nothing.** Every model's MAE shifted by at most 0.004-0.005 between
descriptors — well within normal run-to-run noise, and the ranking
(small MLP > conditional model > PCA+ridge) is identical under both.
**This rules out the target descriptor as the explanation for the small
MLP's advantage.** The hypothesis going into this test — "a richer
descriptor would let the conditional model's structure pay off" — is
not supported by this evidence.

**What this means for next steps:** since swapping the descriptor did
essentially nothing, the small-MLP-beats-conditional-model gap is more
likely explained by one of:
1. The conditional model's separate-encoders-then-fusion architecture
   adds optimization difficulty (more parameters to coordinate) without
   adding expressive power the single-layer MLP lacks, given these two
   input blocks are not complex enough to need separate processing.
2. Both models are already near a performance ceiling set by the data
   itself (45,584 cells, 1,500 genes, still a subsample) rather than by
   either model's capacity — in which case neither architecture change
   nor descriptor richness would move the needle much further without
   more real data or a genuinely different kind of input (a real
   pretrained gene representation, not a hand-engineered statistic of
   any dimensionality).
3. A regularization/overfitting difference not yet measured directly
   (train vs. val vs. test MAE gaps were not compared across models this
   session).

**Not done / explicitly deferred:**
- Directly comparing train/val/test MAE gaps per model to check for
  differential overfitting (hypothesis 3).
- Any change to the conditional model's architecture itself (e.g.
  simplifying it, or testing whether removing the separate encoders in
  favor of simple concatenation-then-MLP, which is close to what the
  small MLP baseline already is, recovers the gap).
- Multi-seed replication of this specific finding.
- Multi-cell-type generalization — remains the next agreed phase.

## 2026-11-10 — Overfitting check: NOT overfitting, an optimization/fit issue

**Why:** after ruling out the target descriptor, the remaining open
hypotheses were (1) the conditional model overfits more than the small
MLP, or (2) a performance ceiling / architecture efficiency issue.
Checked by comparing train/val/test MAE for all three tuned models on
the same mean_std-descriptor comparison (identical 45,584-cell
subsample/split/hyperparameter grids as the earlier fair comparison).

**Infrastructure note (process change):** per the owner's explicit
instruction, this run used a NEW persistent EC2 instance
(`m7i-flex.large`, tagged `virtualcell-working`) that is being kept
running across experiments rather than terminated after each one, to
avoid repeatedly paying the ~15 minute GCS data-fetch cost. It is NOT
terminated as of this entry — see the top of the log (or ask the
session) for its current status before assuming it is still up.

**Command run and actual result:**

| model | train MAE | val MAE | test MAE | train→test gap |
|---|---|---|---|---|
| PCA + ridge | 0.882 | 0.888 | 0.885 | 0.003 |
| Small MLP | 0.829 | 0.862 | 0.859 | 0.030 |
| Conditional model | 0.845 | 0.882 | 0.878 | 0.033 |

Full JSON: `outputs/scale_up_overfitting_check/metrics.json`.

**Conclusive finding: this is NOT an overfitting difference.** The small
MLP and conditional model have nearly identical train→test generalization
gaps (0.030 vs 0.033) — so the conditional model is not overfitting more
than the small MLP in any meaningful sense. What actually distinguishes
them: **the small MLP fits the TRAINING data better too** (0.829 vs
0.845 train MAE) — it is not winning by generalizing better from equal
fit, it is winning by fitting better at every stage (train, val, and
test). This rules out hypothesis (1) and points toward hypothesis (2)
from the previous entry: the conditional model's separate-encoders-then-
fusion architecture is less effective at fitting this data, not more
prone to memorizing it. Plausible causes not yet isolated: harder
optimization landscape from the extra structure, insufficient training
budget (same epoch/patience settings were used for both), or the fusion
step diluting information that concatenation preserves directly.

PCA+ridge, as expected for a regularized linear model, shows essentially
no train-test gap (0.003) — it is not overfitting either, it is simply
underfitting relative to both neural models (worst test MAE of the
three), consistent with it being a linear model applied to a
plausibly-nonlinear relationship.

**Status vs. plan:** this closes out the investigation chain requested
by the owner (richer descriptor → ruled out; overfitting → ruled out).
The honest remaining open question is whether the conditional model's
specific architecture (not its capacity, not overfitting) needs
rethinking, or whether more training budget/different optimization
settings would close the gap — neither tested yet.

**Not done / explicitly deferred:**
- Training the conditional model for longer / with different
  optimizer settings to see if the train-fit gap closes with more
  budget (the sweep used the same max_epochs=100/patience=12 for all
  configs, which may simply be insufficient for the more complex model).
- Simplifying the conditional model's architecture (e.g. testing a
  version closer to simple concatenation instead of separate encoders +
  fusion) to directly test the "fusion dilutes information" hypothesis.
- Multi-cell-type generalization — remains the next agreed phase.

## 2026-11-10 — Longer training/optimization: also ruled out

**Why:** per the owner's requested order (training budget → scale data →
architecture/Milestone 5), tested whether the conditional model's gap vs.
the small MLP is a training-budget/optimization issue — the last
untested hypothesis from the investigation chain (descriptor: ruled out;
overfitting: ruled out).

**Infrastructure:** reused the PERSISTENT EC2 instance (`virtualcell-
working`) kept running across experiments per the owner's instruction —
NOT terminated after this run. Also added local disk caching to
`subsample_and_materialize` (`.cache/subsample_{params}.npz`), so this
run's ~15-minute GCS fetch was the last one needed for this particular
45,584-cell subsample; future experiments on the same scale will load
from cache instantly.

**Command run and actual result** (`scripts/extended_training_conditional.py`,
same 45,584-cell subsample/split as every prior comparison, 256/256
hidden conditional model, 5 configs):

| config | max_epochs | weight_decay | lr | best_epoch | train MAE | test MAE | improvement |
|---|---|---|---|---|---|---|---|
| original_short (reference) | 100 | 1e-5 | 1e-3 | 24 | 0.845 | 0.878 | +17.4% |
| longer_training | 500 | 1e-5 | 1e-3 | 62 | 0.827 | 0.878 | +17.3% |
| longer_training_more_reg | 500 | 1e-4 | 1e-3 | 74 | 0.860 | 0.882 | +17.0% |
| longer_training_less_reg | 500 | 1e-6 | 1e-3 | 47 | 0.825 | **0.875** | **+17.6% (best)** |
| longer_training_lower_lr | 800 | 1e-5 | 3e-4 | 102 | 0.816 | 0.882 | +16.9% |

Reference: small MLP (tuned) test MAE = 0.859, improvement +19.1%.

**Conclusive finding, not smoothed over: none of the five configs close
the gap.** The best variant (less regularization) only reaches 0.875 —
still 0.016 MAE behind the small MLP, and the improvement is marginal
(+17.6% vs the original +17.4%) relative to the ~5x longer training
budget spent getting there. Train MAE kept dropping with more training
(0.845 → 0.816 across configs) but test MAE did NOT track downward with
it — more training just lets the model fit the training set harder
without transferring to better test performance. This rules out
"insufficient training budget" as the explanation, joining the already-
ruled-out target descriptor and overfitting hypotheses.

**Status: all three investigated hypotheses for the gap are now ruled
out** (descriptor richness, overfitting, training budget/regularization).
The remaining live hypotheses are: (a) the architecture itself (separate
encoders + fusion) is less effective at this task than direct
concatenation, independent of training details, or (b) a genuine data-
scale ceiling (45,584 of 221,273 cells) that affects the more complex
model differently than the simple one — untested until now.

**Next step per the owner's agreed order:** scale up the real-data
subsample (more cells/genes) before touching the architecture or
escalating to Milestone 5's pretrained encoder, to test hypothesis (b)
directly.

**Not done / explicitly deferred:**
- Testing a simplified conditional model (e.g., closer to plain
  concatenation) to directly test hypothesis (a).
- Learning rate schedules / different optimizers (only Adam at two
  learning rates was tried).

## 2026-11-10 — Scale-up to ~126k cells: blocked by a persistent, unresolved fetch failure

**Status: NOT resolved this session.** After the owner's explicit go-ahead
to try a different fetch approach, four progressively more invasive
fixes were tried for the same first chunk (800 of 6,000 control cells)
at the larger ~126k-cell target scale. All four failed the same way:
alive, near-zero CPU usage, no progress, no exception, for many minutes.

1. **Explicit handle close + periodic gcsfs cache clear** — did not help.
2. **Smaller chunk size (4000 → 800)** — did not help.
3. **Hard 90s timeout + retry in a worker THREAD** — the thread-based
   retries all failed too; the log revealed why: gcsfs's background
   async prefetcher failed to tear down cleanly once
   ("BackgroundPrefetcher teardown did not complete... it will keep
   running in the background"), and every subsequent same-process retry
   also timed out — the corrupted async/event-loop state appears to
   persist for the rest of that Python process.
4. **Hard 120s timeout + retry in a separate OS PROCESS**
   (`scripts/_fetch_chunk_worker.py`, invoked via `subprocess.run`) —
   this was expected to fully solve it (fresh interpreter, fresh event
   loop every attempt, no possible shared corrupted state). It did NOT:
   all 3 process-isolated attempts also stalled and were killed cleanly
   by the timeout, and the script correctly raised its own `RuntimeError`
   after exhausting retries — confirmed via a clean traceback, no orphan
   processes left behind.

**This rules out in-process Python/asyncio state corruption as the root
cause** (since a genuinely fresh OS process still failed the same way).
The most likely remaining explanation is something at the network/host
level specific to this persistent EC2 instance's connection to Google
Cloud Storage for this exact access pattern (anonymous, scattered
non-contiguous range reads at this scale) — not confirmed, since this
was not investigated further (e.g. no packet capture, no test from a
different instance or region was attempted this session).

**What changed in the codebase (kept, since each is independently sound
engineering even though none fixed the root cause):** memory-bounded
two-pass loading (fixes a real, confirmed OOM at larger scale),
subprocess-based chunk fetching with hard timeout + retry
(`scripts/_fetch_chunk_worker.py`), local subsample caching. These are
genuine improvements over the original single-shot loader and remain in
place for future attempts.

**Honest summary for the owner:** the ~45,584-cell scale-up (Milestone 5
log entries above) remains the actual, trustworthy result set. The
~126k-cell scale-up is blocked, not abandoned by choice — it hit a wall
that four different fixes (targeting the three most likely explanations:
resource leak, chunk size, in-process async corruption) did not resolve,
and the most likely remaining cause (host/network-level) was not
investigated this session.

**Recommended next steps, not yet attempted:**
- Try fetching from a FRESH EC2 instance (not the persistent
  `virtualcell-working` one) to check whether this is instance-specific.
- Try a different, non-random, CONTIGUOUS range of cell positions for
  the first chunk, to check whether scattered/non-contiguous reads
  specifically are implicated.
- Try `gsutil`/`gcloud storage` directly (outside the Python/gcsfs stack
  entirely) for the same byte range, as an independent test of whether
  the problem is gcsfs-specific or a genuine GCS/network issue.
- Check AWS's own network path (e.g. via VPC flow logs or a basic
  `curl`/`traceroute` test to `storage.googleapis.com`) for packet loss
  or routing anomalies specific to this instance/AZ.

The persistent EC2 instance (`virtualcell-working`) remains running per
the owner's instruction — not terminated, idle, no cost accruing beyond
baseline compute while idle.

## 2026-11-10/11: Root cause confirmed; scale-up to ~106k cells completed

**Root cause confirmed (live diagnosis, before any more blind fixes, per
owner request):** inspected the Arc h5ad's HDF5 metadata directly (cheap,
fast, no large download). `X/data` and `X/indices` are chunked at 58,977
elements (~230KB, covering only ~6-7 rows' worth each), uncompressed.
With sample positions scattered roughly uniformly across all 221,273
rows (average gap ~37 rows for a 6,000-cell sample, far exceeding the
~6.76-rows/chunk density), nearly every sampled row lands in its own
separate HDF5 chunk — so fetching N scattered rows costs ~N separate
small HTTP range requests over GCS, latency-dominated and effectively
unbounded at scale. This explains why retries (same request pattern,
same cost) and process isolation (same request pattern, different
process) both failed to fix it earlier — neither reduces the round-trip
count, which was the actual problem.

**Fix:** download the full ~15.5GB file once as a single sequential
transfer (`scripts/download_arc_h5ad.py`, `gcsfs.get()` — completed in
111.8s, 1.1 Gbps), then read all subsequent row slices from local disk
(`--h5ad-path` flag added to `scripts/scale_up_real_arc.py`). The
identical scattered-position fetch that previously stalled indefinitely
completed in 26.3s against the local copy — direct empirical
confirmation the diagnosis was correct.

**Follow-up bug found and fixed during the first resumed run:** the
subsample cache (`.cache/subsample_*.npz`) saved `obs`/`var` via
`np.array(df, dtype=object)`, which flattens a DataFrame to its values
array rather than wrapping it as a single pickled object — so
`cached["obs"].item()` on load raised `ValueError: can only convert an
array of size 1 to a Python scalar`. Fixed by wrapping in a 0-d object
array (`np.empty((), dtype=object); holder[()] = df`) before `savez`.
Deleted the one corrupted cache file and let it re-fetch (~213s) to
produce a correctly-cached file.

**Scale-up run completed:** 106,287 cells, 150 targets, full baseline
ladder with independent hyperparameter tuning per model. Conditional
model's hyperparameter sweep (4 configs, each trained to convergence)
took the bulk of the wall-clock time on the 2-vCPU instance — individual
configs ranged from ~4.5 to ~10 minutes, genuinely CPU-bound (verified
via `ps` CPU% staying high throughout, not stalled) rather than hung.
Full results in `docs/results.md` under "Scale-up to ~106k real cells:
the architecture gap persists" — short version: small MLP still wins
(0.891 test MAE) over the tuned conditional model (0.895), same ranking
as the smaller-scale comparison; the architecture gap did not close or
reverse with ~4x more data.

Per owner instruction, the persistent EC2 instance (`virtualcell-working`,
`i-0a277e6b18a5d5af9`) was **stopped** (not terminated) once this run's
results were pulled back locally, to halt compute billing while
preserving the environment/venv/local h5ad copy/cache for a cheap
resume later.

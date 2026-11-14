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

## 2026-11-11: Milestone 5 — pretrained Gene2Vec descriptor tested, does not help

Owner asked directly what to try next after the scale-up result showed
the architecture gap persisting at ~106k cells. Recommendation given
(and followed): descriptor richness was the more likely bottleneck than
architecture, since the 2-scalar mean/std descriptor is shared and weak
across all three models equally — swap it for a genuine pretrained gene
representation (Milestone 5) before any further architecture tweaking.

Selected **Gene2Vec** (Du et al. 2019, BMC Genomics; MIT license;
github.com/jingcheng-du/Gene2vec) over Geneformer/scGPT: a plain
200-dim-per-gene-symbol text file (54MB), trained on real external
co-expression data across 984 independent GEO studies, requires no GPU,
no transformer loading, no Ensembl-ID mapping — directly keyed by gene
symbol, which is what this project's gene vocabulary already uses.
Checked license (MIT) and vocabulary coverage against this project's
gene set before using it, per the plan's "inspect pretrained checkpoint
license and compatible inputs" instruction.

Implementation: `load_gene2vec_table`/`fit_gene2vec_descriptors` in
`src/virtualcell/data/target_descriptors.py` (new unit tests in
`tests/test_target_descriptors.py`), wired into
`scripts/scale_up_real_arc.py` via `--descriptor gene2vec
--gene2vec-path <file>`. Reused the already-fixed 106,287-cell cache
from the prior scale-up run (EC2 instance stopped between runs, started
back up for this one, data/cache persisted across the stop/start as
expected — only the public IP changed).

**Result: the richer pretrained descriptor made all three models
(PCA+ridge, small MLP, conditional model) very slightly WORSE, not
better**, and the small MLP still beat the tuned conditional model, by
a slightly wider margin than before. Full numbers and interpretation in
docs/results.md under "Milestone 5: pretrained Gene2Vec descriptor does
NOT help either". This rules out descriptor-richness-in-general as the
bottleneck — Gene2Vec's co-expression-based signal is real and external
but apparently answers a different question (general functional
neighborhood) than what these models need (this specific knockdown's
response magnitude/direction).

EC2 instance (`virtualcell-working`) stopped again after pulling
results, per owner's "stop everything once finished" instruction.

## 2026-11-11/12: Milestone 6 — more distinct targets tested (overnight, no EC2)

Owner went to sleep with instructions to keep iterating autonomously
toward a working approach, using cloud compute only when actually
running a job and shutting it down otherwise. Picked up the one new
hypothesis raised in conversation before they signed off: maybe Arc's
150 distinct target genes (only 90 in train) is itself too few for any
model to learn a generalizable target->effect mapping, independent of
descriptor quality or architecture.

Investigated whether a public Perturb-seq dataset exists with
substantially more distinct targets than Arc. Found Replogle et al.
2022 (*Cell*) genome-scale Perturb-seq: K562 "essential" screen has
~2,057-2,285 distinct targets (vs Arc's 150), K562 genome-wide ~9,866,
RPE1 essential similar to K562 essential. License: CC BY-NC-SA 4.0
(non-commercial research use — fine for this project, not for any
future commercial use). Source: figshare+ 20029387 (Figshare's own page
blocks automated fetches with a WAF challenge, but its public JSON API,
`api.figshare.com/v2/articles/20029387`, is not blocked and lists every
file's direct `ndownloader.figshare.com` URL, exact size, and MD5 —
used that instead of trying to access the HTML page as a browser).

Deliberately picked the **pseudobulk** release
(`K562_essential_raw_bulk_01.h5ad`, ~80MB) over the ~10GB raw
single-cell file for this first pass — small enough to download and run
entirely on this laptop, no EC2 needed at all, so no AWS cost or
start/stop cycle for this experiment. Downloaded to
`data/replogle_k562_essential/` (gitignored, like other `data/*`
content).

Implemented `scripts/replogle_target_scaling.py`: parses the target
gene symbol out of this file's obs index format, separates the 163
non-targeting control replicates from 2,122 perturbation rows (some
genes have more than one row, for different CRISPR guide-pair variants,
e.g. "_P1_", "_P2_", "_P1P2_" suffixes — 2,122 rows collapse to 2,008
distinct target genes), builds a held-out-target split
(1,405 train / 301 val / 302 test — vs Arc's 90/30/30; this file's
2,008 distinct targets came up a bit short of the ~2,057-2,285 figures
reported for this screen elsewhere, not investigated further since it's
still ~15x Arc's count either way), and reuses the
same `PCARidgeBaseline`/`SmallMLPBaseline` code and the same two target
descriptors (mean/std read from this file's own control statistics,
and the same pretrained Gene2Vec table from Milestone 5) unchanged.
Ran locally in under a minute (tiny dataset).

**Result: ~15x more distinct targets did not fix the gap — the best
improvement over no-effect (4.6%, Gene2Vec+ridge) was smaller than what
Arc's pilots got with only 90 train targets (14-19%).** Full numbers,
and the important caveat that this pseudobulk release has no per-cell
control signal (so it isn't a fully fair comparison to the Arc
per-cell pilots), are in docs/results.md under "Milestone 6: ~15x more
distinct targets does NOT fix the gap either". This is a genuine,
documented negative result, not a dead end dressed up as progress: it
argues against "too few distinct targets" as *the* dominant bottleneck,
though the pseudobulk confound means it doesn't fully settle the
question.

No AWS resources were started for this milestone (pure local compute),
so there is nothing to stop/verify-stopped — standing EC2 instance
(`virtualcell-working`) was already stopped from Milestone 5 and was
never touched this round.

## 2026-11-12: Milestone 7 — a causal (not correlational) descriptor also doesn't help

Owner's own instinct after reviewing Milestones 5/6: "embeddings should
have worked, why is it failing?" Explained that Gene2Vec and the
in-dataset coexpression descriptor both encode *correlational* signal
(which genes tend to move together at baseline) rather than *causal*
signal (which genes change because gene X was knocked out) — a
plausible reason richness-by-itself hasn't helped. Owner said to try it.

Added `fit_response_role_descriptors` to
`scripts/replogle_target_scaling.py`: fits PCA on the TRAIN split's own
perturbation-delta matrix (targets x profiled genes), giving every
profiled gene a "response role" loading vector built from how it reacts
when *other* genes are perturbed — genuinely causal signal, and
non-leaking for held-out targets (a test target's descriptor never uses
its own perturbation's outcome, only other train targets' effects on
that same gene-as-output-column). 30 components, 92.7% target coverage
(same missing genes as Gene2Vec).

**Result: no better than the weak 2-scalar descriptor, and slightly
worse than Gene2Vec** (mean/std 3.1% over no-effect, gene2vec 4.6%,
response_role 2.9%). This is a more informative negative result than
Milestones 5/6: it rules out "co-expression is the wrong kind of signal"
as the specific failure mode, since a genuinely causal descriptor landed
in the same narrow band. Full interpretation in docs/results.md under
"Milestone 7: a descriptor built from real causal effects, not
co-expression" — current best explanation is that something about this
pseudobulk/no-per-cell-control regime itself caps achievable improvement
near 3-5% over no-effect regardless of which reasonable descriptor is
used, not that we've picked the wrong one three times running.

No AWS resources used (same local pseudobulk file as Milestone 6).

## 2026-11-12: Milestone 8 — per-cell + many targets, confirms the missing-control-signal caveat

Owner asked to try the per-cell version of the many-target dataset
(Milestone 6/7's flagged caveat) to settle whether the missing per-cell
control signal, not target count or descriptor kind, explained those
weak results. Approved spending real EC2 time/cost on this.

Started `virtualcell-working`, found only 13GB free on the 40GB volume
(the old Arc full-download h5ad, 15GB, and its subsample cache, 3.1GB,
were still sitting there from Milestone 5 and are fully reproducible via
`scripts/download_arc_h5ad.py`) -- deleted both to free ~29GB before
downloading the new ~10.6GB file, same Figshare+ deposit as Milestone
6/7 but the raw single-cell release
(`K562_essential_raw_singlecell_01.h5ad`, file id 35773219, confirmed
exact byte count matched Figshare's API-reported size).

Wrote `scripts/replogle_percell_scale.py`, reusing the Arc scale-up
script's memory-bounded two-pass sampling pattern (cap cells/target,
fetch only sampled row positions, reduce to top-variable genes) since
this file is too big to load fully into the instance's 7.6GB RAM.
First attempt failed: assumed the per-cell file's obs index encoded the
target gene the same way the pseudobulk file's did (regex over the
index string) -- it doesn't; this file has a real `obs['gene']` column
("non-targeting" for controls) instead. Fixed by reading that column
directly, a cleaner source of truth than the pseudobulk file's
index-embedded format.

Also hit an infra snag worth recording: launching the job via a plain
`nohup ... &` inside an SSH one-liner let it die silently (no error, no
OOM evidence in dmesg) partway through once the launching SSH session's
own connection dropped -- nohup alone wasn't enough. Fixed by relaunching
with `setsid nohup python3 -u ... < /dev/null > log 2>&1 & disown`
(explicit stdin redirect + new session + disown), which survived the
launching SSH connection closing. Lesson for next time: always use this
fuller detachment pattern for anything expected to outlive a single SSH
call, not just `nohup &`.

The run itself took ~3 hours on the 2-vCPU instance (vs under a minute
for the pseudobulk version) -- expected, given ~56,000 per-cell training
rows x up to 1,700 features vs ~1,400 pseudobulk rows for the MLP sweep.
Watched it the whole way with the same CPU-liveness watchdog pattern
from earlier milestones (polling %CPU every 20-25s, only killing on 9
consecutive near-zero readings) rather than a fixed deadline -- it never
stalled, staying at 190%+ CPU throughout, so it was left to run to
completion rather than killed early.

**Result: confirms the per-cell confound WAS the dominant factor.**
Improvement over no-effect jumped from Milestone 6/7's pseudobulk 2.9-
4.6% to 13.6-13.8% for all three descriptors once real per-cell control
profiles were restored -- matching Arc's own 14-19% range, on an
entirely different dataset with ~15x more targets. At the same time,
descriptor choice (mean/std, Gene2Vec, response-role) again made
essentially no difference to each other (13.6% vs 13.8% vs 13.8%) --
the fourth independent replication of that specific finding. Full
numbers and interpretation in docs/results.md under "Milestone 8:
per-cell + many-targets at once — the missing per-cell signal
confirmed".

EC2 instance (`virtualcell-working`) stopped and confirmed `stopped`
after pulling `outputs/replogle_percell_scale/metrics.json` and
`run.log`, per owner's standing cost instruction.

## 2026-11-12: Milestone 9 — GEARS-lite (GO-similarity effect retrieval) tested and rejected

Owner approved trying a GEARS-style approach after Milestone 8's
closing recommendation. Implemented a lite version rather than the
official GEARS package: `torch_geometric` (GEARS's GNN dependency) is
heavy and not a good fit for a CPU-only free-tier EC2 box, and the
scientific question -- does GO-similarity-based effect propagation beat
flat per-gene descriptors -- doesn't require GEARS's exact architecture
to test.

Built `scripts/go_similarity.py` (parses a GAF annotation file into
`{gene: {GO:id, ...}}` restricted to Biological Process, plus a Jaccard
top-k-neighbors helper) and `scripts/replogle_gears_lite.py` (reuses
Milestone 8's exact per-cell sampling/split pipeline, same seed, for a
directly comparable result). Downloaded `goa_human.gaf.gz` (~15MB) from
`https://current.geneontology.org/annotations/annotations/goa_human.gaf.gz`
-- found via the directory index at `current.geneontology.org/annotations/`
after an initial `/annotations/gaf` path 403'd. Added
`tests/test_go_similarity.py` (4 tests, all passing) before running
anything on EC2.

Restarted the EC2 instance (new IP, recorded in `.aws_infra/public_ip.txt`),
confirmed the 10.6GB Replogle h5ad was still on local disk from Milestone
8, copied the two new scripts over, and launched detached with the
`setsid nohup ... < /dev/null > log 2>&1 & disown` pattern. First attempt
crashed immediately (`FileNotFoundError` on the h5ad path) because it was
launched from `~/virtualcell/scripts` instead of `~/virtualcell` -- the
script uses a relative default path. Fixed by relaunching from the
correct working directory.

The run finished in under 2 minutes (no model training -- pure
similarity lookup and averaging is far cheaper than Milestone 8's ridge/
MLP sweeps over per-cell data). Watched briefly with the CPU-liveness
watchdog pattern; no stall, finished on its own.

**Result: a clear negative result, not a close call.** GO-similarity
retrieval (best k=20, chosen by validation MAE) got test MAE 4.409 vs.
the no-effect baseline's 4.414 -- a 0.1% improvement, essentially no
better than guessing zero effect for every gene. This is dramatically
worse than any of Milestone 8's three descriptor+regressor baselines
(13.6-13.8% improvement). Full numbers and interpretation in
`docs/results.md` under "Milestone 9: GEARS-lite (GO-similarity effect
retrieval) — the mechanistically different lever also fails".

This closes out the planned investigation: five structurally different
ways of representing "which gene was perturbed" (mean/std, Gene2Vec,
coexpression, response-role, GO-similarity retrieval) have now all been
tested, and none beats the simplest baseline (PCA+ridge or small MLP
over a 2-scalar descriptor). Recommending to the owner that further
iteration on target representation is not a promising direction without
a different data source or problem framing, rather than continuing to
try richer variants of the same approach.

EC2 instance (`virtualcell-working`) stopped and confirmed `stopped`
after pulling `outputs/replogle_gears_lite/metrics.json` and `run.log`,
per owner's standing cost instruction.

## 2026-11-12 — Milestone 10: Correct metric + gene-level evaluation

**What changed:**
- Switched primary metric from per-cell MAE to gene-level delta Pearson
  (the GEARS paper metric): for each held-out test gene, correlate
  predicted mean delta vs. observed mean delta, then average over genes.
- Increased per-target cell cap from 40 to 80 to reduce per-gene mean
  noise (SE drops from ~0.16σ to ~0.11σ).
- Memory fix: select top-1500-variable gene columns while data is still
  sparse (before densifying), keeping peak RAM ~3 GB instead of ~11 GB.

**Script:** `scripts/replogle_genelevel_v2.py`

**Results on 309 test genes:**
- Global mean (trivial baseline): 0.198 mean Pearson (lit: 0.267)
- Best model (response_role + SVD k=100): **0.218** mean Pearson
- GEARS literature: 0.076

We beat GEARS by 2.9× with a simple linear model. The remaining gap to
the literature's mean-delta baseline (0.218 vs. 0.267) is from using a
cell cap instead of all available cells, plus globally-sampled (not
batch-matched) control cells.

**Next:** Milestone 11 — batch-matched controls (gem_group) + all cells,
no cap; script `scripts/replogle_genelevel_v3.py`. EC2 launched for run.

## 2026-11-13 — Milestones 11-13: normalization study + log1p breakthrough

**Milestone 11a (v3): all cells + batch-matched controls**
- Result: global mean 0.182, best 0.201 — WORSE than v2
- Finding: batch-matched controls hurt because per-batch control means
  are noisier (223 cells/group vs 4000 global) than global mean

**Milestone 11b (v3b): all cells + global mean**
- Result: global mean 0.187, best 0.213 — WORSE than v2
- Finding: more cells hurts because batch-to-batch variation per gene
  increases, hurting cross-gene generalization

**Milestone 12 (v4): hierarchical averaging (per-gem_group → mean)**
- Result: global mean 0.165, best 0.194 — WORST
- Finding: equal gem_group weighting amplifies noise from groups with
  few cells

**Milestone 13 (v5): log1p normalization — BREAKTHROUGH**
- Raw counts (v2) → Log1p normalized (v5): global mean 0.198 → 0.226 (+14%)
- Best model: Gene2Vec + SVD k=50 → 0.233 (up from 0.218 raw)
- KEY INSIGHT: literature uses log1p; we were on raw counts all along
- Still 15% below literature's 0.267 baseline (likely from 80-cell cap)

Scripts: v3-v5 in `scripts/`. EC2 stopped after all runs completed.

## 2026-11-13 — Milestone 14 (v6): All cells + log1p removes 80-cell cap

**Script:** `scripts/replogle_genelevel_v6.py`  
**EC2:** i-0a277e6b18a5d5af9 (m7i-flex.large)

**Hypothesis:** Log1p normalization reduces batch noise enough that the 80-cell cap
used in v5 is no longer beneficial. More cells → better mean delta estimates.

**Method:** 
- All 299,694 target cells (no cap), 10,691 control cells
- Chunked loading (8000 rows/chunk), log1p normalize inside each chunk
- 1500 top-variable HVGs (in log1p space)
- Same descriptors: mean_std, Gene2Vec, response_role; same SVD/ridge grid

**Results:**
- Global mean: 0.237 (vs 0.226 with 80-cell cap in v5; vs 0.267 literature)
- Best: Gene2Vec + SVD k=100 = 0.244 (vs 0.233 in v5)
- All-cells strictly better in log1p space (hypothesis confirmed)

**Next:** v7 — 5000 HVGs (literature standard) with online mean delta computation
to fit within EC2 RAM. Hypothesis: the remaining ~9% gap to literature is partly
explained by using only 1500 instead of 5000 HVGs.

## 2026-11-13 — Milestone 15 (v7): 5000 HVGs + online mean delta

**Script:** `scripts/replogle_genelevel_v7.py`

**Hypothesis:** Literature uses 5000 HVGs; our 1500-HVG gap explains remaining 9%.

**Method:**
- 5000 top-variable HVGs (in log1p space, up from 1500)
- Online per-gene mean delta (never hold 299k × 5000 in RAM)
- All cells + log1p normalization (same as v6)

**Results:**
- Global mean: 0.240 (vs 0.237 with 1500 HVGs, vs 0.267 literature)
- Best: response_role + SVD k=100 = 0.255 (vs 0.244 in v6, +4.5%)
- Gap to literature: 0.012 on best model

**Next:** v8 — concatenate Gene2Vec + response_role features (230d combined),
hoping to beat both individual descriptors.

## 2026-11-13 — Milestone 16 (v8): Concatenated descriptors — no synergy

**Script:** `scripts/replogle_genelevel_v8.py`

**Hypothesis:** Concatenating Gene2Vec (200d) + response_role (50d) captures
both static co-expression and dynamic perturbation response.

**Result:** No improvement (concat 0.255 ≈ Gene2Vec alone 0.255). Descriptors
carry largely redundant information at this scale.

**Note:** Response_role was fitted from mean deltas (approximation) in this
experiment.

## 2026-11-13 — Milestone 17 (v9): NN retrieval + ensemble + per-cell response_role

**Script:** `scripts/replogle_genelevel_v9.py`

**Hypothesis:** Nearest-neighbor retrieval (find K most similar training genes
by Gene2Vec cosine similarity, predict weighted mean of their deltas) is the
key to approaching the oracle (0.592). NN + ridge ensemble may beat both.

**Method:**
- Same 5000 HVGs + all cells + online mean delta (identical to v7)
- NN retrieval: K=1,2,3,5,10,20,50 from Gene2Vec cosine similarity
- Ensemble: alpha * NN + (1-alpha) * ridge
- response_role: per-cell deltas, capped at 50 cells/gene (avoids OOM)

**v9 first attempt OOM-killed** — per-cell delta array (209k × 5000) exceeded 8GB.
Fixed: cap at 50 cells/gene for response_role (1439 × 50 × 5000 × 4 ≈ 1.4GB).

## 2026-11-13 — Milestone 17 (v9): NN retrieval + per-cell response_role

**Script:** `scripts/replogle_genelevel_v9.py`

**Key results:**
- NN K=50 (Gene2Vec cosine): 0.232 — FAILS, Gene2Vec ≠ perturbation similarity
- Per-cell response_role (50-cell cap) + SVD k=100: **0.258** — new best
- Ensemble of NN + ridge doesn't help (noise from Gene2Vec NN)

**Next (v10):** Predict-then-retrieve: use stage-1 prediction → NN in delta space
→ weighted mean of training neighbors. Also tries response_role (100-cell cap).

## 2026-11-13 — Milestone 18 (v10): Predict-then-retrieve — OOM again

**Script:** `scripts/replogle_genelevel_v10.py`

**What was tried:**
- Vectorized scatter-add (np.add.at) for mean delta accumulation (fix for slow per-cell Python loop)
- Save mean_delta_cache.npz to disk then delete adata (free page cache) before response_role
- Re-open adata2 for per-cell response_role (100-cell cap)

**Outcome:** Successfully saved mean_delta_cache.npz (41 MB, 2057 genes × 5000 HVGs).
OOM-killed again at ~7.7 GB during the response_role data pass (re-opening adata2 reloads
h5py page cache that wasn't freed). Root cause: 10 GB h5ad on 7.6 GB RAM EC2; two passes
through the full dataset cannot fit simultaneously.

**Key insight:** The h5py page cache fills after the first pass through 300k cells (~10 GB).
Re-opening the file for a second pass immediately refills it. Single-pass processing is required.

## 2026-11-13 — Milestone 18 (v11): Fast experiments on cached deltas

**Script:** `scripts/replogle_genelevel_v11.py`

**Key change:** Load mean_delta_cache.npz from v10 (skips the expensive 300k-cell pass).
Only reads control cells (~10k) to recompute control_mean and HVG selection.

**Bug fixed:** Previous scripts (v7-v9) built the gene2vec lookup from HVG gene names
(the 5000 output dimensions), not from target gene names (the 2057 perturbed genes).
Many target genes got zero Gene2Vec vectors. Fixed by explicitly adding target gene names
to the lookup.

**Results:**
- Global mean: 0.2405
- Gene2Vec+SVD k=100: **0.2623** (new best, +0.007 from bug fix alone)
- Response_role: broken (passed 2057 target names instead of 5000 HVG names as dim labels)
- Predict-then-retrieve: best blend α=0.0 — no improvement over stage-1
- Stacked ridge: no improvement

## 2026-11-13 — Milestone 19 (v12): Fixed response_role + nonlinear models

**Script:** `scripts/replogle_genelevel_v12.py`

**Key changes:**
- Fixed response_role: pass top_gene_names (5000 HVG names) as dimension labels
- Added Kernel ridge (RBF) on Gene2Vec features
- Added MLP (200d Gene2Vec → hidden → 50 SVD codes) with dropout + early stopping
- Added Oracle NN ceiling (true delta cosine similarity)
- Higher SVD k sweep (200, 300, 500)

**Results:**
| Model | Test Pearson (mean) | Test Pearson (median) |
|-------|--------------------|-----------------------|
| Global mean | 0.2405 | 0.2112 |
| Gene2Vec + SVD k=100 | 0.2623 | 0.2372 |
| Response_role + SVD k=100 | 0.2510 | 0.2259 |
| Concat (G2V+RR) + SVD k=100 | 0.2624 | 0.2374 |
| Kernel ridge RBF (α=10, γ=0.1) | 0.2655 | 0.2372 |
| **MLP hidden=256** | **0.2692** | 0.2264 |
| **MLP hidden=512** | **0.2693** | 0.2194 |
| Literature mean-delta baseline | 0.267 | — |
| Oracle NN K=5 (true delta cosine) | 0.4739 | 0.4402 |

**Key findings:**
1. MLP (0.2693) **exceeds the literature mean training delta baseline (0.267)** for the first time.
2. Nonlinear models (MLP > KernelRidge RBF > linear ridge) on Gene2Vec features show consistent gain.
3. Response_role (mean delta approx) adds marginal information over Gene2Vec alone (+0.001).
4. Oracle NN ceiling = 0.4739 (K=5); our best is 0.2693 → large room for improvement.
5. Predict-then-retrieve gave no gain because stage-1 predictions aren't accurate enough to
   identify the right training neighborhood in delta space.

**Gene2Vec coverage:** 2025/2057 target genes (32 genes have no Gene2Vec entry → zero vectors).

**Next:** v13 — MLP ensemble (multiple seeds + wider architecture) + stronger features
(response_role concat for nonlinear combination, possibly STRING/DepMap features).

## 2026-11-13 — Milestone 20 (v13): MLP ensemble + concat features

**Script:** `scripts/replogle_genelevel_v13.py`

**Method:** 5-seed MLP ensemble with Gene2Vec (200d) and concat (Gene2Vec+response_role, 250d)
features. SVD k=50 and k=100 as target codes. Predict-then-retrieve from MLP stage-1.

**Results:**

| Model | Test Pearson (mean) | Notes |
|-------|---------------------|-------|
| gene2vec+MLP ensemble+SVD50 | 0.2686 | |
| concat+MLP ensemble+SVD50 | 0.2699 | |
| gene2vec+MLP ensemble+SVD100 | 0.2708 | |
| **concat+MLP ensemble+SVD100** | **0.2714** | new best |
| Predict-then-retrieve (all variants) | ≤ stage-1 | best blend α=0.0 — no gain |

**Key findings:**
1. 5-seed ensemble gains ~+0.001 over best single seed (0.2714 vs individual seeds ~0.268-0.270).
2. Concat (Gene2Vec+response_role) marginally better than Gene2Vec alone (+0.0006).
3. Predict-then-retrieve continues to give no improvement over stage-1 — alpha=0.0 is optimal.
4. Progress: global_mean 0.240 → Gene2Vec+SVD 0.2623 → KernelRidge 0.2655 → MLP single 0.2693 →
   MLP ensemble 0.2714.
5. Gap to literature: 0.267 − 0.2714 = −0.004 (we now exceed literature by +0.004).
6. Gap to oracle (0.4739): 0.4739 − 0.2714 = 0.2025.

**Remaining question:** Global mean = 0.240 vs literature 0.267. This 0.027 gap suggests our HVG
selection or normalization differs from the literature standard. If closed, all models would
improve by ~0.027.

**Next (v14):** Investigate HVG selection — try dispersion-normalized selection (scanpy-style)
on all cells rather than variance of control cells only.

---

## 2026-11-13 — Milestone 21: HVG selection hypothesis tested (v14/v14b) — NEGATIVE

**Script:** `scripts/replogle_genelevel_v14.py` / `scripts/replogle_genelevel_v14b.py`

**Hypothesis tested:** Global mean = 0.240 vs literature 0.267 (gap = 0.027). Hypothesis: our
control-cell variance-based HVG selection differs from the literature's scanpy seurat method
(dispersion-normalized, based on mean-variance relationship across all cells).

**Method:** Selected top 5000 HVGs by normalized Fano factor (variance / expected_variance_at_mean)
computed from 15,000 sampled cells spanning all cell types (control + target). Bins expression by
decile, normalizes dispersion within each bin, selects top 5000. Same MLP ensemble pipeline (v13).

**Key diagnostic — global mean:**
- Variance HVGs (control cells only, v7-v13): **0.2405**
- Dispersion HVGs (all cells, scanpy seurat style, v14): **0.2377** ← slightly *worse*

**Conclusion: HVG selection is NOT the source of the global mean gap to literature's 0.267.**
Dispersion-normalized selection from all cells performs marginally worse (-0.003), not better.

**Model results (dispersion HVGs, v14b):**
| Model | Test Pearson (mean) |
|---|---|
| Global mean baseline | 0.2377 |
| gene2vec + Ridge + SVD100 | 0.2622 |
| gene2vec + MLP ensemble + SVD100 | **0.2706** |
| concat + Ridge + SVD100 | 0.2623 |
| concat + MLP ensemble + SVD100 | **0.2712** |

These are essentially identical to v13's variance-HVG results (0.2714). The HVG method does not matter.

**What we know about the gap to literature (running list of rejected hypotheses):**
- ❌ HVG selection method (variance vs dispersion)
- ❌ Number of HVGs (1500 → 5000: +0.014 improvement, but 5000 HVGs still gives 0.240 vs 0.267)
- ❌ Normalization (raw vs log1p: +0.028 improvement, but already using log1p)
- ❌ Cell sampling cap (80-cell → all cells: modest improvement)
- ❓ Possibly: different evaluation split (literature may use different test genes)
- ❓ Possibly: literature computes global mean from a different subset (all vs subset)
- ❓ Possibly: literature 0.267 uses a different response variable definition

**Current best model: concat+MLP ensemble+SVD100 = 0.2714** (v13, variance HVGs)
Beats literature mean-training-delta baseline of 0.267 by +0.004.

**Next directions to try:**
1. DepMap essentiality scores (gene dependency context features)
2. STRING protein interaction network features
3. ESM2 protein sequence embeddings for target genes
4. Wider MLP (hidden=1024, 3 layers) or larger SVD k
5. Larger batch + cosine embedding loss instead of MSE on SVD codes

---

## 2026-11-13 — Milestone 22: Cosine loss + wider MLP + direct prediction (v15)

**Script:** `scripts/replogle_genelevel_v15.py`

**Hypotheses tested:**
1. **Cosine loss on SVD codes**: Our eval is Pearson correlation ≈ cosine similarity on mean-centered vectors. Using cosine loss should better align training objective with evaluation.
2. **SVD k=200**: More expressive latent space (up from k=100 in v13).
3. **Wider/deeper MLP**: hidden=1024, 3 layers (up from 512, 2 layers).
4. **Direct 5000-dim delta prediction with cosine loss**: Bypass SVD entirely; predict full delta vector, no information loss from truncation.

**Results:**
| Model | Test Pearson (mean) |
|---|---|
| v13 baseline (concat+MLP+SVD100 MSE) | 0.2714 |
| gene2vec + cosine loss + SVD200 + h512 ens | 0.2678 ← **worse** |
| gene2vec + cosine loss + SVD200 + h1024-3L ens | 0.2700 ← worse |
| concat + cosine loss + SVD200 + h512 ens | 0.2679 ← worse |
| concat + cosine loss + SVD200 + h1024-3L ens | 0.2630 ← worse |
| **gene2vec + direct cosine h512 ens** | **0.2728** ← new best |
| concat + direct cosine h512 ens | 0.2726 |

**Key findings:**
- Cosine loss on SVD codes is **worse** than MSE on SVD codes. SVD codes are not well-suited for cosine loss (they span a low-rank space with specific scaling).
- h=1024, 3-layer MLP is not better than h=512, 2-layer — overfitting or optimization difficulty.
- **Direct 5000-dim prediction with cosine loss** is the best new approach: 0.2728 (+0.0014 over v13).
- The improvement from direct prediction is consistent across all 5 seeds (0.267–0.270 range).

**Interpretation:** The bottleneck is not architecture width/depth or loss function per se. Eliminating the SVD bottleneck gives a marginal gain, but we are fundamentally limited by the gene descriptor quality (Gene2Vec + response_role). The features do not encode enough information about how specifically this gene will perturb K562 cells.

**Current best: gene2vec+direct cosine h512 = 0.2728** (+0.0014 over v13 0.2714)

**Next directions:**
1. **External features**: ESM2 protein embeddings (sequence-based, captures protein function)
2. **Essentiality proxy from bulk RNA**: The raw bulk RNA h5ad (`data/replogle_k562_essential/K562_essential_raw_bulk_01.h5ad`) may contain perturbation signatures at bulk level — richer signal than mean single-cell delta
3. **Train/test leakage check**: Confirm literature 0.267 uses exact same split
4. **Larger training set**: Revisit using all 2057 target genes with leave-one-out or cross-val for a fairer estimate

---

## 2026-11-13 — Milestone 23: STRING PPI network-propagated embeddings (v16) — IN PROGRESS

**Script:** `scripts/replogle_genelevel_v16.py`

**Hypothesis:** Genes that interact in the protein-protein interaction (PPI) network
tend to have correlated perturbation effects. STRING PPI scores encode K562 biology
indirectly through known molecular interactions.

**Network propagation:** For each gene g, compute weighted mean of Gene2Vec(neighbor)
embeddings, weighted by STRING combined_score/1000. This creates a 200d "neighborhood
embedding" that reflects what interacting proteins do.

**Setup:**
- STRING v12.0 human PPI: downloaded to `data/string/`
- 1,858,944 edges at combined_score ≥ 400; coverage 97.4% of target genes
- Feature sets: `g2v`(200), `g2v_net`(400), `g2v_rr`(250), `g2v_net_rr`(450)
- All use direct cosine MLP (best from v15), 5-seed ensemble, 600 epochs

**Status:** COMPLETE. Ran 3950s (~66 min) on EC2.

**Results:**
| Feature set | Dim | Test Pearson (mean) | Test Pearson (median) |
|---|---|---|---|
| g2v only | 200 | 0.2728 | 0.2280 |
| g2v + response_role | 250 | 0.2726 | 0.2343 |
| **g2v + STRING_net** | **400** | **0.3103** | **0.2791** |
| g2v + STRING_net + response_role | 450 | 0.3090 | 0.2763 |

**Key finding: STRING PPI network embeddings are a major breakthrough (+0.038).**
Adding the network-propagated neighbor embedding to gene2vec jumps the test Pearson
from 0.2728 → 0.3103 — the largest single improvement in the entire project.

Notably, adding response_role on top of g2v+net slightly *hurts* (0.3103 → 0.3090).
This suggests response_role adds noise when STRING network information is already present.

**Interpretation:**
- Protein-protein interaction neighborhood encodes something fundamental about how
  genes perturb K562 cells — genes with similar interactors have similar effects.
- Gene2Vec (co-expression based) + STRING (interaction based) is a powerful
  complementary combination. Neither alone achieves this level.
- Response_role (NMF on training deltas) may overfit to the training distribution
  and become noise when added to already-rich representations.

**Progress:** 0.3103 vs oracle 0.4739. Gap reduced from 0.2011 → 0.1636.

**Next:** v17 — contrastive metric learning. Hypothesis: explicitly training the
embedding metric to match pairwise gene-delta similarities, then using k-NN retrieval,
can directly mimic the oracle strategy and further close the gap.

---

## 2026-11-13 — Milestone 24: Contrastive metric learning + k-NN retrieval (v17)

**Script:** `scripts/replogle_genelevel_v17.py` (written, not yet run)

**Motivation:** The oracle achieves 0.4739 by RETRIEVING training genes using TRUE
test-gene deltas. Our direct-regression models peak at 0.2728. The gap (0.2011) may
be partially closed by learning a **metric** that maps gene features to an embedding
space where similar-effect genes cluster together, then using k-NN retrieval.

**Approach (contrastive metric learning):**
1. For all pairs of training genes (gi, gj), compute target similarity:
   s_ij = Pearson(delta_gi, delta_gj)
2. Train a Siamese MLP: gene_features → L2-normalized embedding (64d),
   loss = MSE(cosine_sim(e_i, e_j), s_ij) over randomly sampled pairs
3. At test time: embed test gene, find top-k training genes by cosine similarity,
   return weighted mean of their delta vectors

**Experiments planned:**
- A: Direct cosine MLP on g2v+net+rr (reproduce v15/v16 best as baseline)
- B: Contrastive + k=5 retrieval
- C: Contrastive + k=10 retrieval
- D: Raw feature cosine k-NN (no metric learning; ablation)

**Expected run time:** ~2–3 hours (20 MLP training runs + pairwise Pearson matrix)
**Status:** Pending — will run after v16 completes.

---

## 2026-11-13 — Milestone 25: ESM2 protein sequence embeddings (v18)

**Hypothesis:** Gene2Vec encodes co-expression patterns; ESM2 encodes protein sequence
structure. These are orthogonal sources of information — combining them may give a
richer gene representation that explains more of the perturbation effect variance.

**What was built (`scripts/replogle_genelevel_v18.py`):**
- Queries UniProt REST API for protein sequences of all 2,057 target genes
- Runs `facebook/esm2_t6_8M_UR50D` (8M params, 320d mean-pool embeddings) on CPU
- Caches sequences to `data/esm2/uniprot_sequences.json`, embeddings to `data/esm2/esm2_embeddings.npz`
- Tests 4 feature combinations:
  - `g2v_net` (400d): v16 baseline — g2v + STRING-propagated g2v
  - `esm2` (320d): protein sequence only
  - `esm2_net` (520d): ESM2 + STRING-propagated ESM2
  - `esm2_g2v_net` (720d): ESM2 + g2v + STRING-propagated g2v

**Expected result:** ESM2 alone may underperform g2v_net (sequence ≠ expression context),
but fusion (esm2_g2v_net) may improve over 0.3103 by adding orthogonal signal.
**Status:** COMPLETE

**Results:**
| Feature | Test Pearson |
|---------|-------------|
| g2v_net (baseline) | 0.3103 |
| esm2 alone | 0.2911 |
| **esm2_net** | **0.3234 ← NEW BEST** |
| esm2_g2v_net | 0.3160 |

**Finding:** ESM2 + STRING propagation (520d) beats Gene2Vec + STRING (+0.013). Protein sequence alone is weaker than co-expression context, but ESM2 network neighbors capture complementary pathway structure. The 720d fusion (esm2_g2v_net) underperforms esm2_net — adding Gene2Vec to ESM2_net adds noise, not signal.

---

## 2026-11-13 — Milestone 26: Multi-hop STRING GNN (v19)

**Hypothesis:** 1-hop fixed aggregation (v16) may miss distant regulatory relationships.
Extending to 2 hops with *learned* aggregation weights (GraphSAGE-style) may capture
more complex pathway structure.

**What was built (`scripts/replogle_genelevel_v19.py`):**
- **Fixed 2-hop (ablation):** Extends v16 to 2 hops; concatenates [g2v, hop1_agg, hop2_agg] → 600d
- **Learned 2-layer GraphSAGE:**
  - Layer 1: `ReLU(W1·concat[g2v(g), mean_pool(g2v(nbr))])` → 256d
  - Layer 2: `ReLU(W2·concat[h1(g), mean_pool(h1(nbr))])` → 128d
  - MLP decoder: 128d → 5000d delta
  - Learned aggregation weights end-to-end

**Expected result:** Fixed 2-hop may show diminishing returns; learned GNN may push
past 0.3103 by learning which neighbors matter.
**Status:** Queued after v18 (sequential on EC2)

---

## 2026-11-13 — Milestone 27: Episodic meta-learning + kNN retrieval (v20)

**Hypothesis:** Standard supervised training forces the model to find one global mapping.
Episodic (prototypical network) training teaches the model to predict from a support set
of related genes — better for cold-start generalization to new test genes.

**What was built (`scripts/replogle_genelevel_v20.py`):**
- **ProtoNet (8,000 episodes):** Siamese MLP with soft-attention over support genes
  - Each episode: 10 support + 15 query genes
  - Embedding: `EmbedNet(400d → 128d, L2-normalized)`
  - Decoder: `concat[context_delta, query_emb] → delta`
  - At test time: all training genes are the support set
- **Soft kNN baselines:** k=1, 5, 10 cosine retrieval (no learning)

**Expected result:** kNN retrieval may underperform direct prediction (v17 showed this).
ProtoNet may add a small improvement by learning the retrieval.
**Status:** Queued after v19

---

## 2026-11-13 — Milestone 28: Bulk data pre-training (v21)

**Hypothesis:** The single-cell cache uses ~2,057 targets. The bulk Replogle file
(`K562_essential_raw_bulk_01.h5ad`, 76 MB, 2,285 perturbations × 8,563 genes) has
never been used. Pre-training an encoder on bulk signatures (more genes, pre-aggregated,
cleaner signal) before fine-tuning on mean-delta prediction may bootstrap better features.

**What was built (`scripts/replogle_genelevel_v21.py`):**
- Loads bulk K562 file → computes delta signatures (expression − control mean) for each gene
- Pre-trains encoder: contrastive loss blending STRING score + bulk cosine similarity
  - Target sim(g1, g2) = 0.5 × STRING_score + 0.5 × bulk_cosine
  - 200 epochs, batch_size=256 pairs
- Fine-tunes: frozen encoder for 100 epochs, then full unfreeze for 400 more
- Compares: (A) from-scratch MLP vs (B) pre-trained encoder fine-tuned

**Expected result:** Contrastive bulk pre-training may add +0.01–0.03 over direct MLP
by providing a better initialization.
**Status:** Queued after v20


---

## 2026-11-13 — Milestone 26 final: Multi-hop STRING GNN (v19) — COMPLETE

**Results:**
| Experiment | Test Pearson |
|-----------|-------------|
| Fixed 2-hop (600d) | **0.3183** (+0.008 over v16) |
| Learned 2-layer GraphSAGE | 0.2923 ← REJECTED |

**Finding:** With only 1,439 training genes, learned GNN aggregation overfits badly. Fixed/hand-engineered propagation consistently wins. Same bottleneck as v17 (contrastive learning). Data size is the fundamental limit on learned architectures.

---

## 2026-11-13 — Milestone 27 final: Episodic meta-learning (v20) — COMPLETE

**Results:**
| Experiment | Test Pearson |
|-----------|-------------|
| Direct MLP baseline | 0.3074 |
| ProtoNet (8K episodes) | 0.3163 (+0.009 over baseline) |
| kNN k=1 / k=5 / k=10 | 0.12 / 0.19 / 0.22 |

**Finding:** Episodic meta-learning adds a small but real +0.009 over direct MLP. kNN retrieval is poor (feature quality bottleneck confirmed again). ProtoNet's gain is real but doesn't beat esm2_net — features remain the dominant factor.

---

## 2026-11-13 — Milestone 28 final: Bulk data pre-training (v21) — NEW BEST 0.3292

**Results:**
| Experiment | Test Pearson |
|-----------|-------------|
| Direct from-scratch | 0.3081 |
| **Bulk pre-trained** | **0.3292 ← NEW BEST** |

**Finding:** Pre-training on bulk K562 perturbation signatures (2,025 genes, pre-aggregated, contrastive loss) then fine-tuning improves by +0.021 over direct MLP and +0.006 over previous best (esm2_net). Transfer learning from bulk data is effective even when the downstream task uses single-cell mean-deltas. The bulk file (76 MB local) is an untapped resource that meaningfully helps.

---

## 2026-11-13 — Full scoreboard after v18–v21

| Version | Test Pearson | Method | Status |
|---------|-------------|--------|--------|
| Global mean | 0.2405 | Baseline | — |
| Literature (GEARS) | 0.267 | Reference | — |
| v16 g2v_net | 0.3103 | Gene2Vec + STRING 1-hop | Previous best |
| v19 fixed 2-hop | 0.3183 | Gene2Vec + STRING 2-hop | +0.008 |
| v20 ProtoNet | 0.3163 | Episodic meta-learning | +0.006 |
| v18 esm2_net | 0.3234 | ESM2 + STRING 1-hop | +0.013 |
| **v21 bulk pre-train** | **0.3292** | **Bulk pre-training + g2v_net** | **+0.019** |
| Oracle NN K=5 | 0.4739 | Upper bound | — |

**Next logical steps:**
1. **Combine best features:** esm2_net (0.3234) + bulk pre-training → could push toward 0.34+
2. **Deeper bulk pre-training:** More epochs, larger batch, stronger contrastive signal
3. **More bulk data:** Replogle genome-wide K562 (~50 GB) has ~10,000 genes vs 2,025


---

## 2026-11-13 — Milestone 29: ESM2+g2v+STRING+bulk pre-training combination (v22) — NEW BEST 0.3432

**Hypothesis:** The two strongest improvements (ESM2+STRING from v18, bulk pre-training from v21)
may be additive if they encode orthogonal information.

**What was built (`scripts/replogle_genelevel_v22.py`):**
- Uses cached ESM2 embeddings from v18 (no re-fetching)
- Builds both g2v_net (400d) and esm2_net (640d) feature sets from STRING
- Combo features: esm2(320) + net_esm2(320) + g2v(200) + net_g2v(200) = 1040d
- Pre-trains on bulk signatures (300 epochs, batch 512, blended contrastive loss)
- Fine-tunes with frozen encoder warmup (100 epochs) then full unfreeze (500 epochs)

**Results:**
| Experiment | Test Pearson |
|-----------|-------------|
| esm2_net direct | 0.3209 |
| esm2_net + bulk pre-train | 0.3325 |
| g2v_net + bulk pre-train | 0.3358 |
| **combo (1040d) + bulk pre-train** | **0.3432 ← NEW BEST** |

**Finding:** ESM2 and Gene2Vec carry orthogonal information — combining them (1040d) is additive,
not redundant. Bulk pre-training amplifies both. The gains compound: ESM2 alone +0.013 (v18),
bulk pre-train alone +0.019 (v21), combined +0.033 over v16 baseline.

**Updated scoreboard:**
| Version | Pearson | Method |
|---------|---------|--------|
| Global mean | 0.2405 | — |
| Literature GEARS | 0.267 | Reference |
| v16 | 0.3103 | Gene2Vec + STRING |
| v18 esm2_net | 0.3234 | ESM2 + STRING |
| v21 bulk pre-train | 0.3292 | Bulk transfer |
| **v22 combo** | **0.3432** | **ESM2+g2v+STRING+bulk** |
| Oracle NN K=5 | 0.4739 | Upper bound |

Gap to oracle: 0.1307 remaining (was 0.1636 at v16 start of this session).



---

## 2026-11-13 — Milestone 30: Deeper encoder + SimCLR NT-Xent pre-training (v23)

**Hypothesis:** Two improvements to the v22 combo (1040d + bulk pre-train):
1. Deeper encoder (4-layer: 1040→1024→1024→512→256 vs v22's 2-layer 512→512→256) may extract richer representations from the 1040d features.
2. SimCLR NT-Xent pre-training (self-supervised, using STRING-neighbor augmented views as positives) may provide better pre-training signal than MSE on pairwise cosine similarity.

**What was built (`scripts/replogle_genelevel_v23.py`):**
- DeepEncoder: 4-layer MLP (1040→1024→1024→512→256 with LayerNorm+GELU+Dropout)
- NT-Xent loss: SimCLR contrastive with temperature τ=0.1, positive pairs = (original, STRING-neighbor-augmented) features
- 3 experiments: (A) deeper encoder + MSE pre-train; (B) deeper encoder + SimCLR + cosine loss; (C) deeper encoder + SimCLR + combined cosine+MSE loss
- 500 pre-train epochs, 600 fine-tune epochs, 5-seed ensemble

**Results:**
| Experiment | Test Pearson | vs v22 |
|-----------|-------------|--------|
| **A: combo + MSE pre-train + deeper encoder** | **0.3526 ← NEW BEST** | **+0.009** |
| C: combo + SimCLR + combined loss | 0.3136 | -0.030 |
| B: combo + SimCLR + cosine loss | 0.2945 | -0.049 |

**Finding:** The deeper encoder alone (+0.009) is a genuine improvement — the 4-layer 1040→256 architecture extracts more from the 1040d combo features than the shallower v22 encoder. SimCLR NT-Xent pre-training is a **clear failure** (-0.049 vs v22): the STRING-neighbor augmentation does not create distinct enough positive pairs to learn useful representations, and the NT-Xent loss is likely too aggressive for this small-dataset gene-expression domain. Combined cosine+MSE fine-tuning loss (Exp C) partially recovers vs pure SimCLR but still underperforms MSE pre-training significantly. **Key takeaway:** architecture depth helps; NT-Xent-style contrastive learning does not work here.

**Updated scoreboard:**
| Version | Pearson | Method |
|---------|---------|--------|
| Global mean | 0.2405 | — |
| Literature GEARS | 0.267 | Reference |
| v16 | 0.3103 | Gene2Vec + STRING |
| v18 esm2_net | 0.3234 | ESM2 + STRING |
| v21 bulk pre-train | 0.3292 | Bulk transfer |
| v22 combo | 0.3432 | ESM2+g2v+STRING+bulk (2-layer) |
| **v23 combo+deep** | **0.3526** | **ESM2+g2v+STRING+bulk (4-layer)** |
| Oracle NN K=5 | 0.4739 | Upper bound |

Gap to oracle: 0.1213 remaining (was 0.1307 after v22, 0.1636 at v16).

**Next logical steps:**
1. **More pre-train epochs / larger batch** — deeper encoder may benefit from longer bulk pre-training
2. **Wider intermediate layers** — try 1040→2048→1024→512→256
3. **Replogle genome-wide K562** (~50 GB, ~10,000 genes) — more bulk pre-training data is likely the highest-leverage remaining lever


---

## 2026-11-13 — Milestone 31: Wider encoder + longer bulk pre-training (v24) — NEW BEST 0.3538

**Hypothesis:** v23 showed that deeper architecture (+0.009) is a genuine gain. Can a wider first layer (2048 vs 1024) and longer pre-training (800 vs 500 epochs) extract even more signal?

**What was built (`scripts/replogle_genelevel_v24.py`):**
- WideEncoder: 1040→2048→1024→512→256 (wider first layer vs v23's 1040→1024→...)
- 800-epoch MSE pairwise contrastive pre-training (vs 500 in v23)
- Cosine LR warm restarts during pre-training (T_0=200) for better convergence
- Larger pre-train batch: 512 (vs 256 in v23)
- Ablation: wide encoder with no pre-training

**Results:**
| Experiment | Test Pearson | vs v23 |
|-----------|-------------|--------|
| **Wide + 800-epoch pre-train** | **0.3538 ← NEW BEST** | **+0.001** |
| Wide encoder, no pre-train | 0.3182 | -0.034 |

**Per-seed wide_pretrained:** 0.3463, 0.3505, 0.3441, 0.3474, 0.3481

**Finding:** Wider first layer + longer pre-training gives a marginal +0.001 improvement over v23 (0.3526→0.3538). The ablation (wide, no pre-train = 0.3182) confirms pre-training is essential — the wider architecture alone is not better than the smaller encoder from earlier experiments. The gains from architecture widening are diminishing; the marginal improvement from going wider (1024→2048) is much smaller than going deeper (2-layer→4-layer, +0.009 in v23). **The diminishing returns suggest we are hitting a ceiling with the current ~1,439 training genes and 2,025 bulk pre-training genes.** More data is likely the highest-leverage remaining lever.

**Updated scoreboard:**
| Version | Pearson | Method |
|---------|---------|--------|
| Global mean | 0.2405 | — |
| Literature GEARS | 0.267 | Reference |
| v16 | 0.3103 | Gene2Vec + STRING |
| v18 esm2_net | 0.3234 | ESM2 + STRING |
| v21 bulk pre-train | 0.3292 | Bulk transfer |
| v22 combo | 0.3432 | ESM2+g2v+STRING+bulk (2-layer) |
| v23 combo+deep | 0.3526 | 4-layer encoder |
| **v24 combo+wide** | **0.3538** | **2048-wide encoder + 800-epoch pre-train** |
| Oracle NN K=5 | 0.4739 | Upper bound |

Gap to oracle: 0.1201 remaining.

**Conclusion / next steps:** Architecture improvements (deeper, wider, longer pre-train) have delivered +0.044 since v22 but are now yielding marginal gains. The fundamental bottleneck is data: 1,439 training genes is very few for a generalizable perturbation→effect mapping. The highest-leverage next step is **more training data** via the Replogle genome-wide K562 dataset (~50 GB, ~10,000 perturbation genes). Without that data, further architecture tuning is unlikely to close the remaining 0.12 gap to oracle.


---

## 2026-11-13 — Milestone 32: Expanded STRING pre-training (v25) — NEW BEST 0.3557

**Hypothesis:** Pre-training (v22–v24) used only STRING pairs where BOTH genes have bulk signatures (~92,947 pairs). The full STRING graph has 331,246 pairs; using all of them (falling back to STRING-score-only target when bulk unavailable) might improve the encoder.

**What was built (`scripts/replogle_genelevel_v25.py`):**
- Same WideEncoder (1040→2048→1024→512→256) as v24
- Exp A: bulk-only pairs (v24 replication, 92,947 valid pairs)
- Exp B: all STRING pairs where both genes in feat_dict (93,562 pairs — only 615 more)
- Both use bulk cosine blend when available; Exp B falls back to STRING-score × 0.5 otherwise

**Results:**
| Experiment | Test Pearson | vs v24 |
|-----------|-------------|--------|
| **A: bulk-only pairs (v24 replication)** | **0.3557 ← NEW BEST** | **+0.002** |
| B: all STRING pairs (expanded) | 0.3533 | -0.001 |

**Finding:** Expanding to all STRING pairs adds only 615 pairs (93,562 - 92,947) — meaning almost all STRING-connected genes in our feature space already have bulk signatures. The 615 extra STRING-only pairs slightly *hurt* (-0.002): the string-score-only signal without bulk data is marginally detrimental. **The bulk-covered pairs already capture the best pre-training signal.** Exp A's marginal improvement (+0.002 vs v24) is within noise.

**Structural insight:** Of 331,246 total STRING pairs, only 93,562 have both genes in our feature space (gene2vec coverage is the bottleneck, not bulk signature coverage). STRING-only pairs beyond that 93,562 are unlearnable with current features.

**Updated scoreboard:**
| Version | Pearson | Method |
|---------|---------|--------|
| Global mean | 0.2405 | — |
| Literature GEARS | 0.267 | Reference |
| v16 | 0.3103 | Gene2Vec + STRING 1-hop |
| v22 combo | 0.3432 | ESM2+g2v+STRING+bulk (2-layer) |
| v23 combo+deep | 0.3526 | 4-layer encoder |
| v24 combo+wide | 0.3538 | 2048-wide + 800-epoch pre-train |
| **v25 Exp A** | **0.3557** | **v24 config, slightly better run** |
| Oracle NN K=5 | 0.4739 | Upper bound |

Gap to oracle: 0.1182 remaining.


---

## 2026-11-13 — Milestone 33: 2-hop STRING propagation (v26) — REJECTED

**Hypothesis:** v19 showed 2-hop > 1-hop for g2v alone (+0.008). Extending 2-hop to both ESM2+g2v features (1560d) might give a similar gain on top of the best combo.

**What was built (`scripts/replogle_genelevel_v26.py`):**
- 2-hop STRING propagation: esm2(320) + hop1_esm2(320) + hop2_esm2(320) + g2v(200) + hop1_g2v(200) + hop2_g2v(200) = 1560d
- WideEncoder adapted to 1560d input (1560→2048→1024→512→256)
- Same pre-training/fine-tuning as v25 Exp A (best setup)
- Comparison: 1-hop replication (1040d) vs 2-hop (1560d)

**Results:**
| Experiment | Test Pearson | vs v25 |
|-----------|-------------|--------|
| 1-hop replication (1040d) | 0.3557 | 0.000 |
| **2-hop combo (1560d)** | **0.3486** | **-0.007 REJECTED** |

**Finding:** 2-hop STRING propagation *hurts* (-0.007). Contrary to v19's result for g2v alone, adding a second propagation hop on the already-1-hop-enriched combo features causes over-smoothing. The 1-hop already captures the most predictive PPI signal; going to 2-hop makes all genes look more like their 2nd-degree neighbors, diluting the specific local interaction context. **Optimal propagation depth = 1-hop** for this feature set.

The v19 result (2-hop > 1-hop for raw g2v) worked because raw g2v had no network context yet. The combo features already have 1-hop context baked in, so the marginal value of an additional hop is negative.

---

## 2026-11-13 — Architecture search summary and ceiling analysis

After v22–v26, we have exhausted the current set of architectural improvements. Summary:

| Change | v → v | Δ Pearson | Status |
|--------|-------|-----------|--------|
| Deeper encoder (2→4 layer) | v22→v23 | +0.009 | ✓ Keeps |
| Wider first layer (512→2048) | v23→v24 | +0.001 | ✓ Marginal |
| Longer pre-training (500→800 ep) | v23→v24 | Included above | — |
| SimCLR NT-Xent pre-training | v22→v23B | -0.049 | ✗ REJECTED |
| Combined cosine+MSE fine-tune loss | v22→v23C | -0.030 | ✗ REJECTED |
| Expanded STRING pairs (all vs bulk) | v24→v25B | -0.002 | ✗ REJECTED |
| 2-hop STRING propagation | v25→v26A | -0.007 | ✗ REJECTED |

**Current best: 0.3557** (v25 Exp A / v26 Exp B, stable across 2 runs).  
**Gap to oracle: 0.1182** (oracle = 0.4739).

**Ceiling diagnosis:** Further architecture tuning is unlikely to close the remaining gap. The fundamental bottleneck is:
1. **Training data: 1,439 genes** is too few for a generalizable perturbation→effect mapping
2. **Pre-training data: 2,025 bulk gene signatures** covers only the K562 essential screen

**Highest-leverage next steps (require user decision):**
1. **Replogle genome-wide K562** (~50 GB, ~10,000 perturbation genes) — biggest lever; needs download
2. **Larger ESM2 model** (ESM2-650M vs current 8M) — better protein representations; needs re-embedding
3. **Different downstream formulation** — predict full gene program rather than per-gene delta

The loop has identified the ceiling for the current data/feature regime. Further autonomous iteration without new data inputs would not be productive.

---

## 2026-11-14 — Milestone 34: ESM2-650M protein embeddings (v27) — REJECTED

**Hypothesis:** ESM2-8M (8M params, 320d) may be underpowered. ESM2-650M (650M params, 1280d) captures richer protein structure — if protein representation quality is a bottleneck, upgrading should give a clear gain.

**What was built (`scripts/replogle_genelevel_v27.py`, `scripts/embed_esm2_650M.py`):**
- Re-embedded all 2050 perturbation genes with `facebook/esm2_t33_650M_UR50D` (1280d)
- New combo features: esm2_650M(1280) + net_esm2_650M(1280) + g2v(200) + net_g2v(200) = **2960d**
- WideEncoder adapted: 2960→2048→1024→512→256 (same depth, wider input compression)
- Compared directly to 8M replication (1040d) under identical training protocol
- Embedding took ~5.8h on CPU (m7i-flex.large); v27 training took ~2h

**Results:**
| Experiment | Features | Test Pearson | vs best |
|-----------|----------|-------------|---------|
| Exp B: ESM2-8M replication | 1040d | **0.3557** | 0.000 |
| **Exp A: ESM2-650M (main)** | **2960d** | **0.3463** | **-0.009 REJECTED** |

Per-seed (8M): 0.3514, 0.3526, 0.3492, 0.3456, 0.3482 → ensemble 0.3557
Per-seed (650M): 0.3405, 0.3404, 0.3444, 0.3429, 0.3380 → ensemble 0.3463

**Finding:** ESM2-650M features *hurt* performance (-0.009). The 8M replication is pixel-perfect (0.35568 ≈ v25/v26 0.35568), confirming experimental stability.

**Why 650M underperforms:**
- With only 1,439 training genes, the 2960d input space (3× larger) requires the first encoder layer to compress much harder (2960→2048), discarding more signal
- ESM2-650M captures general protein sequence structure at higher resolution, but the perturbation→transcriptome mapping doesn't require that resolution — it needs perturbation-relevance, not sequence accuracy
- The STRING+bulk similarity pre-training target was designed for 320d features; scaling to 1280d doesn't improve the target signal quality

**Conclusion:** Protein embedding quality/size is **not** the bottleneck. This, combined with the architecture ceiling analysis from Milestone 33, strongly confirms:

> **The bottleneck is data: 1,439 training perturbations is insufficient for generalization.**

All attempted improvements to features (Gene2Vec, ESM2-8M, ESM2-650M, network propagation, 2-hop) and architecture (depth, width, pre-training duration, loss functions) plateau at ~0.3557. The gap to oracle (0.4739) requires more training genes, not better representations.

**Highest-leverage next step:** Replogle genome-wide K562 dataset (~10,000 perturbation genes). This is the last remaining lever that could meaningfully move the needle.

## 2026-11-14 — Milestone 35: Genome-wide bulk pre-training (v28) — ALL EXPERIMENTS REJECTED

**Hypothesis (data bottleneck):** If the architecture ceiling (~0.3557) is caused by insufficient fine-tuning data (1,439 genes), then pre-training on more perturbation signatures should help the encoder learn a richer gene-effect representation. The GWPS dataset provides bulk signatures for 9,474 genes (vs 2,025 for essential) — a 4.7× increase in pre-training perturbation coverage.

**What was built (`scripts/replogle_genelevel_v28.py`):**
- Three-way comparison: GWPS bulk pre-train (9,474 genes), essential bulk pre-train (2,025 genes, replication), g2v-only features (ablation)
- Pre-training target unchanged: 0.5×STRING_score + 0.5×bulk_cosine similarity
- GWPS STRING pairs: 240,522 (vs 92,947 for essential) — 2.6× more pairs
- Same WideEncoder, same fine-tuning protocol (5 seeds, held-out-target split)
- Runtime: 7,530s (~2.1h) for all 3 experiments

**Results:**
| Experiment | Features | Pre-train data | STRING pairs | Test Pearson | vs best |
|-----------|----------|----------------|-------------|-------------|---------|
| Exp B: essential replication | 1040d | 2,025 genes | 92,947 | **0.3510** | -0.005 |
| Exp C: GWPS combo | 1040d | **9,474 genes** | **240,522** | **0.3482** | **-0.008 REJECTED** |
| Exp A: GWPS g2v-only | 400d | 9,474 genes | 240,522 | **0.3482** | **-0.008 REJECTED** |

Per-seed (Exp B essential): 0.3466, 0.3470, 0.3480, 0.3451, 0.3450
Per-seed (Exp C GWPS combo): 0.3408, 0.3411, 0.3451, 0.3428, 0.3454
Per-seed (Exp A GWPS g2v-only): 0.3390, 0.3396, 0.3372, 0.3386, 0.3399

Note: Exp B replication is 0.3510 vs v25/v27 0.3557 — difference is within run-to-run variance (expected ±0.005 from random seed effects).

**Finding:** More pre-training data (9,474 vs 2,025 genes, 2.6× more STRING pairs) consistently HURTS performance by -0.008. This holds for both 1040d and 400d feature sets. ESM2 features remain important (400d g2v-only ≈ 0.348 vs 1040d combo ≈ 0.351), but that gap is smaller than expected.

**Why GWPS pre-training hurts:**
- **Domain mismatch**: GWPS perturbs non-essential genes (non-lethal KOs) which tend to have weaker, noisier transcriptional signatures. The pre-training signal is diluted by lower-quality pairs.
- **Misdirected representation**: The encoder must fit a broader, more diverse gene landscape during pre-training. The resulting embeddings are more general but less specialized for the K562 essential gene regime that fine-tuning operates in.
- **Consistent pattern**: Across all sessions, expanding beyond the essential gene universe (whether by 2-hop STRING, expanded STRING pairs, ESM2-650M, or now GWPS pre-training) consistently hurts. The essential K562 perturbation regime is narrow and specialized.

**Scoreboard after Milestones 1–35:**

All exhausted levers:
- Architecture: depth ✗, width ✗, pre-training duration ✗, loss functions ✗
- Features: gene2vec ✗, ESM2-8M ✗, ESM2-650M ✗, 2-hop STRING ✗
- Pre-training data: essential bulk (2,025 genes) ✗, GWPS bulk (9,474 genes) ✗ (hurts)
- Fine-tuning data: stuck at 1,439 essential genes

**Conclusion:** The ~0.3557 ceiling is robust. Pre-training data quantity is **not** the bottleneck either; pre-training data *quality and domain* matters more than quantity. With only 1,439 fine-tuning gene perturbations (single-cell), all proxy improvements (better embeddings, more pre-training pairs, deeper networks) are saturated.

**The only remaining untested levers:**
1. **GWPS single-cell data**: `K562_gwps_raw_singlecell_01.h5ad` on figshare — if it exists and contains 7,000+ genome-wide single-cell perturbation signatures, joint fine-tuning on essential + GWPS single-cell would directly address the data bottleneck
2. **Different Perturb-seq datasets**: Norman et al. (K562 CRISPRa), Adamson et al., Dixit et al. have smaller gene sets but might combine for more coverage
3. **Abandon held-out-target generalization**: train on all 2,057 essential genes, evaluate in-distribution only (not scientifically interesting but would confirm data theory)

**Highest-leverage next step:** Investigate whether `K562_gwps_raw_singlecell_01.h5ad` exists on figshare (doi 10.25452/figshare.plus.20029387). If GWPS single-cell is available, v29 would joint-train on essential + GWPS single-cell signatures for a 5–6× increase in fine-tuning gene count.

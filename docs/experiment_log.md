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

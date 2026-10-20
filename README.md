# VirtualCell

Predicting single-cell transcriptional responses to genetic perturbations,
with rigorous evaluation on **unseen** perturbation targets.

**Research question:** Can conditioning on control-cell state improve gene
expression response prediction for held-out CRISPR perturbations compared
with simple baselines? See `VirtualCell_Codex_Implementation_Plan.md` for
the full research plan, and `docs/experiment_log.md` for what has actually
been run so far.

## Status

Milestone 1 (CPU vertical slice on a synthetic fixture) is implemented.
Real-data integration (Milestone 2+) is not yet started — see
`docs/experiment_log.md`.

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Commands

```bash
pytest -q
python -m virtualcell.cli --output-dir outputs/milestone1_smoke
```

The CLI command runs the full Milestone 1 pipeline on a deterministic
synthetic dataset: builds the held-out-target split, fits the no-effect and
seen-target-mean baselines, evaluates MAE/RMSE/bootstrap CIs, and writes
`outputs/milestone1_smoke/metrics.json`.

## Project layout

```text
configs/      dataset and run configs
docs/         dataset card, methodology, experiment log, results
src/virtualcell/
  data/       adapters, fixtures, splits, preprocessing
  models/     baselines, (planned) conditional and foundation models
  training/   (planned) trainer and losses
  evaluation/ metrics
scripts/      (planned) CLI entry scripts
tests/        pytest suite
```

## Principles

- No fabricated datasets, trained models, GPU scale, or performance claims.
- Real-data paths are configurable but unset by default; synthetic fixtures
  back all current tests and the CLI smoke test.
- Splits are target-grouped with explicit leak-guard tests — see
  `tests/test_splits.py`.

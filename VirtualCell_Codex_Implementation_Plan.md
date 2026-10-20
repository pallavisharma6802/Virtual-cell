# VirtualCell — Codex Implementation Plan

**Goal:** Build a reproducible research project that predicts single-cell transcriptional responses to genetic perturbations, with rigorous evaluation on unseen perturbation targets. 

## Instructions for Codex

Inspect the repository first. Implement incrementally, run tests after each milestone, and document actual work done in `docs/experiment_log.md`. **Never fabricate performance results, datasets, trained models, GPU scale, wet-lab findings, or clinical claims.** If internet/dataset access is unavailable, ask the owner and after confirmation, develop against synthetic fixtures and make real-data paths configurable. Prioritize scientific validity over complex architecture.

## 1. Research question

**Can conditioning on control-cell state improve gene expression response prediction for held-out CRISPR perturbations compared with simple baselines?**

Input: a control-cell expression reference, perturbation target, and optionally cell-context metadata. Output: predicted change in the expression of a fixed, ordered gene vocabulary and a reconstructed post-perturbation expression profile on a documented expression scale.

Hypotheses:
- H1: Conditional modeling beats a no-effect/control-mean baseline on **unseen perturbation targets**.
- H2: Conditioning on control-cell state improves performance over perturbation-only inputs.
- H3: Any gain persists across a harder split (batch/cell context) if metadata permits.

A null result is acceptable; never force positive conclusions.

**Out of scope for MVP:** new billion-parameter pretraining, clinical efficacy, wet-lab validation, cross-species translation, and spatial transcriptomics.

## 2. Data sources and access checks

Investigate:
- [Arc Institute Virtual Cell Atlas / challenge resources](https://github.com/ArcInstitute/arc-virtual-cell-atlas). Verify current task, download links, license, official splits, and format before implementing.
- [scPerturb](https://www.sanderlab.org/scPerturb/). Consider harmonized Perturb-seq data for secondary tests.

Do not assume public data counts, licenses, or schema until verified. Record findings in `docs/dataset_card.md`: dataset name/version, URL, citation, perturbation modality, cells/genes/targets, controls, batches, donors, cell types, preprocessing state, limitations, and usage terms.

Canonical adapter should expose `cell_id`, `perturbation_id`, `is_control`, `cell_type` (if available), `batch_id` (if available), `donor_id` (if available), ordered `gene_ids`, a sparse-compatible expression matrix, and provenance. Use Scanpy/AnnData. Ensure stable gene mapping; recognize raw counts versus normalized/log-transformed data. Fit learned scaling, PCA, filters, and batch correction on **train only**; do not leak held-out data into preprocessing.

**Important:** Perturbed cells generally do not have one-to-one matched control cells. Build appropriate context-matched control distributions/means; document how they are constructed and whether they are available at inference.

## 3. Required splits and leak guards

Implement saved, seeded split manifests:
1. **In-distribution sanity split:** disjoint cells from previously seen targets. Explicitly label as easiest benchmark.
2. **Primary split: held-out targets:** group by perturbation target; train, validation, and test have disjoint target IDs including all replicates. This is the primary research claim.
3. **Secondary context split:** batch/donor/cell-type holdout if actual metadata supports it. Do not call a batch holdout an unseen cell-type test.
4. **Stretch:** external dataset validation with harmonized genes and clear domain-shift caveats.

If a public benchmark has official splits, replicate them separately. Write tests for group isolation, control/reference restrictions, training-only transformations, gene-order consistency, and no test-target embedding lookup learned from labels.

## 4. Models (build in this order)

### A. Simple baselines first
- **No-effect:** predicted delta = 0 relative to appropriate controls.
- **Seen-target mean delta:** training-derived target mean, only for in-distribution test. Cannot work as a held-out-target predictor.
- **PCA + ridge / regularized linear model:** use permitted control-context and **non-leaking target side information**.
- **Small MLP:** match features and evaluation protocol to the conditional model.

**Unseen-target caveat:** An ID embedding trained only on seen perturbation targets cannot represent new targets meaningfully. For genuine zero-shot held-out-target evaluation use generalizable gene-target descriptors (for example an appropriately licensed pretrained gene representation) and document source, coverage, and missing-target behavior. If unavailable, report this as a blocked experiment rather than invent zero-shot performance.

### B. Conditional PyTorch model
- Control encoder: encodes control expression (individual sampled controls or aggregate profile, with documented sampling).
- Perturbation encoder: target-gene features that also exist for test targets.
- Optional context encoder: cell type or other metadata available at inference.
- Fusion: concat + MLP first; optionally cross-attention/transformer later.
- Prediction head: gene-wise expression change (`delta`).
- Reconstruct predicted perturbed expression from control reference in the same numeric space.

Start with Huber or MSE over defined expression deltas. Log train/validation loss, effective config, memory, and time. Add regularization and early stopping. Ablations: control context removed; context metadata removed; simple versus attention fusion; loss choice.

### C. Optional pretrained foundation model
Investigate scGPT or another accessible, appropriately licensed single-cell encoder, checking gene vocabulary and preprocessing compatibility. Compare frozen encoder + trained head against actual adapter/partial fine-tuning (if feasible). State which parameters train. Record checkpoint version, training regime, trainable parameter count, GPU resources, and dataset split. Do not describe trained-head experiments as end-to-end foundation model fine-tuning.

## 5. Evaluation: scientifically credible outcomes

Evaluate under identical gene universes, controls, and test target IDs:
- MAE and RMSE of expression changes (with explicit normalization scale).
- Pearson/Spearman delta-profile correlations per target; handle constant vectors.
- Improvement **relative to no-effect baseline**.
- Differentially expressed gene overlap / precision–recall when DE labels have a valid statistical procedure.
- Per-target metrics, per-context performance, counts and plots, not just pooled averages.
- Grouped bootstrap confidence intervals over perturbations or biological replicates as scientifically appropriate, not naïvely independent cells.

Inspect worst-performing targets, low-count targets, confounding with batches, and systematic artifacts. Optional pathway analyses must document gene sets, species, and version. Never claim biological mechanism, clinical translation, or wet-lab validation from this benchmark.

Hyperparameters selected on validation only; lock test until final evaluation. Report negative results honestly.

## 6. Proposed codebase

```text
virtualcell/
├── README.md
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── dataset.example.yaml
│   ├── baseline.yaml
│   ├── conditional_mlp.yaml
│   └── foundation_encoder.yaml
├── data/                         # gitignored
├── docs/
│   ├── dataset_card.md
│   ├── methodology.md
│   ├── experiment_log.md
│   ├── results.md
│   └── project_summary_for_resume.md
├── src/virtualcell/
│   ├── data/{adapters,preprocess,splits,fixtures}.py
│   ├── models/{baselines,conditional,foundation}.py
│   ├── training/{trainer,losses}.py
│   ├── evaluation/{metrics,biological,report}.py
│   └── cli.py
├── scripts/
│   ├── inspect_dataset.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── generate_report.py
├── tests/
│   ├── test_data_schema.py
│   ├── test_splits.py
│   ├── test_preprocessing.py
│   ├── test_baselines.py
│   ├── test_model_shapes.py
│   └── test_metrics.py
└── outputs/                      # gitignored
```

Stack: Python 3.11+, PyTorch, Scanpy, AnnData, NumPy, Pandas, SciPy, scikit-learn, pytest, Matplotlib, and optionally MLflow. Pin dependencies; support CPU smoke tests, single-GPU training, seeded reproducibility. Add Hugging Face libraries only when used.

Document commands, once implemented:
```bash
python scripts/inspect_dataset.py --config configs/dataset.example.yaml
python scripts/prepare_data.py --config configs/dataset.example.yaml
python scripts/train.py --config configs/baseline.yaml
python scripts/train.py --config configs/conditional_mlp.yaml
python scripts/evaluate.py --run-dir outputs/RUN_ID
python scripts/generate_report.py --runs outputs/
pytest -q
```

Each run saves config, seed, dataset identifier/hash, split manifest, model/checkpoint, hardware details, training curves, per-target CSV/JSON metrics, and figures. Never commit dataset files or large checkpoints by default.

## 7. Milestones and acceptance tests

### Milestone 1 — CPU vertical slice (start here)
- Scaffold package, tests, configs, docs.
- Generate a deterministic tiny synthetic AnnData dataset with controls, perturbations, multiple batches and cell contexts.
- Validate schemas, leakage-free target grouping, preprocessing, no-effect baseline, MAE/RMSE.
- Run end-to-end fixture experiment and output metrics.
- **Done:** `pytest -q` passes on CPU, single command produces metrics, and tests prove no target overlap.

### Milestone 2 — Real biological dataset and baselines
- Verify actual Arc/scPerturb instructions and licensing; write dataset card.
- Implement local-file AnnData adapter, config-driven QC, preprocessing, split manifests.
- Train no-effect, simple linear/PCA, and MLP baselines where appropriate.
- **Done:** real-data run is reproducible with a documented command, and real test target IDs are isolated.

### Milestone 3 — Conditional model
- Implement control encoder + target-side-information encoder + fusion + delta head.
- Train/validate, checkpoint, early stopping, seeds, at least two ablations.
- **Done:** model reloads for inference, comparisons share splits, model-vs-baseline results are reported whether positive or negative.

### Milestone 4 — Research-grade evaluation
- Add per-target correlations, biological response diagnostics, optional DE metrics, group bootstrap CIs, context/batch analysis.
- Investigate at least three clear failure cases.
- **Done:** `docs/results.md` has actual tables, figures, run IDs, sample sizes, conclusions and limitations.

### Milestone 5 — Optional foundation encoder
- Inspect pretrained checkpoint license and compatible inputs.
- Frozen encoder baseline, then actual fine-tuning if compute allows.
- Compare versus simpler model under same protocol.
- **Done:** verified trainable parameter counts and measured compute; or documented blocker.

## 8. Five-week planning horizon

| Week | Priority | Deliverable |
|---|---|---|
| 1 | Repo, fixtures, splits, control baseline | Passing CPU tests |
| 2 | Real data, preprocessing, linear/MLP baselines | Verified benchmark setup |
| 3 | Conditional PyTorch model, ablations | Reproducible checkpoints |
| 4 | Generalization tests and biological analysis | Results and error report |
| 5 | Optional pretrained encoder and polish | Research-style write-up |

This is an estimate, not a commitment; rigorous evaluation takes priority.

## 9. Deliverables

- Reproducible repo with install and run instructions.
- Dataset card; documented train/val/test protocol; leak tests.
- Baseline-vs-model results with uncertainty and per-perturbation analyses.
- Observed-vs-predicted change figures and failure analysis.
- Research summary answering H1–H3, including null results.
- `docs/project_summary_for_resume.md` written **only after actual experiments**, with verifiable tools, scale, results, and no invented metrics.

## 10. Antipatterns to avoid

- Random cell split presented as unseen-perturbation generalization.
- Target-specific learned embeddings for never-seen targets.
- Fit preprocessing or batch correction to train+test for an inductive evaluation.
- Treat unmatched controls as paired samples.
- Mix count-space/log-space targets and predictions.
- Report confidence intervals treating correlated cells as independent.
- Overinterpret predictive signals as validated biological discoveries.
- Build a giant transformer before baselines are reproducible.

## 11. Exact kickoff prompt for Codex

> Implement **Milestone 1 now** in the current repository. Inspect existing files first. Create the project structure, deterministic synthetic AnnData fixture, schema validation, held-out-target split tests, training-only preprocessing, no-effect baseline, MAE/RMSE evaluation, CPU smoke test, and a CLI that writes metrics to `outputs/`. Run `pytest -q` and the smoke-test CLI. Document exact passing commands and artifact paths in the README and `docs/experiment_log.md`. Then inspect the official Arc virtual-cell or scPerturb data documentation and plan the real-data adapter; implement it only where schema and access are verified. Never fabricate dataset measurements or model results. Report completed tasks and blockers plainly.

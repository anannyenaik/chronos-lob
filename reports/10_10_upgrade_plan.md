# ChronosLOB 10/10 Empirical Upgrade Plan

This plan records the next empirical release workstream. It does not add
benchmark outputs, benchmark claims, generated plots or placeholder results.

## 1. Current repository state

### Data contracts and validation

ChronosLOB has typed schemas for book events, snapshots, feature rows,
label rows and data-quality issues. It supports local FI-2010-style loading,
canonical event-log storage, local Binance-style reconstruction from supplied
files and reproducibility manifests with checksums for event logs. The FI-2010
loader is local-file only and intentionally does not download data.

### Feature and label construction

The repository includes past-only microstructure features for imbalance,
microprice, order flow, volatility and regimes. It also includes future-window
labels for mid-price movement, spread, volatility, fill probability and adverse
selection. Leakage checks validate feature/label separation and explicit label
horizons.

### Leakage controls and temporal validation

Temporal train/validation/test splits, walk-forward splits, purging, row
embargoes and train-only quantile binning are implemented. Baseline and neural
training paths use temporal splits and train-only preprocessing rather than
random financial time-series splits.

### Classical and neural modelling

Implemented model infrastructure includes majority-class and scikit-learn
baselines, DeepLOB-style supervised modelling, PyTorch sequence datasets,
event tokenisation, a transformer encoder, self-supervised objectives and
multi-task heads. Current runs are smoke checks or reusable training plumbing,
not real benchmark evidence.

### Calibration and uncertainty

The project supports Brier score, negative log likelihood, reliability bins,
expected calibration error, temperature scaling, confidence filtering,
abstention curves and multi-task calibration summaries. These tools are ready
to consume stored predictions from real experiments.

### Execution-aware validation

The validation layer models explicit research assumptions for fees, spread
costs, aggressive/passive/hybrid execution modes, row-step latency, turnover,
passive-fill proxies, confidence thresholds and simple risk constraints. It is
a deterministic research simulator, not venue execution infrastructure.

### Robustness analysis

Transfer, regime, ablation and sensitivity utilities organise supplied
experiment records. They currently validate analysis contracts and synthetic
smoke paths; they do not create real robustness evidence without upstream
benchmark runs.

### Reproducibility and audit tooling

The repository includes run metadata helpers, config loading, report-archive
generation, release-readiness checks and a strict project audit. The current
validation suite passes and inventories docs, configs, reports, CLI commands
and synthetic fixture labelling.

### Public documentation

The README, roadmap, reproducibility docs, safety statement, CLI reference,
experiment evidence index and reports are polished and scope-aware. They
clearly separate implemented infrastructure from future empirical evidence.

## 2. Main gap

The main remaining gap is real benchmark evidence.

Synthetic fixtures validate plumbing only. They show that loaders, schemas,
feature builders, label builders, splitters, models, calibration utilities,
execution validation and audit tooling can run on tiny deterministic inputs.
They are not benchmark results, market evidence or execution evidence.

No real benchmark performance is currently claimed. The next phase should add
reproducible FI-2010 benchmark experiments using local data supplied by the
user, with traceable configs, data manifests, predictions, calibration
artefacts and explicit execution assumptions.

## 3. Target research question

Can self-supervised order-book representations improve short-horizon
market-state forecasting after leakage-safe validation, calibration analysis
and explicit execution assumptions?

The project should answer this by separating:

- forecast quality
- calibration quality
- cost-aware signal quality
- robustness under ablations and assumptions
- systems performance

## 4. Target empirical artefact layout

The intended experiment directory contract is:

```text
experiments/
  fi2010_midprice_h10/
    config.yaml
    data_manifest.json
    results.json
    predictions.csv or predictions.parquet
    calibration_bins.csv
    execution_sensitivity.csv
    model_card.md
    plots/
      reliability_curve.png
      cost_sensitivity.png
      confusion_matrix.png
      regime_breakdown.png
```

Artefact purposes:

- `config.yaml`: complete experiment configuration, including seed, data
  path, label horizon, split definition, model configuration, calibration
  settings and execution assumptions.
- `data_manifest.json`: local data provenance, row counts, schema details,
  split metadata and checksums where possible.
- `results.json`: machine-readable metrics, split summaries, environment
  metadata and code commit.
- `predictions.csv` or `predictions.parquet`: row-level labels, predictions,
  probabilities, split ids and identifiers needed to recompute metrics.
- `calibration_bins.csv`: reliability-bin records built from validation or
  test predictions according to the configured evaluation protocol.
- `execution_sensitivity.csv`: cost, latency, confidence-threshold and
  turnover sensitivity rows derived from stored predictions.
- `model_card.md`: concise methodological summary, data scope, leakage
  controls, known limits and claim boundaries for the run.
- `plots/`: generated visual summaries derived only from stored artefacts.

This plan does not create the experiment directory or any fake outputs. The
layout above is a contract for future implementation.

## 5. Minimum model comparison

The minimum benchmark comparison should include:

- majority class
- logistic regression
- random forest
- DeepLOB-style model
- transformer
- SSL-pretrained transformer, only if pretraining is actually run and traceable

The minimum metrics should include:

- split
- horizon
- macro F1
- accuracy
- Brier score
- ECE
- turnover
- cost-aware signal metric
- optional latency sensitivity

Modest results are acceptable if the methodology is rigorous, leakage-safe and
fully reproducible.

## 6. Implementation roadmap

### Phase A: Experiment artefact contract

Purpose: define and validate the on-disk contract for experiment outputs
before producing any real benchmark results.

Status: implemented as a schema and inspection layer under
`chronoslob/experiments/`, with a synthetic contract fixture under
`tests/fixtures/` for validation tests only.

Files added or modified: `chronoslob/experiments/`, `chronoslob/cli.py`,
`docs/EXPERIMENT_ARTIFACT_CONTRACT.md`, `docs/CLI_REFERENCE.md` and
`tests/test_experiment_artifact_contract.py`.

CLI command: `python -m chronoslob.cli inspect-experiment-artifacts --experiment PATH`.

Tests expected: schema validation for required files, optional plot files,
prediction columns, calibration bins, execution sensitivity rows and helpful
error messages for incomplete artefacts.

Strict non-goals: do not run models, create benchmark outputs, add fake plots,
download data or weaken existing public wording checks.

### Phase B: FI-2010 local benchmark preparation

Purpose: turn user-supplied local FI-2010 files into a documented benchmark
input with explicit provenance and label/split configuration.

Status: implemented as a local-only preparation layer under
`chronoslob/experiments/fi2010_benchmark.py`, with a config template at
`configs/experiments/fi2010_midprice_h10.yaml` and a CLI command at
`python -m chronoslob.cli prepare-fi2010-benchmark`. The preparation step
produces a data manifest, label distribution summary, temporal split
summary, validation summary and a config snapshot. It does not run a model
or produce `results.json`.

Files added or modified: `chronoslob/experiments/fi2010_benchmark.py`,
`chronoslob/experiments/__init__.py`, `chronoslob/cli.py`,
`configs/experiments/fi2010_midprice_h10.yaml`,
`tests/test_fi2010_benchmark_preparation.py`, `docs/FI2010_BENCHMARK.md`,
`docs/CLI_REFERENCE.md`, `docs/EXPERIMENT_EVIDENCE_INDEX.md` and
`docs/REPRODUCIBILITY.md`.

CLI command: `python -m chronoslob.cli prepare-fi2010-benchmark --config configs/experiments/fi2010_midprice_h10.yaml --data-path PATH --out PATH`.

Tests expected: manifest creation from local paths, checksum calculation,
schema validation, split metadata checks and clear failures for missing data.

Strict non-goals: do not commit FI-2010 data, add download logic, infer licence
terms or fit transforms on validation/test rows.

### Phase C: Paper experiment runner

Purpose: provide one reproducible runner that consumes a config and writes the
standard artefacts for a single benchmark experiment.

Status: implemented as an initial runner under
`chronoslob/experiments/paper_runner.py`. The runner supports the
majority-class baseline (required) and optionally a train-only standardised
logistic regression baseline; stronger model families remain deferred to
Phase D and Phase E. It composes the Phase B preparation step, runs the
selected baselines on a deterministic temporal split and writes the standard
artefacts under the experiment artefact contract.

Files added or modified: `chronoslob/experiments/paper_runner.py`,
`chronoslob/experiments/__init__.py`, `chronoslob/cli.py`,
`tests/test_paper_experiment_runner.py`, `docs/PAPER_EXPERIMENTS.md`,
`docs/CLI_REFERENCE.md`, `docs/EXPERIMENT_EVIDENCE_INDEX.md` and
`docs/REPRODUCIBILITY.md`.

CLI command: `python -m chronoslob.cli run-paper-experiment --config configs/experiments/fi2010_midprice_h10.yaml --data-path PATH --out PATH [--models majority[,logistic]] [--overwrite]`.

Tests expected: deterministic tiny-fixture run, artefact validation, seed
recording, commit recording, split recording, overwrite protection and clear
failures for unsupported models or missing data paths.

Strict non-goals: do not optimise for headline metrics, select models on test
data or merge predictive, calibration and execution metrics into one score.

### Phase D: Classical baseline benchmark suite

Purpose: establish leakage-safe classical baselines as the benchmark floor.

Status: implemented by extending the paper experiment runner with a
classical model registry under `chronoslob/experiments/model_registry.py`.
The runner now supports `majority`, `logistic`, `ridge`, `elastic_net`,
`random_forest` and `gradient_boosting` as short model names exposed
via `--models`. The combined artefact set is unchanged
(`config.yaml`, `data_manifest.json`, `results.json`,
`predictions.csv`, `model_card.md`, `confusion_matrix.json`,
`runner_summary.json`) and `results.json` keeps predictive and
calibration metric groups conceptually separate via
`evidence_streams`.

Files added or modified: `chronoslob/experiments/model_registry.py`,
`chronoslob/experiments/paper_runner.py`,
`chronoslob/experiments/__init__.py`,
`chronoslob/experiments/fi2010_benchmark.py`,
`chronoslob/cli.py`, `configs/experiments/fi2010_midprice_h10.yaml`,
`tests/test_paper_experiment_runner.py`,
`tests/test_classical_paper_models.py`,
`docs/PAPER_EXPERIMENTS.md`, `docs/CLI_REFERENCE.md`,
`docs/EXPERIMENT_EVIDENCE_INDEX.md` and `docs/REPRODUCIBILITY.md`.

CLI command: `python -m chronoslob.cli run-paper-experiment --config
configs/experiments/fi2010_midprice_h10.yaml --data-path PATH --out PATH
[--models majority[,logistic,ridge,elastic_net,random_forest,gradient_boosting]]
[--overwrite]`.

Tests expected: registry recognises every supported classical name,
unsupported names fail clearly, comma-separated CLI lists work, a
multi-model run writes predictions, results and confusion-matrix
entries per model, probabilities are finite where emitted and sum to
one, preprocessing is fit on the training split only, the output
directory validates under the artefact contract and overwrite
protection is preserved.

Strict non-goals: do not tune on test data, add broad hyperparameter
searches, treat synthetic fixture smoke runs as benchmark evidence or
present per-model probabilities for models that do not natively emit
them.

### Phase E: Neural benchmark suite

Purpose: run DeepLOB-style, transformer and traceable SSL-pretrained
transformer variants under the same artefact contract.

Files likely to be added or modified: `chronoslob/training/torch_experiment.py`,
`chronoslob/training/transformer_experiment.py`,
`chronoslob/training/ssl_experiment.py`, `chronoslob/training/paper_experiment.py`,
`configs/experiments/fi2010_neural_benchmarks.yaml`,
`tests/test_fi2010_neural_benchmarks.py`.

CLI command expected: `python -m chronoslob.cli run-fi2010-neural-benchmarks --config configs/experiments/fi2010_neural_benchmarks.yaml`.

Tests expected: small deterministic smoke runs, model-config serialisation,
checkpoint metadata, prediction export and explicit pretraining provenance when
SSL is used.

Strict non-goals: do not call the transformer SSL-pretrained unless the
pretraining artefact exists, and do not claim an exact external-paper
reproduction unless the protocol matches closely enough.

### Phase F: Calibration and execution evidence

Purpose: compute calibration and execution-aware evidence from stored
predictions without retraining or changing model selection.

Files likely to be added or modified: `chronoslob/training/calibration.py`,
`chronoslob/backtest/validation.py`, `chronoslob/cli.py`,
`tests/test_empirical_calibration_execution.py`.

CLI command expected: `python -m chronoslob.cli build-empirical-evidence --experiment experiments/fi2010_midprice_h10`.

Tests expected: Brier score, ECE, reliability bins, confidence filtering,
turnover, cost sensitivity and latency sensitivity recomputed from stored
predictions.

Strict non-goals: do not refit calibrators on test predictions, hide execution
assumptions or report cost-aware metrics as predictive accuracy.

### Phase G: Plot generation and experiment inspection

Purpose: generate plots from validated artefacts and provide a read-only
inspection command for completed experiments.

Files likely to be added or modified: `chronoslob/analysis/plots.py`,
`chronoslob/cli.py`, `docs/CLI_REFERENCE.md`, `tests/test_experiment_plots.py`.

CLI command expected: `python -m chronoslob.cli inspect-experiment --path experiments/fi2010_midprice_h10`.

Tests expected: plot generation from stored CSV/Parquet/JSON artefacts,
read-only inspection, stable filenames and failures when source artefacts are
missing.

Strict non-goals: do not hand-edit plots, generate plots without data or add
visuals that imply unreported benchmark results.

### Phase H: Ablation suite

Purpose: quantify sensitivity to model components, labels, horizons,
pretraining, costs and split assumptions.

Files likely to be added or modified: `chronoslob/analysis/ablations.py`,
`chronoslob/analysis/sensitivity.py`, `configs/experiments/fi2010_ablations.yaml`,
`tests/test_empirical_ablations.py`.

CLI command expected: `python -m chronoslob.cli run-fi2010-ablation-suite --config configs/experiments/fi2010_ablations.yaml`.

Tests expected: ablation-spec validation, paired comparison tables, missing-run
handling and separation between predictive, calibration and execution-aware
metrics.

Strict non-goals: do not add unbounded experiment grids, use test data for
model choice or collapse all evidence into a single ranking.

### Phase I: Systems benchmark suite

Purpose: measure practical research-platform behaviour such as runtime,
memory, artefact size and optional inference latency sensitivity.

Files likely to be added or modified: `chronoslob/utils/benchmarking.py`,
`chronoslob/cli.py`, `reports/systems_benchmark.md`,
`tests/test_systems_benchmark.py`.

CLI command expected: `python -m chronoslob.cli run-systems-benchmark --experiment experiments/fi2010_midprice_h10`.

Tests expected: deterministic benchmark record schema, environment capture,
basic timing sanity checks and no hidden network calls.

Strict non-goals: do not optimise prematurely, compare against hardware not
recorded in the artefacts or present systems metrics as forecast quality.

### Phase J: Empirical report builder

Purpose: build a concise empirical report directly from validated experiment
artefacts.

Files likely to be added or modified: `chronoslob/utils/empirical_report.py`,
`chronoslob/cli.py`, `reports/report_archive/README.md`,
`tests/test_empirical_report_builder.py`.

CLI command expected: `python -m chronoslob.cli build-empirical-report --experiment experiments/fi2010_midprice_h10`.

Tests expected: generated Markdown from artefacts only, table consistency with
`results.json`, plot-link validation and clear omission of unavailable metrics.

Strict non-goals: do not hand-write benchmark tables, invent missing metrics or
edit generated reports to improve narrative fit.

### Phase K: Final public documentation update

Purpose: update public docs only after real artefacts exist and pass the
validation contract.

Files likely to be added or modified: `README.md`, `ROADMAP.md`,
`docs/PROJECT_STATUS.md`, `docs/EXPERIMENT_EVIDENCE_INDEX.md`,
`docs/REPRODUCIBILITY.md`, `reports/README.md`.

CLI command expected: `python -m chronoslob.cli inspect-release-readiness`.

Tests expected: public wording checks, local-link validation, report inventory
checks, audit pass and no unsupported result claims.

Strict non-goals: do not add a major results section before real artefacts
exist, repeat caveats across every document or change package version without
the established release process.

### Phase L: Private profile/interview conversion outside public docs

Purpose: translate the finished empirical work into private talking points and
interview preparation without adding employment-positioning language to the
public repository.

Files likely to be added or modified: none in the public repository.

CLI command expected: none.

Tests expected: public repository remains unchanged and release-readiness scans
continue to pass.

Strict non-goals: do not add private positioning notes, employer-specific
language or interview scripts to the public repository.

## 7. Reproducibility contract

Any future reported metric must trace to:

- config
- data source path or manifest
- checksum where possible
- seed
- split definition
- code commit
- model configuration
- predictions
- calibration artefacts where relevant
- execution assumptions where relevant

Transforms, bucket boundaries, calibrators and model-selection decisions must
be fitted or chosen without using test data.

## 8. Claim boundary

Acceptable public claims include:

- "reproducible FI-2010 benchmark experiment"
- "temporal validation"
- "calibration analysis"
- "execution-aware sensitivity analysis"
- "DeepLOB-style baseline"
- "self-supervised pretraining experiment", only if actually run

Unacceptable public claims include:

- "profitable trading strategy"
- "market-beating alpha"
- "production-grade market-execution system"
- "foundation model", unless genuinely pretrained across multiple
  instruments, regimes and tasks
- "DeepLOB reproduction", unless exact enough to justify the term
- any metric not backed by stored artefacts

## 9. Immediate next implementation recommendation

Implement the experiment artefact contract and validation layer.

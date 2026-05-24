# Paper Experiments

ChronosLOB ships a paper experiment runner that turns a user-supplied
local FI-2010-style file into a validated experiment artefact directory.
The runner is the predictive-quality evidence stream above the FI-2010
benchmark preparation step: it reuses preparation outputs, runs the
classical benchmark suite and writes the standard artefacts defined by
the experiment artefact contract.

The runner does not download data, does not perform any network call,
does not train neural models in this phase and does not invent
benchmark evidence from synthetic fixtures.

## What The Runner Does

`run-paper-experiment` performs these steps in order:

1. Loads the FI-2010 benchmark preparation config.
2. Validates that the supplied local data path exists and is a file.
3. Invokes the Phase B benchmark preparation logic to produce a data
   manifest, label summary, split summary and validation summary into
   a `preparation/` subdirectory of the output directory.
4. Reloads the local file through the same FI-2010 loader configuration
   so that feature and label rows align with the preparation summary.
5. Builds a deterministic temporal train/validation/test split.
6. Fits the requested classical models on the training rows only and
   evaluates them on the held-out test split. Per-model train-only
   standardisation is applied where required by the model.
7. Writes the standard artefact set: `config.yaml`,
   `data_manifest.json`, `results.json`, `predictions.csv`,
   `model_card.md`, `confusion_matrix.json` and `runner_summary.json`.
8. Validates the output directory against the experiment artefact
   contract and refuses to report success if required artefacts are
   missing or invalid.

The runner is deliberately limited: it does not refit calibrators on
test predictions, does not select model hyperparameters on test data,
does not produce plot files in this phase and does not produce
execution-aware sensitivity records in this phase. Those evidence
streams are tracked under later phases of the empirical upgrade plan.

## Currently Supported Models

Phase D establishes a classical benchmark suite. The supported short
model names — used in `--models`, the config file and tests — are:

- `majority` (always required) — deterministic majority-class baseline.
- `logistic` — `LogisticRegression` from scikit-learn on a train-only
  `TrainOnlyStandardScaler` projection of the feature matrix.
- `ridge` — `RidgeClassifier` on train-only standardised features.
  Does not emit calibrated class probabilities; calibration metrics
  for this model are omitted in `results.json`.
- `elastic_net` — `LogisticRegression` with the elastic-net penalty
  (`saga` solver) on train-only standardised features.
- `random_forest` — `RandomForestClassifier` on raw features.
- `gradient_boosting` — `GradientBoostingClassifier` on raw features.

Names are case-folded to lower-case before lookup. Other model
families — DeepLOB-style, transformer and self-supervised transformer —
remain explicitly out of scope for the classical benchmark suite and
are tracked under Phase E.

If `--models` is omitted, the runner defaults to `majority`. The
`majority` baseline must be present in any explicit selection so that
the output is always anchored to an interpretable floor.

## Passing Multiple Models

Models are supplied as a comma-separated list and de-duplicated while
preserving first-seen order:

```bash
python -m chronoslob.cli run-paper-experiment \
  --config configs/experiments/fi2010_midprice_h10.yaml \
  --data-path /local/path/to/fi2010_normalised.csv \
  --out runs/fi2010_midprice_h10 \
  --models majority,logistic,random_forest \
  --overwrite
```

Each model is fitted and evaluated independently. The combined
`predictions.csv` keeps a `model_name` column so rows are traceable
back to the model that produced them. `results.json` records one
`model_results` entry per successfully fitted model and
`confusion_matrix.json` records one `models` entry per successful
model.

If a non-required model fails to fit on the supplied data, the runner
records it as skipped (with the failure reason) in `runner_summary.json`
and the model card, and continues with the remaining models. If every
requested model fails, the runner raises an explicit error rather than
writing an empty result set.

## Difference From Benchmark Preparation

`prepare-fi2010-benchmark` only writes preparation artefacts and never
trains a model. `run-paper-experiment` includes the same preparation
artefacts (under a `preparation/` subdirectory) and additionally writes
the model run artefacts required by the experiment artefact contract:
`results.json`, `predictions.csv`, `model_card.md` and
`confusion_matrix.json`.

A preparation directory will fail
`inspect-experiment-artifacts` because required artefacts are missing,
while a paper experiment directory passes the same validator.

## Expected Outputs

A successful run writes the following layout into the output
directory:

```text
<out_dir>/
  config.yaml
  data_manifest.json
  results.json
  predictions.csv
  model_card.md
  confusion_matrix.json
  runner_summary.json
  preparation/
    config.yaml
    data_manifest.json
    label_summary.json
    split_summary.json
    validation_summary.json
    preparation_summary.json
```

`runner_summary.json` records the requested model list, the models
that ran successfully, any skipped models with their reason, the
metric groups emitted, the data source kind and split counts.

Plots, calibration bins and execution sensitivity records are not
written in this phase; the artefact contract treats them as optional
warnings, so their absence does not invalidate the directory.

## Metric Groups

`results.json` keeps predictive and calibration metrics conceptually
separate via the `evidence_streams` field:

- Predictive metrics include `accuracy`, `macro_f1`, `weighted_f1`,
  `balanced_accuracy`, `matthews_corrcoef`, `n_samples`,
  `class_count_train` and `class_count_test`.
- Calibration metrics include `brier_score`, `log_loss`,
  `expected_calibration_error` and `mean_confidence`, and are emitted
  only for models that produce calibrated class probabilities. Models
  without `predict_proba` (currently `ridge`) record no calibration
  metrics.
- Execution-aware metrics are not computed in this phase.

The `evidence_streams.execution` field is therefore set to
`["not_computed"]` for the classical benchmark suite. This is an
explicit placeholder until Phase F adds execution-sensitivity
artefacts.

## How Artefact Validation Works

After writing the artefacts, the runner calls
`validate_experiment_directory` from `chronoslob.experiments.artifacts`.
This is the same validator exposed by `inspect-experiment-artifacts`.
If any required artefact is missing or has an invalid schema, the
runner raises a clear error and the CLI exits with a non-zero status.

You can re-validate any directory at any time:

```bash
python -m chronoslob.cli inspect-experiment-artifacts \
  --experiment runs/paper_experiment_classical_smoke
```

## Smoke Command

The bundled FI-2010-like fixture under `tests/fixtures/fi2010` exists
only to exercise the runner plumbing. It is not the canonical FI-2010
benchmark and a synthetic fixture smoke run is not benchmark evidence,
market evidence or execution evidence.

```bash
python -m chronoslob.cli run-paper-experiment \
  --config configs/experiments/fi2010_midprice_h10.yaml \
  --data-path tests/fixtures/fi2010/tiny_fi2010_like.csv \
  --out runs/paper_experiment_classical_smoke \
  --models majority,logistic \
  --overwrite
```

`runs/` is in `.gitignore`, so smoke outputs are not committed. The
tiny fixture is too small for tree-based models in a strict comparison
setting; the smoke command therefore exercises `majority` and
`logistic`, while the heavier models are unit-tested through the
registry.

## Supplying A Real Local FI-2010 Path

When a real FI-2010-style file is available locally, replace the
`--data-path` argument with the path to that file:

```bash
python -m chronoslob.cli run-paper-experiment \
  --config configs/experiments/fi2010_midprice_h10.yaml \
  --data-path /local/path/to/fi2010_normalised.csv \
  --out runs/fi2010_midprice_h10 \
  --models majority,logistic,ridge,elastic_net,random_forest,gradient_boosting \
  --overwrite
```

The repository does not ship FI-2010 data and will never download it.
Real benchmark evidence requires a local FI-2010 path and stored
artefacts produced by this runner; the tiny fixture cannot stand in
for that evidence. Users are responsible for obtaining and licensing
any benchmark copy they use.

## Overwrite Protection

By default the runner refuses to write into a non-empty output
directory. Pass `--overwrite` to replace the directory contents. This
keeps accidental result loss explicit while still allowing iterative
runs against the tiny fixture.

## Why Synthetic Fixture Runs Are Not Market Evidence

The fixture file under `tests/fixtures/fi2010/tiny_fi2010_like.csv`
contains a handful of synthetic rows that exist only so the loader,
splitter, feature pipeline, baseline models, metric utilities and
artefact validator can be exercised end-to-end. The numbers it
produces describe fixture plumbing, not market microstructure. The
model card emitted for any fixture run states this explicitly.

## Out Of Scope For This Phase

- DeepLOB-style, transformer and SSL-pretrained transformer
  experiments. These are Phase E.
- Calibration evidence stream as a stored artefact (reliability bins,
  recomputed ECE). The runner records `brier_score`, `log_loss`,
  `expected_calibration_error` and `mean_confidence` in `results.json`
  when probabilities are compatible with the label set, but no
  separate `calibration_bins.csv` is generated in this phase.
- Execution-sensitivity evidence stream (cost, latency, turnover
  sensitivity). Tracked under Phase F.
- Plot generation. Tracked under Phase G.
- Ablation suites and systems benchmarks. Tracked under Phase H and
  Phase I.

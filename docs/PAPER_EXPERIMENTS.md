# Paper Experiments

ChronosLOB ships a paper experiment runner that turns a user-supplied
local FI-2010-style file into a validated experiment artefact directory.
The runner is the predictive-quality evidence stream above the FI-2010
benchmark preparation step: it reuses preparation outputs, runs the
requested classical and neural baselines and writes the standard
artefacts defined by the experiment artefact contract.

The runner does not download data, does not perform any network call
and does not invent benchmark evidence from synthetic fixtures.

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
6. Fits the requested models on the training rows only and evaluates
   them on the held-out test split. Per-model train-only
   standardisation or train-only tokenisation state is applied where
   required by the model.
7. Writes the standard artefact set: `config.yaml`,
   `data_manifest.json`, `results.json`, `predictions.csv`,
   `model_card.md`, `confusion_matrix.json` and `runner_summary.json`.
8. Validates the output directory against the experiment artefact
   contract and refuses to report success if required artefacts are
   missing or invalid.

The runner is deliberately limited: it does not refit calibrators on
test predictions and does not select model hyperparameters on test
data. Phase G now adds optional plot generation from stored
artefacts (see [Plot Generation](#plot-generation)).

Phase F now adds calibration and execution-aware sensitivity evidence:
when the runner produces predictions that contain confidence values it
also writes `calibration_bins.csv`, and when a forward mid-price
return proxy can be constructed from the supplied frame it writes
`execution_sensitivity.csv`. Both artefacts are derived only from
stored held-out prediction rows; no calibrator is fitted on test data
and no model selection uses test data.

## Currently Supported Models

Phase E supports the classical benchmark suite plus two neural paper
runner baselines. The supported short model names used in `--models`,
the config file and tests are:

- `majority` (always required) — deterministic majority-class baseline.
- `logistic` — `LogisticRegression` from scikit-learn on a train-only
  `TrainOnlyStandardScaler` projection of the feature matrix.
- `ridge` — `RidgeClassifier` on train-only standardised features.
  Does not emit class probabilities; calibration metrics
  for this model are omitted in `results.json`.
- `elastic_net` — `LogisticRegression` with the elastic-net penalty
  (`saga` solver) on train-only standardised features.
- `random_forest` — `RandomForestClassifier` on raw features.
- `gradient_boosting` — `GradientBoostingClassifier` on raw features.
- `deeplob_style` - compact DeepLOB-style CNN-LSTM baseline over
  split-contained FI-2010 windows. This is not an exact reproduction
  of the original architecture.
- `transformer` - supervised transformer baseline over deterministic
  snapshot-derived token windows.

Names are case-folded to lower-case before lookup. Other model
families are unsupported unless they appear in the registry. The
self-supervised transformer path is not registered in this phase
because the paper runner does not yet implement genuine train-only
pretraining and supervised fine-tuning.

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
  calibration_bins.csv        # emitted when models produce confidences
  execution_sensitivity.csv   # emitted when a return proxy is available
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
metric groups emitted, the data source kind and split counts. Phase F
adds an `evidence` block to `runner_summary.json` recording whether
`calibration_bins.csv` and `execution_sensitivity.csv` were written,
which models contributed to each artefact and any warning explaining
why an evidence stream was skipped.

When the runner emits calibration or execution evidence it also adds
`calibration_bins.csv` and `execution_sensitivity.csv` to the output
layout below and references them from `results.json`. Plots are
written under `<out_dir>/plots/` only when `--build-plots` is passed
to `run-paper-experiment` or when `build-paper-plots` is invoked
explicitly; the artefact contract treats them as optional artefacts,
so their absence does not invalidate the directory.

## Metric Groups

`results.json` keeps predictive and calibration metrics conceptually
separate via the `evidence_streams` field:

- Predictive metrics include `accuracy`, `macro_f1`, `weighted_f1`,
  `balanced_accuracy`, `matthews_corrcoef`, `n_samples`,
  `class_count_train` and `class_count_test`.
- Calibration metrics include `brier_score`, `log_loss`,
  `expected_calibration_error` and `mean_confidence`, and are emitted
  only for models that produce class probabilities. Models
  without `predict_proba` (currently `ridge`) record no calibration
  metrics. Phase F additionally writes `calibration_bins.csv`
  containing per-model reliability bins (`bin_index`, `bin_lower`,
  `bin_upper`, `count`, `mean_confidence`, `accuracy`,
  `confidence_gap`) computed from held-out test predictions.
- Execution-aware sensitivity metrics include
  `gross_signal_return_proxy`, `net_signal_return_proxy`,
  `turnover_proxy` and `hit_rate_proxy`. They are emitted in
  `execution_sensitivity.csv` whenever the configured forward
  mid-price return proxy can be constructed from the supplied frame.
  These are simplified proxy measurements under explicit cost
  assumptions, not a production backtest.

When evidence cannot be produced (for example when no model emits
probabilities, when the configured price columns are absent or when
the forward horizon falls outside the available frame) the runner
records a clear warning in `runner_summary.json` and the model card
and continues with the remaining artefacts.

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

`runs/` is in `.gitignore`, so smoke outputs are not committed. A CPU
neural smoke run can be exercised with:

```bash
python -m chronoslob.cli run-paper-experiment \
  --config configs/experiments/fi2010_midprice_h10.yaml \
  --data-path tests/fixtures/fi2010/tiny_fi2010_like.csv \
  --out runs/paper_experiment_neural_smoke \
  --models majority,deeplob_style,transformer \
  --overwrite
```

The neural settings used by this command are controlled by the
`neural_settings` section of
`configs/experiments/fi2010_midprice_h10.yaml`, including lookback,
batch size, epoch count, learning rate, model sizes and `device: cpu`.
The fixture remains a synthetic smoke run only; it is not benchmark
evidence.

## Supplying A Real Local FI-2010 Path

When a real FI-2010-style file is available locally, replace the
`--data-path` argument with the path to that file:

```bash
python -m chronoslob.cli run-paper-experiment \
  --config configs/experiments/fi2010_midprice_h10.yaml \
  --data-path /local/path/to/fi2010_normalised.csv \
  --out runs/fi2010_midprice_h10 \
  --models majority,logistic,ridge,elastic_net,random_forest,gradient_boosting,deeplob_style,transformer \
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

## Calibration And Execution Evidence

The calibration artefact is a per-model reliability table built from
held-out test predictions only. Bin edges are deterministic
(`n_bins` equal-width bins on the unit interval), empty bins are
recorded with `count` of 0 and finite placeholder values, and the
runner never refits any calibrator on test data.

The execution-sensitivity artefact is a cost-aware signal-quality
table. For each model that emits confidence values and each
combination of `confidence_threshold`, `cost_bps` and `latency_steps`
defined in the config, the runner records:

- `eligible_predictions` — rows above the threshold with a valid
  forward-return proxy.
- `trade_count_proxy` — eligible rows with a non-zero directional
  sign under the configured class-to-direction map.
- `turnover_proxy` — sum of absolute trade signs across eligible rows.
- `gross_signal_return_proxy` — mean of
  `direction_sign × forward_mid_return_bps` over trade rows.
- `cost_proxy` — the configured cost in basis points.
- `net_signal_return_proxy` — `gross_signal_return_proxy − cost_proxy`.
- `hit_rate_proxy` — fraction of eligible rows where the prediction
  matched the realised label.

The forward-return proxy is built from `bid_price_1` and `ask_price_1`
on the supplied frame and the horizon comes from the experiment
config. Where the proxy is unavailable for a row (for example the
forward horizon falls outside the test window) the row is recorded
with zero counts so the artefact remains traceable. This is an
explicit simplified analysis, not a production backtest, not a
tradable strategy and not live trading evidence.

## Plot Generation

Phase G adds deterministic plot generation from stored artefacts only.
Plots are placed under `<out_dir>/plots/` with stable filenames so
the experiment artefact contract recognises them as optional artefacts:

- `plots/reliability_curve.png` — built from `calibration_bins.csv`,
  one line per model plus a diagonal reference. Skipped with a warning
  when the CSV is absent or has no rows with positive bin counts.
- `plots/cost_sensitivity.png` — built from
  `execution_sensitivity.csv`, plotting `net_signal_return_proxy`
  against `cost_bps` and grouping by model and confidence threshold.
  Skipped with a warning when the CSV is absent or no rows have
  finite values. This is execution-aware sensitivity, not strategy
  performance.
- `plots/confusion_matrix.png` — built from `confusion_matrix.json`,
  one panel per model with axes labelled from the stored class
  identifiers. Skipped with a warning when the JSON is absent or has
  no usable matrix entries.
- `plots/regime_breakdown.png` — only generated when genuine regime
  data is available in stored artefacts (for example a `regime`
  column on `predictions.csv`). Skipped with a clear warning when no
  genuine regime breakdown is present. The runner never fabricates
  regimes from row numbers or timestamps.

Use the same runner with `--build-plots`:

```bash
python -m chronoslob.cli run-paper-experiment \
  --config configs/experiments/fi2010_midprice_h10.yaml \
  --data-path tests/fixtures/fi2010/tiny_fi2010_like.csv \
  --out runs/paper_experiment_plots_smoke \
  --models majority,logistic,deeplob_style,transformer \
  --overwrite \
  --build-plots
```

Or build plots later from a completed experiment directory:

```bash
python -m chronoslob.cli build-paper-plots \
  --experiment runs/paper_experiment_plots_smoke \
  --overwrite
```

A `plot_summary.json` artefact records the experiment directory, the
plots written, the plots skipped and any warnings, with timezone-aware
timestamps and finite, serialisable values only. Plot generation
failures for optional inputs are recorded as warnings and do not
invalidate the experiment artefact directory.

## Paper Ablation Suite

Phase H adds a traceable ablation suite for robustness analysis and
assumption sensitivity. It composes `run-paper-experiment` and writes
aggregate artefacts (`ablation_summary.json`,
`ablation_results.csv`, `ablation_manifest.json`) plus concise
Markdown reports. Child experiment directories are created only for
ablations that actually run; skipped ablations are explicit in the
summary, CSV and reports.

The smoke command is:

```bash
python -m chronoslob.cli run-paper-ablations \
  --config configs/experiments/fi2010_midprice_h10.yaml \
  --data-path tests/fixtures/fi2010/tiny_fi2010_like.csv \
  --out runs/paper_ablation_smoke \
  --models majority,logistic \
  --ablation-set smoke \
  --overwrite
```

The synthetic fixture output is a smoke artefact only and is not
benchmark evidence. See [PAPER_ABLATIONS.md](PAPER_ABLATIONS.md) for
the supported ablation sets, output layout and local FI-2010 usage
pattern.

## Inspecting A Paper Experiment Directory

`inspect-paper-experiment` prints a concise, read-only summary of a
completed experiment directory:

```bash
python -m chronoslob.cli inspect-paper-experiment \
  --experiment runs/paper_experiment_plots_smoke
```

The output lists artefact validation status, requested and skipped
models, evidence stream metric names, prediction/calibration/execution
row counts, plot inventory and whether the run is a synthetic fixture
smoke run. The command does not train a model, run inference or write
new files. Fixture smoke runs remain explicitly labelled as not
benchmark evidence. Real benchmark evidence requires a local FI-2010
path and stored artefacts produced by this runner.

## Out Of Scope For This Phase

- SSL-pretrained transformer experiments inside `run-paper-experiment`.
  The model name is left out of the supported registry until
  train-only pretraining and supervised fine-tuning are implemented
  end to end.
- Systems benchmarks. Tracked under Phase I.

# Paper Experiments

ChronosLOB ships a local paper experiment runner for the FI-2010 mid-price
direction task. It consumes a user-supplied FI-2010-style CSV, applies the
configured split policy, fits the requested baselines on training rows only and
writes a validated experiment artefact directory.

The runner does not download data, does not perform network calls and does not
derive benchmark evidence from the bundled synthetic fixture.

## Runner Contract

`run-paper-experiment` performs the following steps:

1. Loads the FI-2010 benchmark config.
2. Validates the supplied local data path.
3. Writes preparation artefacts under `preparation/`.
4. Builds the configured split. For combined FI-2010 CSV inputs with a
   `split` column, official test rows remain held out while validation is
   carved from official train rows only.
5. Fits the requested models on training rows only.
6. Evaluates on the held-out test split.
7. Writes `config.yaml`, `data_manifest.json`, `results.json`,
   `predictions.csv`, `model_card.md`, `confusion_matrix.json`,
   `runner_summary.json` and any available optional evidence artefacts.
8. Validates the output directory against the experiment artefact contract.

The runner does not select hyperparameters on test data and does not refit
calibrators on held-out predictions.

## Supported Models

Supported short names are:

- `majority` - required deterministic majority-class baseline.
- `logistic` - train-only standardised scikit-learn logistic regression.
- `ridge` - train-only standardised ridge classifier; calibration metrics are
  omitted because it does not emit class probabilities.
- `elastic_net` - train-only standardised elastic-net logistic regression.
- `random_forest` - scikit-learn random forest on the selected feature matrix.
- `gradient_boosting` - scikit-learn gradient boosting on the selected feature
  matrix.
- `deeplob_style` - compact supervised CNN-LSTM baseline over split-contained
  FI-2010 windows; not an exact external-paper reproduction.
- `transformer` - supervised transformer over the normalised FI-2010 matrix
  path.
- `matrix_transformer` - explicit alias for the same normalised matrix
  transformer path.

`transformer` and `matrix_transformer` consume prepared matrix windows
directly. They do not construct `OrderBookSnapshot` or `OrderBookLevel`
instances from z-score-normalised FI-2010 rows. Raw order-book schemas remain
strict, and negative raw quantities remain invalid.

`ssl_transformer` is not registered in the paper runner. It should only be
added after a genuine train-only pretraining and supervised fine-tuning path
exists.

## Evidence Artefacts

The runner keeps evidence streams separate:

- Predictive metrics include accuracy, macro-F1, weighted-F1, balanced
  accuracy, Matthews correlation coefficient and class-count metadata.
- Calibration metrics include Brier score, log loss, mean confidence, expected
  calibration error and `calibration_bins.csv` where probabilities are
  available.
- Execution-aware sensitivity writes `execution_sensitivity.csv` when a
  forward mid-price return proxy can be constructed. These rows are simplified
  proxy measurements under explicit confidence, cost and latency assumptions;
  they are not execution results for live markets.
- Plot generation is optional through `--build-plots` or
  `build-paper-plots`. Plots are derived from stored artefacts only.

`plots/regime_breakdown.png` is generated only when genuine regime data is
present in stored artefacts. It is skipped with a warning otherwise.

## Smoke Command

The bundled fixture under `tests/fixtures/fi2010` is a tiny synthetic file for
exercising the runner path. It is not FI-2010 benchmark evidence.

```bash
python -m chronoslob.cli run-paper-experiment \
  --config configs/experiments/fi2010_midprice_h10.yaml \
  --data-path tests/fixtures/fi2010/tiny_fi2010_like.csv \
  --out runs/paper_experiment_smoke \
  --models majority,logistic \
  --overwrite \
  --build-plots
```

`runs/` is ignored by git, so smoke outputs are not committed.

## Local FI-2010 Usage

After acquiring and converting FI-2010 locally, point `--data-path` at the
converted CSV:

```bash
python -m chronoslob.cli run-paper-experiment \
  --config configs/experiments/fi2010_midprice_h10.yaml \
  --data-path data/processed/fi2010/fold1_combined.csv \
  --out experiments/fi2010_midprice_h10 \
  --models majority,logistic,random_forest,gradient_boosting,deeplob_style,transformer \
  --overwrite \
  --build-plots
```

The repository does not ship FI-2010 data and does not fetch it. Users are
responsible for obtaining and licensing any benchmark copy they use.

## Inspecting Results

Re-validate and summarise a completed directory with:

```bash
python -m chronoslob.cli inspect-paper-experiment \
  --experiment experiments/fi2010_midprice_h10
```

The current committed FI-2010 run includes predictive metrics, calibration
bins, execution-aware sensitivity, reliability/cost/confusion plots and a
model card. The accompanying ablation and systems benchmark suites are
documented in [PAPER_ABLATIONS.md](PAPER_ABLATIONS.md) and
[SYSTEM_BENCHMARKS.md](SYSTEM_BENCHMARKS.md).

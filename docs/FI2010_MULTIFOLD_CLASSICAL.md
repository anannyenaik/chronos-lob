# FI-2010 Multi-Fold Classical Benchmarks

This runner evaluates classical baselines across prepared FI-2010 folds and
writes fold-level plus aggregate evidence. It is the classical evidence layer
only: no neural or self-supervised models are run.

## Prerequisite

Run multi-fold preparation first so each fold has a split-aware combined CSV:

```bash
python -m chronoslob.cli prepare-fi2010-multifold \
  --config configs/experiments/fi2010_multifold.yaml \
  --extracted-root data/raw/fi2010/extracted/BenchmarkDatasets \
  --processed-root data/processed/fi2010 \
  --out runs/fi2010_multifold_prepare \
  --folds all
```

## Command

```bash
python -m chronoslob.cli run-fi2010-multifold-classical \
  --config configs/experiments/fi2010_multifold.yaml \
  --processed-root data/processed/fi2010 \
  --out experiments/fi2010_multifold_classical \
  --models majority,logistic,ridge,elastic_net,random_forest,gradient_boosting \
  --folds all \
  --overwrite
```

Use `--folds 1,2` or a shorter `--models` list for a subset. The runner refuses
to replace a non-empty output directory unless `--overwrite` is supplied.

## Artefacts

```text
summary.json
results_by_fold.csv
results_summary.csv
calibration_summary.csv
execution_summary.csv
model_failures.json
folds/
  fold_1/
    results.json
    confusion_matrix.json
    calibration_bins.csv
    execution_sensitivity.csv
    model_card.md
```

Full prediction rows are not written by default.

## Metrics

`results_by_fold.csv` records accuracy, macro F1, MCC, Brier score where
available and expected calibration error where available for validation and
official test splits. `results_summary.csv` reports means and standard
deviations across folds. Calibration and execution-sensitivity summaries remain
separate evidence streams.

Execution-sensitivity values are labelled as proxy diagnostics. They include
confidence threshold, cost in basis points, latency steps, eligible prediction
count, trade-count proxy, turnover proxy, gross signal-return proxy, cost proxy,
net signal-return proxy and hit-rate proxy.

## Split Handling

Each prepared fold CSV must contain the configured `split` column. Official test
rows stay out of preprocessing, fitting and validation. Validation rows are
carved only from the tail of official train rows. Feature standardisation, when
needed, is fitted only on the remaining train rows.

## Limitations

Only `majority`, `logistic`, `ridge`, `elastic_net`, `random_forest` and
`gradient_boosting` are supported here. DeepLOB-style, transformer and
matrix-transformer names fail clearly in this runner.

The output is benchmark evidence under the recorded local data and config. No
profitability, deployment or market-execution claim is made.

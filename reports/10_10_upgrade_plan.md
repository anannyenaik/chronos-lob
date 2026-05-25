# ChronosLOB FI-2010 Empirical Release Notes

This note records the empirical-release status after the real FI-2010 fold-1
milestone. It is a maintenance note for the repository, not a narrative paper.

## Current Status

The main FI-2010 evidence stream is complete for the official NoAuction ZScore
fold-1 train/test pair. The data was verified locally, converted through the
official matrix adapter and evaluated through the combined CSV `split` column.
Official test rows remain held out from preprocessing, fitting, validation and
model-selection decisions.

Committed evidence now includes:

- `experiments/fi2010_midprice_h10/` for the main paper experiment
- `experiments/fi2010_midprice_h10_ablations/` for controlled ablations
- `experiments/fi2010_midprice_h10_systems/` for local systems measurements
- `reports/chronoslob_empirical_report.md` for the generated artefact report

## Technical Scope

The main experiment includes majority, logistic, random forest, gradient
boosting, DeepLOB-style and transformer baselines. In the current artefacts,
gradient boosting is the strongest model by macro-F1. The transformer result is
produced through the supervised normalised FI-2010 matrix path.

Raw order-book schemas remain strict: raw `OrderBookLevel` quantities are still
validated as non-negative, and z-score-normalised FI-2010 rows are not coerced
into raw order-book snapshot objects.

`ssl_transformer` remains unsupported in the paper runner. The ablation suite
records SSL pretraining as skipped until a traceable train-only pretraining and
supervised fine-tuning path exists.

## Evidence Streams

- Predictive metrics are stored in `results.json`.
- Calibration evidence is stored in `calibration_bins.csv`.
- Execution-aware sensitivity is stored in `execution_sensitivity.csv`.
- Plots are generated from stored artefacts and include reliability, cost
  sensitivity and confusion-matrix views.
- Ablations record run and skipped statuses explicitly.
- Systems benchmark reports record local loader throughput, matrix feature
  preparation, runner timing, inference latency and a small resource profile.

## Remaining Research Work

- Extend FI-2010 coverage across additional folds.
- Implement SSL pretraining and fine-tuning inside the paper runner before
  reporting SSL model results.
- Add adapters for additional limit order book datasets when data access and
  licensing are available.
- Improve execution modelling beyond the current proxy sensitivity layer.
- Add regime analysis only when genuine regime data is present in stored
  artefacts.

## Release Boundary

The repository presents ChronosLOB as research software for offline market
microstructure experiments. It does not present trading, live-execution or
broker-integration claims. Any reported metric must trace to a config, data
source, seed, split definition, code commit where available and stored
artefacts.

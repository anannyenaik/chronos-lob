# Project Status

Current package version: `0.1.0`.

ChronosLOB is a research platform for limit order book representation
learning, short-horizon market-state forecasting, calibration and
execution-aware validation.

## Implemented

- Local loaders, schemas, validators and small synthetic fixtures.
- Past-only feature generation and explicit future-horizon labels with
  no-look-ahead checks.
- Temporal, walk-forward and purged or embargoed splitters; train-only
  preprocessing; metadata-only experiment registry.
- Classical, DeepLOB-style and transformer model code paths.
- Self-supervised masked-field and next-field objectives plus
  multi-task fine-tuning infrastructure.
- Calibration, uncertainty and confidence-filtering diagnostics.
- Execution-aware validation with explicit assumptions for fees, spread
  costs, latency, turnover, passive fill proxies and risk constraints.
- Transfer, regime, ablation and sensitivity analysis utilities.
- Local audit, release-readiness inspection and evidence-archive
  builder.
- Real FI-2010 fold-1 evidence under official split-aware evaluation,
  including predictive metrics, calibration bins, execution-aware
  sensitivity, ablations and local systems measurements.
- Normalised FI-2010 matrix support for the supervised transformer
  paper-runner path.

## Current Evidence

The committed FI-2010 artefacts live under
[`experiments/fi2010_midprice_h10/`](../experiments/fi2010_midprice_h10/),
[`experiments/fi2010_midprice_h10_ablations/`](../experiments/fi2010_midprice_h10_ablations/)
and
[`experiments/fi2010_midprice_h10_systems/`](../experiments/fi2010_midprice_h10_systems/).
The generated artefact report is
[`reports/chronoslob_empirical_report.md`](../reports/chronoslob_empirical_report.md).

The current main experiment includes majority, logistic, random forest,
gradient boosting, DeepLOB-style and normalised-matrix transformer baselines.
`ssl_transformer` is not reported as a model result.

## Data Assumptions

- The repository ships no real exchange data; users provide any
  FI-2010 or public venue data locally.
- Synthetic fixtures exist only to exercise loaders, schemas, replay,
  features, labels and models on a tiny deterministic input.
- Crypto-style reconstruction examples are local engineering
  demonstrations and should not be treated as equivalent to
  equity-market behaviour.

## Current Limitations

- Execution-aware validation is a simplified research simulation. It
  does not model venue-specific queue priority, production-grade
  partial-fill behaviour or market impact.
- The current FI-2010 evidence covers fold 1 only.
- `ssl_transformer` remains unsupported in the paper runner.
- Robustness analysis beyond stored ablations requires additional
  documented experiment records.
- See [SAFETY_AND_LIMITATIONS.md](SAFETY_AND_LIMITATIONS.md) for the
  full scope statement.

## Next

Future work is focused on broader FI-2010 folds, a traceable SSL
pretraining/fine-tuning runner, additional limit order book dataset
adapters where data access allows, richer execution modelling and genuine
regime analysis.

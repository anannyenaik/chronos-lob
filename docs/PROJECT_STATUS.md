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

## Experimental Utilities

Some utilities are deliberately minimal and intended as building blocks:
the experiment registry, the adverse-selection summary and the
sensitivity-curve helpers expose deterministic infrastructure and are
expected to grow as real experiment runs accumulate.

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
- Robustness analysis utilities organise supplied experiment records;
  they do not produce evidence on their own.
- See [SAFETY_AND_LIMITATIONS.md](SAFETY_AND_LIMITATIONS.md) for the
  full scope statement.

## Next

Empirical experiments on real and locally hosted datasets with
documented provenance, temporal splits, seeds and stored outputs. From
those runs the predictive, calibration and execution-aware validation
streams can be reported as separate evidence.

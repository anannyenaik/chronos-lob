# Roadmap

ChronosLOB is developed as research software for market microstructure
modelling: reproducible data pipelines, leakage-safe labels, sequence models,
calibration and execution-aware evaluation.

## Completed

- Typed market-data contracts for order book snapshots, event records, feature
  rows, label rows and data-quality issues.
- Local FI-2010-style loading, official FI-2010 matrix conversion and offline
  replay utilities for supplied local files.
- Past-only microstructure features and future-window labels with dedicated
  no-look-ahead tests.
- Temporal, walk-forward, purged and embargoed splitters with train-only
  preprocessing.
- Classical baselines, DeepLOB-style supervised modelling and transformer
  infrastructure.
- Calibration, uncertainty and confidence-filtering diagnostics.
- Execution-aware validation utilities for spread costs, fees, latency,
  turnover, fill proxies and simple risk constraints.
- Robustness utilities for transfer summaries, regimes, ablations and
  sensitivity analysis.
- Reproducibility infrastructure: typed configs, deterministic smoke checks,
  release-readiness inspection, strict audit checks and experiment artefact
  validation.
- Real FI-2010 fold-1 empirical evidence on the official NoAuction ZScore
  train/test pair, using official split-aware evaluation.
- Normalised FI-2010 matrix support for the supervised transformer paper-runner
  path, without weakening raw order-book schema validation.
- FI-2010 calibration artefacts, execution-aware sensitivity rows, plots,
  controlled ablations and local systems benchmark artefacts.
- An evidence-backed empirical artefact report generated from stored FI-2010
  experiment, ablation and systems benchmark outputs.

## In Progress and Next

- Broaden FI-2010 coverage beyond the current fold-1 evidence set while keeping
  official split-aware evaluation and train-only preprocessing intact.
- Add a genuine SSL pretraining and supervised fine-tuning runner before
  reporting any `ssl_transformer` result.
- Extend data adapters for LOBSTER, ITCH or other limit order book formats when
  data access and licensing allow.
- Improve execution modelling with richer queue-position, partial-fill,
  latency and market-impact assumptions while keeping the current offline
  research boundary clear.
- Add richer regime analysis based on genuine stored regime features rather
  than row-number or timestamp-derived substitutes.
- Continue tightening report generation so public tables, plots and summaries
  are rebuilt directly from stored artefacts.

## Evaluation Principles

- Treat prediction, calibration and execution-aware sensitivity as separate
  evidence streams.
- Prefer temporal, official or walk-forward evaluation over random splits.
- Fit scalers, encoders, thresholds and calibrators on training data only.
- Keep data provenance, seeds, split definitions and code versions visible.
- Record limitations as modelling assumptions, not as hidden footnotes.

## Out of Scope

ChronosLOB focuses on reproducible research infrastructure and offline
evaluation. The current codebase deliberately excludes broker integration,
order placement, live trading connectivity and operational execution systems so
the research layer remains auditable and dataset-driven.

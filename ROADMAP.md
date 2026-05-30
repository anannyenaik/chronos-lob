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
- Real FI-2010 multi-fold classical evidence on the official NoAuction ZScore
  folds 1-5, using official split-aware evaluation.
- A completed one-epoch matched neural full grid across folds 1-5, horizons
  10/20/50 and seeds 0-2 comparing supervised, masked-SSL and next-field-SSL
  transformer variants; this is matched comparison evidence and supports no SSL
  improvement claim.
- A first partial-real proper-training neural subset, separate from the
  one-epoch full grid, covering fold 1, horizon 10, seed 0, lookback 10 and all
  three objectives with validation-only early stopping, best-checkpoint
  restoration and per-epoch curves. In this exact slice SSL did not improve
  macro-F1 or MCC; masked reconstruction improved ECE only in that row.
- A dedicated SSL failure-analysis report (`reports/ssl_failure_analysis/`, built
  by `analyse-fi2010-ssl-results`) generated from retained lightweight comparison
  tables only. It separates the completed one-epoch matched grid from the partial
  longer-training subset and records that the completed grid supports no broad SSL
  improvement; the only positive predictive-metric signal is narrow to fold 1,
  horizon 50 in the partial subset, where calibration worsened.
- A separate, earlier 25-epoch reduced-scope supervised neural FI-2010 benchmark
  across folds 1-5, with a single-seed and lookback-20 caveat, reported
  separately from the matched grid.
- Normalised FI-2010 matrix support for the supervised transformer paper-runner
  path, without weakening raw order-book schema validation.
- FI-2010 calibration artefacts, statistical uncertainty, brutal ablations,
  execution-aware proxy diagnostics and external benchmark protocol context.
- A richer execution-aware proxy analysis (`analyse-fi2010-execution-v3`) over the
  retained execution-v3 tables, covering confidence filtering, active fraction,
  turnover, cost, latency, fill and adverse-selection proxies, with regime
  diagnostics explicitly skipped and no PnL or live-trading claim.
- A final empirical report generated from stored multi-fold FI-2010 classical,
  the matched one-epoch neural full grid, the separate reduced-scope neural
  benchmark, uncertainty, ablation, execution-proxy and external-context
  artefacts.

## In Progress and Next

- Broaden the proper-training neural subset beyond the current fold-1,
  horizon-10, seed-0, lookback-10 slice, while keeping official split-aware
  evaluation, train-only preprocessing and validation-only model selection
  intact.
- Extend evidence beyond FI-2010 to other limit order book datasets where data
  access and licensing allow.
- Broaden genuine train-only SSL pretraining and supervised fine-tuning
  evidence before making any SSL improvement claim.
- Extend data adapters for LOBSTER, ITCH or other limit order book formats when
  data access and licensing allow.
- Improve execution modelling with richer queue-position, partial-fill,
  latency and market-impact assumptions while keeping the current offline
  research boundary clear.
- Add richer regime analysis based on genuine stored regime features rather
  than row-number or timestamp-derived substitutes.
- Continue tightening report generation so public tables and summaries are
  rebuilt directly from stored artefacts.

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

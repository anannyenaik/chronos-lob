# Roadmap

This roadmap summarises implemented components and future work for ChronosLOB.
It is intentionally conservative: planned work is not presented as available
functionality until it is implemented, tested and documented.

## Completed Components

- Project scaffold, packaging, CI and local validation tooling.
- Typed schemas for market events, order books, features, labels and
  data-quality issues.
- Local FI-2010-style loading and validation helpers.
- Past-only microstructure feature generation.
- Future-window label generation and no-look-ahead checks.
- Temporal, walk-forward and purged or embargoed splitters.
- Metadata-only experiment registry utilities.
- Classical baseline interfaces, train-only preprocessing and metrics.
- PyTorch sequence-window datasets and loaders.
- DeepLOB-style supervised baseline plumbing.
- Offline Binance-style order book reconstruction.
- Canonical JSONL event-log storage and replay utilities.
- Deterministic event tokenisation and transformer input preparation.
- Supervised transformer encoder architecture.
- Self-supervised masked-field and next-field objectives.
- Multi-task fine-tuning infrastructure.
- Calibration, uncertainty and confidence-filtering diagnostics.
- Execution-aware validation utilities with explicit simplified assumptions for
  costs, latency, turnover, risk constraints and adverse selection.
- Transfer, regime, ablation and sensitivity analysis summaries.
- Local project audit, public release-readiness inspection and technical
  evidence archive generation.

## Future Work

- Manual technical report written from reproducible experiment outputs.
- Real data experiment runs with documented provenance, temporal splits, seeds,
  code versions and output paths.
- Calibration and execution-aware validation reports generated from real
  experiment artefacts.
- More detailed robustness analysis once real experiment records exist.
- Optional documentation improvements for public review.

## Out Of Scope

- Live data ingestion or order placement.
- Broker or exchange integration.
- Production queue-position, partial-fill or market impact modelling.
- Portfolio optimisation.
- Dashboard outputs.
- Notebook outputs with embedded results.
- Fake benchmark tables, placeholder metrics or manually invented plots.

## Release Principle

ChronosLOB should remain a reproducible experiment artefact. Any claim about
results must be backed by auditable inputs, configs, seeds, code versions and
stored outputs, with limitations stated alongside the result.

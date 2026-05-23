# Roadmap

## Completed

- Data contracts and validators for events, order books, features,
  labels and quality issues; local FI-2010-style loading and offline
  Binance-style order book reconstruction.
- Past-only microstructure feature generation, future-window labels and
  no-look-ahead checks.
- Temporal, walk-forward and purged or embargoed splitters, train-only
  preprocessing and metadata-only experiment registry utilities.
- Classical baselines, DeepLOB-style supervised baseline, PyTorch
  sequence-window datasets and loaders.
- Canonical JSONL event-log storage, deterministic event tokenisation
  and transformer input preparation.
- Supervised transformer encoder, masked-field and next-field
  self-supervised objectives and multi-task fine-tuning infrastructure.
- Calibration, uncertainty and confidence-filtering diagnostics.
- Execution-aware validation utilities with explicit assumptions for
  fees, spread costs, latency, turnover, risk constraints and
  passive-fill proxies.
- Transfer, regime, ablation and sensitivity analysis utilities.
- Local repository audit, release-readiness inspection and technical
  evidence archive builder.

## In Progress and Next

- Empirical experiments on real and locally hosted datasets with
  documented data provenance, temporal splits and seeds.
- Result artefact management for predictive, calibration and
  execution-aware validation outputs as separate evidence streams.
- Reporting generated directly from reproducible experiment runs rather
  than from hand-edited tables.

## Future Extensions

- Additional pretraining objectives and downstream evaluation protocols.
- Broader public dataset adapters under the same leakage-safe contract.
- Deeper robustness analysis across regimes and instruments.

## Out of Scope

- Live data ingestion or order placement.
- Broker or exchange integration.
- Production queue-position, partial-fill or market impact modelling.
- Portfolio optimisation.

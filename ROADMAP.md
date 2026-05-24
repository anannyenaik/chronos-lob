# Roadmap

ChronosLOB is developed as a research-engineering platform for market microstructure modelling: reproducible data pipelines, leakage-safe labels, sequence models, calibration and execution-aware evaluation.

## Completed

- **Typed market-data contracts** for order book snapshots, event records, feature rows, label rows and data-quality issues.
- **Local data adapters** for FI-2010-style benchmark files and offline Binance-style order book replay from synthetic fixtures.
- **Order book tooling** for local book state, deterministic replay, canonical JSONL event logs and manifest hashing.
- **Microstructure feature generation** covering mid-price, spread, microprice, depth, imbalance, realised volatility, event intensity and regime-style summaries.
- **Future-window label construction** for direction, return buckets, volatility regime, spread widening, passive-fill proxy and adverse-selection proxy.
- **Leakage controls** including feature-label separation, no-look-ahead validation, temporal splits, walk-forward splits, purging, embargoing and train-only preprocessing.
- **Baseline modelling stack** with majority class, logistic regression, ridge, elastic-net logistic, random forest, gradient boosting and a DeepLOB-style CNN-LSTM.
- **Sequence learning infrastructure** with PyTorch datasets, sequence-window batching, event tokenisation and transformer-ready field-wise token channels.
- **Transformer modelling stack** including a supervised encoder, masked-field self-supervised objectives, next-field prediction and multi-task fine-tuning heads.
- **Calibration and uncertainty tooling** including temperature scaling, Brier score, expected calibration error, reliability bins, confidence filtering and abstention curves.
- **Execution-aware validation utilities** for spread costs, fees, latency, turnover, inventory limits, risk constraints, passive-fill assumptions and adverse-selection tracking.
- **Robustness analysis utilities** for regime summaries, transfer matrices, ablations, sensitivity curves and unified experiment records.
- **Reproducibility infrastructure** including typed configs, deterministic smoke checks, audit commands, CI, release-readiness checks and an experiment evidence archive.

## In Progress and Next

- Run documented experiments on real locally hosted datasets with clear data provenance, temporal splits, seeds and saved configurations.
- Produce evidence-backed result tables for predictive performance, calibration quality and execution-aware validation as separate outputs.
- Add model-card-style experiment summaries that record dataset, split, horizon, label definition, fitted preprocessing, calibration method and execution assumptions.
- Strengthen report generation so plots and tables are rebuilt from stored experiment artefacts rather than edited manually.
- Expand cross-instrument and regime-shift evaluations once suitable local datasets are available.

## Research Extensions

- Add richer self-supervised objectives such as future-state reconstruction, contrastive market-state learning and multi-horizon pretraining.
- Extend dataset adapters for additional public or institutionally available limit order book formats under the same validation contract.
- Add embedding analysis for learned market states, including regime clustering and instrument-transfer diagnostics.
- Benchmark supervised-only training against self-supervised pretraining across instruments, horizons and market regimes.
- Extend ablation studies across feature groups, token fields, model size, lookback length, latency assumptions and confidence thresholds.
- Improve calibration analysis with conformal-style confidence sets and task-specific abstention policies.

## Engineering Extensions

- Add checkpointed experiment runners for longer neural training jobs.
- Add richer artefact tracking for metrics, predictions, calibration curves, execution traces and sensitivity outputs.
- Add optional notebook dashboards for inspecting embeddings, calibration, latency sensitivity and regime-specific performance.
- Improve large-file data handling while keeping raw market data outside version control.
- Add benchmark profiles for runtime, memory use and inference latency.

## Evaluation Principles

- Treat prediction, calibration and execution-aware validation as separate evidence streams.
- Prefer temporal and walk-forward evaluation over random splits.
- Fit scalers, encoders, thresholds and calibrators on training data only.
- Keep execution assumptions explicit and configurable.
- Avoid static claims that are not backed by reproducible runs.
- Present limitations as modelling assumptions, not afterthoughts.

## Out of Scope

ChronosLOB focuses on reproducible research infrastructure and offline
evaluation. The current codebase deliberately keeps live trading
connectivity, broker integration, order placement and production
market-impact modelling outside the core platform so the research layer
remains auditable and dataset-driven.

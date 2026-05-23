# Limitations

ChronosLOB is a research-engineering project for market microstructure
modelling, leakage-safe forecasting and execution-aware validation. It is not
financial advice, not a live trading system and not a production execution
platform.

No real model result, FI-2010 benchmark result, live venue result or execution
performance claim is committed to this repository. Smoke outputs are synthetic
plumbing checks only.

## Public Data Limitations

Public limit order book data can support reproducible research, but it may have
restricted coverage, simplified message semantics, survivorship effects,
preprocessing choices, missing venue context or unclear timestamp conventions.
Users must document the exact data source, preprocessing assumptions and label
construction used by any future experiment.

## Synthetic Fixture Caveat

The fixtures under `tests/fixtures/` are synthetic. They exist to exercise
loaders, schemas, replay, features, labels, models, calibration, execution
validation and analysis plumbing. They are not real market data and must not be
reported as benchmark evidence, market evidence or execution evidence.

## FI-2010 Caveat

The FI-2010-style loader reads local user-provided files only. The repository
does not download or bundle FI-2010 data and does not claim replication of any
published benchmark result. Different mirrors and preprocessing conventions may
change feature layouts, labels and splits.

## Binance And Crypto Caveat

Offline Binance-style reconstruction utilities are local engineering
demonstrations against supplied files. The bundled Binance-style fixtures are
synthetic. Crypto market microstructure should not be overclaimed as directly
equivalent to equity-market behaviour.

## Forecasting Versus Tradability

Forecast accuracy, macro-F1, MCC, NLL, Brier score, calibration error and
confidence-filtering diagnostics do not automatically imply cost-adjusted signal
quality. Prediction quality, uncertainty quality and execution-aware validation
must remain separate in reports.

## Simplified Execution Validation

The execution-aware validation layer is deterministic research-simulation
infrastructure. It supports aggressive, passive and hybrid modes, fees, spread
costs, row-step latency, turnover, confidence-threshold sweeps, passive fill
proxies, adverse-selection labels and simple risk constraints.

It does not implement live trading, broker or exchange integration,
venue-specific queue priority, production partial-fill realism, a production
queue model, portfolio optimisation or a market impact model. Any future
execution report must state these assumptions explicitly.

## Modelling Limitations

Classical baselines, DeepLOB-style supervised plumbing, transformer architecture,
self-supervised objectives and multi-task fine-tuning infrastructure are
implemented, but no real benchmark performance is reported. Smoke losses and
accuracies from synthetic fixtures validate code paths only.

## Robustness Analysis Limitations

Transfer, regime, ablation and sensitivity utilities organise supplied
experiment records. They do not generate evidence by themselves. Real robustness
analysis requires real upstream experiment records produced from documented
configs, seeds, data versions and code versions.

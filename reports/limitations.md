# Limitations

ChronosLOB is research software for limit order book modelling,
leakage-safe forecasting and execution-aware validation. The full
scope statement is in
[docs/SAFETY_AND_LIMITATIONS.md](../docs/SAFETY_AND_LIMITATIONS.md);
this note adds the technical caveats that are most relevant when
extending the platform.

## Public Data

Public limit order book data can support reproducible research but may
have restricted coverage, simplified message semantics, survivorship
effects, preprocessing choices, missing venue context or unclear
timestamp conventions. Any experiment should document its exact data
source, preprocessing steps and label construction.

## Synthetic Fixtures

Files under `tests/fixtures/` are synthetic. They exist to exercise
loaders, schemas, replay, features, labels, models, calibration,
execution-aware validation and analysis code paths.

## FI-2010

The FI-2010-style loader reads local user-provided files only. The
repository does not download or bundle FI-2010 data. Different mirrors
and preprocessing conventions can change feature layouts, labels and
splits, so any experiment must record the exact mirror in use.

## Binance and Crypto

Offline Binance-style reconstruction utilities are local engineering
demonstrations against supplied files. Bundled Binance-style fixtures
are synthetic. Crypto market microstructure is not directly equivalent
to equity-market behaviour.

## Forecasting Versus Tradability

Forecast accuracy, macro-F1, MCC, NLL, Brier score, calibration error
and confidence-filtering diagnostics do not in themselves characterise
cost-aware signal quality. Predictive, calibration and execution-aware
validation evidence remain separate streams.

## Simplified Execution Validation

The execution-aware validation layer is a deterministic research
simulation. It supports aggressive, passive and hybrid modes, fees,
spread costs, row-step latency, turnover, confidence-threshold sweeps,
passive fill proxies, adverse-selection labels and simple risk
constraints.

It is not a live trading system, does not integrate with brokers or
exchanges, and does not implement venue-specific queue priority,
production-grade partial fills, a production queue model, portfolio
optimisation or a market impact model.

## Robustness Analysis

Transfer, regime, ablation and sensitivity utilities organise supplied
experiment records. They do not produce evidence on their own; real
analysis requires upstream experiment records from documented configs,
seeds, data versions and code commits.

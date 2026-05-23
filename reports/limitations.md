# Limitations

This repository currently contains scaffold code, canonical schemas, a local
FI-2010-style loader, a microstructure feature engine, a future-window label
engine, leakage-control utilities, classical baselines, PyTorch sequence
datasets, DeepLOB-style and transformer model plumbing, self-supervised
objectives, supervised multi-task fine-tuning infrastructure and calibration
utilities. No execution-aware simulator, backtest or real benchmark result has
been implemented before Phase 16. Phase 16 now adds simplified execution-aware
validation utilities for spread costs, fees, latency, turnover, passive
fill-proxy assumptions, adverse-selection labels and risk constraints. These
utilities are deterministic research-simulation infrastructure only.

No model results exist yet. No trading performance is claimed.

ChronosLOB is a research-engineering project for studying market microstructure
representations, leakage-safe forecasting and execution-aware validation. It is not
financial advice and should not be presented as a deployable trading system.

Public limit order book data can be useful for reproducible research, but it has
limitations. Datasets may have restricted coverage, simplified message semantics,
survivorship effects, missing venue context or preprocessing choices that are not
fully observable.

Crypto market microstructure differs from equities. Public crypto data may support
engineering demonstrations, but results on crypto venues should not be overclaimed
as directly equivalent to equity-market behaviour.

Any backtests or execution-validation runs are simplified research simulations
unless explicitly proven otherwise. Queue position, latency, market impact,
maker/taker fees, partial fills, order priority, exchange-specific matching
rules and data delays all require careful assumptions.

The project should separate forecast quality from tradability. Accuracy,
cross-entropy or calibration improvements do not automatically imply
cost-adjusted signal quality or profitable execution.

The supervised transformer encoder added in Phase 12 is architecture and
plumbing only. It consumes field-wise tokenised market microstructure batches
and produces classification logits, but it has no real-label training run, no
calibration, no execution simulation and no backtest. Smoke-training paths use
deterministic synthetic labels derived from the window index and the final
token's side; these labels carry no market information and must not be
reported as forecast or trading evidence.

The self-supervised pretraining wrapper added in Phase 13 is pretraining
infrastructure only. It implements masked field modelling and one-step
next-field prediction over the same field-wise token batches, using targets
derived entirely from the token sequence (no supervised market labels). The
contrastive objective is deferred. Loss values produced by the SSL smoke
runner verify only that the wrapper builds, accepts masked and next-field
targets, and supports backward passes through the wrapped encoder; they do
not measure forecast quality, alpha, Sharpe, profitability, tradability or
execution viability.

The multi-task fine-tuning layer added in Phase 14 is supervised training
infrastructure only. It reuses the field-wise transformer backbone and adds
classification heads for direction, return quantile, volatility regime, spread
widening, fill-proxy and adverse-selection proxy labels. Its smoke runner uses
a tiny synthetic event-log fixture and reports only plumbing diagnostics.
Those losses and accuracies are not market evidence and must not be presented
as alpha, tradability, profitability, Sharpe or execution viability.

The calibration and uncertainty layer added in Phase 15 is probabilistic
forecasting infrastructure only. It implements negative log-likelihood, Brier
score, expected calibration error, reliability-bin data, temperature scaling,
confidence filtering and abstention analysis for classifier outputs. Its smoke
runner uses synthetic logits and labels only, fits temperature on a synthetic
calibration subset and evaluates a separate synthetic subset. Confidence
filtering is not a trading strategy, abstention analysis is not an execution
backtest, and calibration diagnostics do not claim alpha, Sharpe, profitability
or execution viability. Execution-aware validation is intentionally left for
Phase 16.

The execution-aware validation layer added in Phase 16 is simulation
infrastructure only. It supports aggressive, passive and hybrid execution modes,
spread/fee cost decomposition, row-step latency, turnover accounting,
confidence-threshold sweeps, passive fill-proxy checks, adverse-selection labels
and simple inventory, trade-count, turnover and drawdown constraints. Its
synthetic smoke runner uses deterministic synthetic market states and signals.
Simulated PnL is an execution-validation metric, not a profitability claim.
The layer does not implement live trading, broker or exchange integration,
production market impact, venue-specific queue priority, partial-fill realism,
portfolio optimisation or deployable trading decisions.

The transfer, regime and ablation analysis layer added in Phase 17 is
analysis infrastructure only. It organises and summarises supplied
experiment-result records into regime breakdowns, transfer matrices,
ablation comparisons and sensitivity curves. It does not train models,
does not run backtests and does not generate evidence by itself. The
layer keeps predictive metrics (accuracy, macro-F1, MCC, NLL, Brier
score, ECE), calibration diagnostics and execution-validation metrics
clearly separated, and explicitly refuses any combined or magic score
field. Regime thresholds are explicit by default; data-derived
boundaries require a separate `fit_regime_boundaries` call driven by
training or calibration data only. The synthetic smoke runner uses
deterministic synthetic records and is plumbing only; its outputs are
not market evidence, alpha evidence, tradability evidence or live
performance. Real robustness analysis requires real upstream
experiment records.

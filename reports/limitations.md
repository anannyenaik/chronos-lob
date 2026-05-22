# Limitations

This repository currently contains only scaffold code. No data loader, feature
engine, label generator, model, training loop or execution-aware simulator has been
implemented yet.

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

Future backtests will be simplified research simulations unless explicitly proven
otherwise. Queue position, latency, market impact, maker/taker fees, partial fills,
order priority, exchange-specific matching rules and data delays all require careful
assumptions.

The project should separate forecast quality from tradability. Accuracy,
cross-entropy or calibration improvements do not automatically imply cost-adjusted
signal quality or profitable execution.

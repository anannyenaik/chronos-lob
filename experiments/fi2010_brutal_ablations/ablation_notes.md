# FI-2010 Brutal Ablation Notes

## Purpose

These ablations stress where the supervised FI-2010 signal survives and where it breaks across feature groups, model class, lookback, horizon, calibration threshold and execution cost or latency assumptions.

## Families run

- `feature_groups`
- `horizon`
- `model_class`
- `calibration`
- `execution`

## Families skipped

- `lookback`

## Reading the deltas

- `delta_vs_baseline` is `metric_value - baseline_value` for the family baseline.
- feature_groups baseline is `all_features`; a large negative delta means that feature subset loses signal.
- model_class baseline is `gradient_boosting`; a negative delta means the simpler model class is weaker.
- horizon baseline is the configured target horizon; positive deltas mark easier horizons.
- calibration and execution deltas are measured against the zero threshold, zero cost and zero latency reference.

## Strongest and weakest findings

- Most robust feature group: `price_only` (mean test macro-F1 delta -0.0676 vs all_features).
- Weakest feature group: `top_of_book_only` (mean test macro-F1 delta -0.0746 vs all_features).

## Skipped ablations

- `lookback/lookback_sweep`: neural lookback sweep not requested; this is CPU-expensive, so pass --neural-lookbacks (and --max-epochs) to execute it

## Execution metrics are proxies

Execution cost and latency numbers are simplified proxy diagnostics under stated assumptions, not a backtest.

## Boundaries

- Diagnostic only; no profitability or live tradability claim is made.
- Execution numbers are simplified proxy diagnostics, not a backtest.
- No foundation-model or leading-benchmark claim.
- No self-supervised pretraining result is reported.
- Neural superiority over the classical baseline is not asserted.

# Execution-v3 Proxy Analysis

Builder version `fi2010-execution-aware-proxy-analysis/v1`.

This report is a richer, reviewer-facing summary of the FI-2010 execution-v3 outputs. It is an
offline execution-aware proxy diagnostic. It is not PnL, not live-trading evidence and not a
production execution simulator. The confidence, cost, latency, fill and adverse-selection
diagnostics test whether predictive metrics survive simple execution-like frictions; cost-adjusted
values are cost-adjusted proxies only.

It is generated only from retained lightweight execution-v3 tables. The heavy raw per-run prediction
arrays were deleted to save storage and are not required by this analysis.

- Source execution-v3 directory: `experiments/fi2010_execution_v3`.
- Payoff mode: `unit_payoff`; cost mode: `unit_proxy`.
- Run groups summarised: 135.
- Smoke-test inputs: no.

## Confidence Filtering and Active Fraction

Mean retained fraction, active fraction, abstention fraction and cost-adjusted proxy by confidence
threshold (averaged over folds, seeds and objectives at each horizon shown below by objective):

| objective | threshold | active fraction |
| --- | --- | --- |
| masked_reconstruction | 0.33 | 0.6512 |
| masked_reconstruction | 0.35 | 0.6447 |
| masked_reconstruction | 0.40 | 0.5600 |
| masked_reconstruction | 0.45 | 0.4170 |
| masked_reconstruction | 0.50 | 0.2689 |
| masked_reconstruction | 0.55 | 0.1618 |
| masked_reconstruction | 0.60 | 0.0853 |
| masked_reconstruction | 0.65 | 0.0393 |
| masked_reconstruction | 0.70 | 0.0147 |
| masked_reconstruction | 0.75 | 0.0043 |
| masked_reconstruction | 0.80 | 0.0016 |
| masked_reconstruction | 0.85 | 0.0006 |
| masked_reconstruction | 0.90 | 0.0000 |
| masked_reconstruction | 0.95 | 0.0000 |
| next_field | 0.33 | 0.3456 |
| next_field | 0.35 | 0.3413 |
| next_field | 0.40 | 0.2581 |
| next_field | 0.45 | 0.1858 |
| next_field | 0.50 | 0.1306 |
| next_field | 0.55 | 0.0868 |
| next_field | 0.60 | 0.0585 |
| next_field | 0.65 | 0.0441 |
| next_field | 0.70 | 0.0288 |
| next_field | 0.75 | 0.0182 |
| next_field | 0.80 | 0.0114 |
| next_field | 0.85 | 0.0053 |
| next_field | 0.90 | 0.0026 |
| next_field | 0.95 | 0.0026 |
| supervised | 0.33 | 0.6352 |
| supervised | 0.35 | 0.6282 |
| supervised | 0.40 | 0.5307 |
| supervised | 0.45 | 0.3883 |
| supervised | 0.50 | 0.2493 |
| supervised | 0.55 | 0.1406 |
| supervised | 0.60 | 0.0676 |
| supervised | 0.65 | 0.0264 |
| supervised | 0.70 | 0.0081 |
| supervised | 0.75 | 0.0020 |
| supervised | 0.80 | 0.0008 |
| supervised | 0.85 | 0.0004 |
| supervised | 0.90 | 0.0000 |
| supervised | 0.95 | 0.0000 |

Higher confidence thresholds retain fewer predictions and lower the active fraction; the active
fraction is the share of all samples that remain directional (non-abstaining) after filtering.

## Turnover Proxy

Mean signal-change-rate turnover proxy by confidence threshold:

| objective | threshold | turnover proxy |
| --- | --- | --- |
| masked_reconstruction | 0.33 | 0.6512 |
| masked_reconstruction | 0.35 | 0.6448 |
| masked_reconstruction | 0.40 | 0.5600 |
| masked_reconstruction | 0.45 | 0.4170 |
| masked_reconstruction | 0.50 | 0.2689 |
| masked_reconstruction | 0.55 | 0.1618 |
| masked_reconstruction | 0.60 | 0.0853 |
| masked_reconstruction | 0.65 | 0.0393 |
| masked_reconstruction | 0.70 | 0.0147 |
| masked_reconstruction | 0.75 | 0.0043 |
| masked_reconstruction | 0.80 | 0.0016 |
| masked_reconstruction | 0.85 | 0.0006 |
| masked_reconstruction | 0.90 | 0.0000 |
| masked_reconstruction | 0.95 | 0.0000 |
| next_field | 0.33 | 0.3456 |
| next_field | 0.35 | 0.3414 |
| next_field | 0.40 | 0.2582 |
| next_field | 0.45 | 0.1858 |
| next_field | 0.50 | 0.1306 |
| next_field | 0.55 | 0.0868 |
| next_field | 0.60 | 0.0585 |
| next_field | 0.65 | 0.0441 |
| next_field | 0.70 | 0.0288 |
| next_field | 0.75 | 0.0182 |
| next_field | 0.80 | 0.0114 |
| next_field | 0.85 | 0.0053 |
| next_field | 0.90 | 0.0026 |
| next_field | 0.95 | 0.0026 |
| supervised | 0.33 | 0.6352 |
| supervised | 0.35 | 0.6282 |
| supervised | 0.40 | 0.5307 |
| supervised | 0.45 | 0.3883 |
| supervised | 0.50 | 0.2493 |
| supervised | 0.55 | 0.1406 |
| supervised | 0.60 | 0.0676 |
| supervised | 0.65 | 0.0264 |
| supervised | 0.70 | 0.0081 |
| supervised | 0.75 | 0.0020 |
| supervised | 0.80 | 0.0008 |
| supervised | 0.85 | 0.0004 |
| supervised | 0.90 | 0.0000 |
| supervised | 0.95 | 0.0000 |

Turnover proxy falls as the confidence threshold rises, because fewer active directional signals
remain. This is a turnover proxy, not a realised order count.

## Cost Sensitivity

Mean gross proxy, cost-adjusted proxy and degradation percentage across the fee and
spread-multiplier grid (averaged over objectives and horizons):

| fee bps | spread x | gross proxy | cost-adjusted proxy | degradation % |
| --- | --- | --- | --- | --- |
| 0.0 | 0.0 | -6159.0 | -6159.0 | 0.00 |
| 0.0 | 0.5 | -6159.0 | -6256.1 | 1.91 |
| 0.0 | 1.0 | -6159.0 | -6353.1 | 3.83 |
| 0.0 | 2.0 | -6159.0 | -6547.2 | 7.65 |
| 1.0 | 0.0 | -6159.0 | -6161.0 | 0.04 |
| 1.0 | 0.5 | -6159.0 | -6258.0 | 1.95 |
| 1.0 | 1.0 | -6159.0 | -6355.1 | 3.86 |
| 1.0 | 2.0 | -6159.0 | -6549.2 | 7.69 |
| 2.0 | 0.0 | -6159.0 | -6162.9 | 0.08 |
| 2.0 | 0.5 | -6159.0 | -6259.9 | 1.99 |
| 2.0 | 1.0 | -6159.0 | -6357.0 | 3.90 |
| 2.0 | 2.0 | -6159.0 | -6551.1 | 7.73 |
| 5.0 | 0.0 | -6159.0 | -6168.7 | 0.19 |
| 5.0 | 0.5 | -6159.0 | -6265.8 | 2.10 |
| 5.0 | 1.0 | -6159.0 | -6362.8 | 4.02 |
| 5.0 | 2.0 | -6159.0 | -6556.9 | 7.84 |
| 10.0 | 0.0 | -6159.0 | -6178.4 | 0.38 |
| 10.0 | 0.5 | -6159.0 | -6275.5 | 2.30 |
| 10.0 | 1.0 | -6159.0 | -6372.5 | 4.21 |
| 10.0 | 2.0 | -6159.0 | -6566.6 | 8.03 |

Degradation percentage rises monotonically with assumed cost. These are cost-adjusted proxies, not
realised PnL.

## Latency Sensitivity

Mean cost-adjusted proxy degradation versus latency 0 and mean directional hit rate by horizon and
row-step latency lag (averaged over objectives):

| horizon | latency step | net degradation vs lag 0 | hit rate |
| --- | --- | --- | --- |
| 10 | 0 | 0.00 | 0.2061 |
| 10 | 1 | -76.44 | 0.2153 |
| 10 | 2 | -276.64 | 0.2011 |
| 10 | 5 | -372.87 | 0.1969 |
| 10 | 10 | -423.58 | 0.2023 |
| 20 | 0 | 0.00 | 0.3098 |
| 20 | 1 | -226.13 | 0.3011 |
| 20 | 2 | -601.84 | 0.2880 |
| 20 | 5 | -783.36 | 0.2834 |
| 20 | 10 | -967.04 | 0.2768 |
| 50 | 0 | 0.00 | 0.4125 |
| 50 | 1 | -216.78 | 0.4087 |
| 50 | 2 | -486.80 | 0.4039 |
| 50 | 5 | -1684.51 | 0.3830 |
| 50 | 10 | -2229.09 | 0.3738 |

Latency is a row-step diagnostic shifted only within the same fold and partition. It is a latency
sensitivity proxy, not a live latency measurement.

## Fill-Assumption Sensitivity

Mean fill fraction, directional hit rate on filled trades and cost-adjusted proxy by proxy fill mode
(averaged over objectives and horizons):

| fill mode | fill fraction | hit rate | cost-adjusted proxy |
| --- | --- | --- | --- |
| abstain_only | 0.0000 | n/a | 0.0 |
| aggressive_crossing | 1.0000 | 0.3094 | -6770.4 |
| passive_conservative | 0.0156 | 0.3429 | -57.5 |
| passive_optimistic | 1.0000 | 0.3094 | -6566.6 |

Fill modes are proxy assumptions, not exchange-confirmed executions.

## Adverse-Selection Proxy

Filled-weighted adverse-selection proxy rate by confidence bucket (averaged over objectives,
horizons and fill assumptions):

| confidence bucket | adverse-selection proxy rate |
| --- | --- |
| 0.33-0.50 | 0.2566 |
| 0.50-0.70 | 0.2235 |
| 0.70-0.85 | 0.1714 |
| 0.85-1.00 | 0.2139 |

The adverse-selection proxy uses a label/future-move proxy, not measured adverse selection against
real fills.

## Regime Diagnostics

Status: `skipped`.

Regime execution diagnostics are skipped. The retained execution-v3 tables and the underlying
FI-2010 prediction artefacts do not carry regime labels or snapshot market-context columns, so
supported snapshot-derived proxy regimes cannot be built without regenerating the neural grid with
additional context columns.

Fields required before supported snapshot-derived proxy regimes could be added (recorded as future
work, not invented here):

- spread or relative_spread (snapshot-derived spread proxy)
- imbalance or order_imbalance (snapshot-derived imbalance proxy)
- top-of-book depth columns (snapshot-derived liquidity proxy)
- a future signed-return or move column for a volatility proxy

## Claim Assessment

| claim | status | scope |
| --- | --- | --- |
| execution_proxy_diagnostics_implemented | supported | retained execution-v3 confidence-filtering and cost tables |
| execution_proxy_cost_sensitivity | supported | cost_sensitivity_summary across fee and spread-multiplier settings |
| execution_proxy_latency_sensitivity | supported | latency_sensitivity_summary across row-step latency lags |
| execution_proxy_fill_sensitivity | supported | fill_assumption_summary across aggressive/passive/abstain proxy modes |
| execution_proxy_adverse_selection | supported | adverse_selection_proxy_summary by confidence bucket and fill assumption |
| execution_proxy_regime_diagnostics | skipped | regime diagnostics require regime labels or snapshot context columns |
| execution_proxy_profitability_or_live_trading | forbidden | blocked by release policy; these diagnostics are offline proxies only |

## Figures

| figure | title | path |
| --- | --- | --- |
| active_fraction_vs_confidence | Active fraction vs confidence threshold | reports/execution_v3_analysis/active_fraction_vs_confidence.png |
| cost_adjusted_proxy_vs_confidence | Cost-adjusted proxy vs confidence threshold | reports/execution_v3_analysis/cost_adjusted_proxy_vs_confidence.png |
| turnover_proxy_vs_confidence | Turnover proxy vs confidence threshold | reports/execution_v3_analysis/turnover_proxy_vs_confidence.png |
| latency_degradation_by_horizon | Cost-adjusted proxy degradation by latency and horizon | reports/execution_v3_analysis/latency_degradation_by_horizon.png |
| adverse_selection_by_confidence_bucket | Adverse-selection proxy by confidence bucket | reports/execution_v3_analysis/adverse_selection_by_confidence_bucket.png |
| fill_assumption_sensitivity | Cost-adjusted proxy by fill assumption | reports/execution_v3_analysis/fill_assumption_sensitivity.png |

## What This Does Not Claim

This analysis does not claim profitability, realised PnL, tradable alpha, live trading, production
execution quality, true event-level order-flow imbalance or queue-position modelling. It is a
descriptive offline proxy diagnostic over stored FI-2010 metrics under leakage-safe evaluation.

# Execution Centrepiece

This report is an execution-aware proxy diagnostic built from retained execution-v3 analysis tables and retained full-grid aggregate summaries.
Deleted raw prediction arrays are not required and are not read.

It is an offline diagnostic, not PnL, not live-trading evidence and not a production execution simulator.

## Inputs

| input | path or status |
| --- | --- |
| execution_v3_analysis | reports/execution_v3_analysis |
| execution_v3 | experiments/fi2010_execution_v3 |
| neural_full_grid | experiments/fi2010_neural_full_grid |
| raw_predictions_required | false |
| payoff_mode | unit_payoff |
| cost_mode | unit_proxy |
| run_group_count | 135 |
| execution_v3_status | not smoke-test |
| execution_manifest_loaded | true |

## What Predictive Metrics Show

| objective | horizon | macro-F1 | ECE | accuracy |
| --- | --- | --- | --- | --- |
| masked_reconstruction | 10 | 0.3233 | 0.1425 | 0.4061 |
| next_field | 10 | 0.2733 | 0.0847 | 0.5570 |
| supervised | 10 | 0.3336 | 0.1165 | 0.4477 |
| masked_reconstruction | 20 | 0.3547 | 0.1281 | 0.3931 |
| next_field | 20 | 0.2805 | 0.0956 | 0.4626 |
| supervised | 20 | 0.3711 | 0.0999 | 0.4091 |
| masked_reconstruction | 50 | 0.4148 | 0.0854 | 0.4368 |
| next_field | 50 | 0.3823 | 0.0846 | 0.4116 |
| supervised | 50 | 0.4180 | 0.0733 | 0.4416 |

Calibration is represented by retained ECE in the full-grid aggregate summary. Threshold-level ECE is unavailable because retained confidence-filtering tables do not include thresholded calibration bins.

## What Confidence Filtering Changes

| threshold | active fraction | turnover proxy | cost-adjusted proxy |
| --- | --- | --- | --- |
| 0.33 | 0.5440 | 0.5440 | -6159.0 |
| 0.35 | 0.5381 | 0.5381 | -6065.4 |
| 0.40 | 0.4496 | 0.4496 | -4848.5 |
| 0.45 | 0.3304 | 0.3304 | -3382.6 |
| 0.50 | 0.2163 | 0.2163 | -2011.7 |
| 0.55 | 0.1297 | 0.1297 | -1103.5 |
| 0.60 | 0.0705 | 0.0705 | -555.2 |
| 0.65 | 0.0366 | 0.0366 | -289.1 |
| 0.70 | 0.0172 | 0.0172 | -146.9 |
| 0.75 | 0.0082 | 0.0082 | -74.1 |
| 0.80 | 0.0046 | 0.0046 | -45.5 |
| 0.85 | 0.0021 | 0.0021 | -18.0 |
| 0.90 | 0.0009 | 0.0009 | -4.3 |
| 0.95 | 0.0013 | 0.0013 | 11.8 |

## Metric-To-Proxy Gap

Selected objective/horizon rows show the gap between retained forecast metrics and execution-aware signal-quality proxy diagnostics.

| pretraining_objective | horizon | predictive_macro_f1 | predictive_ece | predictive_accuracy | confidence_filtered_ece | active_fraction_at_0_50 | turnover_proxy_at_0_50 | threshold_macro_f1_at_0_50 | cost_adjusted_proxy_at_0_50 | active_fraction_at_0_70 | turnover_proxy_at_0_70 | threshold_macro_f1_at_0_70 | cost_adjusted_proxy_at_0_70 | active_fraction_at_0_85 | turnover_proxy_at_0_85 | threshold_macro_f1_at_0_85 | cost_adjusted_proxy_at_0_85 | representative_fee_bps | representative_spread_multiplier | representative_cost_adjusted_proxy | representative_cost_degradation_pct | representative_latency_step | latency_degradation_vs_lag0 | latency_directional_hit_rate | high_confidence_bucket | high_confidence_fill_assumption | high_confidence_adverse_selection_proxy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masked_reconstruction | 10.0000 | 0.3233 | 0.1425 | 0.4061 | unavailable: retained confidence-threshold tables do not include ECE | 0.1600 | 0.1600 | 0.3096 | -3328.4 | 0.0024 | 0.0024 | 0.3051 | -62.6923 | 0.0000 | 0.0000 | 0.3247 | 0.0000 | 2.0000 | 1.0000 | -10406.6 | 1.7790 | 10.0000 | -485.3 | 0.1990 | unavailable | unavailable | unavailable |
| masked_reconstruction | 20.0000 | 0.3547 | 0.1281 | 0.3931 | unavailable: retained confidence-threshold tables do not include ECE | 0.2497 | 0.2497 | 0.3513 | -3748.6 | 0.0087 | 0.0087 | 0.3419 | -165.5 | 0.0000 | 0.0000 | 0.3238 | 0.0000 | 2.0000 | 1.0000 | -9252.0 | 2.5360 | 10.0000 | -1055.5 | 0.2727 | unavailable | unavailable | unavailable |
| masked_reconstruction | 50.0000 | 0.4148 | 0.0854 | 0.4368 | unavailable: retained confidence-threshold tables do not include ECE | 0.3969 | 0.3969 | 0.4208 | -1843.1 | 0.0330 | 0.0330 | 0.4286 | -101.3 | 0.0017 | 0.0017 | 0.3256 | 12.9231 | 2.0000 | 1.0000 | -5220.6 | 6.3944 | 10.0000 | -2236.5 | 0.3764 | 0.85-1.00 | aggressive_crossing | 0.2019 |
| next_field | 10.0000 | 0.2733 | 0.0847 | 0.5570 | unavailable: retained confidence-threshold tables do not include ECE | 0.0535 | 0.0535 | 0.2685 | -1167.7 | 0.0091 | 0.0091 | 0.2973 | -187.5 | 0.0007 | 0.0007 | 0.2778 | -8.8889 | 2.0000 | 1.0000 | -2507.4 | 1.6610 | 10.0000 | -37.2000 | 0.2004 | 0.85-1.00 | aggressive_crossing | 0.1237 |
| next_field | 20.0000 | 0.2805 | 0.0956 | 0.4626 | unavailable: retained confidence-threshold tables do not include ECE | 0.0713 | 0.0713 | 0.2817 | -1166.3 | 0.0231 | 0.0231 | 0.3233 | -405.6 | 0.0101 | 0.0101 | 0.3364 | -175.1 | 2.0000 | 1.0000 | -2944.8 | 3.6634 | 10.0000 | -337.5 | 0.2826 | 0.85-1.00 | aggressive_crossing | 0.1372 |
| next_field | 50.0000 | 0.3823 | 0.0846 | 0.4116 | unavailable: retained confidence-threshold tables do not include ECE | 0.2671 | 0.2671 | 0.3863 | -758.9 | 0.0543 | 0.0543 | 0.3867 | -219.8 | 0.0053 | 0.0053 | 0.2950 | 3.9000 | 2.0000 | 1.0000 | -4933.4 | 6.9192 | 10.0000 | -1974.1 | 0.3669 | 0.85-1.00 | aggressive_crossing | 0.2770 |
| supervised | 10.0000 | 0.3336 | 0.1165 | 0.4477 | unavailable: retained confidence-threshold tables do not include ECE | 0.1245 | 0.1245 | 0.3272 | -2156.5 | 0.0022 | 0.0022 | 0.3212 | -47.2500 | 0.0000 | 0.0000 | 0.3237 | 0.0000 | 2.0000 | 1.0000 | -8400.1 | 1.9585 | 10.0000 | -748.2 | 0.2076 | unavailable | unavailable | unavailable |
| supervised | 20.0000 | 0.3711 | 0.0999 | 0.4091 | unavailable: retained confidence-threshold tables do not include ECE | 0.2030 | 0.2030 | 0.3821 | -2607.7 | 0.0069 | 0.0069 | 0.3758 | -105.5 | 0.0000 | 0.0000 | 0.3236 | 0.0000 | 2.0000 | 1.0000 | -8404.1 | 2.7230 | 10.0000 | -1508.2 | 0.2751 | unavailable | unavailable | unavailable |
| supervised | 50.0000 | 0.4180 | 0.0733 | 0.4416 | unavailable: retained confidence-threshold tables do not include ECE | 0.4204 | 0.4204 | 0.4281 | -1327.8 | 0.0152 | 0.0152 | 0.4187 | -26.8667 | 0.0012 | 0.0012 | 0.3073 | 5.5000 | 2.0000 | 1.0000 | -5144.0 | 7.4855 | 10.0000 | -2476.7 | 0.3781 | 0.85-1.00 | aggressive_crossing | 0.3297 |

## Cost And Latency

Cost-adjusted proxy and latency sensitivity are retained proxy diagnostics. They are not realised execution outcomes.

| pretraining_objective | horizon | reference_cost_adjusted_proxy | representative_fee_bps | representative_spread_multiplier | representative_cost_adjusted_proxy | representative_cost_degradation_pct | max_cost_fee_bps | max_cost_spread_multiplier | max_cost_adjusted_proxy | max_cost_degradation_pct | representative_latency_step | latency_degradation_vs_lag0 | latency_directional_hit_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masked_reconstruction | 10.0000 | -10224.2 | 2.0000 | 1.0000 | -10406.6 | 1.7790 | 10.0000 | 2.0000 | -10599.8 | 3.6626 | 10.0000 | -485.3 | 0.1990 |
| masked_reconstruction | 20.0000 | -9021.9 | 2.0000 | 1.0000 | -9252.0 | 2.5360 | 10.0000 | 2.0000 | -9495.5 | 5.2211 | 10.0000 | -1055.5 | 0.2727 |
| masked_reconstruction | 50.0000 | -4922.3 | 2.0000 | 1.0000 | -5220.6 | 6.3944 | 10.0000 | 2.0000 | -5536.4 | 13.1649 | 10.0000 | -2236.5 | 0.3764 |
| next_field | 10.0000 | -2465.5 | 2.0000 | 1.0000 | -2507.4 | 1.6610 | 10.0000 | 2.0000 | -2551.7 | 3.4197 | 10.0000 | -37.2000 | 0.2004 |
| next_field | 20.0000 | -2870.7 | 2.0000 | 1.0000 | -2944.8 | 3.6634 | 10.0000 | 2.0000 | -3023.2 | 7.5423 | 10.0000 | -337.5 | 0.2826 |
| next_field | 50.0000 | -4668.3 | 2.0000 | 1.0000 | -4933.4 | 6.9192 | 10.0000 | 2.0000 | -5214.0 | 14.2454 | 10.0000 | -1974.1 | 0.3669 |
| supervised | 10.0000 | -8243.2 | 2.0000 | 1.0000 | -8400.1 | 1.9585 | 10.0000 | 2.0000 | -8566.3 | 4.0322 | 10.0000 | -748.2 | 0.2076 |
| supervised | 20.0000 | -8180.4 | 2.0000 | 1.0000 | -8404.1 | 2.7230 | 10.0000 | 2.0000 | -8641.0 | 5.6061 | 10.0000 | -1508.2 | 0.2751 |
| supervised | 50.0000 | -4834.7 | 2.0000 | 1.0000 | -5144.0 | 7.4855 | 10.0000 | 2.0000 | -5471.6 | 15.4113 | 10.0000 | -2476.7 | 0.3781 |

## Adverse-Selection Proxy

The adverse-selection proxy is reported by confidence bucket and fill assumption. It is a label or future-move proxy, not measured adverse selection from exchange-confirmed fills.

| pretraining_objective | horizon | confidence_bucket | fill_assumption | total_filled | total_adverse | mean_adverse_fraction | weighted_adverse_fraction | adverse_selection_mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masked_reconstruction | 10.0000 | 0.33-0.50 | aggressive_crossing | 183417.0 | 34267.0 | 0.1868 | 0.1868 | label_proxy |
| masked_reconstruction | 10.0000 | 0.33-0.50 | passive_optimistic | 183417.0 | 34267.0 | 0.1868 | 0.1868 | label_proxy |
| masked_reconstruction | 10.0000 | 0.50-0.70 | aggressive_crossing | 83719.0 | 13641.0 | 0.1629 | 0.1629 | label_proxy |
| masked_reconstruction | 10.0000 | 0.50-0.70 | passive_optimistic | 83719.0 | 13641.0 | 0.1629 | 0.1629 | label_proxy |
| masked_reconstruction | 10.0000 | 0.70-0.85 | aggressive_crossing | 1165.0 | 94.0000 | 0.0807 | 0.0807 | label_proxy |
| masked_reconstruction | 10.0000 | 0.70-0.85 | passive_conservative | 175.0 | 15.0000 | 0.0857 | 0.0857 | label_proxy |
| masked_reconstruction | 10.0000 | 0.70-0.85 | passive_optimistic | 1165.0 | 94.0000 | 0.0807 | 0.0807 | label_proxy |
| masked_reconstruction | 20.0000 | 0.33-0.50 | aggressive_crossing | 204712.0 | 52779.0 | 0.2578 | 0.2578 | label_proxy |
| masked_reconstruction | 20.0000 | 0.33-0.50 | passive_optimistic | 204712.0 | 52779.0 | 0.2578 | 0.2578 | label_proxy |
| masked_reconstruction | 20.0000 | 0.50-0.70 | aggressive_crossing | 129329.0 | 28562.0 | 0.2208 | 0.2208 | label_proxy |
| masked_reconstruction | 20.0000 | 0.50-0.70 | passive_optimistic | 129329.0 | 28562.0 | 0.2208 | 0.2208 | label_proxy |
| masked_reconstruction | 20.0000 | 0.70-0.85 | aggressive_crossing | 4238.0 | 682.0 | 0.1609 | 0.1609 | label_proxy |

## Fill Assumptions

| pretraining_objective | horizon | fill_mode | mean_fill_fraction | mean_directional_hit_rate | mean_cost_adjusted_proxy |
| --- | --- | --- | --- | --- | --- |
| masked_reconstruction | 10.0000 | abstain_only | 0.0000 | unavailable | 0.0000 |
| masked_reconstruction | 10.0000 | aggressive_crossing | 1.0000 | 0.2124 | -10787.6 |
| masked_reconstruction | 10.0000 | passive_conservative | 0.0007 | 0.3014 | -8.3225 |
| masked_reconstruction | 10.0000 | passive_optimistic | 1.0000 | 0.2124 | -10599.8 |
| masked_reconstruction | 20.0000 | abstain_only | 0.0000 | unavailable | 0.0000 |
| masked_reconstruction | 20.0000 | aggressive_crossing | 1.0000 | 0.2947 | -9732.3 |
| masked_reconstruction | 20.0000 | passive_conservative | 0.0028 | 0.4225 | -43.5982 |
| masked_reconstruction | 20.0000 | passive_optimistic | 1.0000 | 0.2947 | -9495.5 |
| masked_reconstruction | 50.0000 | abstain_only | 0.0000 | unavailable | 0.0000 |
| masked_reconstruction | 50.0000 | aggressive_crossing | 1.0000 | 0.4152 | -5843.5 |

## Figures

| figure | status | path |
| --- | --- | --- |
| forecasting_vs_signal_quality | completed | reports/execution_centrepiece/forecasting_vs_signal_quality.png |

## Unavailable Fields

| field | reason |
| --- | --- |
| confidence_filtered_ece | unavailable: retained confidence-threshold tables do not include ECE |
| raw_predictions | not required and not read; deleted raw predictions are unavailable |
| realised_execution | unavailable: offline diagnostic has no broker or venue fills |
| supported_regime_diagnostics | unavailable: retained tables lack regime labels or snapshot context |

Regime diagnostics remain skipped:

Regime execution diagnostics are skipped. The retained execution-v3 tables and the underlying
FI-2010 prediction artefacts do not carry regime labels or snapshot market-context columns, so
supported snapshot-derived proxy regimes cannot be built without regenerating the neural grid with
additional context columns.

## Claim Assessment

| claim | status |
| --- | --- |
| PnL | forbidden |
| active_fraction_analysis | supported |
| adverse_selection_confidence_analysis | supported |
| confidence_filtering_tradeoff_analysis | supported |
| forecasting_vs_signal_quality_gap_analysis | supported |
| latency_cost_gap_analysis | supported |
| live_trading | forbidden |
| profitability_or_tradability | forbidden |
| turnover_proxy_analysis | supported |

The upstream execution-v3 claim assessment is retained and remains consistent with this centrepiece.

## Conclusion

The execution centrepiece does not establish profitability or tradability.
It shows why forecast metrics must be interpreted alongside calibration, confidence filtering, active fraction, turnover, latency, cost and adverse-selection proxy diagnostics.

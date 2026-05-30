# FI-2010 Feature Ablation And Stability Analysis

This report strengthens the retained feature-ablation evidence into a scoped feature-stability analysis.

snapshot_order_flow_proxy is a labelled snapshot proxy derived from FI-2010 matrices. It should not be interpreted as true event-level order-flow imbalance.

These diagnostics are not causal feature importance and should not be read as universal feature importance across all models or horizons.

## Completed Scope

| field | value |
| --- | --- |
| evidence status | partial_real |
| completed fits | 2580 |
| failed fits | 0 |
| folds | fold_1, fold_2, fold_3, fold_4, fold_5 |
| horizons | 10, 20, 50 |
| seeds | 0, 1, 2 |
| models | gradient_boosting, logistic, ridge |
| raw predictions retained | no |

## Snapshot Proxy Finding

- Horizon-10 logistic/ridge status: supported (30/30 matched rows degraded when removed).
- Horizon-20/50 status: supported (65/65 matched rows degraded when removed).
- Non-linear slice status: supported.

Execution-aware ablation diagnostics require retained prediction-level outputs or a targeted rerun.

## Strongest Mean Remove-One-Group Effects

| feature group | mean delta macro-F1 | mean delta MCC | degradation fraction | stability score |
| --- | --- | --- | --- | --- |
| snapshot_order_flow_proxy | -0.1510 | -0.1966 | 1.0000 | 1.0000 |
| size_levels | -0.0087 | -0.0115 | 0.8667 | 1.0000 |
| spread | -0.0075 | -0.0045 | 0.8400 | 1.0000 |
| price_levels | -0.0060 | -0.0067 | 0.7000 | 1.0000 |
| liquidity_concentration | 0.0004 | 0.0004 | 0.3667 | 0.1250 |
| top_of_book_imbalance | 0.0002 | 0.0003 | 0.1800 | 0.1333 |
| volatility_proxy | 0.0001 | 0.0007 | 0.4667 | 0.1833 |
| depth_imbalance | -0.0000 | 0.0001 | 0.4300 | 0.5167 |
| depth_slope | 0.0000 | 0.0001 | 0.2333 | 0.2667 |
| microprice | 0.0000 | 0.0000 | 0.5667 | 0.3083 |

## Claim Assessment

| claim | status | reason |
| --- | --- | --- |
| feature_ablation_infrastructure | supported | required feature-ablation summary and delta tables were loaded |
| horizon10_logistic_ridge_snapshot_proxy_importance | supported | removing snapshot_order_flow_proxy degraded macro-F1 in every matched row |
| broader_horizon_snapshot_proxy_importance | supported | removing snapshot_order_flow_proxy degraded macro-F1 in every matched row |
| nonlinear_model_feature_stability | supported | removing snapshot_order_flow_proxy degraded macro-F1 in every matched row |
| execution_aware_ablation_diagnostics | needs_prediction_outputs | Execution-aware ablation diagnostics require retained prediction-level outputs or a targeted rerun. |
| causal_feature_importance | forbidden | ablation deltas are associational diagnostics, not causal evidence |
| true_event_level_ofi | forbidden | snapshot_order_flow_proxy is a labelled snapshot proxy derived from FI-2010 matrices. It should not be interpreted as true event-level order-flow imbalance. |

## Snapshot Proxy Scope Rows

`snapshot_order_flow_proxy_scope.csv` contains 100 matched remove-one-group rows across horizons 10, 20, 50.

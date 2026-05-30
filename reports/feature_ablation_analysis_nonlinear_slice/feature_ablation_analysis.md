# FI-2010 Feature Ablation And Stability Analysis

This report strengthens the retained feature-ablation evidence into a scoped feature-stability analysis.

snapshot_order_flow_proxy is a labelled snapshot proxy derived from FI-2010 matrices. It should not be interpreted as true event-level order-flow imbalance.

These diagnostics are not causal feature importance and should not be read as universal feature importance across all models or horizons.

## Completed Scope

| field | value |
| --- | --- |
| evidence status | partial_real |
| completed fits | 60 |
| failed fits | 0 |
| folds | fold_1, fold_2, fold_3, fold_4, fold_5 |
| horizons | 10, 50 |
| seeds | 0 |
| models | gradient_boosting |
| raw predictions retained | no |

## Snapshot Proxy Finding

- Horizon-10 logistic/ridge status: needs_real_evidence (0/0 matched rows degraded when removed).
- Horizon-20/50 status: supported (5/5 matched rows degraded when removed).
- Non-linear slice status: supported.

Execution-aware ablation diagnostics require retained prediction-level outputs or a targeted rerun.

## Strongest Mean Remove-One-Group Effects

| feature group | mean delta macro-F1 | mean delta MCC | degradation fraction | stability score |
| --- | --- | --- | --- | --- |
| snapshot_order_flow_proxy | -0.3369 | -0.4348 | 1.0000 | 1.0000 |
| spread | -0.0023 | -0.0024 | 0.6000 | 0.8750 |
| depth_imbalance | -0.0004 | -0.0003 | 0.4000 | 0.9500 |
| top_of_book_imbalance | -0.0000 | -0.0001 | 0.3000 | 0.8500 |

## Claim Assessment

| claim | status | reason |
| --- | --- | --- |
| feature_ablation_infrastructure | supported | required feature-ablation summary and delta tables were loaded |
| horizon10_logistic_ridge_snapshot_proxy_importance | needs_real_evidence | no matching snapshot_order_flow_proxy rows were available |
| broader_horizon_snapshot_proxy_importance | supported | removing snapshot_order_flow_proxy degraded macro-F1 in every matched row |
| nonlinear_model_feature_stability | supported | removing snapshot_order_flow_proxy degraded macro-F1 in every matched row |
| execution_aware_ablation_diagnostics | needs_prediction_outputs | Execution-aware ablation diagnostics require retained prediction-level outputs or a targeted rerun. |
| causal_feature_importance | forbidden | ablation deltas are associational diagnostics, not causal evidence |
| true_event_level_ofi | forbidden | snapshot_order_flow_proxy is a labelled snapshot proxy derived from FI-2010 matrices. It should not be interpreted as true event-level order-flow imbalance. |

## Snapshot Proxy Scope Rows

`snapshot_order_flow_proxy_scope.csv` contains 10 matched remove-one-group rows across horizons 10, 50.

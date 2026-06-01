# SSL-v2 Analysis

Analysis version: `fi2010-ssl-v2-analysis/v1`.

## Scope

- evidence level: `complete_real`
- scope label: `folds_1_2_3_4_5_h10_h50_complete_real`
- folds: [1, 2, 3, 4, 5]
- horizons: [10, 50]
- seeds: [0]
- lookbacks: [50]
- objectives: ['supervised', 'market_state_multitask']

SSL-v2 was added because the first-generation SSL analysis found that random field reconstruction and next-field prediction did not broadly improve downstream predictive or calibration metrics.

## Predictive Metrics

| horizon | fold | delta_macro_f1 | delta_mcc | delta_ece | delta_brier_score |
| --- | --- | --- | --- | --- | --- |
| 10 | 1 | 0.079425 | 0.081668 | 0.009567 | 0.027942 |
| 50 | 1 | -0.081638 | -0.075793 | 0.092834 | 0.055660 |
| 10 | 2 | 0.000000 | 0.000000 | 0.043880 | 0.008728 |
| 50 | 2 | -0.107155 | -0.136067 | 0.036254 | 0.107275 |
| 10 | 3 | -0.058421 | 0.021356 | 0.003233 | -0.037280 |
| 50 | 3 | 0.031548 | 0.040894 | -0.017545 | -0.039277 |
| 10 | 4 | 0.001026 | 0.008603 | -0.010822 | -0.012669 |
| 50 | 4 | -0.016546 | -0.017729 | 0.020665 | 0.022998 |
| 10 | 5 | 0.407412 | 0.617875 | -0.116031 | -0.404400 |
| 50 | 5 | 0.021318 | 0.025349 | -0.009436 | -0.030377 |

## Horizon and Fold Deltas

The CSV artefacts `ssl_v2_delta_by_horizon.csv` and `ssl_v2_delta_by_fold.csv` are the canonical grouped summaries.

## Claim Assessment

| claim | status | scope |
| --- | --- | --- |
| ssl_v2_objective_implemented | supported | code and stored benchmark artefacts |
| ssl_v2_evaluated | supported | evidence level complete_real |
| ssl_v2_predictive_improvement | supported | exact stored scope; evidence level complete_real |
| ssl_v2_calibration_improvement | unsupported | exact stored scope; evidence level complete_real |
| broad_ssl_improvement | unsupported | blocked by existing SSL-v1 failure analysis and scoped SSL-v2 evidence |
| foundation_model | forbidden | not claimed |
| sota | forbidden | not claimed |

## Conservative Interpretation

This analysis reports predictive and calibration deltas only. It does not claim profitability, live trading, broad SSL improvement, market-wide generalisation, a foundation model, or state-of-the-art performance.

Grouped CSV deltas are available for exact numeric inspection.

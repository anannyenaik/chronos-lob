# SSL-v2 Analysis

Analysis version: `fi2010-ssl-v2-analysis/v1`.

## Scope

- evidence level: `partial_real`
- scope label: `limited_ssl_v2_partial_real_slice`
- folds: [1]
- horizons: [10, 50]
- seeds: [0]
- lookbacks: [50]
- objectives: ['supervised', 'masked_reconstruction', 'market_state_multitask']

SSL-v2 was added because the first-generation SSL analysis found that random field reconstruction and next-field prediction did not broadly improve downstream predictive or calibration metrics.

## Predictive Metrics

| horizon | fold | delta_macro_f1 | delta_mcc | delta_ece | delta_brier_score |
| --- | --- | --- | --- | --- | --- |
| 10 | 1 | 0.079425 | 0.081668 | 0.009567 | 0.027942 |
| 50 | 1 | -0.081638 | -0.075793 | 0.092834 | 0.055660 |

## Horizon and Fold Deltas

The CSV artefacts `ssl_v2_delta_by_horizon.csv` and `ssl_v2_delta_by_fold.csv` are the canonical grouped summaries.

## Claim Assessment

| claim | status | scope |
| --- | --- | --- |
| ssl_v2_objective_implemented | supported | code and stored benchmark artefacts |
| ssl_v2_evaluated | supported | evidence level partial_real |
| ssl_v2_predictive_improvement | unsupported | exact stored scope; evidence level partial_real |
| ssl_v2_calibration_improvement | unsupported | exact stored scope; evidence level partial_real |
| broad_ssl_improvement | unsupported | blocked by existing SSL-v1 failure analysis and scoped SSL-v2 evidence |
| foundation_model | forbidden | not claimed |
| sota | forbidden | not claimed |

## Conservative Interpretation

This analysis reports predictive and calibration deltas only. It does not claim profitability, live trading, broad SSL improvement, market-wide generalisation, a foundation model, or state-of-the-art performance.

Grouped CSV deltas are available for exact numeric inspection.

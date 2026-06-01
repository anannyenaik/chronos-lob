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
The current closure is seed 0 only; the multi-seed harness exists, but seeds 1 and 2 are deferred.

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

## Aggregate, Seed, Horizon and Fold Deltas

Canonical grouped summaries: `ssl_v2_delta_overall.csv`, `ssl_v2_delta_by_seed.csv`, `ssl_v2_delta_by_horizon.csv`, `ssl_v2_delta_by_fold.csv` and `ssl_v2_delta_by_fold_horizon.csv`.

| matched rows | mean delta macro-F1 | mean delta MCC | mean delta ECE | mean delta Brier |
| --- | --- | --- | --- | --- |
| 10 | 0.027697 | 0.056615 | 0.005260 | -0.030140 |

## Confidence-Filtered Diagnostics

Selective-prediction deltas versus the matched supervised baseline at confidence thresholds 0.33, 0.50, 0.70,
0.85, 0.95. Active fraction, abstention and active-example counts are recorded in
ssl_v2_confidence_filtered.csv.

| threshold | SSL-v2 active fraction | SSL-v2 macro-F1 | delta macro-F1 | delta MCC |
| --- | --- | --- | --- | --- |
| 0.330000 | 1.000000 | 0.541445 | 0.027697 | 0.056615 |
| 0.500000 | 0.840373 | 0.534295 | 0.016880 | 0.047870 |
| 0.700000 | 0.465708 | 0.727487 | 0.079238 | 0.107886 |
| 0.850000 | 0.309376 | 0.871091 | 0.012722 | 0.011142 |
| 0.950000 | 0.189772 | 0.836651 | -0.076442 | -0.112217 |

## Execution-Aware Proxy Diagnostics

Active fraction is reported above through the confidence-filtered diagnostics. The remaining execution-aware
proxies are deferred.

- deferred proxies: turnover_proxy, cost_adjusted_proxy, latency_sensitivity, adverse_selection_proxy
- computable from retained artefacts: False
- prediction-level artefacts required: True
- execution hook: `build-execution-centrepiece` (execution_centrepiece)

Reason:

The retained SSL-v2 benchmark artefacts are summary-light. The turnover, cost-adjusted, latency-sensitivity
and adverse-selection proxies need per-run test predictions, which are stored as git-ignored heavy artefacts
and are not part of the retained set.

Storage-light design note for future runs:

Future SSL-v2 runs can emit per-(run, threshold) execution summary rows at evaluation time - active fraction,
signal-change turnover proxy, cost-adjusted proxy, a latency-step degradation sweep and an adverse-selection
proxy - and persist only the aggregated per-threshold table, mirroring the execution-v3 *_summary.csv tables,
without ever storing raw per-row predictions. analyse-fi2010-ssl-v2-results would then aggregate those tables
and the execution centrepiece could ingest them through its existing summary-light interface.

Claim boundary: Execution proxies are offline signal-quality diagnostics; no PnL, live-trading, tradability or
production execution claim is implied.

## Claim Assessment

| claim | status | scope |
| --- | --- | --- |
| ssl_v2_objective_implemented | supported | code and stored benchmark artefacts |
| ssl_v2_evaluated | supported | folds [1, 2, 3, 4, 5], horizons [10, 50], seeds [0], lookbacks [50]; evidence level complete_real |
| ssl_v2_predictive_improvement | supported | folds [1, 2, 3, 4, 5], horizons [10, 50], seeds [0], lookbacks [50]; evidence level complete_real |
| ssl_v2_calibration_improvement | unsupported | folds [1, 2, 3, 4, 5], horizons [10, 50], seeds [0], lookbacks [50]; evidence level complete_real |
| broad_ssl_improvement | unsupported | blocked by existing SSL-v1 failure analysis and scoped SSL-v2 evidence |
| foundation_model | forbidden | not claimed |
| sota | forbidden | not claimed |

## Conservative Interpretation

This analysis reports predictive and calibration deltas only. It does not claim profitability, live trading, broad SSL improvement, market-wide generalisation, a foundation model, or state-of-the-art performance.

Grouped CSV deltas are available for exact numeric inspection.

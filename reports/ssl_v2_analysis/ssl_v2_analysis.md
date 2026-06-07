# SSL-v2 Analysis

Analysis version: `fi2010-ssl-v2-analysis/v1`.

## Scope

- evidence level: `complete_real`
- scope label: `folds_1_2_3_4_5_h10_h50_seeds_0_1_2_complete_real`
- folds: [1, 2, 3, 4, 5]
- horizons: [10, 50]
- seeds: [0, 1, 2]
- lookbacks: [50]
- objectives: ['supervised', 'market_state_multitask']

SSL-v2 was added because the first-generation SSL analysis found that random field reconstruction and next-field prediction did not broadly improve downstream predictive or calibration metrics.
The current closure covers the exact stored folds, horizons, seeds and lookbacks listed above.

The SSL-v2 benchmark is complete for the stored FI-2010 scope: folds 1–5, horizons 10/50, seeds 0–2 and
lookback 50. Across 30 matched comparison cells, SSL-v2 has positive mean deltas for macro-F1, MCC, ECE and
Brier, supporting scoped predictive and calibration improvement for this exact retained scope. The evidence is
mixed by seed and horizon, including negative mean macro-F1 deltas for seed 1 and horizon 50, so broad SSL
improvement remains unsupported.

The seed-1 and seed-2 SSL-v2 refresh was executed as independent Slurm array jobs on Durham University
Hamilton/NCC HPC. Retained summaries, provenance and claim assessments are committed; large checkpoints, raw
predictions and cluster logs are intentionally excluded. GPU determinism warnings are documented, and bitwise
reproducibility is not claimed.

## Predictive Metrics

| horizon | fold | delta_macro_f1 | delta_mcc | delta_ece | delta_brier_score |
| --- | --- | --- | --- | --- | --- |
| 10 | 1 | 0.079425 | 0.081668 | 0.009567 | 0.027942 |
| 10 | 1 | 0.000205 | 0.013493 | 0.091742 | 0.022049 |
| 10 | 1 | -0.006047 | -0.018864 | -0.031358 | -0.018494 |
| 50 | 1 | -0.081638 | -0.075793 | 0.092834 | 0.055660 |
| 50 | 1 | -0.059603 | -0.078475 | 0.012906 | 0.021922 |
| 50 | 1 | -0.119027 | -0.174057 | -0.007558 | 0.077108 |
| 10 | 2 | 0.000000 | 0.000000 | 0.043880 | 0.008728 |
| 10 | 2 | 0.403756 | 0.557519 | -0.156813 | -0.393456 |
| 10 | 2 | -0.003179 | 0.028675 | 0.006916 | -0.147299 |
| 50 | 2 | -0.107155 | -0.136067 | 0.036254 | 0.107275 |
| 50 | 2 | -0.013745 | -0.031717 | 0.009079 | 0.035635 |
| 50 | 2 | -0.099986 | -0.157729 | -0.021754 | 0.091452 |
| 10 | 3 | -0.058421 | 0.021356 | 0.003233 | -0.037280 |
| 10 | 3 | -0.414251 | -0.531533 | -0.020667 | 0.246543 |
| 10 | 3 | 0.008503 | 0.042013 | 0.016708 | -0.019416 |
| 50 | 3 | 0.031548 | 0.040894 | -0.017545 | -0.039277 |
| 50 | 3 | -0.030451 | -0.040626 | 0.020825 | 0.043040 |
| 50 | 3 | -0.011297 | -0.017931 | 0.028161 | 0.026965 |
| 10 | 4 | 0.001026 | 0.008603 | -0.010822 | -0.012669 |
| 10 | 4 | -0.009744 | -0.003308 | 0.006328 | 0.004500 |
| 10 | 4 | 0.014242 | 0.007425 | -0.027643 | -0.004107 |
| 50 | 4 | -0.016546 | -0.017729 | 0.020665 | 0.022998 |
| 50 | 4 | 0.016185 | 0.018359 | -0.032074 | -0.028714 |
| 50 | 4 | -0.024825 | -0.032193 | 0.026489 | 0.030998 |
| 10 | 5 | 0.407412 | 0.617875 | -0.116031 | -0.404400 |
| 10 | 5 | 0.004507 | 0.013897 | 0.000217 | -0.012108 |
| 10 | 5 | 0.411067 | 0.628876 | -0.088127 | -0.392906 |
| 50 | 5 | 0.021318 | 0.025349 | -0.009436 | -0.030377 |
| 50 | 5 | 0.006570 | 0.009104 | 0.003643 | -0.005511 |
| 50 | 5 | -0.013295 | -0.021336 | 0.020372 | 0.016980 |

## Aggregate, Seed, Horizon and Fold Deltas

Canonical grouped summaries: `ssl_v2_delta_overall.csv`, `ssl_v2_delta_by_seed.csv`, `ssl_v2_delta_by_horizon.csv`, `ssl_v2_delta_by_fold.csv` and `ssl_v2_delta_by_fold_horizon.csv`.

| matched rows | mean delta macro-F1 | mean delta MCC | mean delta ECE | mean delta Brier |
| --- | --- | --- | --- | --- |
| 30 | 0.011218 | 0.025925 | -0.003000 | -0.023541 |

Aggregate support is not uniform across strata: negative mean macro-F1 for seed(s) 1; negative mean macro-F1
for horizon(s) 50.

## Confidence-Filtered Diagnostics

Selective-prediction deltas versus the matched supervised baseline at confidence thresholds 0.33, 0.50, 0.70,
0.85, 0.95. Active fraction, abstention and active-example counts are recorded in
ssl_v2_confidence_filtered.csv.

| threshold | SSL-v2 active fraction | SSL-v2 macro-F1 | delta macro-F1 | delta MCC |
| --- | --- | --- | --- | --- |
| 0.330000 | 1.000000 | 0.560161 | 0.011218 | 0.025925 |
| 0.500000 | 0.796608 | 0.569784 | 0.009843 | 0.023636 |
| 0.700000 | 0.491696 | 0.677291 | 0.001713 | 0.011071 |
| 0.850000 | 0.328947 | 0.834543 | -0.025392 | -0.035415 |
| 0.950000 | 0.196189 | 0.858355 | -0.053474 | -0.090879 |

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
| ssl_v2_evaluated | supported | folds [1, 2, 3, 4, 5], horizons [10, 50], seeds [0, 1, 2], lookbacks [50]; evidence level complete_real |
| ssl_v2_predictive_improvement | supported | folds [1, 2, 3, 4, 5], horizons [10, 50], seeds [0, 1, 2], lookbacks [50]; evidence level complete_real |
| ssl_v2_calibration_improvement | supported | folds [1, 2, 3, 4, 5], horizons [10, 50], seeds [0, 1, 2], lookbacks [50]; evidence level complete_real |
| broad_ssl_improvement | unsupported | blocked by existing SSL-v1 failure analysis and scoped SSL-v2 evidence |
| foundation_model | forbidden | not claimed |
| sota | forbidden | not claimed |

## Conservative Interpretation

This analysis reports predictive and calibration deltas only. It does not claim profitability, live trading, broad SSL improvement, market-wide generalisation, a foundation model, or state-of-the-art performance.

Grouped CSV deltas are available for exact numeric inspection.

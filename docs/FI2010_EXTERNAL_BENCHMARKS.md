# FI-2010 External Benchmark Context

## Purpose

This document explains how ChronosLOB's FI-2010 evidence can be compared with
published FI-2010 and limit order book forecasting work. It is a protocol
context layer, not a leaderboard. No unverified external numeric metrics are
copied into this repository.

## FI-2010 Dataset Context

FI-2010 was introduced by Ntakaris, Magris, Kanniainen, Gabbouj and Iosifidis
for mid-price direction forecasting on high-frequency limit order book data.
The public release contains normalised `.txt` matrices, auction and no-auction
variants, three normalisation set-ups and anchored day-based cross-validation
folds. The files include limit order book rows, hand-crafted feature rows and
labels for horizons `k = 10, 20, 30, 50, 100`.

Primary pointers:

- Dataset metadata:
  <https://www.research.ed.ac.uk/en/datasets/benchmark-dataset-for-mid-price-forecasting-of-limit-order-book-d/>
- Dataset/paper preprint: <https://arxiv.org/abs/1705.03233>

## ChronosLOB Protocol Summary

The current stored ChronosLOB evidence uses official FI-2010 NoAuction ZScore
folds `1` through `5`, horizon `label_10`, and split-aware evaluation from the
prepared CSV `split` column. Validation rows are carved only from the official
train partition. Fittable preprocessing is fit on train rows only. Full
prediction rows and model checkpoints are not written by the current
multi-fold evidence runs.

Classification metrics treat the official labels as categorical classes.
Execution proxy diagnostics additionally use the configured direction map in
`configs/experiments/fi2010_multifold.yaml`: `1 -> +1`, `2 -> -1`,
`3 -> 0`. Any external execution-like comparison must align this mapping
before comparing signs or proxy returns.

## Comparison Dimensions

| Dimension | ChronosLOB current setting | Why it matters |
| --- | --- | --- |
| Dataset variant | FI-2010 NoAuction ZScore | Papers may use auction data, min-max or decimal-precision variants. |
| Auction/no-auction setting | NoAuction only | Auction-period inclusion changes the sample distribution. |
| Normalisation | Official z-score matrices plus train-only model preprocessing where applicable | Feature scales and fitted preprocessing can change model behaviour. |
| Folds used | Official folds `1` to `5` | Some papers report more folds, fewer folds or a different aggregation. |
| Prediction horizon | `label_10` | FI-2010 also provides `label_20`, `label_30`, `label_50` and `label_100`. |
| Label mapping | Categorical labels preserved for macro-F1; execution proxies use the configured direction map | Class order and direction semantics affect sign-based diagnostics. |
| Train/test split protocol | Official train/test split column; validation carved from train only | Random or reshuffled splits are not comparable. |
| Metrics | Accuracy, macro-F1, MCC, Brier score and ECE; execution proxies separate | Papers may report accuracy, F1 variants or portfolio/execution metrics. |
| Preprocessing | Test rows excluded from fitting, validation and model selection | Direct comparison needs the same leakage boundary. |
| Model class | Classical baselines plus reduced-scope supervised neural baselines | Architecture names alone do not imply identical training protocol. |
| Calibration/execution diagnostics | Calibration and execution v2 are proxy diagnostics, labelled separately | Most external forecasting papers do not report the same diagnostics. |

## Why Direct Metric Comparison May Be Invalid

Direct metric comparison can be invalid when any of the following differ:

- dataset variant, auction setting or normalisation;
- fold set, fold aggregation or train/test split semantics;
- prediction horizon or class mapping;
- metric definition, averaging rule or class ordering;
- lookback length, seed count, training budget or early-stopping policy;
- whether reported values are predictive metrics, calibration diagnostics,
  execution proxies or portfolio objectives.

## Where Comparison Is Meaningful

Comparison is meaningful when the external source and ChronosLOB use the same
FI-2010 variant, horizon, label semantics, split protocol, folds and metric
definition. At present, the safest use is qualitative protocol comparison:
ChronosLOB can show which parts of the protocol match, which parts differ and
which local metrics are available from stored artefacts.

## Where Comparison Is Not Meaningful

Comparison is not meaningful for claims about live deployment, broker or
exchange execution, profitability, portfolio allocation, cross-asset transfer
or self-supervised results. It is also not meaningful to compare ChronosLOB's
execution v2 proxy diagnostics with paper metrics unless the external source
reports the same assumptions and proxy definitions.

## Current ChronosLOB Result Snapshot

- Classical: `gradient_boosting`, test macro-F1 `0.4654 ± 0.0039` across the
  five stored NoAuction ZScore folds.
- Reduced-scope supervised neural: `matrix_transformer`, test macro-F1
  `0.7337 ± 0.0280` across the same folds.
- Neural caveat: the stored neural result is single-seed (`0`), single-lookback
  (`20`) evidence with a reduced CPU scope. It is not a neural superiority
  claim.
- No SSL result is reported.

## External Benchmark Table

| Reference | Dataset/protocol | Model family | Reported metric type | Comparability caveat |
| --- | --- | --- | --- | --- |
| [Ntakaris et al. FI-2010 dataset and baselines](https://arxiv.org/abs/1705.03233) | FI-2010 anchored day-based folds across release variants | Classical ML baselines | Classification metrics | Direct only when variant, fold set, horizon, label mapping and metric definition match. |
| [Tsantekidis et al. stationary-feature LOB forecasting](https://arxiv.org/abs/1810.09965) | Limit order book forecasting with paper-specific feature construction | CNN/LSTM-style deep learning | Classification metrics | Feature construction and split protocol must be aligned before numeric comparison. |
| [Zhang, Zohren and Roberts DeepLOB](https://arxiv.org/abs/1808.03668) | FI-2010 benchmark plus additional market data experiments | CNN-LSTM | Predictive classification metrics | ChronosLOB's `deeplob_style` is a local baseline, not an exact reproduction; no paper metric is copied here. |
| [Wallbridge TransLOB](https://arxiv.org/abs/2003.00130) | FI-2010 transformer-style benchmark protocol | Causal CNN plus attention | Predictive classification metrics | Protocol is not reproduced locally; use as architecture context unless the full protocol is matched. |
| [Sangadiev et al. DeepFolio](https://arxiv.org/abs/2008.12152) | FI-2010 plus portfolio-allocation setting | CNN-based forecasting and allocation | Classification and portfolio metric types | Portfolio objectives and allocation metrics are not directly comparable to ChronosLOB forecasting diagnostics. |

This table intentionally records metric types and caveats only. Numeric
external paper metrics should be added only after they are verified from the
source and matched to the same protocol dimensions above.

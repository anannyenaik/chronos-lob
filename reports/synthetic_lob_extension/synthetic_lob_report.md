# ChronosLOB Synthetic Event-Level Extension

This report is generated from a deterministic synthetic limit-order-book
event simulator. It demonstrates event-level pipeline support under
controlled synthetic regimes. It is not real-market evidence, it does not
show tradability or returns, and it does not change any FI-2010 limitation:
FI-2010 snapshots still expose only snapshot proxies, never true event-level
order flow.

## Overview

| field | value |
| --- | --- |
| generated_at | 2026-05-30T15:31:24.469213+00:00 |
| symbol | SYNTH |
| seed | 0 |
| event_count | 21010 |
| snapshot_count | 4202 |
| regimes | stable_liquid, buy_pressure, high_volatility, sell_pressure, low_liquidity, wide_spread, cancellation_shock |
| target | future_mid_direction |
| leakage_check_ok | True |
| replay_ok | True |
| evidence_level | synthetic_controlled |

## Event Schema

| field | meaning |
| --- | --- |
| timestamp | synthetic strictly increasing event time (UTC) |
| sequence_id | strictly increasing integer event index |
| event_type | ADD, CANCEL or TRADE |
| side | BID or ASK (aggressor side for TRADE) |
| price | tick-aligned price level |
| quantity | non-negative size added, cancelled or executed |
| regime_id / regime_name | known ground-truth regime label |
| latent_mid | latent reference mid used by the simulator |

## Regimes And Event-Level Feature Behaviour

Each regime is a controlled synthetic environment with known intensities
and imbalance. The table reports mean event-level features per regime.

| regime_name | row_count | mean_event_intensity | mean_spread | mean_event_order_flow_imbalance | mean_cancellation_imbalance | mean_trade_imbalance |
| --- | --- | --- | --- | --- | --- | --- |
| stable_liquid | 602 | 1.0 | 0.013422 | 0.009876 | -0.135535 | 0.000443 |
| high_volatility | 600 | 1.0 | 0.013667 | -0.04814 | 0.138112 | 0.111803 |
| low_liquidity | 600 | 1.0 | 0.016617 | 0.092231 | -0.136479 | -0.026068 |
| wide_spread | 600 | 1.0 | 0.020333 | 0.011273 | -0.06439 | -0.111138 |
| buy_pressure | 600 | 1.0 | 0.016933 | -0.029325 | 0.626007 | 0.49271 |
| sell_pressure | 600 | 1.0 | 0.016667 | -0.076532 | -0.502013 | -0.346893 |
| cancellation_shock | 600 | 1.0 | 0.0218 | 0.002396 | 0.006552 | -0.069488 |

## Deterministic Replay Checks

Replay rebuilds the book from the event stream alone and validates
invariants: bid below ask, non-negative depth and continuous sequence ids.

| check | value |
| --- | --- |
| event_count | 21010 |
| snapshot_count | 4202 |
| crossed_snapshot_count | 0 |
| negative_depth_event_count | 0 |
| sequence_gap_count | 0 |
| ok | True |

## Supported Event-Level Features

These features require event messages and are computed only on synthetic
event streams here. FI-2010 snapshots cannot support them.

- event_order_flow_imbalance: signed order flow from adds and cancels
- cancellation_imbalance: bid-versus-ask cancellation pressure
- trade_imbalance: buyer- versus seller-initiated executed volume
- event_intensity and add/cancel/trade rates
- spread, relative_spread and microprice_offset
- depth_imbalance_l1 and depth_imbalance_l5
- realised_volatility_proxy from latent mid changes

## Labels

Labels summarise a window strictly after each feature timestamp, so they
do not leak into contemporaneous features.

| label | class_value | count | fraction |
| --- | --- | --- | --- |
| future_mid_direction | 0 | 1966 | 0.47011 |
| future_mid_direction | 1 | 519 | 0.124103 |
| future_mid_direction | 2 | 1697 | 0.405787 |
| future_return_bucket | 1 | 4182 | 1.0 |
| volatility_regime | 0 | 2091 | 0.5 |
| volatility_regime | 1 | 2091 | 0.5 |
| spread_widening | 0 | 2657 | 0.635342 |
| spread_widening | 1 | 1525 | 0.364658 |
| adverse_selection_proxy | 0 | 1840 | 0.439981 |
| adverse_selection_proxy | 1 | 2342 | 0.560019 |
| regime_label | 0 | 602 | 0.14395 |
| regime_label | 1 | 600 | 0.143472 |
| regime_label | 2 | 600 | 0.143472 |
| regime_label | 3 | 600 | 0.143472 |
| regime_label | 4 | 600 | 0.143472 |
| regime_label | 5 | 600 | 0.143472 |
| regime_label | 6 | 580 | 0.13869 |
| next_regime_id | 0 | 582 | 0.139168 |
| next_regime_id | 1 | 600 | 0.143472 |
| next_regime_id | 2 | 600 | 0.143472 |
| next_regime_id | 3 | 600 | 0.143472 |
| next_regime_id | 4 | 600 | 0.143472 |
| next_regime_id | 5 | 600 | 0.143472 |
| next_regime_id | 6 | 600 | 0.143472 |

## Baseline Diagnostics

Small baselines confirm the synthetic features and labels flow through a
standard train/validate/test protocol. This is platform and data
validation on controlled synthetic regimes, not a real-market result.

Target: `future_mid_direction`. Held-out regimes: high_volatility, cancellation_shock.

| model_name | split | n_samples | accuracy | macro_f1 | mcc |
| --- | --- | --- | --- | --- | --- |
| majority | validation | 836 | 0.3062 | 0.1563 | 0.0000 |
| majority | test | 837 | 0.4612 | 0.2104 | 0.0000 |
| logistic | validation | 836 | 0.6112 | 0.4133 | 0.2818 |
| logistic | test | 837 | 0.4886 | 0.3445 | 0.0844 |
| ridge | validation | 836 | 0.6100 | 0.4139 | 0.2915 |
| ridge | test | 837 | 0.5018 | 0.3539 | 0.1056 |
| gradient_boosting | validation | 836 | 0.5694 | 0.4582 | 0.2715 |
| gradient_boosting | test | 837 | 0.5412 | 0.4805 | 0.1830 |
| majority | regime_holdout_test | 1180 | 0.3068 | 0.1565 | 0.0000 |
| logistic | regime_holdout_test | 1180 | 0.5339 | 0.3707 | 0.2845 |
| ridge | regime_holdout_test | 1180 | 0.5424 | 0.3766 | 0.2886 |
| gradient_boosting | regime_holdout_test | 1180 | 0.5754 | 0.4962 | 0.3437 |

## Synthetic Regime Stress-Test Diagnostics

Execution-aware proxy diagnostics broken down by known regime. These are
controlled stress tests on synthetic data, not real-market execution
evidence and not a returns or tradability claim.

| regime_name | n_samples | accuracy | active_fraction | filtered_accuracy | turnover_proxy | latency_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| all | 837 | 0.48865 | 0.879331 | 0.506793 | 0.161483 | 0.483871 |
| wide_spread | 257 | 0.435798 | 0.712062 | 0.469945 | 0.144531 | 0.435798 |
| cancellation_shock | 580 | 0.512069 | 0.953448 | 0.518987 | 0.169257 | 0.505172 |

## Claim Assessment

| claim | status | reason |
| --- | --- | --- |
| synthetic_event_level_pipeline | supported | Deterministic generation, replay, features and labels are produced. |
| synthetic_event_level_features | supported | Order-flow, cancellation and trade imbalance are computed from events. |
| synthetic_regime_diagnostics | supported | Per-regime execution-aware proxy diagnostics over known regimes. |
| no_lookahead_labels | supported | Labels use windows strictly after the feature timestamp. |
| real_market_event_level_generalisation | unsupported | All data is synthetic; nothing here transfers to real markets. |
| synthetic_to_real_transfer | unsupported | Synthetic results do not transfer to real markets. |
| live_trading_or_profitability | forbidden | No returns, tradability or live trading is implied or claimed. |
| fi2010_true_event_level_ofi | forbidden | FI-2010 snapshots expose only proxies, never true event-level order flow. |

## Limitations

- All data here is synthetic; it is not real-market evidence.
- Results do not transfer to real markets and imply no returns.
- This extension does not change any FI-2010 limitation.
- FI-2010 snapshots still expose only snapshot proxies, not event-level order flow.
- No live trading, profitability or tradability is implied.

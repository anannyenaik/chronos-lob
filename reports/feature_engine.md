# Feature Engine

This note documents the microstructure feature engine implemented in
Phase 3 of ChronosLOB. It is deliberately conservative: every feature
here is computed strictly from information available at or before the
snapshot timestamp `t`. No labels are produced. No models are trained.
No trading claims are made.

## Past-only principle

Every feature in this engine uses only past information:

- single-snapshot features only inspect the snapshot at `t`;
- OFI between two snapshots compares the previous snapshot at `t-1`
  against the current snapshot at `t`;
- rolling realised volatility at index `t` uses only `prices[t - window
  + 1 : t + 1]`;
- rolling event intensity at index `t` uses only timestamps in
  `[timestamps[t] - window_seconds, timestamps[t]]`.

The pipeline never reads any element of the input sequence after the
index it is computing. The unit tests for rolling volatility assert
this directly by constructing inputs whose future values would change
the result if leakage occurred.

## Implemented features

### Top-of-book price features (`chronoslob.features.microprice`)

- **mid_price** = `(best_bid + best_ask) / 2`
- **spread** = `best_ask - best_bid`
- **relative_spread** = `spread / mid_price`
- **microprice** = `(best_ask * bid_qty + best_bid * ask_qty)
  / (bid_qty + ask_qty)`
- **best_bid_price**, **best_ask_price**, **best_bid_quantity**,
  **best_ask_quantity** are also exposed as feature columns.

Crossed top-of-book pairs raise unless `allow_crossed=True` is set
explicitly. Crossed books are never silently re-ordered.

### Depth and imbalance features (`chronoslob.features.imbalance`)

- **bid_depth_{d}**, **ask_depth_{d}** = sum of resting quantities in
  the first `d` levels on each side.
- **depth_imbalance_{d}** = `(bid_depth - ask_depth) / (bid_depth +
  ask_depth)` (raises on a zero denominator).
- **queue_imbalance** = the same formula applied to best-bid/best-ask
  quantities.
- **depth_slope** (per side) = OLS slope of cumulative quantity against
  distance from best price. Simple linear-regression-style proxy that
  needs at least two levels.
- **liquidity_concentration** (per side) = share of side-quantity held
  in the top-`n` levels (`top / total`).

Per-depth keys are always named after the *requested* depth. When the
snapshot has fewer levels than requested, the available levels are used
but the key still reflects the request — this keeps column names stable
across snapshots with varying numbers of levels.

### Order-flow features (`chronoslob.features.order_flow`)

A deliberately simple, top-of-book OFI approximation derived from
two consecutive snapshots. Per-side rules:

- Bid contribution:
  - `+current_bid_qty` when the best bid price strictly improves;
  - `current_bid_qty - previous_bid_qty` when the best bid price is
    unchanged;
  - `-previous_bid_qty` when the best bid price worsens.
- Ask contribution:
  - `-current_ask_qty` when the best ask price strictly improves
    (moves down);
  - `-(current_ask_qty - previous_ask_qty)` when the best ask price is
    unchanged;
  - `+previous_ask_qty` when the best ask price worsens (moves up).
- `OFI = bid_contribution + ask_contribution`.

`compute_trade_imbalance_from_events` is the analogue for trade events:
`(buy_qty - sell_qty) / (buy_qty + sell_qty)` over `TRADE` events whose
`side` field carries the aggressor side (`BID` → buyer-initiated,
`ASK` → seller-initiated, missing side → ignored). Aggressor side is
*not* inferred from price.

### Volatility and intensity (`chronoslob.features.volatility`)

- **log returns** = `log(p_t / p_{t-1})` (positive finite prices only).
- **realised volatility** = `sqrt(sum(log_return^2))` over the chosen
  window. The estimator was chosen for simplicity and reproducibility
  on a fixed sample; it is *not* annualised here.
- **rolling realised volatility** = the same estimator applied
  past-only at each index, using the most recent `window` prices in
  `prices[:t + 1]`.
- **event intensity** = events-per-second in a trailing window ending at
  the latest timestamp.
- **rolling event intensity** = the same trailing window applied per
  index. Timestamps must be timezone-aware and non-decreasing.

### Regimes (`chronoslob.features.regimes`)

Simple categorical flags. Thresholds are either configured directly or
estimated from a feature frame's quantiles via
`compute_regime_thresholds_from_frame`:

- `classify_spread_regime` → `wide_spread` / `normal_spread`.
- `classify_volatility_regime` → `low_volatility` / `medium_volatility`
  / `high_volatility`.
- `classify_liquidity_regime` → `low_liquidity` / `normal_liquidity`.
- `classify_imbalance_regime` → `bid_heavy` / `ask_heavy` / `balanced`.

These are coarse, interpretable hints — they are **not** trading
signals.

## Pipeline

`chronoslob.features.pipeline.FeaturePipelineConfig` controls which
families are computed and at which depths. The pipeline emits either:

- a single `FeatureRow` via `build_features_from_snapshot`, or
- a `pandas.DataFrame` of one row per snapshot via
  `build_feature_frame_from_snapshots` and
  `build_feature_frame_from_fi2010`.

The DataFrame contains `timestamp`, `symbol`, the configured feature
columns and a non-feature `split` column when the dataset carries one.
Labels are never included.

Synthetic timestamps (those emitted by the FI-2010 loader when the
source file has no timestamp column) are detected through the
`synthetic_time` snapshot metadata flag. By default, time-window
features (rolling event intensity) are skipped in that case and the
skip is recorded in `frame.attrs["skipped_time_features"]`. Set
`FeaturePipelineConfig.allow_synthetic_timestamps_for_time_features=True`
to opt into computing them anyway.

`validate_feature_frame` performs a quick audit of the resulting frame:

- non-empty;
- `timestamp` and `symbol` columns present;
- feature columns are numeric;
- no infinite values;
- NaNs only when `allow_nan=True`;
- no column starts with `label` or `y_` when `feature_columns` is
  inferred from the frame.

## FI-2010 normalisation caveat

The canonical FI-2010 matrix is pre-normalised. Treat absolute spread,
depth and volatility numbers from FI-2010 inputs as **structural**
diagnostics rather than as real exchange figures. The bundled test
fixture uses synthetic, human-readable prices for clarity, but its
feature outputs should still not be interpreted as real market
microstructure.

## What this engine is not

- **Not labels.** Future-looking quantities live in
  `chronoslob.labels`, which is implemented in Phase 4. The feature
  pipeline refuses to ship label-like columns.
- **Not signals.** Regime flags and queue-imbalance numbers are
  diagnostic features, not entry/exit rules.
- **Not full OFI.** The OFI approximation here summarises the change at
  the touch between consecutive snapshots. It does not reconstruct
  every cancel, add and trade between two snapshots and therefore does
  not match an event-level OFI implementation when the venue's
  intermediate events were dense.
- **Not regime detection.** The regime helpers compare scalars against
  quantile thresholds. They do not perform clustering, HMM fitting or
  any other statistical regime model.

## Limitations and follow-ups

- Trade imbalance assumes the upstream feed labels the aggressor side
  on TRADE events. We do not infer aggressor side from price.
- Depth-slope is an OLS slope on `(distance, cumulative quantity)`.
  It is not calibrated and should not be interpreted as price impact.
- Volatility uses `sqrt(sum(log_return^2))`, the simplest realised
  volatility estimator. Annualisation, bias correction and microstructure-
  noise robustness are deferred.
- Event-intensity windows count events strictly; they are not weighted
  by event type.
- The pipeline does not yet bucket snapshots into fixed time bars; one
  row in equals one row out.
- No leakage tests for downstream labels are implemented yet. Those
  are scheduled for Phase 4.

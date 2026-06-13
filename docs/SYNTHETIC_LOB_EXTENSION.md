# Synthetic Event-Level Extension

ChronosLOB is primarily an FI-2010 snapshot-matrix platform. FI-2010 exposes
normalised order-book levels and labels only; it does not expose event messages,
so it can support snapshot proxies but never true event-level order flow.

The synthetic event-level extension complements FI-2010 by adding a small,
storage-light, deterministic synthetic limit-order-book event simulator and
replay pipeline. It demonstrates that the platform can handle event-level
microstructure data with known regimes and supported event-level diagnostics.

Everything here is synthetic. It is a controlled stress-test environment, not
real-market evidence. It does not show tradability or returns and it does not
change any FI-2010 limitation.

## Why synthetic first

- Avoids external data and API fragility.
- Avoids large storage.
- Gives known ground-truth regimes.
- Enables genuine event-level features (add/cancel/trade counts, event imbalance,
  cancellation and trade imbalance).
- Enables controlled stress tests for leakage, regime shift and execution-aware
  proxy diagnostics.

## Event schema

Each event is a canonical `BookEvent` with synthetic metadata:

| field | meaning |
| --- | --- |
| `timestamp` | synthetic strictly increasing event time (UTC) |
| `sequence_id` | strictly increasing integer event index |
| `event_type` | `ADD`, `CANCEL` or `TRADE` |
| `side` | `BID` or `ASK` (aggressor side for `TRADE`) |
| `price` | tick-aligned price level |
| `quantity` | non-negative size added, cancelled or executed |
| `regime_id` / `regime_name` | known ground-truth regime label |
| `latent_mid` | latent reference mid used by the simulator |

Generation is deterministic given a seed: the same configuration always yields
identical events. The generator maintains a coherent book while emitting, so the
event stream alone fully determines an uncrossed book on replay.

## Regimes

The library provides known regimes, each with controlled add/cancel/trade
intensities, bid/ask imbalance, spread tendency, latent volatility and depth
concentration:

- `stable_liquid`
- `high_volatility`
- `low_liquidity`
- `wide_spread`
- `buy_pressure`
- `sell_pressure`
- `cancellation_shock`

## Deterministic replay

Replay rebuilds the book from the event stream alone and validates invariants:
best bid below best ask, non-negative depth, monotonic price levels and
continuous sequence ids. Violations are recorded explicitly in a replay quality
report rather than silently ignored.

## Supported event-level features

Computed on synthetic event streams only:

- `event_order_flow_imbalance`: signed order flow from adds and cancels
- `cancellation_imbalance`: bid-versus-ask cancellation pressure
- `trade_imbalance`: buyer- versus seller-initiated executed volume
- `event_intensity` and add/cancel/trade rates
- `spread`, `relative_spread`, `microprice_offset`
- `depth_imbalance_l1`, `depth_imbalance_l5`
- `realised_volatility_proxy`

FI-2010 still supports only snapshot proxies; these event-level features are not
available for FI-2010 and are valid only on synthetic event streams here.

## Labels

Labels summarise a window strictly after each feature timestamp, so they do not
leak into the contemporaneous features:

- `future_mid_direction`, `future_return_bucket`
- `volatility_regime`, `spread_widening`
- `adverse_selection_proxy`
- `regime_label`, `next_regime_id`

A no-lookahead check verifies that every label references a strictly future
snapshot.

## Baselines and regime stress-test diagnostics

Small baselines (majority, logistic, ridge, gradient boosting) run under a
chronological train/validate/test protocol and a regime-holdout protocol that
trains on some regimes and tests on held-out regimes. Per-regime execution-aware
proxy diagnostics (confidence filtering, active fraction, turnover proxy, latency
sensitivity, adverse-selection proxy) are reported using the known regimes.

These are platform and data validation diagnostics on controlled synthetic
regimes, not real-market results.

## Running it

```bash
python -m chronoslob.cli run-synthetic-lob-benchmark --overwrite --make-figures
# fast smoke version
python -m chronoslob.cli run-synthetic-lob-benchmark --smoke --overwrite
```

Artefacts are written under `reports/synthetic_lob_extension/`:
`synthetic_lob_report.md`, `summary.json`, `synthetic_data_summary.json`,
`synthetic_replay_quality.json`, `synthetic_feature_summary.csv`,
`synthetic_label_summary.csv`, `synthetic_benchmark_summary.csv`,
`synthetic_regime_diagnostics.csv`, `synthetic_claim_assessment.json`,
compact event/snapshot samples and `figure_manifest.json`.

## What this does and does not support

Supported (synthetic only): event-level pipeline, event-level features,
known-regime stress-test diagnostics and no-lookahead labels.

Not supported and not claimed: real-market generalisation, transfer of synthetic
results to real markets, live trading, returns or tradability, and true
event-level order flow on FI-2010. This extension complements FI-2010 rather
than replacing it and leaves FI-2010 limitations unchanged.

# Microstructure Features

ChronosLOB exposes an FI-2010 microstructure feature registry in
`chronoslob.features.registry` and a leakage-safe builder in
`chronoslob.features.microstructure_fi2010`.

FI-2010 NoAuction is a normalised snapshot benchmark. It does not provide true
event messages, trades, cancellations or queue position in the converted
snapshot CSV layout, so the registry records those concepts as unsupported.
All `snapshot_order_flow_proxy` columns are snapshot-to-snapshot visible-book
deltas only. `snapshot_order_flow_proxy` is a labelled snapshot proxy derived
from FI-2010 matrices. It should not be interpreted as true event-level
order-flow imbalance.

## Groups

| Group | Kind | FI-2010 support | Formula or mapping | Limitations |
| --- | --- | --- | --- | --- |
| `price_levels` | raw | supported | `bid_price_N`, `ask_price_N` | Normalised benchmark values may not be native exchange prices. |
| `size_levels` | raw | supported | `bid_quantity_N`/`ask_quantity_N` or size aliases | Displayed snapshot size only. |
| `top_of_book` | raw | supported | best bid/ask price and size from level 1 | Snapshot only. |
| `spread` | derived | supported | `ask_price_1 - bid_price_1`; relative spread divides by midprice | Relative spread requires finite non-zero midprice. |
| `midprice` | derived | supported | `(ask_price_1 + bid_price_1) / 2` | Current row only. |
| `microprice` | derived | supported | `(ask_1 * bid_size_1 + bid_1 * ask_size_1) / (bid_size_1 + ask_size_1)` | Undefined denominator is filled after metadata records missing values. |
| `top_of_book_imbalance` | derived | supported | `(bid_size_1 - ask_size_1) / (bid_size_1 + ask_size_1)` | Level-1 displayed size only. |
| `depth_imbalance` | derived | supported where levels exist | same imbalance over levels 1, 5 and 10 | Visible depth only. |
| `depth_slope` | derived | supported with at least two levels | linear slope of displayed sizes across levels | Shape proxy, not queue dynamics. |
| `liquidity_concentration` | derived | supported where levels exist | top-1/top-5 visible depth divided by total visible depth | Hidden liquidity is not observed. |
| `snapshot_order_flow_proxy` | proxy | supported as proxy | same-row-safe deltas of visible price/size columns versus previous row in the same partition | Labelled snapshot proxy; does not identify submissions, trades or cancellations. |
| `volatility_proxy` | rolling | supported | rolling standard deviation of past/current midprice returns within partition | Uses no future rows. |
| `time_context` | derived | skipped unless timestamp/session columns exist | time-of-day seconds and session position | Canonical FI-2010 matrices do not include true timestamps. |

Explicitly unsupported registry entries include `true_order_flow_imbalance`,
`cancellation_imbalance`, `trade_imbalance` and `queue_position`.

## Leakage Controls

- Label and future-horizon columns are excluded before feature construction.
- Rolling volatility uses grouped `shift(1)` and rolling windows over current
  and past rows only.
- Snapshot delta proxies reset at fold/split/session boundaries.
- Feature rows preserve `row_id` so labels and predictions can be joined
  exactly.
- Strict registry validation fails when an explicitly requested group is
  unsupported or maps to no source/generated columns.

Run:

```bash
python -m chronoslob.cli audit-fi2010-features --path tests/fixtures/fi2010/tiny_fi2010_like.csv
```

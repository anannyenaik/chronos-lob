# ChronosLOB Binance Spot Aggregated L2 Replay

This sample report is generated from small Binance-shaped synthetic
fixtures using the same local snapshot-plus-diff replay contract used
for user-supplied Binance Spot captures. It demonstrates the offline
aggregated depth-stream engineering path; the fixture run is not
exchange-data evidence.
The extension is scoped to crypto-market data engineering support:
not equity-market evidence, not live trading, and it does not establish
profitability, tradability or predictive success.
It complements the FI-2010 and synthetic evidence.

Binance diff-depth updates are aggregated level updates, not individual
order-event data. This extension does not attribute individual trades or
cancellations and does not model queue position from diff-depth alone.

## Overview

| field | value |
| --- | --- |
| generated_at | 2026-06-07T01:08:35.814319+00:00 |
| symbol | TESTUSDT |
| venue | binance |
| evidence_level | binance_l2_fixture_replay |
| snapshot_last_update_id | 100 |
| diff_event_count | 3 |
| applied_event_count | 3 |
| snapshot_count | 3 |
| feature_row_count | 3 |
| replay_ok | True |

## Data Source

The replay consumes a local REST-style depth snapshot and a local JSONL
stream of diff-depth updates. Tests and the default sample use small
Binance-shaped synthetic fixtures; users may supply their own captured
snapshot and diff files.

| field | value |
| --- | --- |
| snapshot_source | tests/fixtures/binance/synthetic_snapshot.json |
| updates_source | tests/fixtures/binance/synthetic_diff_updates.jsonl |
| fixture_data | True |

## Snapshot-Plus-Diff Replay Method

- Load a REST-style snapshot and read its lastUpdateId.
- Buffer diff-depth updates from the JSONL stream.
- Discard events whose final update id u <= snapshot lastUpdateId.
- Start at the first event where U <= lastUpdateId + 1 <= u.
- Require update-id continuity after start; verify pu == prior u when present.
- Upsert a price level for positive quantity; remove it for quantity zero.
- Record gaps, stale events and crossed books instead of silently repairing.

## Replay Quality And Book Invariants

Reconstruction rebuilds the book from the snapshot and diff stream and
validates update continuity, stale-event skipping, non-negative depth and
best bid below best ask. Findings are recorded rather than silently fixed.

| check | value |
| --- | --- |
| started_correctly | True |
| event_count | 3 |
| applied_event_count | 3 |
| skipped_stale_count | 0 |
| gap_count | 0 |
| crossed_count | 0 |
| not_initialised_count | 0 |
| invalid_quantity_count | 0 |
| all_snapshots_uncrossed | True |
| final_update_id | 115 |
| final_bid_levels | 3 |
| final_ask_levels | 4 |
| ok | True |

## Supported Event-Level Features

These features are computed from aggregated diff-depth level updates and
the reconstructed snapshots. Update and depth imbalances reflect aggregate
level changes, not individual orders.

| feature | mean | min | max |
| --- | --- | --- | --- |
| spread | 0.5 | 0.5 | 0.5 |
| relative_spread | 0.004988 | 0.004988 | 0.004988 |
| mid_price | 100.25 | 100.25 | 100.25 |
| microprice | 100.203963 | 100.192308 | 100.227273 |
| microprice_offset | -0.046037 | -0.057692 | -0.022727 |
| depth_imbalance_l1 | -0.184149 | -0.230769 | -0.090909 |
| depth_imbalance_l5 | -0.206941 | -0.422222 | -0.090909 |
| event_intensity | 1.5 | 1.0 | 2.0 |
| update_count | 1.666667 | 1.0 | 2.0 |
| bid_update_imbalance | 0.511111 | 0.2 | 1.0 |
| added_depth_imbalance | 0.347319 | -0.230769 | 1.0 |
| removed_depth_imbalance | 0.666667 | 0.0 | 1.0 |
| order_flow_update_imbalance | 0.241123 | -0.411765 | 1.0 |

## Unsupported Features

Diff-depth data alone cannot support the following. They are not computed
and must not be claimed from this extension.

| feature | reason |
| --- | --- |
| trade_imbalance | no trade stream is supplied; diff-depth carries only level updates. |
| true_cancellation_imbalance | removed levels are aggregate deletions, not individual cancellations. |
| queue_position | aggregated level updates expose no per-order queue position. |

## Storage Footprint

Only compact summaries are retained. Large raw captures are written to a
git-ignored local directory and are never committed.

| field | value |
| --- | --- |
| snapshot_bytes | 224 |
| updates_bytes | 466 |
| retained_artefacts | small summaries, CSVs and JSON only |
| raw_capture_policy | large raw captures are git-ignored by default |

## Claim Assessment

| claim | status | reason |
| --- | --- | --- |
| binance_l2_replay_pipeline | supported | Snapshot-plus-diff replay reconstructs the book and passes the continuity and invariant checks. |
| real_captured_aggregated_l2_stream_path | needs_real_evidence | The bundled sample uses Binance-shaped synthetic fixtures; the same offline parser/replay path accepts user-supplied local Binance captures. |
| binance_update_continuity_validation | supported | Update-id bracketing, stale-event and gap checks are enforced. |
| binance_order_book_invariants | supported | Non-negative depth and best bid below best ask are validated. |
| real_market_predictive_success | unsupported | No predictive or returns evidence is produced by replay. |
| equity_market_generalisation | unsupported | This is crypto-venue data and does not transfer to equities. |
| live_trading_or_profitability | forbidden | Replay is offline; no live trading or returns are implied. |
| individual_order_queue_position | forbidden | Aggregated level updates cannot expose individual queue position. |
| true_trades_or_cancellations_from_diff_depth | unsupported | Diff-depth carries aggregate level changes, not trades or cancels. |

## Limitations

- Binance L2 evidence is crypto-market engineering evidence, not equity evidence.
- Diff-depth updates are aggregated level updates, not individual order events.
- No true trade, cancellation or queue-position attribution is possible here.
- It does not establish profitability, tradability or predictive success.
- It is not live trading; replay is offline from local files.
- It complements FI-2010 and synthetic evidence and changes no FI-2010 limitation.

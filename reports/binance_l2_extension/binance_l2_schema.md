# Binance L2 Extension Schema

Typed schemas for the real captured aggregated L2 replay path. Binance
diff-depth updates are aggregated level updates, not individual order
events.

## Depth Snapshot

| field | meaning |
| --- | --- |
| symbol | instrument symbol, e.g. TESTUSDT |
| last_update_id | snapshot lastUpdateId from the REST depth payload |
| bids / asks | price/quantity levels (bids descending, asks ascending) |
| timestamp / received_timestamp | optional exchange / capture timestamps |

## Diff-Depth Update

| field | meaning |
| --- | --- |
| event_type | depthUpdate (Binance 'e') |
| event_time / transaction_time | Binance 'E' / 'T' epoch-ms timestamps |
| symbol | instrument symbol (Binance 's') |
| first_update_id | U: first update id covered by the event |
| final_update_id | u: final update id covered by the event |
| previous_final_update_id | pu: prior event final update id when present |
| bids / asks | aggregated level updates; quantity 0 removes a level |
| received_timestamp | optional local capture timestamp |

## Reconstructed Book Snapshot

| field | meaning |
| --- | --- |
| timestamp | timestamp of the applied diff event |
| symbol / venue | instrument symbol and 'binance' venue |
| bids / asks | reconstructed price levels after applying the diff |
| sequence_id | final update id of the most recent applied diff |

## Replay Quality Report

| field | meaning |
| --- | --- |
| started_correctly | snapshot/diff bracketing succeeded |
| applied_event_count | number of diffs applied to the book |
| skipped_stale_count | diffs discarded as stale |
| gap_count | update-id continuity gaps recorded |
| crossed_count | crossed-book findings recorded |
| final_bid_levels / final_ask_levels | depth of the final reconstructed book |
| ok | true when no continuity or invariant violation was found |

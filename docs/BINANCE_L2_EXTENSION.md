# Binance Spot Aggregated L2 Replay

ChronosLOB includes a storage-light, offline path that ingests a Binance Spot
L2 depth snapshot plus a diff-depth update stream, reconstructs the local order
book deterministically and validates update continuity and book invariants. It
adds an aggregated crypto-venue depth-stream engineering path alongside FI-2010
snapshots and the synthetic event-level extension. The committed sample uses
Binance-shaped synthetic fixtures; user-supplied local captures are required
for exchange-data evidence.

## What this is, and what it is not

- It is an offline aggregated crypto-market order-book ingestion and replay
  path. Fixture runs are engineering checks; local captures are required for
  exchange-data evidence.
- Binance diff-depth updates are **aggregated level updates**, not individual
  order-event data. A positive quantity upserts a price level; a zero quantity
  removes it.
- It is **not** equity-market evidence, **not** live trading, and **not**
  profitability, tradability or predictive-success evidence.
- It does **not** model queue position and does **not** attribute individual
  trades or cancellations from diff-depth alone.
- It complements the FI-2010 snapshot evidence and the synthetic event-level
  extension and changes no FI-2010 limitation: FI-2010 snapshots still expose
  only snapshot proxies, never true event-level order flow.

## Snapshot-plus-diff replay

The replay follows the standard Binance local-book procedure:

1. Load a REST-style depth snapshot and read its `lastUpdateId`.
2. Buffer diff-depth updates from the JSONL stream.
3. Discard events whose final update id `u <= lastUpdateId`.
4. Start at the first event where `U <= lastUpdateId + 1 <= u`.
5. Require update-id continuity after the start; when `pu` is present, verify
   `pu == previous u`.
6. Apply bid/ask level updates: positive quantity upserts a level, zero removes
   it.

Gaps, stale events and crossed books are recorded as findings rather than
silently repaired. The reconstruction lives in
[`chronoslob/book/reconstruction.py`](../chronoslob/book/reconstruction.py) and
the extension orchestration in
[`chronoslob/binance_l2/`](../chronoslob/binance_l2/).

## Replay quality and invariants

The replay quality report records: whether replay started correctly, the number
of applied versus skipped (stale) events, update-id gaps, crossed-book findings,
an enforced-zero invalid-quantity count (the schema rejects negative or
non-finite quantities at parse time), a best-bid-below-best-ask check across all
emitted snapshots, the final book depth and an overall `ok` flag.

## Supported features

Computed from aggregated diff-depth level updates and the reconstructed
snapshots: `spread`, `relative_spread`, `mid_price`, `microprice`,
`microprice_offset`, `depth_imbalance_l1`, `depth_imbalance_l5`,
`event_intensity`, `update_count`, `bid_update_imbalance`,
`added_depth_imbalance`, `removed_depth_imbalance` and an aggregate
`order_flow_update_imbalance`. The order-flow-style feature is an aggregate
level-update imbalance, not an individual-order order-flow imbalance.

## Unsupported features

Diff-depth data alone cannot support these, so they are not computed and must
not be claimed:

- `trade_imbalance`: no trade stream is supplied with diff-depth.
- true cancellation imbalance: removed levels are aggregate deletions, not
  individual cancellations; the supported signal is `removed_depth_imbalance`.
- queue position: aggregated level updates expose no per-order queue position.

## Running it

The replay is offline and uses local files only. With no paths supplied it uses
the small bundled Binance-shaped synthetic fixtures:

```
python -m chronoslob.cli replay-binance-l2-sample --out reports/binance_l2_extension --make-figures
```

Supply your own captured files with `--snapshot` and `--updates`:

```
python -m chronoslob.cli replay-binance-l2-sample \
  --snapshot path/to/snapshot.json --updates path/to/diff_depth.jsonl --out reports/binance_l2_extension
```

A live capture command is intentionally not bundled; users supply a REST
snapshot JSON and a diff-depth JSONL file. This keeps tests offline and avoids a
network dependency. Large raw captures should be written to the git-ignored
`data/binance_l2_captures/` directory and never committed.

## Fixtures

Tiny Binance-shaped synthetic fixtures live under
[`tests/fixtures/binance/`](../tests/fixtures/binance/): a snapshot, a valid
diff stream, a stale-event stream, a sequence-gap stream and a crossed-book
update. They are synthetic and must never be presented as real exchange data.

## Storage footprint

Only compact summaries, CSVs, small JSON files and tiny figures are retained
under `reports/binance_l2_extension/`. Raw captures are git-ignored by default.

## Claim boundaries

The extension's claim assessment marks the replay pipeline, update-continuity
validation and order-book invariants as supported when replay checks pass. A
fixture-only sample is explicitly marked as needing a user-supplied local
Binance capture before the real stream-path claim is supported. Real-market
predictive success and equity-market generalisation are unsupported; recovering
true trades or cancellations from diff-depth is unsupported; and live trading,
profitability and individual-order queue position are forbidden.

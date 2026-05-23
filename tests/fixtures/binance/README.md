# Synthetic Binance-style fixtures

All files in this directory are **synthetic**. They are not real
Binance market data.

They exist only to exercise the offline order book reconstruction
code in `chronoslob.data.binance`, `chronoslob.book.local_order_book`,
`chronoslob.book.reconstruction` and `chronoslob.book.replay`. They
must never be presented as benchmark data, as live exchange data or
as evidence of any trading performance.

The fixtures are deliberately tiny so that tests stay fast and the
files remain easy to audit by eye.

## Files

- `synthetic_snapshot.json` — a small depth snapshot for the
  fictional `TESTUSDT` symbol. The `lastUpdateId` is 100.
- `synthetic_diff_updates.jsonl` — a short, valid sequence of diff
  updates (`U`, `u`, `pu`) that exercises a quantity change, a
  zero-quantity deletion and a new price level.
- `synthetic_gap_updates.jsonl` — the same shape as the valid
  sequence but with a deliberate update-id gap so reconstruction
  records a `GAP_DETECTED` issue.
- `synthetic_crossed_updates.jsonl` — a single diff update whose
  ask quantity moves below the best bid so the reconstruction code
  records a `CROSSED_BOOK` issue.

## What this directory does not contain

- Real Binance REST or WebSocket payloads.
- Captured production data of any kind.
- API keys, credentials or remote URLs.

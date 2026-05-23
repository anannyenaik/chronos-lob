# Offline Binance-style order book reconstruction

This note documents the offline order book reconstruction layer
introduced in Phase 8. It uses UK English. It is deliberately narrow:
the layer reconstructs local order book state from a Binance-style
snapshot plus a sequence of diff updates, surfaces continuity issues
and converts the result to the canonical
`chronoslob.data.schemas.OrderBookSnapshot` shape.

The reconstruction code makes **no network calls**, opens **no
WebSocket connections** and contains **no REST clients**. All inputs
are local files supplied by the caller.

## Scope

The phase covers:

- Typed schemas for Binance-style REST depth snapshots and diff
  events (`chronoslob.data.binance`).
- In-memory order book state with explicit update and trimming
  semantics (`chronoslob.book.local_order_book.LocalOrderBook`).
- Reconstruction logic with update-id continuity checks, stale event
  skipping, gap detection and crossed-book surfacing
  (`chronoslob.book.reconstruction`).
- Deterministic replay helpers that load fixtures from local files
  (`chronoslob.book.replay`).
- Conversion to the canonical Phase 1 `OrderBookSnapshot` schema.
- A read-only CLI inspection command,
  `python -m chronoslob.cli inspect-binance-replay`, that operates on
  the bundled synthetic fixtures.

The phase does not implement live ingestion, REST polling, WebSocket
clients, model training, transformers, self-supervised pretraining,
backtests, PnL or any other trading-system components.

## Snapshot plus diff reconstruction

The reconstruction follows the standard snapshot plus diff pattern:

1. A depth snapshot supplies the initial state and a `lastUpdateId`.
2. A stream of diff events updates the book in order. Each diff event
   carries a first (`U`) and final (`u`) update id, and a Binance
   USDT-margined futures stream additionally carries the previous
   final update id (`pu`).
3. The first diff event applied to the book is the one that brackets
   the snapshot, i.e. `U <= lastUpdateId + 1 <= u`. Diff events whose
   final update id is at or below `lastUpdateId` are stale and are
   skipped.
4. Each diff event sets the resting quantity at a price level when
   the quantity is strictly positive and removes the level when the
   quantity is zero.

This implementation uses dictionaries keyed by price for each side so
that updates are applied in O(1) and re-sorting is done only when the
canonical snapshot is built.

## Update id continuity

`chronoslob.book.reconstruction.has_update_gap` is the single source
of truth for the continuity check between successive events:

- When `previous_final_update_id` is supplied on the event, continuity
  requires it to equal the book's `last_update_id`. This matches the
  USDT-margined futures continuity rule.
- Otherwise continuity requires `first_update_id == last_update_id + 1`.

Both checks are deliberately strict. The reconstruction code does
**not** silently bridge gaps: a violation is recorded as a
`ReconstructionIssue` with `status == GAP_DETECTED`.

## Stale event skipping

Stale events are detected by
`chronoslob.book.reconstruction.is_stale_event`: any event whose
`final_update_id` does not exceed the book's `last_update_id` is
ignored. A `STALE_EVENT_SKIPPED` issue is recorded so the caller can
audit how many events were dropped relative to the snapshot.

## Gap detection and stop behaviour

When `reconstruct_order_book` is called with `stop_on_gap=True` (the
default) a detected gap stops reconstruction at the offending event
and the result is returned with `ok == False`. When
`stop_on_gap=False` reconstruction continues by treating the
offending event as the new baseline; the issue is still recorded so
that downstream tools can decide whether to discard the partial
reconstruction.

## Crossed-book handling

A diff that causes the best bid to be at or above the best ask is
treated as a data-quality issue. `LocalOrderBook.apply_diff` raises a
`ValueError` mentioning "crossed"; the reconstruction code catches
that error, records a `ReconstructionIssue` with `status ==
CROSSED_BOOK` and either stops (default) or continues based on the
`allow_crossed` flag. The canonical `OrderBookSnapshot` schema does
not reject crossed books at construction time, so a caller that
prefers crossed bookkeeping can still emit and inspect them.

## Deterministic replay

`chronoslob.book.replay.replay_binance_jsonl` loads a snapshot
JSON file and a diff JSONL file from local paths, applies them
through `reconstruct_order_book` and returns the resulting
`ReconstructionResult`. The function is deterministic: repeated
invocations with the same input files yield byte-identical canonical
snapshots and identical issue lists. There is no randomness, no
network access and no global state.

`summarise_replay_result` returns a small summary dictionary suitable
for printing in a CLI or storing alongside experiment metadata.

## Fixture-only scope

The synthetic fixtures bundled under `tests/fixtures/binance/` are
deliberately tiny and are explicitly marked as synthetic in the
adjacent `README.md`. They exist to exercise the reconstruction code
and to keep the test suite fast. They are not real Binance data and
must not be presented as such.

## Limitations versus live ingestion

The phase is intentionally narrow. The following items are explicitly
out of scope:

- No live WebSocket client. Reconstruction operates on captured
  fixtures only.
- No REST snapshot fetcher. The snapshot is read from a local JSON
  file the user supplies.
- No automatic resynchronisation after a gap. The recommended response
  is to capture a fresh snapshot offline and rerun the reconstruction.
- No multi-symbol or multi-venue orchestration. Each `LocalOrderBook`
  is bound to a single symbol and a single venue label.
- No latency modelling, no order matching, no PnL accounting and no
  transaction cost model.

## No trading claims

This reconstruction layer is an engineering and research demonstration
that ChronosLOB can handle event-stream data deterministically. It is
not a trading system, does not produce trading signals and does not
imply any profitability or deployable execution capability.

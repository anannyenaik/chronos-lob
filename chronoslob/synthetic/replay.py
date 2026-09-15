"""Deterministic replay of synthetic events into order-book snapshots.

The replay component is intentionally independent of the generator: it consumes
only the canonical event stream and rebuilds the book from scratch. This lets it
act as a check that the emitted stream is coherent (no crossed book, no negative
depth, continuous sequence ids).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from chronoslob.data.schemas import (
    BookEvent,
    EventType,
    OrderBookLevel,
    OrderBookSnapshot,
    Side,
)

__all__ = [
    "ReplayQualityReport",
    "ReplayResult",
    "replay_events_to_snapshots",
]


@dataclass
class ReplayQualityReport:
    """Quality findings from replaying a synthetic event stream."""

    event_count: int = 0
    snapshot_count: int = 0
    crossed_snapshot_count: int = 0
    negative_depth_event_count: int = 0
    sequence_gap_count: int = 0
    empty_book_snapshot_count: int = 0
    unknown_event_type_count: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return ``True`` when no invariant violations were recorded."""
        return (
            self.crossed_snapshot_count == 0
            and self.negative_depth_event_count == 0
            and self.sequence_gap_count == 0
            and self.unknown_event_type_count == 0
            and not self.errors
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation."""
        return {
            "event_count": self.event_count,
            "snapshot_count": self.snapshot_count,
            "crossed_snapshot_count": self.crossed_snapshot_count,
            "negative_depth_event_count": self.negative_depth_event_count,
            "sequence_gap_count": self.sequence_gap_count,
            "empty_book_snapshot_count": self.empty_book_snapshot_count,
            "unknown_event_type_count": self.unknown_event_type_count,
            "ok": self.ok,
            "errors": list(self.errors),
        }


@dataclass
class ReplayResult:
    """Snapshots and quality findings produced by a replay run."""

    snapshots: list[OrderBookSnapshot]
    quality: ReplayQualityReport


def replay_events_to_snapshots(
    events: Sequence[BookEvent],
    *,
    tick_size: float,
    max_levels_per_side: int = 10,
    snapshot_interval: int = 5,
    symbol: str = "SYNTH",
) -> ReplayResult:
    """Replay ``events`` into top-``max_levels_per_side`` snapshots.

    A snapshot is captured every ``snapshot_interval`` events and once more at
    the end of the stream. Book invariants are validated as the stream is
    replayed and summarised in the returned quality report.
    """
    if tick_size <= 0.0:
        raise ValueError("tick_size must be positive")
    if snapshot_interval < 1:
        raise ValueError("snapshot_interval must be >= 1")

    bids: dict[int, float] = {}
    asks: dict[int, float] = {}
    quality = ReplayQualityReport(event_count=len(events))
    snapshots: list[OrderBookSnapshot] = []
    previous_sequence: int | None = None

    for index, event in enumerate(events):
        previous_sequence = _check_sequence(event, previous_sequence, quality)
        _apply_event(event, bids, asks, tick_size, quality)
        is_last = index == len(events) - 1
        if (index + 1) % snapshot_interval == 0 or is_last:
            snapshot = _build_snapshot(
                event, bids, asks, tick_size, max_levels_per_side, symbol
            )
            snapshots.append(snapshot)
            _check_snapshot(snapshot, quality)

    quality.snapshot_count = len(snapshots)
    return ReplayResult(snapshots=snapshots, quality=quality)


def _check_sequence(
    event: BookEvent,
    previous_sequence: int | None,
    quality: ReplayQualityReport,
) -> int | None:
    sequence = event.sequence_id
    if sequence is None:
        return previous_sequence
    if previous_sequence is not None and sequence != previous_sequence + 1:
        quality.sequence_gap_count += 1
        quality.errors.append(
            f"sequence discontinuity: {previous_sequence} -> {sequence}"
        )
    return sequence


def _apply_event(
    event: BookEvent,
    bids: dict[int, float],
    asks: dict[int, float],
    tick_size: float,
    quality: ReplayQualityReport,
) -> None:
    if event.price is None or event.quantity is None or event.side is None:
        return
    price_tick = round(event.price / tick_size)
    quantity = event.quantity
    if event.event_type is EventType.ADD:
        book = bids if event.side is Side.BID else asks
        book[price_tick] = book.get(price_tick, 0.0) + quantity
    elif event.event_type is EventType.CANCEL:
        book = bids if event.side is Side.BID else asks
        _reduce(book, price_tick, quantity, quality)
    elif event.event_type is EventType.TRADE:
        # The recorded side is the aggressor; the passive resting side is the
        # opposite book and is the one whose liquidity is consumed.
        passive = asks if event.side is Side.BID else bids
        _reduce(passive, price_tick, quantity, quality)
    else:
        quality.unknown_event_type_count += 1
        quality.errors.append(f"unsupported event type for replay: {event.event_type}")


def _reduce(
    book: dict[int, float],
    price_tick: int,
    quantity: float,
    quality: ReplayQualityReport,
) -> None:
    resting = book.get(price_tick)
    if resting is None:
        return
    remaining = resting - quantity
    if remaining < -1e-6:
        quality.negative_depth_event_count += 1
        quality.errors.append(
            f"reduce below zero at tick {price_tick}: {resting} - {quantity}"
        )
    if remaining <= 1e-9:
        del book[price_tick]
    else:
        book[price_tick] = remaining


def _build_snapshot(
    event: BookEvent,
    bids: dict[int, float],
    asks: dict[int, float],
    tick_size: float,
    max_levels_per_side: int,
    symbol: str,
) -> OrderBookSnapshot:
    bid_levels = [
        OrderBookLevel(price=round(price * tick_size, 6), quantity=bids[price])
        for price in sorted(bids, reverse=True)[:max_levels_per_side]
    ]
    ask_levels = [
        OrderBookLevel(price=round(price * tick_size, 6), quantity=asks[price])
        for price in sorted(asks)[:max_levels_per_side]
    ]
    metadata: dict[str, str | int | float | bool] = {"synthetic": True}
    for key in ("regime_id", "regime_name", "latent_mid"):
        value = event.metadata.get(key)
        if value is not None:
            metadata[key] = value
    if event.sequence_id is not None:
        metadata["sequence_id"] = event.sequence_id
    return OrderBookSnapshot(
        timestamp=event.timestamp,
        symbol=symbol,
        venue="synthetic",
        bids=bid_levels,
        asks=ask_levels,
        sequence_id=event.sequence_id,
        metadata=metadata,
    )


def _check_snapshot(snapshot: OrderBookSnapshot, quality: ReplayQualityReport) -> None:
    if not snapshot.bids or not snapshot.asks:
        quality.empty_book_snapshot_count += 1
        return
    if snapshot.is_crossed:
        quality.crossed_snapshot_count += 1
        best_bid = snapshot.best_bid.price if snapshot.best_bid else None
        best_ask = snapshot.best_ask.price if snapshot.best_ask else None
        quality.errors.append(f"crossed snapshot: bid={best_bid} ask={best_ask}")

"""Snapshot-plus-diff order book reconstruction.

This module replays Binance-style snapshot and diff-depth fixtures
against a :class:`LocalOrderBook` while explicitly checking update-id
continuity. It surfaces gaps, stale events and crossed-book conditions
as :class:`ReconstructionIssue` records rather than silently repairing
them. All inputs are local objects — no network access is performed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from chronoslob.book.local_order_book import (
    LocalOrderBook,
    LocalOrderBookConfig,
)
from chronoslob.data.binance import (
    BinanceDepthSnapshot,
    BinanceDiffDepthEvent,
)
from chronoslob.data.schemas import MetadataDict, OrderBookSnapshot

__all__ = [
    "ReconstructionIssue",
    "ReconstructionResult",
    "ReconstructionStatus",
    "has_update_gap",
    "is_stale_event",
    "reconstruct_order_book",
    "should_apply_first_diff",
]


class ReconstructionStatus(StrEnum):
    """Status codes used in :class:`ReconstructionIssue`."""

    OK = "OK"
    GAP_DETECTED = "GAP_DETECTED"
    STALE_EVENT_SKIPPED = "STALE_EVENT_SKIPPED"
    NOT_INITIALISED = "NOT_INITIALISED"
    CROSSED_BOOK = "CROSSED_BOOK"


@dataclass(frozen=True)
class ReconstructionIssue:
    """A single data-quality finding raised during reconstruction."""

    status: str
    message: str
    event_update_id: int | None = None
    expected_update_id: int | None = None
    metadata: MetadataDict = field(default_factory=dict)


@dataclass
class ReconstructionResult:
    """Result of a deterministic reconstruction run."""

    snapshots: list[OrderBookSnapshot] = field(default_factory=list)
    issues: list[ReconstructionIssue] = field(default_factory=list)
    final_update_id: int | None = None
    ok: bool = True

    @property
    def gap_count(self) -> int:
        """Return the number of gap issues recorded."""
        return sum(
            1
            for issue in self.issues
            if issue.status == ReconstructionStatus.GAP_DETECTED
        )

    @property
    def crossed_count(self) -> int:
        """Return the number of crossed-book issues recorded."""
        return sum(
            1
            for issue in self.issues
            if issue.status == ReconstructionStatus.CROSSED_BOOK
        )

    @property
    def n_snapshots(self) -> int:
        """Return the number of canonical snapshots emitted."""
        return len(self.snapshots)


# ---------------------------------------------------------------------------
# Continuity helpers
# ---------------------------------------------------------------------------


def should_apply_first_diff(
    snapshot_last_update_id: int,
    event: BinanceDiffDepthEvent,
) -> bool:
    """Return ``True`` if ``event`` is the first applicable diff after a snapshot.

    Implements the standard Binance condition ``U <= lastUpdateId + 1 <= u``.
    Stale events whose final update id does not exceed
    ``snapshot_last_update_id`` are not applicable.
    """
    if event.final_update_id <= snapshot_last_update_id:
        return False
    return (
        event.first_update_id <= snapshot_last_update_id + 1
        and snapshot_last_update_id + 1 <= event.final_update_id
    )


def is_stale_event(
    last_update_id: int, event: BinanceDiffDepthEvent
) -> bool:
    """Return ``True`` if ``event`` is entirely behind ``last_update_id``."""
    return event.final_update_id <= last_update_id


def has_update_gap(
    last_update_id: int, event: BinanceDiffDepthEvent
) -> bool:
    """Return ``True`` if a continuity gap is detected against ``last_update_id``.

    If ``event.previous_final_update_id`` is supplied, continuity requires
    it to equal ``last_update_id``. Otherwise the event's
    ``first_update_id`` is expected to be exactly
    ``last_update_id + 1``.
    """
    if event.previous_final_update_id is not None:
        return event.previous_final_update_id != last_update_id
    return event.first_update_id > last_update_id + 1


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def _expected_update_id(last_update_id: int) -> int:
    return last_update_id + 1


def reconstruct_order_book(
    snapshot: BinanceDepthSnapshot,
    events: Sequence[BinanceDiffDepthEvent],
    *,
    max_depth: int | None = None,
    stop_on_gap: bool = True,
    allow_crossed: bool = False,
) -> ReconstructionResult:
    """Reconstruct a local book deterministically from a snapshot and diffs.

    Events are processed in the order supplied. Stale events arriving
    before the first applicable diff are skipped and recorded. Gaps are
    detected via :func:`has_update_gap` and recorded as issues; when
    ``stop_on_gap`` is true reconstruction stops at the gap. Crossed
    books raise inside the local book, are caught and recorded; when
    ``allow_crossed`` is false reconstruction stops on the crossed event.
    """
    config = LocalOrderBookConfig(
        symbol=snapshot.symbol,
        venue="binance",
        max_depth=max_depth,
        allow_crossed=allow_crossed,
    )
    book = LocalOrderBook(config)
    book.load_snapshot(snapshot)

    result = ReconstructionResult(final_update_id=book.last_update_id)

    first_diff_applied = False

    for event in events:
        if event.symbol != snapshot.symbol:
            result.issues.append(
                ReconstructionIssue(
                    status=ReconstructionStatus.NOT_INITIALISED,
                    message=(
                        "diff event symbol does not match snapshot symbol; "
                        f"event={event.symbol!r}, snapshot={snapshot.symbol!r}"
                    ),
                    event_update_id=event.final_update_id,
                    metadata={"event_symbol": event.symbol},
                )
            )
            result.ok = False
            break

        last_update_id = book.last_update_id
        assert last_update_id is not None  # load_snapshot guarantees this

        if not first_diff_applied:
            if is_stale_event(last_update_id, event):
                result.issues.append(
                    ReconstructionIssue(
                        status=ReconstructionStatus.STALE_EVENT_SKIPPED,
                        message=(
                            "skipping stale diff event "
                            f"(u={event.final_update_id}) at or behind "
                            f"snapshot lastUpdateId={last_update_id}"
                        ),
                        event_update_id=event.final_update_id,
                        expected_update_id=_expected_update_id(last_update_id),
                    )
                )
                continue
            if not should_apply_first_diff(last_update_id, event):
                result.issues.append(
                    ReconstructionIssue(
                        status=ReconstructionStatus.GAP_DETECTED,
                        message=(
                            "first diff event does not bracket "
                            f"lastUpdateId+1={last_update_id + 1}; "
                            f"U={event.first_update_id}, u={event.final_update_id}"
                        ),
                        event_update_id=event.final_update_id,
                        expected_update_id=_expected_update_id(last_update_id),
                    )
                )
                result.ok = False
                if stop_on_gap:
                    break
                # If continuing despite a gap, treat this event as the
                # new baseline so further continuity checks have a
                # reference point.
                first_diff_applied = True
                _safe_apply(book, event, result, allow_crossed=allow_crossed)
                if result.crossed_count and not allow_crossed:
                    break
                result.snapshots.append(book.to_snapshot())
                result.final_update_id = book.last_update_id
                continue
            if has_update_gap(last_update_id, event):
                result.issues.append(
                    ReconstructionIssue(
                        status=ReconstructionStatus.GAP_DETECTED,
                        message=(
                            "first applicable diff has inconsistent previous "
                            f"update id; expected pu={last_update_id}, "
                            f"got pu={event.previous_final_update_id}"
                        ),
                        event_update_id=event.final_update_id,
                        expected_update_id=_expected_update_id(last_update_id),
                    )
                )
                result.ok = False
                if stop_on_gap:
                    break
            first_diff_applied = True
        else:
            if is_stale_event(last_update_id, event):
                result.issues.append(
                    ReconstructionIssue(
                        status=ReconstructionStatus.STALE_EVENT_SKIPPED,
                        message=(
                            "skipping stale diff event "
                            f"(u={event.final_update_id}) at or behind "
                            f"lastUpdateId={last_update_id}"
                        ),
                        event_update_id=event.final_update_id,
                        expected_update_id=_expected_update_id(last_update_id),
                    )
                )
                continue
            if has_update_gap(last_update_id, event):
                result.issues.append(
                    ReconstructionIssue(
                        status=ReconstructionStatus.GAP_DETECTED,
                        message=(
                            "update id continuity broken; "
                            f"expected next update at {last_update_id + 1}, "
                            f"got U={event.first_update_id}, "
                            f"u={event.final_update_id}, "
                            f"pu={event.previous_final_update_id}"
                        ),
                        event_update_id=event.final_update_id,
                        expected_update_id=_expected_update_id(last_update_id),
                    )
                )
                result.ok = False
                if stop_on_gap:
                    break

        applied = _safe_apply(book, event, result, allow_crossed=allow_crossed)
        if not applied:
            if not allow_crossed:
                break
            continue
        result.snapshots.append(book.to_snapshot())
        result.final_update_id = book.last_update_id

    return result


def _safe_apply(
    book: LocalOrderBook,
    event: BinanceDiffDepthEvent,
    result: ReconstructionResult,
    *,
    allow_crossed: bool,
) -> bool:
    """Apply ``event`` to ``book``, recording a crossed-book issue on failure."""
    try:
        book.apply_diff(event)
    except ValueError as exc:
        if "crossed" not in str(exc).lower():
            raise
        result.issues.append(
            ReconstructionIssue(
                status=ReconstructionStatus.CROSSED_BOOK,
                message=str(exc),
                event_update_id=event.final_update_id,
            )
        )
        result.ok = False
        if allow_crossed:
            return False
        return False
    return True

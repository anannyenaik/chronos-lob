"""Rich replay-quality summary for Binance-style snapshot-plus-diff replay.

This module turns a :class:`~chronoslob.book.reconstruction.ReconstructionResult`
into a compact, JSON-serialisable quality report with the granular checks the
Binance L2 extension exposes: whether replay started correctly, how many diff
events were applied versus skipped, update-continuity gaps, stale events,
crossed-book findings, book-invariant checks and final book depth.

All inputs are in-memory objects loaded from local files; nothing here makes a
network call. Binance diff-depth updates are aggregated level updates, not
individual order events, so this report never attributes individual trades,
cancellations or queue positions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from chronoslob.book.reconstruction import (
    ReconstructionResult,
    ReconstructionStatus,
)
from chronoslob.data.binance import BinanceDiffDepthEvent
from chronoslob.data.schemas import OrderBookSnapshot

__all__ = [
    "BinanceReplayQualityReport",
    "build_replay_quality_report",
]

# Message fragments emitted by reconstruct_order_book when the very first
# applicable diff fails to bracket lastUpdateId+1 or has an inconsistent
# previous update id. Used only to decide whether replay started correctly.
_FIRST_DIFF_FAILURE_FRAGMENTS = (
    "first diff event does not bracket",
    "first applicable diff has inconsistent previous",
)


@dataclass
class BinanceReplayQualityReport:
    """Granular quality findings from a Binance L2 snapshot-plus-diff replay."""

    event_count: int = 0
    applied_event_count: int = 0
    skipped_stale_count: int = 0
    gap_count: int = 0
    crossed_count: int = 0
    not_initialised_count: int = 0
    invalid_quantity_count: int = 0
    started_correctly: bool = True
    all_snapshots_uncrossed: bool = True
    final_update_id: int | None = None
    final_bid_levels: int = 0
    final_ask_levels: int = 0
    issue_count: int = 0
    issues_by_status: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return ``True`` when no invariant or continuity violation was found."""
        return (
            self.started_correctly
            and self.all_snapshots_uncrossed
            and self.gap_count == 0
            and self.crossed_count == 0
            and self.not_initialised_count == 0
            and self.invalid_quantity_count == 0
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation."""
        return {
            "event_count": self.event_count,
            "applied_event_count": self.applied_event_count,
            "skipped_stale_count": self.skipped_stale_count,
            "gap_count": self.gap_count,
            "crossed_count": self.crossed_count,
            "not_initialised_count": self.not_initialised_count,
            "invalid_quantity_count": self.invalid_quantity_count,
            "started_correctly": self.started_correctly,
            "all_snapshots_uncrossed": self.all_snapshots_uncrossed,
            "final_update_id": self.final_update_id,
            "final_bid_levels": self.final_bid_levels,
            "final_ask_levels": self.final_ask_levels,
            "issue_count": self.issue_count,
            "issues_by_status": dict(self.issues_by_status),
            "ok": self.ok,
            "notes": list(self.notes),
        }


def _count_issues(result: ReconstructionResult) -> dict[str, int]:
    counts: dict[str, int] = {}
    for issue in result.issues:
        counts[issue.status] = counts.get(issue.status, 0) + 1
    return counts


def _started_correctly(result: ReconstructionResult) -> bool:
    for issue in result.issues:
        if issue.status != ReconstructionStatus.GAP_DETECTED:
            continue
        message = issue.message.lower()
        if any(fragment in message for fragment in _FIRST_DIFF_FAILURE_FRAGMENTS):
            return False
    return True


def _final_depth(snapshots: Sequence[OrderBookSnapshot]) -> tuple[int, int]:
    if not snapshots:
        return 0, 0
    final = snapshots[-1]
    return len(final.bids), len(final.asks)


def build_replay_quality_report(
    result: ReconstructionResult,
    events: Sequence[BinanceDiffDepthEvent],
) -> BinanceReplayQualityReport:
    """Summarise a reconstruction run into a granular quality report.

    ``events`` is the full ordered list of loaded diff events (including stale
    ones); ``result`` is the outcome of replaying them. The number of applied
    events equals the number of emitted snapshots because the reconstruction
    appends exactly one snapshot per applied diff.
    """
    counts = _count_issues(result)
    skipped_stale = counts.get(ReconstructionStatus.STALE_EVENT_SKIPPED, 0)
    not_initialised = counts.get(ReconstructionStatus.NOT_INITIALISED, 0)
    final_bid_levels, final_ask_levels = _final_depth(result.snapshots)
    all_uncrossed = not any(snapshot.is_crossed for snapshot in result.snapshots)

    notes = [
        "Binance diff-depth updates are aggregated level updates, not individual "
        "order events; counts below are level-update counts.",
        "Diff-event quantities are validated as finite and non-negative at parse "
        "time, so invalid_quantity_count is enforced to be zero upstream.",
    ]
    if skipped_stale and result.n_snapshots == 0 and _started_correctly(result):
        notes.append(
            "All supplied diff events were stale relative to the snapshot; the book "
            "remained at the snapshot state."
        )

    return BinanceReplayQualityReport(
        event_count=len(events),
        applied_event_count=result.n_snapshots,
        skipped_stale_count=skipped_stale,
        gap_count=result.gap_count,
        crossed_count=result.crossed_count,
        not_initialised_count=not_initialised,
        invalid_quantity_count=0,
        started_correctly=_started_correctly(result),
        all_snapshots_uncrossed=all_uncrossed,
        final_update_id=result.final_update_id,
        final_bid_levels=final_bid_levels,
        final_ask_levels=final_ask_levels,
        issue_count=len(result.issues),
        issues_by_status=counts,
        notes=notes,
    )

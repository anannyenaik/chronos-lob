"""Supported event-level features for Binance-style L2 replay.

Features are computed from aggregated diff-depth level updates and the
reconstructed book snapshots they produce. Binance diff-depth data is
*aggregated level-update* data, not individual order messages: a positive
quantity upserts a price level and a zero quantity removes it. The module
therefore exposes ``added``/``removed`` depth imbalances and an aggregate
update-flow imbalance, and deliberately does **not** expose true trade
imbalance, individual cancellation attribution or queue-position features,
because diff-depth alone cannot support them.

Every feature row uses only information at or before its snapshot's update id;
the trailing event window ends at the snapshot's own applied diff, so there is
no look-ahead. No network access occurs anywhere in this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import pandas as pd

from chronoslob.data.binance import BinanceDiffDepthEvent
from chronoslob.data.schemas import OrderBookSnapshot

__all__ = [
    "BINANCE_FEATURE_COLUMNS",
    "UNSUPPORTED_FEATURES",
    "build_binance_feature_frame",
    "build_update_continuity_frame",
]

# Numeric, model-ready features supported from aggregated diff-depth replay.
BINANCE_FEATURE_COLUMNS: tuple[str, ...] = (
    "spread",
    "relative_spread",
    "mid_price",
    "microprice",
    "microprice_offset",
    "depth_imbalance_l1",
    "depth_imbalance_l5",
    "event_intensity",
    "update_count",
    "bid_update_imbalance",
    "added_depth_imbalance",
    "removed_depth_imbalance",
    "order_flow_update_imbalance",
)

# Features that diff-depth data alone cannot support, with the reason.
UNSUPPORTED_FEATURES: tuple[tuple[str, str], ...] = (
    (
        "trade_imbalance",
        "no trade stream is supplied; diff-depth carries only level updates.",
    ),
    (
        "true_cancellation_imbalance",
        "removed levels are aggregate deletions, not individual cancellations.",
    ),
    (
        "queue_position",
        "aggregated level updates expose no per-order queue position.",
    ),
)

_IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "update_id",
    "timestamp",
    "best_bid",
    "best_ask",
)


def _all_columns() -> list[str]:
    return [*_IDENTIFIER_COLUMNS, *BINANCE_FEATURE_COLUMNS]


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return float(numerator) / float(denominator)


def _event_aggregates(event: BinanceDiffDepthEvent) -> dict[str, float]:
    bid_updates = len(event.bids)
    ask_updates = len(event.asks)
    added_bid = sum(level.quantity for level in event.bids if level.quantity > 0.0)
    added_ask = sum(level.quantity for level in event.asks if level.quantity > 0.0)
    removed_bid = float(sum(1 for level in event.bids if level.quantity == 0.0))
    removed_ask = float(sum(1 for level in event.asks if level.quantity == 0.0))
    return {
        "bid_updates": float(bid_updates),
        "ask_updates": float(ask_updates),
        "update_count": float(bid_updates + ask_updates),
        "added_bid": float(added_bid),
        "added_ask": float(added_ask),
        "removed_bid": removed_bid,
        "removed_ask": removed_ask,
    }


def _book_features(snapshot: OrderBookSnapshot) -> dict[str, float]:
    best_bid = snapshot.best_bid
    best_ask = snapshot.best_ask
    if best_bid is None or best_ask is None:
        return {
            "spread": 0.0,
            "relative_spread": 0.0,
            "mid_price": float("nan"),
            "microprice": float("nan"),
            "microprice_offset": 0.0,
            "depth_imbalance_l1": 0.0,
            "depth_imbalance_l5": 0.0,
        }
    mid = (best_bid.price + best_ask.price) / 2.0
    spread = best_ask.price - best_bid.price
    bid_size = best_bid.quantity
    ask_size = best_ask.quantity
    size_total = bid_size + ask_size
    if size_total > 0.0:
        microprice = (best_bid.price * ask_size + best_ask.price * bid_size) / size_total
    else:
        microprice = mid
    bid_depth5 = float(sum(level.quantity for level in snapshot.bids[:5]))
    ask_depth5 = float(sum(level.quantity for level in snapshot.asks[:5]))
    return {
        "spread": spread,
        "relative_spread": _safe_ratio(spread, mid),
        "mid_price": mid,
        "microprice": microprice,
        "microprice_offset": microprice - mid,
        "depth_imbalance_l1": _safe_ratio(bid_size - ask_size, size_total),
        "depth_imbalance_l5": _safe_ratio(bid_depth5 - ask_depth5, bid_depth5 + ask_depth5),
    }


def _window_intensity(
    window_events: Sequence[BinanceDiffDepthEvent],
) -> float:
    """Return events per second over the window, falling back to event count."""
    count = len(window_events)
    if count == 0:
        return 0.0
    times: list[datetime] = []
    for event in window_events:
        stamp = event.transaction_time or event.event_time
        if stamp is not None:
            times.append(stamp)
    if len(times) >= 2:
        span = (max(times) - min(times)).total_seconds()
        if span > 0.0:
            return float(count) / span
    return float(count)


def _aligned_events(
    events: Sequence[BinanceDiffDepthEvent],
    snapshots: Sequence[OrderBookSnapshot],
) -> list[BinanceDiffDepthEvent]:
    """Return the diff events applied to produce ``snapshots``, in order.

    Each emitted snapshot carries ``sequence_id == event.final_update_id`` of
    the applied diff, so snapshots align to events by final update id. The scan
    is monotone to avoid accidentally matching a later stale duplicate update id.
    """
    aligned: list[BinanceDiffDepthEvent] = []
    cursor = 0
    for snapshot in snapshots:
        update_id = snapshot.sequence_id
        if update_id is None:
            continue
        while cursor < len(events):
            event = events[cursor]
            cursor += 1
            if event.final_update_id == update_id:
                aligned.append(event)
                break
    return aligned


def build_binance_feature_frame(
    events: Sequence[BinanceDiffDepthEvent],
    snapshots: Sequence[OrderBookSnapshot],
    *,
    window_events: int = 20,
) -> pd.DataFrame:
    """Build a past-only event-level feature frame aligned to ``snapshots``.

    Each row pairs a reconstructed snapshot with the aggregated diff event that
    produced it. Book-shape features come from the snapshot; event-flow features
    come from a trailing window of applied diff events ending at that snapshot.
    """
    if window_events < 1:
        raise ValueError("window_events must be >= 1")
    if not snapshots:
        return pd.DataFrame(columns=_all_columns())

    aligned = _aligned_events(events, snapshots)
    rows: list[dict[str, float | int | str]] = []
    for index, snapshot in enumerate(snapshots):
        if index >= len(aligned):
            break
        start = max(0, index + 1 - window_events)
        window = aligned[start : index + 1]
        aggregates = [_event_aggregates(event) for event in window]
        bid_updates = sum(item["bid_updates"] for item in aggregates)
        ask_updates = sum(item["ask_updates"] for item in aggregates)
        added_bid = sum(item["added_bid"] for item in aggregates)
        added_ask = sum(item["added_ask"] for item in aggregates)
        removed_bid = sum(item["removed_bid"] for item in aggregates)
        removed_ask = sum(item["removed_ask"] for item in aggregates)

        flow_numerator = (added_bid - added_ask) + (removed_ask - removed_bid)
        flow_denominator = added_bid + added_ask + removed_bid + removed_ask

        row: dict[str, float | int | str] = dict(_book_features(snapshot))
        row["event_intensity"] = _window_intensity(window)
        row["update_count"] = _event_aggregates(aligned[index])["update_count"]
        row["bid_update_imbalance"] = _safe_ratio(
            bid_updates - ask_updates, bid_updates + ask_updates
        )
        row["added_depth_imbalance"] = _safe_ratio(added_bid - added_ask, added_bid + added_ask)
        row["removed_depth_imbalance"] = _safe_ratio(
            removed_bid - removed_ask, removed_bid + removed_ask
        )
        row["order_flow_update_imbalance"] = _safe_ratio(flow_numerator, flow_denominator)

        best_bid = snapshot.best_bid
        best_ask = snapshot.best_ask
        row["update_id"] = int(snapshot.sequence_id) if snapshot.sequence_id is not None else -1
        row["timestamp"] = snapshot.timestamp.isoformat()
        row["best_bid"] = best_bid.price if best_bid is not None else float("nan")
        row["best_ask"] = best_ask.price if best_ask is not None else float("nan")
        rows.append(row)

    return pd.DataFrame(rows, columns=_all_columns())


def build_update_continuity_frame(
    events: Sequence[BinanceDiffDepthEvent],
) -> pd.DataFrame:
    """Build a per-event update-continuity audit table.

    For each diff event the table records its update-id triple and whether the
    previous final update id (``pu``) matches the prior event's final update id
    and whether final update ids increase strictly. This is a structural audit
    of the supplied stream, independent of snapshot bracketing.
    """
    columns = (
        "index",
        "first_update_id",
        "final_update_id",
        "previous_final_update_id",
        "prior_final_update_id",
        "pu_matches_prior_u",
        "u_strictly_increasing",
    )
    rows: list[dict[str, object]] = []
    prior_final: int | None = None
    for index, event in enumerate(events):
        pu = event.previous_final_update_id
        pu_matches = (
            None if pu is None or prior_final is None else bool(pu == prior_final)
        )
        increasing = (
            None if prior_final is None else bool(event.final_update_id > prior_final)
        )
        rows.append(
            {
                "index": index,
                "first_update_id": event.first_update_id,
                "final_update_id": event.final_update_id,
                "previous_final_update_id": "" if pu is None else pu,
                "prior_final_update_id": "" if prior_final is None else prior_final,
                "pu_matches_prior_u": "" if pu_matches is None else pu_matches,
                "u_strictly_increasing": "" if increasing is None else increasing,
            }
        )
        prior_final = event.final_update_id
    return pd.DataFrame(rows, columns=list(columns))

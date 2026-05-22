"""Passive-fill proxy labels.

These labels are intentionally simple research proxies. They use top-of-book
changes to ask whether a passive order *might* have filled; they do not model
queue position, order size, latency, partial fills or venue matching rules.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from chronoslob.data.schemas import OrderBookLevel, OrderBookSnapshot, Side

__all__ = [
    "compute_passive_fill_proxy",
    "compute_passive_fill_proxy_series",
]


def _validate_horizon(horizon: int) -> int:
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise TypeError("horizon must be an integer")
    if horizon <= 0:
        raise ValueError(f"horizon must be strictly positive; got {horizon!r}")
    return horizon


def _validate_index(index: int) -> int:
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("index must be an integer")
    if index < 0:
        raise ValueError(f"index must be non-negative; got {index!r}")
    return index


def _validate_missing(missing: str) -> str:
    if missing not in {"drop", "none"}:
        raise ValueError("missing must be one of {'drop', 'none'}")
    return missing


def _validate_side(side: Side) -> Side:
    try:
        resolved = Side(side)
    except ValueError as exc:
        raise ValueError("side must be Side.BID or Side.ASK") from exc
    if resolved not in {Side.BID, Side.ASK}:  # pragma: no cover - defensive
        raise ValueError("side must be Side.BID or Side.ASK")
    return resolved


def _is_timezone_aware(timestamp: datetime) -> bool:
    return timestamp.tzinfo is not None and timestamp.tzinfo.utcoffset(timestamp) is not None


def _validate_snapshots(snapshots: Sequence[OrderBookSnapshot]) -> list[OrderBookSnapshot]:
    if not snapshots:
        raise ValueError("snapshots must be non-empty")
    cleaned = list(snapshots)
    previous: datetime | None = None
    for position, snapshot in enumerate(cleaned):
        if not isinstance(snapshot, OrderBookSnapshot):
            raise TypeError(
                "snapshots must contain OrderBookSnapshot objects; "
                f"got {type(snapshot).__name__} at index {position}"
            )
        if not _is_timezone_aware(snapshot.timestamp):
            raise ValueError(f"snapshots[{position}].timestamp must be timezone-aware")
        if previous is not None and snapshot.timestamp < previous:
            raise ValueError("snapshots must be ordered by non-decreasing timestamp")
        if snapshot.best_bid is None or snapshot.best_ask is None:
            raise ValueError("top-of-book bid and ask levels are required")
        snapshot.assert_not_crossed()
        previous = snapshot.timestamp
    return cleaned


def _find_level(levels: Sequence[OrderBookLevel], price: float) -> OrderBookLevel | None:
    for level in levels:
        if level.price == price:
            return level
    return None


def _bid_fill_observed(
    snapshot: OrderBookSnapshot,
    original_price: float,
    original_quantity: float,
    *,
    require_price_touch: bool,
) -> bool:
    best_bid = snapshot.best_bid
    if best_bid is None:  # pragma: no cover - validated earlier
        raise ValueError("missing best bid")
    if best_bid.price > original_price:
        return True
    if require_price_touch:
        return best_bid.price == original_price and best_bid.quantity < original_quantity
    level = _find_level(snapshot.bids, original_price)
    return level is not None and level.quantity < original_quantity


def _ask_fill_observed(
    snapshot: OrderBookSnapshot,
    original_price: float,
    original_quantity: float,
    *,
    require_price_touch: bool,
) -> bool:
    best_ask = snapshot.best_ask
    if best_ask is None:  # pragma: no cover - validated earlier
        raise ValueError("missing best ask")
    if best_ask.price < original_price:
        return True
    if require_price_touch:
        return best_ask.price == original_price and best_ask.quantity < original_quantity
    level = _find_level(snapshot.asks, original_price)
    return level is not None and level.quantity < original_quantity


def compute_passive_fill_proxy(
    snapshots: Sequence[OrderBookSnapshot],
    index: int,
    horizon: int,
    side: Side,
    *,
    require_price_touch: bool = True,
) -> bool:
    """Return whether a passive top-of-book order might have filled.

    For a passive buy, the proxy fires if future best-bid quantity decreases
    at the original best-bid price, or if the best bid moves above that price.
    For a passive sell, it fires if future best-ask quantity decreases at the
    original best-ask price, or if the best ask moves below that price.
    """
    seq = _validate_snapshots(snapshots)
    idx = _validate_index(index)
    h = _validate_horizon(horizon)
    resolved_side = _validate_side(side)
    future_index = idx + h
    if idx >= len(seq):
        raise IndexError(f"index {idx} is outside snapshots of length {len(seq)}")
    if future_index >= len(seq):
        raise IndexError(
            "insufficient future data for requested horizon: "
            f"index {idx} + horizon {h} >= length {len(seq)}"
        )

    current = seq[idx]
    if resolved_side is Side.BID:
        best_bid = current.best_bid
        if best_bid is None:  # pragma: no cover - validated earlier
            raise ValueError("missing best bid")
        original_price = best_bid.price
        original_quantity = best_bid.quantity
        for future_snapshot in seq[idx + 1 : future_index + 1]:
            if _bid_fill_observed(
                future_snapshot,
                original_price,
                original_quantity,
                require_price_touch=require_price_touch,
            ):
                return True
        return False

    best_ask = current.best_ask
    if best_ask is None:  # pragma: no cover - validated earlier
        raise ValueError("missing best ask")
    original_price = best_ask.price
    original_quantity = best_ask.quantity
    for future_snapshot in seq[idx + 1 : future_index + 1]:
        if _ask_fill_observed(
            future_snapshot,
            original_price,
            original_quantity,
            require_price_touch=require_price_touch,
        ):
            return True
    return False


def compute_passive_fill_proxy_series(
    snapshots: Sequence[OrderBookSnapshot],
    horizon: int,
    side: Side,
    *,
    missing: str = "drop",
) -> list[bool | None]:
    """Return passive-fill proxy labels under an explicit missing policy."""
    seq = _validate_snapshots(snapshots)
    h = _validate_horizon(horizon)
    resolved_side = _validate_side(side)
    policy = _validate_missing(missing)
    output: list[bool | None] = []
    for idx in range(len(seq)):
        if idx + h >= len(seq):
            if policy == "none":
                output.append(None)
            continue
        output.append(
            compute_passive_fill_proxy(seq, idx, h, resolved_side)
        )
    return output

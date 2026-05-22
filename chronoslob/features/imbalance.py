"""Order book depth and imbalance features.

This module computes interpretable depth-based features from a single
:class:`~chronoslob.data.schemas.OrderBookSnapshot`. The features are
backward-looking by construction (a snapshot is a state at time ``t``)
and therefore safe to use as inputs at ``t``.

The implementations follow the conventions documented in
``reports/feature_engine.md``:

* ``depth_imbalance`` uses the canonical
  ``(bid_depth - ask_depth) / (bid_depth + ask_depth)`` form;
* zero denominators raise ``ValueError`` instead of silently returning 0
  so a single empty side of the book cannot masquerade as a balanced
  market.
* ``depth_slope`` is a deliberately simple linear-regression-like proxy
  on (distance, cumulative quantity). It is documented as such; it is
  not a calibrated liquidity-curve model.
"""

from __future__ import annotations

from chronoslob.data.schemas import (
    OrderBookLevel,
    OrderBookSnapshot,
    Side,
    is_finite_number,
)

__all__ = [
    "compute_depth",
    "compute_depth_imbalance",
    "compute_depth_slope",
    "compute_level_imbalances",
    "compute_liquidity_concentration",
    "compute_queue_imbalance",
]


def _validate_depth(depth: int | None) -> int | None:
    if depth is None:
        return None
    if isinstance(depth, bool) or not isinstance(depth, int):
        raise TypeError("depth must be an int or None")
    if depth <= 0:
        raise ValueError(f"depth must be strictly positive when provided; got {depth!r}")
    return depth


def compute_depth(
    levels: list[OrderBookLevel],
    depth: int | None = None,
) -> float:
    """Return the cumulative quantity over the first ``depth`` levels.

    If ``depth`` is ``None`` all available levels are summed. Levels are
    assumed to already be ordered by the schema (best price first); this
    function does not sort.
    """
    validated_depth = _validate_depth(depth)
    chosen = levels if validated_depth is None else levels[:validated_depth]
    total = 0.0
    for level in chosen:
        total += float(level.quantity)
    return total


def compute_depth_imbalance(
    bid_levels: list[OrderBookLevel],
    ask_levels: list[OrderBookLevel],
    depth: int | None = None,
) -> float:
    """Return ``(bid_depth - ask_depth) / (bid_depth + ask_depth)``.

    Raises ``ValueError`` when the denominator is zero. The caller may
    catch this and substitute a sentinel; we intentionally do not silently
    return 0.0 because that hides a structurally empty side of the book.
    """
    bid_depth = compute_depth(bid_levels, depth)
    ask_depth = compute_depth(ask_levels, depth)
    denominator = bid_depth + ask_depth
    if denominator <= 0.0:
        raise ValueError(
            "depth imbalance requires bid_depth + ask_depth > 0; "
            f"got bid_depth={bid_depth!r}, ask_depth={ask_depth!r}"
        )
    return (bid_depth - ask_depth) / denominator


def compute_queue_imbalance(
    best_bid_quantity: float,
    best_ask_quantity: float,
) -> float:
    """Return the top-of-book queue imbalance.

    Uses the same formula as :func:`compute_depth_imbalance` but applied
    only to the best-bid / best-ask quantities. Raises ``ValueError`` if
    both quantities are zero.
    """
    if not is_finite_number(best_bid_quantity):
        raise ValueError(
            f"best_bid_quantity must be a finite number; got {best_bid_quantity!r}"
        )
    if not is_finite_number(best_ask_quantity):
        raise ValueError(
            f"best_ask_quantity must be a finite number; got {best_ask_quantity!r}"
        )
    bid_qty = float(best_bid_quantity)
    ask_qty = float(best_ask_quantity)
    if bid_qty < 0.0 or ask_qty < 0.0:
        raise ValueError(
            "best_bid_quantity and best_ask_quantity must be non-negative; "
            f"got bid={bid_qty!r}, ask={ask_qty!r}"
        )
    denominator = bid_qty + ask_qty
    if denominator <= 0.0:
        raise ValueError(
            "queue imbalance requires best_bid_quantity + best_ask_quantity > 0; "
            f"got bid={bid_qty!r}, ask={ask_qty!r}"
        )
    return (bid_qty - ask_qty) / denominator


def compute_level_imbalances(
    snapshot: OrderBookSnapshot,
    depths: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Return per-depth bid/ask depths and depth imbalances plus queue imbalance.

    For each ``d`` in ``depths`` the returned dictionary contains:

    * ``bid_depth_d``
    * ``ask_depth_d``
    * ``depth_imbalance_d`` (only when ``bid_depth_d + ask_depth_d > 0``)

    The keys are named after the *requested* depth even when fewer
    levels are available on the snapshot. This makes it easier to assemble
    feature frames whose column names are stable across snapshots with
    varying numbers of levels. The ``queue_imbalance`` is always included
    when both top-of-book levels exist.
    """
    if not isinstance(snapshot, OrderBookSnapshot):
        raise TypeError("snapshot must be an OrderBookSnapshot")
    if not depths:
        raise ValueError("depths must contain at least one positive integer")
    for d in depths:
        _validate_depth(d)
        if d is None or d <= 0:  # pragma: no cover - guarded above
            raise ValueError(f"depths must be strictly positive; got {d!r}")

    out: dict[str, float] = {}
    for d in depths:
        bid_depth = compute_depth(snapshot.bids, d)
        ask_depth = compute_depth(snapshot.asks, d)
        out[f"bid_depth_{d}"] = bid_depth
        out[f"ask_depth_{d}"] = ask_depth
        denominator = bid_depth + ask_depth
        if denominator > 0.0:
            out[f"depth_imbalance_{d}"] = (bid_depth - ask_depth) / denominator

    best_bid = snapshot.best_bid
    best_ask = snapshot.best_ask
    if best_bid is not None and best_ask is not None:
        bid_qty = float(best_bid.quantity)
        ask_qty = float(best_ask.quantity)
        if bid_qty + ask_qty > 0.0:
            out["queue_imbalance"] = (bid_qty - ask_qty) / (bid_qty + ask_qty)
    return out


def _select_side_levels(
    snapshot: OrderBookSnapshot, side: Side
) -> list[OrderBookLevel]:
    if side is Side.BID:
        return snapshot.bids
    if side is Side.ASK:
        return snapshot.asks
    raise TypeError(f"side must be a Side enum; got {side!r}")


def compute_depth_slope(
    snapshot: OrderBookSnapshot,
    side: Side,
    depth: int | None = None,
) -> float:
    """Return a simple liquidity-slope proxy for one side of the book.

    The slope is the ordinary-least-squares regression coefficient of
    cumulative quantity against absolute distance from the best price
    on the given side. A larger value indicates that liquidity grows
    quickly with distance from the touch.

    This is intentionally a simple, auditable approximation; it is
    *not* a calibrated liquidity-impact curve. The function requires at
    least two usable levels and raises ``ValueError`` otherwise.
    """
    levels = _select_side_levels(snapshot, side)
    validated_depth = _validate_depth(depth)
    if validated_depth is not None:
        levels = levels[:validated_depth]
    if len(levels) < 2:
        raise ValueError(
            "depth slope requires at least 2 levels on the requested side; "
            f"got {len(levels)}"
        )
    best_price = float(levels[0].price)
    distances: list[float] = []
    cumulative: list[float] = []
    running_total = 0.0
    for level in levels:
        running_total += float(level.quantity)
        distances.append(abs(float(level.price) - best_price))
        cumulative.append(running_total)

    n = float(len(distances))
    mean_x = sum(distances) / n
    mean_y = sum(cumulative) / n
    numerator = sum(
        (x - mean_x) * (y - mean_y)
        for x, y in zip(distances, cumulative, strict=True)
    )
    denominator = sum((x - mean_x) ** 2 for x in distances)
    if denominator == 0.0:
        raise ValueError(
            "depth slope is undefined when all level distances coincide"
        )
    return numerator / denominator


def compute_liquidity_concentration(
    snapshot: OrderBookSnapshot,
    side: Side,
    top_n: int = 3,
) -> float:
    """Return the share of side-quantity held in the top ``top_n`` levels.

    Formula: ``sum(top_n quantities) / sum(all quantities on that side)``.
    Requires ``top_n >= 1`` and a strictly positive total quantity on the
    requested side; otherwise raises ``ValueError``.
    """
    if isinstance(top_n, bool) or not isinstance(top_n, int):
        raise TypeError("top_n must be an int")
    if top_n <= 0:
        raise ValueError(f"top_n must be strictly positive; got {top_n!r}")

    levels = _select_side_levels(snapshot, side)
    if not levels:
        raise ValueError(
            f"liquidity concentration requires at least one level on side {side.value}"
        )
    total = sum(float(level.quantity) for level in levels)
    if total <= 0.0:
        raise ValueError(
            "liquidity concentration requires a strictly positive total quantity; "
            f"got {total!r}"
        )
    top = sum(float(level.quantity) for level in levels[:top_n])
    return top / total

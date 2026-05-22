"""Order-flow imbalance (OFI) features from snapshot deltas and trades.

This module implements a deliberately simple, top-of-book OFI
approximation built from consecutive
:class:`~chronoslob.data.schemas.OrderBookSnapshot` pairs, and a
top-level trade-imbalance helper for sequences of trade
:class:`~chronoslob.data.schemas.BookEvent` records.

The OFI approximation here is a *first-order* signal:

* For the bid side the contribution is:
    - ``+current_bid_qty`` when the best bid price strictly improves;
    - ``current_bid_qty - previous_bid_qty`` when the best bid price is
      unchanged;
    - ``-previous_bid_qty`` when the best bid price worsens.
* For the ask side the contribution is:
    - ``-current_ask_qty`` when the best ask price strictly improves
      (moves down);
    - ``-(current_ask_qty - previous_ask_qty)`` when the best ask price is
      unchanged;
    - ``+previous_ask_qty`` when the best ask price worsens (moves up).
* ``OFI = bid_contribution + ask_contribution``.

This is *not* a full event-level OFI reconstruction. It is an
interpretable proxy that summarises the change at the touch between
consecutive snapshots. Limitations are documented in
``reports/feature_engine.md``.

For trades, this module assumes the *upstream* side convention: a
``BookEvent`` with ``side == Side.BID`` is treated as a buy (i.e. resting
on the bid is being lifted by an aggressor on the buy side)... see the
function docstring for the exact convention. We do not infer the
aggressor side from price.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise

from chronoslob.data.schemas import BookEvent, EventType, OrderBookSnapshot, Side

__all__ = [
    "compute_order_flow_imbalance_from_snapshots",
    "compute_order_flow_imbalance_series",
    "compute_trade_imbalance_from_events",
]


def _check_snapshot_has_top(snapshot: OrderBookSnapshot, name: str) -> None:
    if snapshot.best_bid is None or snapshot.best_ask is None:
        raise ValueError(
            f"{name} snapshot must have at least one bid and one ask level "
            "to compute order-flow imbalance"
        )


def compute_order_flow_imbalance_from_snapshots(
    previous: OrderBookSnapshot,
    current: OrderBookSnapshot,
    *,
    allow_crossed: bool = False,
) -> float:
    """Return a simple top-of-book OFI between two consecutive snapshots.

    ``current.timestamp`` must not be before ``previous.timestamp`` and
    both snapshots must have a top-of-book on each side. Crossed
    snapshots are rejected unless ``allow_crossed`` is set.

    See the module docstring for the exact contribution rules. The
    returned scalar shares its sign with conventional OFI definitions:
    a positive value indicates net inflow on the bid (buyers more
    aggressive) and a negative value indicates net inflow on the ask
    (sellers more aggressive).
    """
    if not isinstance(previous, OrderBookSnapshot):
        raise TypeError("previous must be an OrderBookSnapshot")
    if not isinstance(current, OrderBookSnapshot):
        raise TypeError("current must be an OrderBookSnapshot")
    _check_snapshot_has_top(previous, "previous")
    _check_snapshot_has_top(current, "current")
    if current.timestamp < previous.timestamp:
        raise ValueError(
            "current.timestamp must not be before previous.timestamp; "
            f"got previous={previous.timestamp!r}, current={current.timestamp!r}"
        )
    if not allow_crossed:
        previous.assert_not_crossed()
        current.assert_not_crossed()

    prev_bid_price = float(previous.best_bid.price)  # type: ignore[union-attr]
    prev_bid_qty = float(previous.best_bid.quantity)  # type: ignore[union-attr]
    prev_ask_price = float(previous.best_ask.price)  # type: ignore[union-attr]
    prev_ask_qty = float(previous.best_ask.quantity)  # type: ignore[union-attr]
    cur_bid_price = float(current.best_bid.price)  # type: ignore[union-attr]
    cur_bid_qty = float(current.best_bid.quantity)  # type: ignore[union-attr]
    cur_ask_price = float(current.best_ask.price)  # type: ignore[union-attr]
    cur_ask_qty = float(current.best_ask.quantity)  # type: ignore[union-attr]

    if cur_bid_price > prev_bid_price:
        bid_contribution = cur_bid_qty
    elif cur_bid_price == prev_bid_price:
        bid_contribution = cur_bid_qty - prev_bid_qty
    else:
        bid_contribution = -prev_bid_qty

    if cur_ask_price < prev_ask_price:
        ask_contribution = -cur_ask_qty
    elif cur_ask_price == prev_ask_price:
        ask_contribution = -(cur_ask_qty - prev_ask_qty)
    else:
        ask_contribution = prev_ask_qty

    return bid_contribution + ask_contribution


def compute_order_flow_imbalance_series(
    snapshots: Sequence[OrderBookSnapshot],
    *,
    first_value_is_nan: bool = False,
    allow_crossed: bool = False,
) -> list[float]:
    """Return per-snapshot OFI values aligned to ``snapshots``.

    The output has the same length as ``snapshots``. The first value has
    no previous snapshot to compare to and defaults to ``0.0``; pass
    ``first_value_is_nan=True`` to receive ``float('nan')`` instead.

    Timestamps must be non-decreasing across ``snapshots``. Crossed
    snapshots are rejected unless ``allow_crossed`` is set.
    """
    seq = list(snapshots)
    if not seq:
        return []
    for i in range(1, len(seq)):
        if seq[i].timestamp < seq[i - 1].timestamp:
            raise ValueError(
                "snapshots must be ordered by non-decreasing timestamp; "
                f"index {i} ({seq[i].timestamp!r}) precedes "
                f"index {i - 1} ({seq[i - 1].timestamp!r})"
            )
    out: list[float] = []
    out.append(math.nan if first_value_is_nan else 0.0)
    for previous, current in pairwise(seq):
        out.append(
            compute_order_flow_imbalance_from_snapshots(
                previous, current, allow_crossed=allow_crossed
            )
        )
    return out


def compute_trade_imbalance_from_events(
    events: Sequence[BookEvent],
    *,
    allow_empty: bool = False,
) -> float:
    """Return ``(buy_qty - sell_qty) / (buy_qty + sell_qty)`` over trade events.

    Convention: a ``BookEvent`` with ``event_type == TRADE`` and
    ``side == Side.BID`` is treated as a buyer-initiated trade (the
    aggressor lifted the offer / consumed the resting bid is treated as
    a *passive bid* fill). Implementations differ across venues, so this
    module documents and uses the simple convention "side is the
    aggressor side": ``BID`` -> buyer-initiated, ``ASK`` -> seller-initiated.
    Events with no side are ignored. Non-trade events are ignored.

    Raises ``ValueError`` if no usable trade quantity is found, unless
    ``allow_empty=True`` in which case ``float('nan')`` is returned.
    """
    buy_qty = 0.0
    sell_qty = 0.0
    for event in events:
        if not isinstance(event, BookEvent):
            raise TypeError(
                f"events must contain BookEvent records; got {type(event).__name__}"
            )
        if event.event_type is not EventType.TRADE:
            continue
        if event.side is None:
            continue
        if event.quantity is None:
            continue
        quantity = float(event.quantity)
        if quantity <= 0.0:
            continue
        if event.side is Side.BID:
            buy_qty += quantity
        elif event.side is Side.ASK:
            sell_qty += quantity
    denominator = buy_qty + sell_qty
    if denominator <= 0.0:
        if allow_empty:
            return math.nan
        raise ValueError(
            "trade imbalance requires at least one usable trade quantity; "
            "got none"
        )
    return (buy_qty - sell_qty) / denominator

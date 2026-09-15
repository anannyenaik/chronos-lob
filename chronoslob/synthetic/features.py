"""Genuine event-level features for synthetic order-book streams.

These features require event messages (submissions, cancellations and trades)
and therefore *cannot* be computed from FI-2010 snapshots, which only expose
snapshot proxies. They are valid only on synthetic event streams here; they make
no claim about real markets.

Every feature row uses only information at or before its snapshot timestamp: the
trailing event window ends at the snapshot's own event, so there is no
look-ahead into future events.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from chronoslob.data.schemas import BookEvent, EventType, OrderBookSnapshot, Side

__all__ = [
    "EVENT_FEATURE_COLUMNS",
    "build_event_feature_frame",
]

# Numeric, model-ready event-level feature columns. Identifier and ground-truth
# state columns (sequence_id, timestamp, regime_id, regime_name, mid_price) are
# emitted alongside but are not part of this feature list.
EVENT_FEATURE_COLUMNS: tuple[str, ...] = (
    "event_intensity",
    "add_rate",
    "cancel_rate",
    "trade_rate",
    "event_order_flow_imbalance",
    "cancellation_imbalance",
    "trade_imbalance",
    "spread",
    "relative_spread",
    "microprice_offset",
    "depth_imbalance_l1",
    "depth_imbalance_l5",
    "realised_volatility_proxy",
)


def build_event_feature_frame(
    events: Sequence[BookEvent],
    snapshots: Sequence[OrderBookSnapshot],
    *,
    window_events: int = 50,
) -> pd.DataFrame:
    """Build a past-only event-level feature frame aligned to ``snapshots``.

    For each snapshot the trailing window of ``window_events`` events (ending at
    the snapshot's own sequence id) drives the event-flow features, while the
    snapshot itself drives the book-shape features.
    """
    if window_events < 1:
        raise ValueError("window_events must be >= 1")
    if not snapshots:
        return pd.DataFrame(columns=_all_columns())

    arrays = _event_arrays(events)
    rows: list[dict[str, float | int | str]] = []
    for snapshot in snapshots:
        sequence_id = snapshot.sequence_id
        if sequence_id is None:
            continue
        end = sequence_id + 1
        start = max(0, end - window_events)
        row = _window_features(arrays, start, end)
        row.update(_book_features(snapshot))
        row["sequence_id"] = int(sequence_id)
        row["timestamp"] = snapshot.timestamp.isoformat()
        row["regime_id"] = int(snapshot.metadata.get("regime_id", -1))
        row["regime_name"] = str(snapshot.metadata.get("regime_name", "unknown"))
        rows.append(row)

    frame = pd.DataFrame(rows, columns=_all_columns())
    return frame


def _all_columns() -> list[str]:
    return [
        "sequence_id",
        "timestamp",
        "regime_id",
        "regime_name",
        "mid_price",
        *EVENT_FEATURE_COLUMNS,
    ]


def _event_arrays(events: Sequence[BookEvent]) -> dict[str, np.ndarray]:
    count = len(events)
    is_add = np.zeros(count, dtype=bool)
    is_cancel = np.zeros(count, dtype=bool)
    is_trade = np.zeros(count, dtype=bool)
    is_bid = np.zeros(count, dtype=bool)
    is_buy_trade = np.zeros(count, dtype=bool)
    quantity = np.zeros(count, dtype=float)
    latent_mid = np.zeros(count, dtype=float)
    for index, event in enumerate(events):
        quantity[index] = event.quantity or 0.0
        latent_mid[index] = float(event.metadata.get("latent_mid", 0.0))
        is_bid[index] = event.side is Side.BID
        if event.event_type is EventType.ADD:
            is_add[index] = True
        elif event.event_type is EventType.CANCEL:
            is_cancel[index] = True
        elif event.event_type is EventType.TRADE:
            is_trade[index] = True
            is_buy_trade[index] = str(event.metadata.get("aggressor_side", "")) == "BID"
    return {
        "is_add": is_add,
        "is_cancel": is_cancel,
        "is_trade": is_trade,
        "is_bid": is_bid,
        "is_buy_trade": is_buy_trade,
        "quantity": quantity,
        "latent_mid": latent_mid,
    }


def _window_features(
    arrays: dict[str, np.ndarray],
    start: int,
    end: int,
) -> dict[str, float | int | str]:
    is_add = arrays["is_add"][start:end]
    is_cancel = arrays["is_cancel"][start:end]
    is_trade = arrays["is_trade"][start:end]
    is_bid = arrays["is_bid"][start:end]
    is_buy_trade = arrays["is_buy_trade"][start:end]
    quantity = arrays["quantity"][start:end]
    latent_mid = arrays["latent_mid"][start:end]

    window = max(1, end - start)
    add_count = int(is_add.sum())
    cancel_count = int(is_cancel.sum())
    trade_count = int(is_trade.sum())
    total = add_count + cancel_count + trade_count

    add_bid = float(quantity[is_add & is_bid].sum())
    add_ask = float(quantity[is_add & ~is_bid].sum())
    cancel_bid = float(quantity[is_cancel & is_bid].sum())
    cancel_ask = float(quantity[is_cancel & ~is_bid].sum())
    buy_trade = float(quantity[is_trade & is_buy_trade].sum())
    sell_trade = float(quantity[is_trade & ~is_buy_trade].sum())

    # Net order-flow imbalance: liquidity added on the bid and cancelled on the
    # ask both push the imbalance towards buying pressure.
    flow_numerator = (add_bid - add_ask) + (cancel_ask - cancel_bid)
    flow_denominator = add_bid + add_ask + cancel_bid + cancel_ask

    realised_vol = float(np.std(np.diff(latent_mid))) if latent_mid.size >= 2 else 0.0

    return {
        "event_intensity": float(total) / float(window),
        "add_rate": _ratio(add_count, total),
        "cancel_rate": _ratio(cancel_count, total),
        "trade_rate": _ratio(trade_count, total),
        "event_order_flow_imbalance": _safe_ratio(flow_numerator, flow_denominator),
        "cancellation_imbalance": _safe_ratio(cancel_bid - cancel_ask, cancel_bid + cancel_ask),
        "trade_imbalance": _safe_ratio(buy_trade - sell_trade, buy_trade + sell_trade),
        "realised_volatility_proxy": realised_vol,
    }


def _book_features(snapshot: OrderBookSnapshot) -> dict[str, float | int | str]:
    best_bid = snapshot.best_bid
    best_ask = snapshot.best_ask
    if best_bid is None or best_ask is None:
        return {
            "mid_price": float("nan"),
            "spread": 0.0,
            "relative_spread": 0.0,
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
        "mid_price": mid,
        "spread": spread,
        "relative_spread": _safe_ratio(spread, mid),
        "microprice_offset": microprice - mid,
        "depth_imbalance_l1": _safe_ratio(bid_size - ask_size, size_total),
        "depth_imbalance_l5": _safe_ratio(bid_depth5 - ask_depth5, bid_depth5 + ask_depth5),
    }


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return float(numerator) / float(denominator)

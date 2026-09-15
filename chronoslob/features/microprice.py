"""Top-of-book price features.

This module implements the canonical top-of-book price features used as
building blocks throughout the feature engine:

* mid-price
* spread
* relative spread
* microprice
* a small dictionary of summary features for an entire snapshot

Each function operates on inputs that are available *at* the snapshot
timestamp and therefore carries no look-ahead information. Inputs are
validated explicitly so that downstream pipeline stages can rely on the
returned values being finite real numbers (no NaN, no inf, no negatives
where they would not make sense).
"""

from __future__ import annotations

import math

from chronoslob.data.schemas import OrderBookSnapshot, is_finite_number

__all__ = [
    "compute_microprice",
    "compute_mid_price",
    "compute_relative_spread",
    "compute_snapshot_price_features",
    "compute_spread",
]


def _validate_price(value: float, *, name: str) -> float:
    """Return ``value`` as a strictly positive finite float or raise."""
    if not is_finite_number(value):
        raise ValueError(f"{name} must be a finite number; got {value!r}")
    numeric = float(value)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be strictly positive; got {numeric!r}")
    return numeric


def _validate_quantity(value: float, *, name: str) -> float:
    """Return ``value`` as a non-negative finite float or raise."""
    if not is_finite_number(value):
        raise ValueError(f"{name} must be a finite number; got {value!r}")
    numeric = float(value)
    if numeric < 0.0:
        raise ValueError(f"{name} must be non-negative; got {numeric!r}")
    return numeric


def compute_mid_price(
    best_bid: float,
    best_ask: float,
    *,
    allow_crossed: bool = False,
) -> float:
    """Return the arithmetic mid-price ``(best_bid + best_ask) / 2``.

    Both prices must be finite and strictly positive. The bid must be
    strictly below the ask unless ``allow_crossed`` is set, in which case
    crossed top-of-book pairs are still computed but not silently fixed.
    """
    bid = _validate_price(best_bid, name="best_bid")
    ask = _validate_price(best_ask, name="best_ask")
    if not allow_crossed and bid >= ask:
        raise ValueError(
            "best_bid must be strictly less than best_ask; "
            f"got bid={bid!r}, ask={ask!r}"
        )
    return (bid + ask) / 2.0


def compute_spread(
    best_bid: float,
    best_ask: float,
    *,
    allow_crossed: bool = False,
) -> float:
    """Return the absolute spread ``best_ask - best_bid``.

    The spread may be zero or negative only when ``allow_crossed=True``
    explicitly opts into a crossed book. Otherwise the inputs must satisfy
    ``best_bid < best_ask``.
    """
    bid = _validate_price(best_bid, name="best_bid")
    ask = _validate_price(best_ask, name="best_ask")
    if not allow_crossed and bid >= ask:
        raise ValueError(
            "best_bid must be strictly less than best_ask; "
            f"got bid={bid!r}, ask={ask!r}"
        )
    return ask - bid


def compute_relative_spread(
    best_bid: float,
    best_ask: float,
    *,
    allow_crossed: bool = False,
) -> float:
    """Return ``spread / mid_price``.

    Both prices must be strictly positive. The mid-price is always strictly
    positive for valid bid/ask pairs, so the ratio is well defined.
    """
    spread = compute_spread(best_bid, best_ask, allow_crossed=allow_crossed)
    mid = compute_mid_price(best_bid, best_ask, allow_crossed=allow_crossed)
    return spread / mid


def compute_microprice(
    best_bid: float,
    best_ask: float,
    bid_quantity: float,
    ask_quantity: float,
    *,
    allow_crossed: bool = False,
) -> float:
    """Return the size-weighted microprice.

    The microprice is the quantity-weighted average that tilts toward the
    side with the larger opposing quantity:

        microprice = (best_ask * bid_qty + best_bid * ask_qty)
                     / (bid_qty + ask_qty)

    Quantities must be finite and non-negative, and the denominator must
    be strictly positive (raises ``ValueError`` if both quantities are
    zero).
    """
    bid = _validate_price(best_bid, name="best_bid")
    ask = _validate_price(best_ask, name="best_ask")
    if not allow_crossed and bid >= ask:
        raise ValueError(
            "best_bid must be strictly less than best_ask; "
            f"got bid={bid!r}, ask={ask!r}"
        )
    bid_qty = _validate_quantity(bid_quantity, name="bid_quantity")
    ask_qty = _validate_quantity(ask_quantity, name="ask_quantity")
    denominator = bid_qty + ask_qty
    if denominator <= 0.0:
        raise ValueError(
            "microprice requires bid_quantity + ask_quantity > 0; "
            f"got bid_qty={bid_qty!r}, ask_qty={ask_qty!r}"
        )
    return (ask * bid_qty + bid * ask_qty) / denominator


def compute_snapshot_price_features(
    snapshot: OrderBookSnapshot,
    *,
    allow_crossed: bool = False,
) -> dict[str, float]:
    """Return a dictionary of top-of-book price features for ``snapshot``.

    The returned mapping always contains finite ``float`` values. Keys:

    * ``mid_price``
    * ``spread``
    * ``relative_spread``
    * ``best_bid_price``
    * ``best_ask_price``
    * ``best_bid_quantity``
    * ``best_ask_quantity``
    * ``microprice`` (only when the top quantities sum to a positive value)

    Raises ``ValueError`` if either side of the book is empty. Crossed
    books raise unless ``allow_crossed=True``; crossed books are never
    silently re-ordered.
    """
    if not isinstance(snapshot, OrderBookSnapshot):
        raise TypeError("snapshot must be an OrderBookSnapshot")
    best_bid = snapshot.best_bid
    best_ask = snapshot.best_ask
    if best_bid is None or best_ask is None:
        raise ValueError(
            "snapshot must have at least one bid and one ask level to "
            "compute price features"
        )

    bid_price = float(best_bid.price)
    ask_price = float(best_ask.price)
    bid_qty = float(best_bid.quantity)
    ask_qty = float(best_ask.quantity)

    features: dict[str, float] = {
        "best_bid_price": bid_price,
        "best_ask_price": ask_price,
        "best_bid_quantity": bid_qty,
        "best_ask_quantity": ask_qty,
        "mid_price": compute_mid_price(
            bid_price, ask_price, allow_crossed=allow_crossed
        ),
        "spread": compute_spread(
            bid_price, ask_price, allow_crossed=allow_crossed
        ),
        "relative_spread": compute_relative_spread(
            bid_price, ask_price, allow_crossed=allow_crossed
        ),
    }
    if bid_qty + ask_qty > 0.0 and math.isfinite(bid_qty + ask_qty):
        features["microprice"] = compute_microprice(
            bid_price,
            ask_price,
            bid_qty,
            ask_qty,
            allow_crossed=allow_crossed,
        )
    return features

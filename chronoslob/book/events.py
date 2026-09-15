"""Pure helpers for working with order book levels and snapshots.

These helpers operate on the typed schemas defined in
:mod:`chronoslob.data.schemas` and are intentionally side-effect-free so that
they can be reused by future reconstruction, feature-engineering and
execution-simulation phases without importing heavier modules.
"""

from __future__ import annotations

from itertools import pairwise

from chronoslob.data.schemas import OrderBookLevel, OrderBookSnapshot, Side

__all__ = [
    "has_duplicate_prices",
    "sort_levels_for_side",
    "top_of_book",
    "validate_book_side_order",
]


def sort_levels_for_side(
    levels: list[OrderBookLevel],
    side: Side,
) -> list[OrderBookLevel]:
    """Return a new list of levels sorted by price for the given ``side``.

    ``BID`` levels are sorted in descending price order so that the best
    (highest) bid is first. ``ASK`` levels are sorted in ascending price
    order so that the best (lowest) ask is first. The input list is not
    mutated.
    """
    if not isinstance(side, Side):
        raise TypeError(f"side must be Side, got {type(side).__name__}")
    descending = side is Side.BID
    return sorted(levels, key=lambda level: level.price, reverse=descending)


def validate_book_side_order(
    levels: list[OrderBookLevel],
    side: Side,
) -> None:
    """Raise ``ValueError`` if ``levels`` are not correctly ordered for ``side``.

    ``BID`` levels must be strictly descending by price; ``ASK`` levels must
    be strictly ascending. Lists of zero or one level are considered valid.
    """
    if not isinstance(side, Side):
        raise TypeError(f"side must be Side, got {type(side).__name__}")
    if len(levels) < 2:
        return
    if side is Side.BID:
        for previous, current in pairwise(levels):
            if current.price >= previous.price:
                raise ValueError(
                    "bid levels must be strictly descending by price; "
                    f"got {previous.price!r} followed by {current.price!r}"
                )
    else:
        for previous, current in pairwise(levels):
            if current.price <= previous.price:
                raise ValueError(
                    "ask levels must be strictly ascending by price; "
                    f"got {previous.price!r} followed by {current.price!r}"
                )


def has_duplicate_prices(levels: list[OrderBookLevel]) -> bool:
    """Return ``True`` if any two levels share the same price."""
    seen: set[float] = set()
    for level in levels:
        if level.price in seen:
            return True
        seen.add(level.price)
    return False


def top_of_book(
    snapshot: OrderBookSnapshot,
) -> tuple[OrderBookLevel | None, OrderBookLevel | None]:
    """Return ``(best_bid, best_ask)`` for ``snapshot``.

    Either or both elements may be ``None`` if the corresponding side of the
    book is empty.
    """
    return snapshot.best_bid, snapshot.best_ask

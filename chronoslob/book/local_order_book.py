"""Local order book state management.

This module provides a small, in-memory limit order book that can be
loaded from a Binance-style depth snapshot and updated with subsequent
diff-depth events. It is deliberately ignorant of the Binance
update-id continuity rules — full reconstruction continuity lives in
:mod:`chronoslob.book.reconstruction`.

The class is intended for offline replay of locally captured fixtures
and never makes any network calls.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

from chronoslob.data.schemas import (
    MetadataDict,
    OrderBookLevel,
    OrderBookSnapshot,
    Side,
    ensure_utc_datetime,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from chronoslob.data.binance import (
        BinanceDepthSnapshot,
        BinanceDiffDepthEvent,
    )

__all__ = [
    "LocalOrderBook",
    "LocalOrderBookConfig",
]


class LocalOrderBookConfig(BaseModel):
    """Configuration for :class:`LocalOrderBook`."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    symbol: str = Field(..., min_length=1)
    venue: str = "binance"
    max_depth: int | None = None
    allow_crossed: bool = False

    @field_validator("symbol", "venue")
    @classmethod
    def _validate_strings(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("string fields must be non-empty")
        return value

    @field_validator("max_depth")
    @classmethod
    def _validate_max_depth(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("max_depth must be a positive int or None")
        if value <= 0:
            raise ValueError(f"max_depth must be positive; got {value!r}")
        return value


class LocalOrderBook:
    """In-memory limit order book replayed from local fixtures.

    The book stores resting quantity per price level on each side. A
    quantity of ``0`` in a diff event removes the price level; a
    strictly positive quantity sets or replaces the level. After every
    update the book is trimmed to ``config.max_depth`` if configured and
    its crossed state is validated unless explicitly allowed.
    """

    def __init__(self, config: LocalOrderBookConfig) -> None:
        self._config = config
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}
        self._last_update_id: int | None = None
        self._timestamp: datetime | None = None
        self._initialised: bool = False

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> LocalOrderBookConfig:
        """Return the immutable configuration."""
        return self._config

    @property
    def symbol(self) -> str:
        """Return the configured symbol."""
        return self._config.symbol

    @property
    def venue(self) -> str:
        """Return the configured venue."""
        return self._config.venue

    @property
    def last_update_id(self) -> int | None:
        """Return the most recently applied update id, if any."""
        return self._last_update_id

    @property
    def timestamp(self) -> datetime | None:
        """Return the most recently applied event timestamp, if any."""
        return self._timestamp

    @property
    def is_initialised(self) -> bool:
        """Return ``True`` once a snapshot has been loaded."""
        return self._initialised

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all internal state."""
        self._bids.clear()
        self._asks.clear()
        self._last_update_id = None
        self._timestamp = None
        self._initialised = False

    def load_snapshot(self, snapshot: BinanceDepthSnapshot) -> None:
        """Replace internal state from a Binance-style depth snapshot."""
        if snapshot.symbol != self._config.symbol:
            raise ValueError(
                "snapshot symbol does not match configured symbol; "
                f"snapshot={snapshot.symbol!r}, config={self._config.symbol!r}"
            )
        self._bids = {level.price: level.quantity for level in snapshot.bids}
        self._asks = {level.price: level.quantity for level in snapshot.asks}
        self._last_update_id = snapshot.last_update_id
        self._timestamp = (
            snapshot.timestamp
            if snapshot.timestamp is not None
            else snapshot.received_timestamp
        )
        self._initialised = True
        self._trim_to_max_depth()
        if not self._config.allow_crossed:
            self.assert_not_crossed()

    def apply_diff(self, event: BinanceDiffDepthEvent) -> None:
        """Apply a Binance-style diff event to the in-memory book."""
        if not self._initialised:
            raise RuntimeError(
                "cannot apply diff event before load_snapshot has been called"
            )
        if event.symbol != self._config.symbol:
            raise ValueError(
                "event symbol does not match configured symbol; "
                f"event={event.symbol!r}, config={self._config.symbol!r}"
            )

        for level in event.bids:
            if level.quantity == 0.0:
                self._bids.pop(level.price, None)
            else:
                self._bids[level.price] = level.quantity
        for level in event.asks:
            if level.quantity == 0.0:
                self._asks.pop(level.price, None)
            else:
                self._asks[level.price] = level.quantity

        self._last_update_id = event.final_update_id
        new_timestamp = event.transaction_time or event.event_time
        if new_timestamp is not None:
            self._timestamp = ensure_utc_datetime(new_timestamp)
        elif event.received_timestamp is not None:
            self._timestamp = ensure_utc_datetime(event.received_timestamp)

        self._trim_to_max_depth()

        if not self._config.allow_crossed:
            self.assert_not_crossed()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def best_bid(self) -> OrderBookLevel | None:
        """Return the highest bid level, or ``None`` if no bids are stored."""
        if not self._bids:
            return None
        price = max(self._bids)
        return OrderBookLevel(price=price, quantity=self._bids[price])

    def best_ask(self) -> OrderBookLevel | None:
        """Return the lowest ask level, or ``None`` if no asks are stored."""
        if not self._asks:
            return None
        price = min(self._asks)
        return OrderBookLevel(price=price, quantity=self._asks[price])

    def levels(
        self, side: Side, depth: int | None = None
    ) -> list[OrderBookLevel]:
        """Return sorted price levels for ``side`` optionally trimmed to ``depth``."""
        if not isinstance(side, Side):
            raise TypeError(f"side must be Side, got {type(side).__name__}")
        if depth is not None:
            if isinstance(depth, bool) or not isinstance(depth, int):
                raise TypeError("depth must be a positive int or None")
            if depth <= 0:
                raise ValueError(f"depth must be positive; got {depth!r}")
        source = self._bids if side is Side.BID else self._asks
        sorted_prices = sorted(source, reverse=side is Side.BID)
        if depth is not None:
            sorted_prices = sorted_prices[:depth]
        return [
            OrderBookLevel(price=price, quantity=source[price])
            for price in sorted_prices
        ]

    def depth_counts(self) -> dict[str, int]:
        """Return the number of stored price levels per side."""
        return {"bids": len(self._bids), "asks": len(self._asks)}

    def is_crossed(self) -> bool:
        """Return ``True`` if the best bid is at or above the best ask."""
        if not self._bids or not self._asks:
            return False
        return max(self._bids) >= min(self._asks)

    def assert_not_crossed(self) -> None:
        """Raise ``ValueError`` if the book is crossed."""
        if self.is_crossed():
            best_bid_price = max(self._bids)
            best_ask_price = min(self._asks)
            raise ValueError(
                "local order book is crossed; "
                f"best_bid={best_bid_price!r}, best_ask={best_ask_price!r}"
            )

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_snapshot(
        self,
        timestamp: datetime | None = None,
        *,
        extra_metadata: MetadataDict | None = None,
    ) -> OrderBookSnapshot:
        """Return a canonical :class:`OrderBookSnapshot` view of the book.

        ``timestamp`` overrides the book's current timestamp when supplied.
        If neither is available, a ``RuntimeError`` is raised because
        canonical snapshots require a timezone-aware timestamp.
        """
        resolved_timestamp = (
            ensure_utc_datetime(timestamp) if timestamp is not None else self._timestamp
        )
        if resolved_timestamp is None:
            raise RuntimeError(
                "cannot build OrderBookSnapshot without a timestamp; "
                "supply one or load events that carry timestamps"
            )

        bids = self.levels(Side.BID)
        asks = self.levels(Side.ASK)
        metadata: MetadataDict = {"source": "binance_local_book"}
        if extra_metadata:
            for key, value in extra_metadata.items():
                metadata[key] = value

        return OrderBookSnapshot(
            timestamp=resolved_timestamp,
            symbol=self._config.symbol,
            venue=self._config.venue,
            bids=bids,
            asks=asks,
            sequence_id=self._last_update_id,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trim_to_max_depth(self) -> None:
        max_depth = self._config.max_depth
        if max_depth is None:
            return
        if len(self._bids) > max_depth:
            top_prices = sorted(self._bids, reverse=True)[:max_depth]
            self._bids = {price: self._bids[price] for price in top_prices}
        if len(self._asks) > max_depth:
            top_prices = sorted(self._asks)[:max_depth]
            self._asks = {price: self._asks[price] for price in top_prices}

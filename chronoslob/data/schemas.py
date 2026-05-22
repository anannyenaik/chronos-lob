"""Canonical data schemas for ChronosLOB.

This module defines the typed contracts used across the project for market
events, order book snapshots, feature rows, label rows and data-quality
findings. The schemas exist to make later phases (loaders, reconstruction,
feature generation, label generation, training and execution simulation) build
on a single, validated representation.

Design notes
------------
* Timestamps are required to be timezone-aware. Naive datetimes are rejected
  rather than silently localised.
* Prices and quantities are stored as ``float`` for now. Decimal arithmetic
  may be reconsidered later if exact tick-level accounting is required.
* ``FeatureRow`` and ``LabelRow`` are intentionally kept separate to reduce
  the risk of labels leaking into feature inputs.
* Models use Pydantic v2 and forbid unexpected fields by default so that
  upstream bugs surface loudly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from itertools import pairwise

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# Allowed scalar types inside metadata mappings. Nested structures are
# rejected to keep on-disk serialisation predictable.
MetadataScalar = str | int | float | bool
MetadataDict = dict[str, MetadataScalar]

# A loose union representing per-record market data. Loaders may produce
# either snapshot-shaped records or event-shaped records.
MarketDataRecord = "OrderBookSnapshot | BookEvent"

_ALLOWED_SEVERITIES = frozenset({"info", "warning", "error"})


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def ensure_utc_datetime(value: datetime) -> datetime:
    """Return ``value`` as a timezone-aware UTC datetime.

    Naive datetimes are rejected. Aware datetimes in a non-UTC timezone are
    converted to UTC rather than rewritten in place so that downstream
    comparisons are unambiguous.
    """
    if not isinstance(value, datetime):
        raise TypeError(f"expected datetime, got {type(value).__name__}")
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError("datetime must be timezone-aware; got naive value")
    if value.utcoffset() == UTC.utcoffset(value):
        return value
    return value.astimezone(UTC)


def is_finite_number(value: object) -> bool:
    """Return ``True`` only for finite ``int`` or ``float`` values.

    Booleans are excluded even though they are ``int`` subclasses, because
    they should not be treated as numeric features or prices.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def validate_metadata(metadata: Mapping[str, object]) -> MetadataDict:
    """Validate that ``metadata`` only contains simple JSON-like scalars.

    Nested mappings, sequences, ``None``, non-finite floats and unknown types
    are rejected. The function returns a new dictionary so callers may safely
    mutate the result without affecting the caller's input.
    """
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")

    cleaned: MetadataDict = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise ValueError("metadata keys must be non-empty strings")
        if isinstance(value, bool):
            cleaned[key] = value
            continue
        if isinstance(value, int):
            cleaned[key] = value
            continue
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(
                    f"metadata value for key {key!r} must be finite; got {value!r}"
                )
            cleaned[key] = value
            continue
        if isinstance(value, str):
            cleaned[key] = value
            continue
        raise TypeError(
            f"metadata value for key {key!r} has unsupported type {type(value).__name__}"
        )
    return cleaned


def _validate_timestamp(value: datetime | None) -> datetime | None:
    """Internal helper used by Pydantic validators."""
    if value is None:
        return None
    return ensure_utc_datetime(value)


def _validate_finite_float(value: float, *, name: str, allow_zero: bool = True) -> float:
    """Reject NaN/inf and optionally non-positive floats."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite; got {value!r}")
    if not allow_zero and numeric <= 0.0:
        raise ValueError(f"{name} must be strictly positive; got {numeric!r}")
    return numeric


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Side(StrEnum):
    """Order book side."""

    BID = "BID"
    ASK = "ASK"

    @classmethod
    def _missing_(cls, value: object) -> Side | None:
        if isinstance(value, str):
            normalised = value.strip().upper()
            for member in cls:
                if member.value == normalised:
                    return member
        return None


class EventType(StrEnum):
    """Event types observed in market data feeds.

    Names are chosen to be compatible with FI-2010-style snapshots, Binance
    depth updates, LOBSTER/ITCH-style messages and synthetic event streams.
    """

    ADD = "ADD"
    CANCEL = "CANCEL"
    MODIFY = "MODIFY"
    TRADE = "TRADE"
    SNAPSHOT = "SNAPSHOT"
    DEPTH_UPDATE = "DEPTH_UPDATE"

    @classmethod
    def _missing_(cls, value: object) -> EventType | None:
        if isinstance(value, str):
            normalised = value.strip().upper()
            for member in cls:
                if member.value == normalised:
                    return member
        return None


# Event types that almost always carry a side. ``TRADE`` is allowed without
# a side because some public datasets do not expose aggressor information.
_EVENT_TYPES_REQUIRING_SIDE = frozenset(
    {EventType.ADD, EventType.CANCEL, EventType.MODIFY, EventType.DEPTH_UPDATE}
)


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------


class _ChronosModel(BaseModel):
    """Common Pydantic configuration for ChronosLOB schemas."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=False,
        str_strip_whitespace=False,
        validate_assignment=True,
    )


# ---------------------------------------------------------------------------
# Order book schemas
# ---------------------------------------------------------------------------


class OrderBookLevel(_ChronosModel):
    """A single price level on one side of the order book."""

    price: float = Field(..., description="Strictly positive price level.")
    quantity: float = Field(..., description="Non-negative resting quantity.")
    order_count: int | None = Field(
        default=None,
        description="Optional number of orders resting at this level (>= 0 when present).",
    )

    @field_validator("price")
    @classmethod
    def _validate_price(cls, value: float) -> float:
        return _validate_finite_float(value, name="price", allow_zero=False)

    @field_validator("quantity")
    @classmethod
    def _validate_quantity(cls, value: float) -> float:
        numeric = _validate_finite_float(value, name="quantity", allow_zero=True)
        if numeric < 0.0:
            raise ValueError(f"quantity must be non-negative; got {numeric!r}")
        return numeric

    @field_validator("order_count")
    @classmethod
    def _validate_order_count(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("order_count must be an int or None")
        if value < 0:
            raise ValueError(f"order_count must be non-negative; got {value!r}")
        return value


def _check_side_order(levels: list[OrderBookLevel], side: Side) -> None:
    """Raise ``ValueError`` if ``levels`` are not ordered correctly for ``side``."""
    if len(levels) < 2:
        return
    if side is Side.BID:
        for previous, current in pairwise(levels):
            if current.price >= previous.price:
                raise ValueError(
                    "bid levels must be strictly descending by price; "
                    f"got {previous.price!r} followed by {current.price!r}"
                )
    elif side is Side.ASK:
        for previous, current in pairwise(levels):
            if current.price <= previous.price:
                raise ValueError(
                    "ask levels must be strictly ascending by price; "
                    f"got {previous.price!r} followed by {current.price!r}"
                )
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown side {side!r}")


def _check_no_duplicate_prices(levels: list[OrderBookLevel], side: Side) -> None:
    seen: set[float] = set()
    for level in levels:
        if level.price in seen:
            raise ValueError(
                f"duplicate price {level.price!r} on {side.value} side"
            )
        seen.add(level.price)


class OrderBookSnapshot(_ChronosModel):
    """A full or partial limit order book snapshot at a single timestamp.

    Bid levels must be strictly descending by price; ask levels must be
    strictly ascending. Duplicate prices on either side are rejected. The
    snapshot is allowed to be empty on either side to support partial test
    fixtures, but ordering is not silently fixed for non-empty sides.
    """

    timestamp: datetime
    symbol: str = Field(..., min_length=1)
    venue: str | None = None
    bids: list[OrderBookLevel] = Field(default_factory=list)
    asks: list[OrderBookLevel] = Field(default_factory=list)
    sequence_id: int | None = None
    received_timestamp: datetime | None = None
    metadata: MetadataDict = Field(default_factory=dict)

    @field_validator("timestamp", "received_timestamp")
    @classmethod
    def _validate_timestamps(cls, value: datetime | None) -> datetime | None:
        return _validate_timestamp(value)

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("symbol must be a non-empty string")
        return value

    @field_validator("venue")
    @classmethod
    def _validate_venue(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("venue must be a non-empty string when provided")
        return value

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: object) -> MetadataDict:
        if value is None:
            return {}
        try:
            return validate_metadata(value)  # type: ignore[arg-type]
        except TypeError as exc:
            # Pydantic only wraps ValueError/AssertionError into ValidationError,
            # so surface type errors as value errors for consistent reporting.
            raise ValueError(str(exc)) from exc

    @model_validator(mode="after")
    def _validate_book_invariants(self) -> OrderBookSnapshot:
        _check_no_duplicate_prices(self.bids, Side.BID)
        _check_no_duplicate_prices(self.asks, Side.ASK)
        _check_side_order(self.bids, Side.BID)
        _check_side_order(self.asks, Side.ASK)
        # Crossed books are detectable via ``is_crossed`` and ``assert_not_crossed``
        # but are not rejected at construction time. This lets quality checks
        # surface them later without preventing ingestion of edge-case data.
        return self

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def best_bid(self) -> OrderBookLevel | None:
        """Return the highest bid level, or ``None`` if no bids are present."""
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> OrderBookLevel | None:
        """Return the lowest ask level, or ``None`` if no asks are present."""
        return self.asks[0] if self.asks else None

    @property
    def mid_price(self) -> float | None:
        """Return ``(best_bid + best_ask) / 2`` when both sides are present."""
        bid = self.best_bid
        ask = self.best_ask
        if bid is None or ask is None:
            return None
        return (bid.price + ask.price) / 2.0

    @property
    def spread(self) -> float | None:
        """Return ``best_ask - best_bid`` when both sides are present."""
        bid = self.best_bid
        ask = self.best_ask
        if bid is None or ask is None:
            return None
        return ask.price - bid.price

    @property
    def is_crossed(self) -> bool:
        """Return ``True`` if the best bid is at or above the best ask."""
        bid = self.best_bid
        ask = self.best_ask
        if bid is None or ask is None:
            return False
        return bid.price >= ask.price

    # ------------------------------------------------------------------
    # Explicit validation hooks
    # ------------------------------------------------------------------

    def assert_not_crossed(self, *, allow_crossed: bool = False) -> None:
        """Raise ``ValueError`` if the book is crossed unless explicitly allowed."""
        if allow_crossed:
            return
        if self.is_crossed:
            raise ValueError(
                "best bid must be strictly less than best ask; book is crossed "
                f"(best_bid={self.best_bid.price if self.best_bid else None!r}, "
                f"best_ask={self.best_ask.price if self.best_ask else None!r})"
            )


# ---------------------------------------------------------------------------
# Event schema
# ---------------------------------------------------------------------------


class BookEvent(_ChronosModel):
    """An event-level update to the limit order book.

    The schema is intentionally permissive about optional fields so that it
    can represent rows from a wide range of public datasets, but it still
    enforces type-safety on the fields it does carry. Loaders should fill in
    as many fields as the source actually provides.
    """

    timestamp: datetime
    event_type: EventType
    symbol: str = Field(..., min_length=1)
    side: Side | None = None
    price: float | None = None
    quantity: float | None = None
    sequence_id: int | None = None
    order_id: str | None = None
    trade_id: str | None = None
    received_timestamp: datetime | None = None
    metadata: MetadataDict = Field(default_factory=dict)

    @field_validator("timestamp", "received_timestamp")
    @classmethod
    def _validate_timestamps(cls, value: datetime | None) -> datetime | None:
        return _validate_timestamp(value)

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("symbol must be a non-empty string")
        return value

    @field_validator("price")
    @classmethod
    def _validate_price(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return _validate_finite_float(value, name="price", allow_zero=False)

    @field_validator("quantity")
    @classmethod
    def _validate_quantity(cls, value: float | None) -> float | None:
        if value is None:
            return None
        numeric = _validate_finite_float(value, name="quantity", allow_zero=True)
        if numeric < 0.0:
            raise ValueError(f"quantity must be non-negative; got {numeric!r}")
        return numeric

    @field_validator("order_id", "trade_id")
    @classmethod
    def _validate_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise ValueError("identifier must be a non-empty string when provided")
        return value

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: object) -> MetadataDict:
        if value is None:
            return {}
        try:
            return validate_metadata(value)  # type: ignore[arg-type]
        except TypeError as exc:
            # Pydantic only wraps ValueError/AssertionError into ValidationError,
            # so surface type errors as value errors for consistent reporting.
            raise ValueError(str(exc)) from exc

    @model_validator(mode="after")
    def _validate_event_invariants(self) -> BookEvent:
        if self.event_type in _EVENT_TYPES_REQUIRING_SIDE and self.side is None:
            raise ValueError(
                f"event_type {self.event_type.value} requires a side"
            )
        return self


# ---------------------------------------------------------------------------
# Feature and label schemas
# ---------------------------------------------------------------------------


def _validate_feature_mapping(
    value: Mapping[str, object],
    *,
    container_name: str,
) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{container_name} must be a mapping")
    cleaned: dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{container_name} names must be non-empty strings")
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(
                f"{container_name}[{key!r}] must be a finite number, "
                f"got {type(item).__name__}"
            )
        numeric = float(item)
        if not math.isfinite(numeric):
            raise ValueError(
                f"{container_name}[{key!r}] must be finite; got {item!r}"
            )
        cleaned[key] = numeric
    return cleaned


class FeatureRow(_ChronosModel):
    """A single timestamped feature vector.

    ``timestamp`` is the time at which the feature row was emitted. Features
    must only use information available at or before ``timestamp``. The
    optional ``horizon_origin_timestamp`` records the originating event time
    for derived features and, if present, must not be after ``timestamp``.
    """

    timestamp: datetime
    symbol: str = Field(..., min_length=1)
    features: dict[str, float] = Field(default_factory=dict)
    horizon_origin_timestamp: datetime | None = None
    metadata: MetadataDict = Field(default_factory=dict)

    @field_validator("timestamp", "horizon_origin_timestamp")
    @classmethod
    def _validate_timestamps(cls, value: datetime | None) -> datetime | None:
        return _validate_timestamp(value)

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("symbol must be a non-empty string")
        return value

    @field_validator("features", mode="before")
    @classmethod
    def _validate_features(cls, value: object) -> dict[str, float]:
        if value is None:
            return {}
        return _validate_feature_mapping(value, container_name="feature")  # type: ignore[arg-type]

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: object) -> MetadataDict:
        if value is None:
            return {}
        try:
            return validate_metadata(value)  # type: ignore[arg-type]
        except TypeError as exc:
            # Pydantic only wraps ValueError/AssertionError into ValidationError,
            # so surface type errors as value errors for consistent reporting.
            raise ValueError(str(exc)) from exc

    @model_validator(mode="after")
    def _validate_horizon(self) -> FeatureRow:
        if (
            self.horizon_origin_timestamp is not None
            and self.horizon_origin_timestamp > self.timestamp
        ):
            raise ValueError(
                "horizon_origin_timestamp must not be after timestamp; "
                f"got {self.horizon_origin_timestamp!r} > {self.timestamp!r}"
            )
        return self


LabelValue = int | float | str | bool


def _validate_label_mapping(value: Mapping[str, object]) -> dict[str, LabelValue]:
    if not isinstance(value, Mapping):
        raise TypeError("labels must be a mapping")
    cleaned: dict[str, LabelValue] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("label names must be non-empty strings")
        if isinstance(item, bool):
            cleaned[key] = item
            continue
        if isinstance(item, int):
            cleaned[key] = item
            continue
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(
                    f"labels[{key!r}] must be finite when numeric; got {item!r}"
                )
            cleaned[key] = item
            continue
        if isinstance(item, str):
            cleaned[key] = item
            continue
        raise TypeError(
            f"labels[{key!r}] has unsupported type {type(item).__name__}"
        )
    return cleaned


class LabelRow(_ChronosModel):
    """A single timestamped label vector.

    Label rows are deliberately separated from ``FeatureRow`` to make
    leakage easier to audit. ``horizon_start`` and ``horizon_end`` describe
    the future window the labels summarise. ``horizon_start`` may equal
    ``timestamp`` (e.g. for an open-ended window starting now), but it must
    not be before ``timestamp``. ``horizon_end`` must be strictly after
    ``horizon_start``.
    """

    timestamp: datetime
    symbol: str = Field(..., min_length=1)
    labels: dict[str, LabelValue] = Field(default_factory=dict)
    horizon_start: datetime
    horizon_end: datetime
    metadata: MetadataDict = Field(default_factory=dict)

    @field_validator("timestamp", "horizon_start", "horizon_end")
    @classmethod
    def _validate_timestamps(cls, value: datetime) -> datetime:
        normalised = _validate_timestamp(value)
        # _validate_timestamp returns None for None; these three fields are
        # required so a None return indicates a programming error.
        assert normalised is not None
        return normalised

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("symbol must be a non-empty string")
        return value

    @field_validator("labels", mode="before")
    @classmethod
    def _validate_labels(cls, value: object) -> dict[str, LabelValue]:
        if value is None:
            return {}
        return _validate_label_mapping(value)  # type: ignore[arg-type]

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: object) -> MetadataDict:
        if value is None:
            return {}
        try:
            return validate_metadata(value)  # type: ignore[arg-type]
        except TypeError as exc:
            # Pydantic only wraps ValueError/AssertionError into ValidationError,
            # so surface type errors as value errors for consistent reporting.
            raise ValueError(str(exc)) from exc

    @model_validator(mode="after")
    def _validate_horizon(self) -> LabelRow:
        if self.horizon_start < self.timestamp:
            raise ValueError(
                "horizon_start must be at or after timestamp; "
                f"got {self.horizon_start!r} < {self.timestamp!r}"
            )
        if self.horizon_end <= self.horizon_start:
            raise ValueError(
                "horizon_end must be strictly after horizon_start; "
                f"got {self.horizon_end!r} <= {self.horizon_start!r}"
            )
        return self


# ---------------------------------------------------------------------------
# Data-quality findings
# ---------------------------------------------------------------------------


class DataQualityIssue(_ChronosModel):
    """A single data-quality finding raised during ingestion or validation."""

    timestamp: datetime | None = None
    symbol: str | None = None
    severity: str
    code: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    metadata: MetadataDict = Field(default_factory=dict)

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamps(cls, value: datetime | None) -> datetime | None:
        return _validate_timestamp(value)

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("symbol must be a non-empty string when provided")
        return value

    @field_validator("severity")
    @classmethod
    def _validate_severity(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("severity must be a string")
        normalised = value.strip().lower()
        if normalised not in _ALLOWED_SEVERITIES:
            raise ValueError(
                f"severity must be one of {sorted(_ALLOWED_SEVERITIES)}; got {value!r}"
            )
        return normalised

    @field_validator("code", "message")
    @classmethod
    def _validate_required_strings(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("code and message must be non-empty strings")
        return value

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: object) -> MetadataDict:
        if value is None:
            return {}
        try:
            return validate_metadata(value)  # type: ignore[arg-type]
        except TypeError as exc:
            # Pydantic only wraps ValueError/AssertionError into ValidationError,
            # so surface type errors as value errors for consistent reporting.
            raise ValueError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


__all__ = [
    "BookEvent",
    "DataQualityIssue",
    "EventType",
    "FeatureRow",
    "LabelRow",
    "LabelValue",
    "MarketDataRecord",
    "MetadataDict",
    "MetadataScalar",
    "OrderBookLevel",
    "OrderBookSnapshot",
    "Side",
    "ensure_utc_datetime",
    "is_finite_number",
    "validate_metadata",
]

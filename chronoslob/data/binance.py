"""Offline Binance-style depth schemas and parsers.

This module defines typed, validated schemas for Binance-style depth
snapshots and diff-depth updates and provides parsers that read **local
files only**. No network calls, REST requests or WebSocket connections
are made anywhere in this module.

The schemas intentionally do not import any networking dependency. They
exist so that downstream reconstruction code (see
:mod:`chronoslob.book.reconstruction`) can operate on a stable typed
contract regardless of how the underlying raw bytes were captured. Test
fixtures live under ``tests/fixtures/binance/`` and are clearly marked
as synthetic.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from chronoslob.data.schemas import (
    MetadataDict,
    OrderBookLevel,
    OrderBookSnapshot,
    ensure_utc_datetime,
)

__all__ = [
    "BinanceDepthLevel",
    "BinanceDepthSnapshot",
    "BinanceDiffDepthEvent",
    "load_binance_diff_events_jsonl",
    "load_binance_snapshot_json",
    "parse_binance_diff_event",
    "parse_binance_snapshot",
    "to_order_book_snapshot",
]


# Deterministic placeholder timestamp used only when fixtures supply
# neither an exchange nor a received timestamp. The metadata field
# ``synthetic_time`` is set to ``True`` so downstream code can tell that
# the timestamp is not from a real venue.
_SYNTHETIC_DEFAULT_TIMESTAMP = datetime(1970, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------


class _BinanceModel(BaseModel):
    """Common Pydantic configuration for Binance-style schemas."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=False,
        validate_assignment=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_float(value: object, *, name: str) -> float:
    """Convert ``value`` to a finite float without accepting booleans."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, not bool")
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{name} must be a non-empty numeric string")
        try:
            numeric = float(text)
        except ValueError as exc:
            raise ValueError(f"{name} must be numeric; got {value!r}") from exc
    else:
        raise TypeError(
            f"{name} must be numeric or numeric string; got {type(value).__name__}"
        )
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return numeric


def _coerce_int(value: object, *, name: str) -> int:
    """Convert ``value`` to an int without accepting booleans."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int, not bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{name} must be a non-empty integer string")
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer; got {value!r}") from exc
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{name} must be an integer-valued number; got {value!r}")
        return int(value)
    raise TypeError(
        f"{name} must be an int, integer string or integer float; got {type(value).__name__}"
    )


def _datetime_from_milliseconds(value: object, *, name: str) -> datetime:
    """Convert a millisecond epoch value to a UTC ``datetime``."""
    millis = _coerce_int(value, name=name)
    if millis < 0:
        raise ValueError(f"{name} must be a non-negative epoch in ms; got {millis!r}")
    return datetime.fromtimestamp(millis / 1000.0, tz=UTC)


def _coerce_optional_datetime(value: object | None, *, name: str) -> datetime | None:
    """Best-effort conversion of an incoming timestamp to a UTC ``datetime``."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return ensure_utc_datetime(value)
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a datetime or epoch; got bool")
    if isinstance(value, (int, float)):
        return _datetime_from_milliseconds(value, name=name)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # Accept ISO-8601 strings or numeric epoch strings.
        try:
            return _datetime_from_milliseconds(text, name=name)
        except (TypeError, ValueError):
            pass
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(
                f"{name} must be an ISO-8601 datetime or epoch ms; got {value!r}"
            ) from exc
        return ensure_utc_datetime(parsed)
    raise TypeError(f"{name} has unsupported timestamp type {type(value).__name__}")


def _parse_levels(
    raw_levels: object,
    *,
    container_name: str,
) -> list[BinanceDepthLevel]:
    """Parse a list of ``[price, quantity]`` pairs into ``BinanceDepthLevel`` objects."""
    if raw_levels is None:
        return []
    if not isinstance(raw_levels, Iterable) or isinstance(raw_levels, (str, bytes)):
        raise TypeError(f"{container_name} must be an iterable of level pairs")
    parsed: list[BinanceDepthLevel] = []
    for index, raw_level in enumerate(raw_levels):
        location = f"{container_name}[{index}]"
        if isinstance(raw_level, BinanceDepthLevel):
            parsed.append(raw_level)
            continue
        if isinstance(raw_level, Mapping):
            if "price" not in raw_level or "quantity" not in raw_level:
                raise ValueError(
                    f"{location} mapping must include 'price' and 'quantity'"
                )
            price = _coerce_float(raw_level["price"], name=f"{location}.price")
            quantity = _coerce_float(
                raw_level["quantity"], name=f"{location}.quantity"
            )
        elif isinstance(raw_level, Sequence) and not isinstance(raw_level, (str, bytes)):
            items = list(raw_level)
            if len(items) < 2:
                raise ValueError(
                    f"{location} must contain at least price and quantity"
                )
            price = _coerce_float(items[0], name=f"{location}.price")
            quantity = _coerce_float(items[1], name=f"{location}.quantity")
        else:
            raise TypeError(
                f"{location} must be a mapping or [price, quantity] pair; "
                f"got {type(raw_level).__name__}"
            )
        parsed.append(BinanceDepthLevel(price=price, quantity=quantity))
    return parsed


def _check_no_duplicate_prices(
    levels: Sequence[BinanceDepthLevel], *, side: str
) -> None:
    seen: set[float] = set()
    for level in levels:
        if level.price in seen:
            raise ValueError(
                f"duplicate price {level.price!r} on {side} side"
            )
        seen.add(level.price)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class BinanceDepthLevel(_BinanceModel):
    """A single Binance-style depth level."""

    price: float = Field(..., description="Strictly positive price.")
    quantity: float = Field(..., description="Non-negative resting quantity.")

    @field_validator("price")
    @classmethod
    def _validate_price(cls, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("price must be a finite number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"price must be finite; got {value!r}")
        if numeric <= 0.0:
            raise ValueError(f"price must be strictly positive; got {numeric!r}")
        return numeric

    @field_validator("quantity")
    @classmethod
    def _validate_quantity(cls, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("quantity must be a finite number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"quantity must be finite; got {value!r}")
        if numeric < 0.0:
            raise ValueError(f"quantity must be non-negative; got {numeric!r}")
        return numeric


class BinanceDepthSnapshot(_BinanceModel):
    """A Binance-style REST depth snapshot.

    Bids and asks are accepted in any order from the raw payload. The
    parser :func:`parse_binance_snapshot` sorts them into canonical
    descending (bids) and ascending (asks) order, but constructing this
    model directly with already-sorted lists is also supported. Duplicate
    price levels on the same side are rejected.
    """

    symbol: str = Field(..., min_length=1)
    last_update_id: int = Field(..., ge=0)
    bids: list[BinanceDepthLevel] = Field(default_factory=list)
    asks: list[BinanceDepthLevel] = Field(default_factory=list)
    timestamp: datetime | None = None
    received_timestamp: datetime | None = None

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("symbol must be a non-empty string")
        return value

    @field_validator("last_update_id")
    @classmethod
    def _validate_last_update_id(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("last_update_id must be an int")
        if value < 0:
            raise ValueError(f"last_update_id must be >= 0; got {value!r}")
        return value

    @field_validator("timestamp", "received_timestamp")
    @classmethod
    def _validate_timestamps(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return ensure_utc_datetime(value)

    @model_validator(mode="after")
    def _validate_invariants(self) -> BinanceDepthSnapshot:
        _check_no_duplicate_prices(self.bids, side="bid")
        _check_no_duplicate_prices(self.asks, side="ask")
        return self


class BinanceDiffDepthEvent(_BinanceModel):
    """A Binance-style ``depthUpdate`` diff event."""

    event_type: str = "depthUpdate"
    event_time: datetime | None = None
    transaction_time: datetime | None = None
    symbol: str = Field(..., min_length=1)
    first_update_id: int = Field(..., ge=0)
    final_update_id: int = Field(..., ge=0)
    previous_final_update_id: int | None = None
    bids: list[BinanceDepthLevel] = Field(default_factory=list)
    asks: list[BinanceDepthLevel] = Field(default_factory=list)
    received_timestamp: datetime | None = None

    @field_validator("event_type")
    @classmethod
    def _validate_event_type(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("event_type must be a non-empty string")
        return value

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("symbol must be a non-empty string")
        return value

    @field_validator("first_update_id", "final_update_id")
    @classmethod
    def _validate_update_id(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("update id must be an int")
        if value < 0:
            raise ValueError(f"update id must be >= 0; got {value!r}")
        return value

    @field_validator("previous_final_update_id")
    @classmethod
    def _validate_previous_final_update_id(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("previous_final_update_id must be an int or None")
        if value < 0:
            raise ValueError(
                f"previous_final_update_id must be >= 0; got {value!r}"
            )
        return value

    @field_validator("event_time", "transaction_time", "received_timestamp")
    @classmethod
    def _validate_timestamps(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return ensure_utc_datetime(value)

    @model_validator(mode="after")
    def _validate_invariants(self) -> BinanceDiffDepthEvent:
        if self.first_update_id > self.final_update_id:
            raise ValueError(
                "first_update_id must be <= final_update_id; "
                f"got {self.first_update_id} > {self.final_update_id}"
            )
        _check_no_duplicate_prices(self.bids, side="bid")
        _check_no_duplicate_prices(self.asks, side="ask")
        return self


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_binance_snapshot(
    payload: Mapping[str, object],
    *,
    symbol: str | None = None,
    timestamp: datetime | None = None,
    received_timestamp: datetime | None = None,
) -> BinanceDepthSnapshot:
    """Parse a Binance-style REST depth snapshot payload.

    The payload is expected to contain ``lastUpdateId`` and ``bids``/``asks``
    arrays of ``[price, quantity]`` pairs. ``symbol`` may be supplied
    explicitly or inferred from the payload via the ``symbol`` or ``s``
    keys. Bids are sorted descending and asks ascending in the returned
    snapshot.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")

    inferred_symbol = symbol
    if inferred_symbol is None:
        for key in ("symbol", "s"):
            raw = payload.get(key)
            if isinstance(raw, str) and raw.strip():
                inferred_symbol = raw.strip()
                break
    if inferred_symbol is None:
        raise ValueError(
            "symbol must be supplied explicitly or via the payload 'symbol'/'s' key"
        )

    if "lastUpdateId" in payload:
        last_update_id = _coerce_int(payload["lastUpdateId"], name="lastUpdateId")
    elif "last_update_id" in payload:
        last_update_id = _coerce_int(
            payload["last_update_id"], name="last_update_id"
        )
    else:
        raise ValueError("payload must include 'lastUpdateId' or 'last_update_id'")

    bids = _parse_levels(payload.get("bids", []), container_name="bids")
    asks = _parse_levels(payload.get("asks", []), container_name="asks")

    bids.sort(key=lambda level: level.price, reverse=True)
    asks.sort(key=lambda level: level.price)

    resolved_timestamp = timestamp
    if resolved_timestamp is None and "timestamp" in payload:
        resolved_timestamp = _coerce_optional_datetime(
            payload["timestamp"], name="timestamp"
        )
    resolved_received = received_timestamp
    if resolved_received is None and "received_timestamp" in payload:
        resolved_received = _coerce_optional_datetime(
            payload["received_timestamp"], name="received_timestamp"
        )

    return BinanceDepthSnapshot(
        symbol=inferred_symbol,
        last_update_id=last_update_id,
        bids=bids,
        asks=asks,
        timestamp=resolved_timestamp,
        received_timestamp=resolved_received,
    )


def _extract(
    payload: Mapping[str, object], *keys: str, default: object = None
) -> object:
    for key in keys:
        if key in payload:
            return payload[key]
    return default


def parse_binance_diff_event(payload: Mapping[str, object]) -> BinanceDiffDepthEvent:
    """Parse a Binance-style ``depthUpdate`` diff event payload.

    Accepts both raw Binance stream keys (``e``, ``E``, ``T``, ``s``,
    ``U``, ``u``, ``pu``, ``b``, ``a``) and clearer fixture aliases
    (``event_type``, ``event_time``, ``transaction_time``, ``symbol``,
    ``first_update_id``, ``final_update_id``, ``previous_final_update_id``,
    ``bids``, ``asks``).
    """
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")

    raw_event_type = _extract(payload, "event_type", "e", default="depthUpdate")
    if not isinstance(raw_event_type, str):
        raise TypeError("event_type must be a string")

    raw_symbol = _extract(payload, "symbol", "s")
    if not isinstance(raw_symbol, str) or not raw_symbol.strip():
        raise ValueError("event must include a non-empty symbol")

    if "first_update_id" in payload:
        first_update_id = _coerce_int(
            payload["first_update_id"], name="first_update_id"
        )
    elif "U" in payload:
        first_update_id = _coerce_int(payload["U"], name="U")
    else:
        raise ValueError("event must include 'first_update_id' or 'U'")

    if "final_update_id" in payload:
        final_update_id = _coerce_int(
            payload["final_update_id"], name="final_update_id"
        )
    elif "u" in payload:
        final_update_id = _coerce_int(payload["u"], name="u")
    else:
        raise ValueError("event must include 'final_update_id' or 'u'")

    previous_final_update_id: int | None = None
    if "previous_final_update_id" in payload:
        raw_previous = payload["previous_final_update_id"]
        if raw_previous is not None:
            previous_final_update_id = _coerce_int(
                raw_previous, name="previous_final_update_id"
            )
    elif "pu" in payload:
        raw_previous = payload["pu"]
        if raw_previous is not None:
            previous_final_update_id = _coerce_int(raw_previous, name="pu")

    event_time: datetime | None = None
    if "event_time" in payload:
        event_time = _coerce_optional_datetime(
            payload["event_time"], name="event_time"
        )
    elif "E" in payload:
        event_time = _coerce_optional_datetime(payload["E"], name="E")

    transaction_time: datetime | None = None
    if "transaction_time" in payload:
        transaction_time = _coerce_optional_datetime(
            payload["transaction_time"], name="transaction_time"
        )
    elif "T" in payload:
        transaction_time = _coerce_optional_datetime(payload["T"], name="T")

    received_timestamp: datetime | None = None
    if "received_timestamp" in payload:
        received_timestamp = _coerce_optional_datetime(
            payload["received_timestamp"], name="received_timestamp"
        )

    raw_bids = _extract(payload, "bids", "b", default=[])
    raw_asks = _extract(payload, "asks", "a", default=[])
    bids = _parse_levels(raw_bids, container_name="bids")
    asks = _parse_levels(raw_asks, container_name="asks")

    return BinanceDiffDepthEvent(
        event_type=raw_event_type,
        event_time=event_time,
        transaction_time=transaction_time,
        symbol=raw_symbol.strip(),
        first_update_id=first_update_id,
        final_update_id=final_update_id,
        previous_final_update_id=previous_final_update_id,
        bids=bids,
        asks=asks,
        received_timestamp=received_timestamp,
    )


# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------


def load_binance_snapshot_json(
    path: str | Path,
    *,
    symbol: str | None = None,
) -> BinanceDepthSnapshot:
    """Load a Binance-style depth snapshot from a local JSON file.

    The file is read with UTF-8 encoding from disk only — no network
    access. ``symbol`` may be passed explicitly when the JSON payload
    does not carry it.
    """
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"failed to parse {file_path} as JSON: {exc.msg} at line {exc.lineno}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{file_path} must contain a JSON object at the top level")
    return parse_binance_snapshot(payload, symbol=symbol)


def load_binance_diff_events_jsonl(
    path: str | Path,
) -> list[BinanceDiffDepthEvent]:
    """Load a list of Binance-style diff events from a local JSONL file.

    Blank lines are skipped. Malformed JSON or events that fail
    validation raise ``ValueError`` with the offending line number.
    """
    file_path = Path(path)
    events: list[BinanceDiffDepthEvent] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"malformed JSON in {file_path} on line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"{file_path} line {line_number} must be a JSON object"
                )
            try:
                events.append(parse_binance_diff_event(payload))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid diff event in {file_path} on line {line_number}: {exc}"
                ) from exc
    return events


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def _trim_levels(
    levels: Sequence[BinanceDepthLevel], depth: int | None
) -> list[BinanceDepthLevel]:
    if depth is None:
        return list(levels)
    if depth <= 0:
        raise ValueError(f"depth must be a positive int when supplied; got {depth!r}")
    return list(levels[:depth])


def to_order_book_snapshot(
    snapshot: BinanceDepthSnapshot,
    *,
    depth: int | None = None,
) -> OrderBookSnapshot:
    """Convert a Binance snapshot to the canonical :class:`OrderBookSnapshot`.

    ``timestamp`` is taken from ``snapshot.timestamp`` if present,
    otherwise from ``snapshot.received_timestamp``. When neither is
    available the snapshot is given a deterministic UTC fixture
    timestamp and the resulting metadata records ``synthetic_time=True``
    so downstream code can distinguish reconstructed fixture data from
    real venue data.
    """
    bids = _trim_levels(snapshot.bids, depth)
    asks = _trim_levels(snapshot.asks, depth)

    canonical_bids = [
        OrderBookLevel(price=level.price, quantity=level.quantity) for level in bids
    ]
    canonical_asks = [
        OrderBookLevel(price=level.price, quantity=level.quantity) for level in asks
    ]

    metadata: MetadataDict = {"source": "binance_snapshot"}
    chosen_timestamp = snapshot.timestamp or snapshot.received_timestamp
    if chosen_timestamp is None:
        chosen_timestamp = _SYNTHETIC_DEFAULT_TIMESTAMP
        metadata["synthetic_time"] = True

    return OrderBookSnapshot(
        timestamp=chosen_timestamp,
        symbol=snapshot.symbol,
        venue="binance",
        bids=canonical_bids,
        asks=canonical_asks,
        sequence_id=snapshot.last_update_id,
        received_timestamp=snapshot.received_timestamp,
        metadata=metadata,
    )

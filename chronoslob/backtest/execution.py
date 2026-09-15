"""Typed execution-simulation data contracts.

The classes in this module describe inputs and outputs for simplified
execution-aware validation. They are deliberately local, deterministic and
offline: no order placement, exchange connectivity or live trading behaviour is
implemented here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

__all__ = [
    "ExecutionDecision",
    "ExecutionFill",
    "ExecutionMode",
    "ExecutionResult",
    "MarketState",
    "PredictionSignal",
    "TradeSide",
    "side_sign",
]


class TradeSide(StrEnum):
    """Prediction or order side used by the research simulation."""

    BUY = "buy"
    SELL = "sell"


class ExecutionMode(StrEnum):
    """Supported simplified execution modes."""

    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"
    HYBRID = "hybrid"


def _coerce_side(side: TradeSide | str) -> TradeSide:
    if isinstance(side, TradeSide):
        return side
    if isinstance(side, str):
        try:
            return TradeSide(side)
        except ValueError as exc:
            raise ValueError(f"unsupported trade side: {side!r}") from exc
    raise TypeError("side must be a TradeSide or string value")


def _coerce_mode(mode: ExecutionMode | str) -> ExecutionMode:
    if isinstance(mode, ExecutionMode):
        return mode
    if isinstance(mode, str):
        try:
            return ExecutionMode(mode)
        except ValueError as exc:
            raise ValueError(f"unsupported execution mode: {mode!r}") from exc
    raise TypeError("mode must be an ExecutionMode or string value")


def _validate_timestamp(timestamp: datetime, *, name: str) -> datetime:
    if not isinstance(timestamp, datetime):
        raise TypeError(f"{name} must be a datetime")
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return timestamp


def _validate_finite_float(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite float")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _validate_positive_float(value: float, *, name: str) -> float:
    numeric = _validate_finite_float(value, name=name)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be positive")
    return numeric


def _validate_non_negative_float(value: float, *, name: str) -> float:
    numeric = _validate_finite_float(value, name=name)
    if numeric < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return numeric


def _validate_probability(value: float, *, name: str) -> float:
    numeric = _validate_finite_float(value, name=name)
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must satisfy 0 <= {name} <= 1")
    return numeric


def _validate_metadata(metadata: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(metadata)


def _validate_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validate_symbol(symbol: str) -> str:
    if not isinstance(symbol, str):
        raise TypeError("symbol must be a string")
    cleaned = symbol.strip()
    if not cleaned:
        raise ValueError("symbol must not be empty")
    return cleaned


def side_sign(side: TradeSide | str) -> int:
    """Return +1 for buy-side exposure and -1 for sell-side exposure."""
    return 1 if _coerce_side(side) is TradeSide.BUY else -1


@dataclass(frozen=True)
class PredictionSignal:
    """Prediction-like input consumed by execution-aware validation."""

    timestamp: datetime
    symbol: str
    side: TradeSide
    confidence: float
    score: float
    horizon: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "timestamp",
            _validate_timestamp(self.timestamp, name="timestamp"),
        )
        object.__setattr__(self, "symbol", _validate_symbol(self.symbol))
        object.__setattr__(self, "side", _coerce_side(self.side))
        object.__setattr__(
            self,
            "confidence",
            _validate_probability(self.confidence, name="confidence"),
        )
        object.__setattr__(self, "score", _validate_finite_float(self.score, name="score"))
        object.__setattr__(self, "horizon", _validate_positive_int(self.horizon, name="horizon"))
        object.__setattr__(
            self,
            "metadata",
            _validate_metadata(self.metadata, name="metadata"),
        )


@dataclass(frozen=True)
class MarketState:
    """Market-state-like row used for simplified execution simulation."""

    timestamp: datetime
    symbol: str
    mid_price: float
    best_bid: float
    best_ask: float
    spread: float
    bid_size: float
    ask_size: float
    volatility: float | None = None
    fill_probability: float | None = None
    adverse_selection_label: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "timestamp",
            _validate_timestamp(self.timestamp, name="timestamp"),
        )
        object.__setattr__(self, "symbol", _validate_symbol(self.symbol))
        mid_price = _validate_positive_float(self.mid_price, name="mid_price")
        best_bid = _validate_positive_float(self.best_bid, name="best_bid")
        best_ask = _validate_positive_float(self.best_ask, name="best_ask")
        spread = _validate_non_negative_float(self.spread, name="spread")
        if best_ask < best_bid:
            raise ValueError("best_ask must be greater than or equal to best_bid")
        implied_spread = best_ask - best_bid
        if not math.isclose(implied_spread, spread, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("spread must equal best_ask - best_bid")
        if mid_price < best_bid or mid_price > best_ask:
            raise ValueError("mid_price must lie inside the best bid/ask range")
        object.__setattr__(self, "mid_price", mid_price)
        object.__setattr__(self, "best_bid", best_bid)
        object.__setattr__(self, "best_ask", best_ask)
        object.__setattr__(self, "spread", spread)
        object.__setattr__(
            self,
            "bid_size",
            _validate_non_negative_float(self.bid_size, name="bid_size"),
        )
        object.__setattr__(
            self,
            "ask_size",
            _validate_non_negative_float(self.ask_size, name="ask_size"),
        )
        if self.volatility is not None:
            object.__setattr__(
                self,
                "volatility",
                _validate_non_negative_float(self.volatility, name="volatility"),
            )
        if self.fill_probability is not None:
            object.__setattr__(
                self,
                "fill_probability",
                _validate_probability(self.fill_probability, name="fill_probability"),
            )
        if self.adverse_selection_label is not None and not isinstance(
            self.adverse_selection_label,
            bool,
        ):
            raise TypeError("adverse_selection_label must be a bool when provided")
        object.__setattr__(
            self,
            "metadata",
            _validate_metadata(self.metadata, name="metadata"),
        )


@dataclass(frozen=True)
class ExecutionDecision:
    """Validated decision produced before fill simulation."""

    timestamp: datetime
    symbol: str
    mode: ExecutionMode
    side: TradeSide
    should_trade: bool
    reason: str
    confidence: float
    order_price: float | None
    quantity: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "timestamp",
            _validate_timestamp(self.timestamp, name="timestamp"),
        )
        object.__setattr__(self, "symbol", _validate_symbol(self.symbol))
        object.__setattr__(self, "mode", _coerce_mode(self.mode))
        object.__setattr__(self, "side", _coerce_side(self.side))
        if not isinstance(self.should_trade, bool):
            raise TypeError("should_trade must be a bool")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")
        object.__setattr__(
            self,
            "confidence",
            _validate_probability(self.confidence, name="confidence"),
        )
        if self.order_price is not None:
            object.__setattr__(
                self,
                "order_price",
                _validate_positive_float(self.order_price, name="order_price"),
            )
        quantity = _validate_non_negative_float(self.quantity, name="quantity")
        if self.should_trade and quantity <= 0.0:
            raise ValueError("quantity must be positive when should_trade is true")
        if self.should_trade and self.order_price is None:
            raise ValueError("order_price is required when should_trade is true")
        object.__setattr__(self, "quantity", quantity)


@dataclass(frozen=True)
class ExecutionFill:
    """Fill outcome produced by the simplified execution assumptions."""

    timestamp: datetime
    symbol: str
    mode: ExecutionMode
    side: TradeSide
    filled: bool
    fill_price: float | None
    quantity: float
    fees: float
    spread_cost: float
    slippage: float
    latency_steps: int
    adverse_selection: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "timestamp",
            _validate_timestamp(self.timestamp, name="timestamp"),
        )
        object.__setattr__(self, "symbol", _validate_symbol(self.symbol))
        object.__setattr__(self, "mode", _coerce_mode(self.mode))
        object.__setattr__(self, "side", _coerce_side(self.side))
        if not isinstance(self.filled, bool):
            raise TypeError("filled must be a bool")
        if self.fill_price is not None:
            object.__setattr__(
                self,
                "fill_price",
                _validate_positive_float(self.fill_price, name="fill_price"),
            )
        quantity = _validate_non_negative_float(self.quantity, name="quantity")
        if self.filled and quantity <= 0.0:
            raise ValueError("quantity must be positive when filled is true")
        if self.filled and self.fill_price is None:
            raise ValueError("fill_price is required when filled is true")
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "fees", _validate_non_negative_float(self.fees, name="fees"))
        object.__setattr__(
            self,
            "spread_cost",
            _validate_non_negative_float(self.spread_cost, name="spread_cost"),
        )
        object.__setattr__(
            self,
            "slippage",
            _validate_non_negative_float(self.slippage, name="slippage"),
        )
        object.__setattr__(
            self,
            "latency_steps",
            _validate_non_negative_int(self.latency_steps, name="latency_steps"),
        )
        if not isinstance(self.adverse_selection, bool):
            raise TypeError("adverse_selection must be a bool")
        object.__setattr__(
            self,
            "metadata",
            _validate_metadata(self.metadata, name="metadata"),
        )


@dataclass(frozen=True)
class ExecutionResult:
    """Cost-aware simulated result for one prediction signal."""

    decision: ExecutionDecision
    fill: ExecutionFill
    price_move: float
    realised_return: float
    gross_pnl: float
    net_pnl: float
    cost: float
    inventory_after: float
    turnover: float

    def __post_init__(self) -> None:
        if not isinstance(self.decision, ExecutionDecision):
            raise TypeError("decision must be an ExecutionDecision")
        if not isinstance(self.fill, ExecutionFill):
            raise TypeError("fill must be an ExecutionFill")
        for name in (
            "price_move",
            "realised_return",
            "gross_pnl",
            "net_pnl",
            "inventory_after",
        ):
            object.__setattr__(
                self,
                name,
                _validate_finite_float(getattr(self, name), name=name),
            )
        object.__setattr__(self, "cost", _validate_non_negative_float(self.cost, name="cost"))
        object.__setattr__(
            self,
            "turnover",
            _validate_non_negative_float(self.turnover, name="turnover"),
        )
        if not math.isclose(
            self.gross_pnl - self.cost,
            self.net_pnl,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("net_pnl must equal gross_pnl - cost")

"""Turnover and position-path utilities for simulated fills."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from chronoslob.backtest.execution import ExecutionFill, ExecutionResult, side_sign

__all__ = [
    "TurnoverSummary",
    "compute_average_holding_period",
    "compute_position_path",
    "compute_trade_count",
    "compute_turnover",
]


@dataclass(frozen=True)
class TurnoverSummary:
    """Deterministic turnover summary over filled simulated trades."""

    n_trades: int
    total_quantity: float
    total_notional: float
    turnover: float
    turnover_basis: str
    average_trade_quantity: float | None
    average_trade_notional: float | None
    average_holding_period: float | None
    position_path: list[float]

    def to_dict(self) -> dict[str, float | int | str | list[float] | None]:
        """Return a serialisable representation."""
        return {
            "n_trades": self.n_trades,
            "total_quantity": self.total_quantity,
            "total_notional": self.total_notional,
            "turnover": self.turnover,
            "turnover_basis": self.turnover_basis,
            "average_trade_quantity": self.average_trade_quantity,
            "average_trade_notional": self.average_trade_notional,
            "average_holding_period": self.average_holding_period,
            "position_path": list(self.position_path),
        }


def _as_fill(trade: ExecutionFill | ExecutionResult) -> ExecutionFill:
    if isinstance(trade, ExecutionFill):
        return trade
    if isinstance(trade, ExecutionResult):
        return trade.fill
    raise TypeError("trades must contain ExecutionFill or ExecutionResult instances")


def _fills(trades: Sequence[ExecutionFill | ExecutionResult]) -> list[ExecutionFill]:
    if not isinstance(trades, Sequence):
        raise TypeError("trades must be a sequence")
    return [_as_fill(trade) for trade in trades]


def compute_trade_count(trades: Sequence[ExecutionFill | ExecutionResult]) -> int:
    """Count filled simulated trades."""
    return sum(1 for fill in _fills(trades) if fill.filled and fill.quantity > 0.0)


def compute_position_path(trades: Sequence[ExecutionFill | ExecutionResult]) -> list[float]:
    """Return inventory after each supplied fill/result row."""
    position = 0.0
    path: list[float] = []
    for fill in _fills(trades):
        if fill.filled:
            position += side_sign(fill.side) * fill.quantity
        path.append(position)
    return path


def compute_average_holding_period(
    trades: Sequence[ExecutionFill | ExecutionResult],
) -> float | None:
    """Compute a simple index-step holding period for closed positions."""
    position = 0.0
    open_index: int | None = None
    periods: list[int] = []
    for index, fill in enumerate(_fills(trades)):
        previous_position = position
        if fill.filled:
            position += side_sign(fill.side) * fill.quantity
        if abs(previous_position) <= 1e-12 and abs(position) > 1e-12:
            open_index = index
        if (
            open_index is not None
            and abs(previous_position) > 1e-12
            and abs(position) <= 1e-12
        ):
            periods.append(index - open_index)
            open_index = None
    if not periods:
        return None
    return sum(periods) / len(periods)


def compute_turnover(
    trades: Sequence[ExecutionFill | ExecutionResult],
    *,
    use_notional: bool = False,
) -> TurnoverSummary:
    """Summarise absolute traded quantity and notional turnover."""
    fills = _fills(trades)
    filled = [fill for fill in fills if fill.filled and fill.quantity > 0.0]
    n_trades = len(filled)
    total_quantity = sum(fill.quantity for fill in filled)
    total_notional = sum((fill.fill_price or 0.0) * fill.quantity for fill in filled)
    turnover = total_notional if use_notional else total_quantity
    return TurnoverSummary(
        n_trades=n_trades,
        total_quantity=total_quantity,
        total_notional=total_notional,
        turnover=turnover,
        turnover_basis="notional" if use_notional else "quantity",
        average_trade_quantity=(
            total_quantity / n_trades if n_trades > 0 else None
        ),
        average_trade_notional=(
            total_notional / n_trades if n_trades > 0 else None
        ),
        average_holding_period=compute_average_holding_period(fills),
        position_path=compute_position_path(fills),
    )

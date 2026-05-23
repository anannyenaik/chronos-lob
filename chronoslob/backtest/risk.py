"""Deterministic risk constraints for execution-aware validation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from chronoslob.backtest.execution import TradeSide, side_sign

__all__ = [
    "RiskCheck",
    "RiskConfig",
    "RiskState",
    "apply_drawdown_limit",
    "apply_inventory_limit",
    "apply_turnover_limit",
    "should_abstain_for_risk",
    "update_risk_state",
]


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


def _validate_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class RiskConfig:
    """Inventory, turnover and drawdown limits for the simulation."""

    inventory_limit: float | None = None
    max_trades: int | None = None
    max_turnover: float | None = None
    max_drawdown: float | None = None
    turnover_notional: bool = False

    def __post_init__(self) -> None:
        if self.inventory_limit is not None:
            object.__setattr__(
                self,
                "inventory_limit",
                _validate_non_negative_float(
                    self.inventory_limit,
                    name="inventory_limit",
                ),
            )
        if self.max_trades is not None:
            object.__setattr__(
                self,
                "max_trades",
                _validate_positive_int(self.max_trades, name="max_trades"),
            )
        if self.max_turnover is not None:
            object.__setattr__(
                self,
                "max_turnover",
                _validate_non_negative_float(self.max_turnover, name="max_turnover"),
            )
        if self.max_drawdown is not None:
            object.__setattr__(
                self,
                "max_drawdown",
                _validate_non_negative_float(self.max_drawdown, name="max_drawdown"),
            )
        if not isinstance(self.turnover_notional, bool):
            raise TypeError("turnover_notional must be a bool")


@dataclass(frozen=True)
class RiskState:
    """Mutable-by-replacement risk state tracked across validation rows."""

    inventory: float = 0.0
    trade_count: int = 0
    turnover: float = 0.0
    cumulative_net_pnl: float = 0.0
    peak_net_pnl: float = 0.0
    max_drawdown: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "inventory",
            _validate_finite_float(self.inventory, name="inventory"),
        )
        if isinstance(self.trade_count, bool) or not isinstance(self.trade_count, int):
            raise TypeError("trade_count must be an integer")
        if self.trade_count < 0:
            raise ValueError("trade_count must be non-negative")
        object.__setattr__(
            self,
            "turnover",
            _validate_non_negative_float(self.turnover, name="turnover"),
        )
        object.__setattr__(
            self,
            "cumulative_net_pnl",
            _validate_finite_float(self.cumulative_net_pnl, name="cumulative_net_pnl"),
        )
        object.__setattr__(
            self,
            "peak_net_pnl",
            _validate_finite_float(self.peak_net_pnl, name="peak_net_pnl"),
        )
        object.__setattr__(
            self,
            "max_drawdown",
            _validate_non_negative_float(self.max_drawdown, name="max_drawdown"),
        )


@dataclass(frozen=True)
class RiskCheck:
    """Risk constraint outcome with an explicit abstention reason."""

    should_abstain: bool
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.should_abstain, bool):
            raise TypeError("should_abstain must be a bool")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")


def _turnover_increment(
    *,
    quantity: float,
    price: float,
    config: RiskConfig,
) -> float:
    quantity = _validate_non_negative_float(quantity, name="quantity")
    price = _validate_positive_float(price, name="price")
    return price * quantity if config.turnover_notional else quantity


def apply_inventory_limit(
    state: RiskState,
    side: TradeSide | str,
    quantity: float,
    config: RiskConfig,
) -> RiskCheck:
    """Check whether the proposed trade would breach the inventory cap."""
    if not isinstance(state, RiskState):
        raise TypeError("state must be a RiskState")
    if not isinstance(config, RiskConfig):
        raise TypeError("config must be a RiskConfig")
    quantity = _validate_non_negative_float(quantity, name="quantity")
    if config.inventory_limit is None:
        return RiskCheck(False, "inventory_limit_not_configured")
    projected_inventory = state.inventory + side_sign(side) * quantity
    if abs(projected_inventory) > config.inventory_limit:
        return RiskCheck(True, "inventory_limit_exceeded")
    return RiskCheck(False, "inventory_within_limit")


def apply_turnover_limit(
    state: RiskState,
    proposed_turnover: float,
    config: RiskConfig,
) -> RiskCheck:
    """Check whether proposed turnover would breach the configured cap."""
    if not isinstance(state, RiskState):
        raise TypeError("state must be a RiskState")
    if not isinstance(config, RiskConfig):
        raise TypeError("config must be a RiskConfig")
    proposed_turnover = _validate_non_negative_float(
        proposed_turnover,
        name="proposed_turnover",
    )
    if config.max_turnover is None:
        return RiskCheck(False, "turnover_limit_not_configured")
    if state.turnover + proposed_turnover > config.max_turnover:
        return RiskCheck(True, "turnover_limit_exceeded")
    return RiskCheck(False, "turnover_within_limit")


def apply_drawdown_limit(
    state: RiskState,
    config: RiskConfig,
    *,
    projected_net_pnl: float | None = None,
) -> RiskCheck:
    """Check the optional simulated drawdown cap."""
    if not isinstance(state, RiskState):
        raise TypeError("state must be a RiskState")
    if not isinstance(config, RiskConfig):
        raise TypeError("config must be a RiskConfig")
    if config.max_drawdown is None:
        return RiskCheck(False, "drawdown_limit_not_configured")
    if projected_net_pnl is None:
        drawdown = state.max_drawdown
    else:
        projected = state.cumulative_net_pnl + _validate_finite_float(
            projected_net_pnl,
            name="projected_net_pnl",
        )
        peak = max(state.peak_net_pnl, projected)
        drawdown = max(state.max_drawdown, peak - projected)
    if drawdown > config.max_drawdown:
        return RiskCheck(True, "drawdown_limit_exceeded")
    return RiskCheck(False, "drawdown_within_limit")


def should_abstain_for_risk(
    state: RiskState,
    side: TradeSide | str,
    quantity: float,
    price: float,
    config: RiskConfig,
) -> RiskCheck:
    """Return the first risk abstention reason for a proposed trade."""
    inventory_check = apply_inventory_limit(state, side, quantity, config)
    if inventory_check.should_abstain:
        return inventory_check
    if config.max_trades is not None and state.trade_count >= config.max_trades:
        return RiskCheck(True, "max_trades_exceeded")
    proposed_turnover = _turnover_increment(
        quantity=quantity,
        price=price,
        config=config,
    )
    turnover_check = apply_turnover_limit(state, proposed_turnover, config)
    if turnover_check.should_abstain:
        return turnover_check
    drawdown_check = apply_drawdown_limit(state, config)
    if drawdown_check.should_abstain:
        return drawdown_check
    return RiskCheck(False, "risk_constraints_passed")


def update_risk_state(
    state: RiskState,
    side: TradeSide | str,
    quantity: float,
    price: float,
    net_pnl: float,
    config: RiskConfig,
) -> RiskState:
    """Return updated risk state after one filled simulated trade."""
    if not isinstance(state, RiskState):
        raise TypeError("state must be a RiskState")
    if not isinstance(config, RiskConfig):
        raise TypeError("config must be a RiskConfig")
    quantity = _validate_non_negative_float(quantity, name="quantity")
    price = _validate_positive_float(price, name="price")
    net_pnl = _validate_finite_float(net_pnl, name="net_pnl")
    turnover_increment = _turnover_increment(
        quantity=quantity,
        price=price,
        config=config,
    )
    new_inventory = state.inventory + side_sign(side) * quantity
    new_trade_count = state.trade_count + (1 if quantity > 0.0 else 0)
    new_turnover = state.turnover + turnover_increment
    new_cumulative_net_pnl = state.cumulative_net_pnl + net_pnl
    new_peak_net_pnl = max(state.peak_net_pnl, new_cumulative_net_pnl)
    current_drawdown = new_peak_net_pnl - new_cumulative_net_pnl
    return RiskState(
        inventory=new_inventory,
        trade_count=new_trade_count,
        turnover=new_turnover,
        cumulative_net_pnl=new_cumulative_net_pnl,
        peak_net_pnl=new_peak_net_pnl,
        max_drawdown=max(state.max_drawdown, current_drawdown),
    )

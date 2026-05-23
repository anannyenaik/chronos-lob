"""Explicit cost models for simplified execution-aware validation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from chronoslob.backtest.execution import ExecutionMode

__all__ = [
    "CostBreakdown",
    "ExecutionCostConfig",
    "FeeModel",
    "SpreadCostModel",
    "estimate_aggressive_cost",
    "estimate_passive_cost",
    "estimate_total_cost",
]

AGGRESSIVE_HALF_SPREAD = "half_spread"
AGGRESSIVE_FULL_SPREAD = "full_spread"
SUPPORTED_AGGRESSIVE_SPREAD_CONVENTIONS = {
    AGGRESSIVE_HALF_SPREAD,
    AGGRESSIVE_FULL_SPREAD,
}


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


@dataclass(frozen=True)
class FeeModel:
    """Fixed and proportional fees for one filled simulated trade."""

    fixed_fee_per_trade: float = 0.0
    proportional_fee_bps: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fixed_fee_per_trade",
            _validate_non_negative_float(
                self.fixed_fee_per_trade,
                name="fixed_fee_per_trade",
            ),
        )
        object.__setattr__(
            self,
            "proportional_fee_bps",
            _validate_non_negative_float(
                self.proportional_fee_bps,
                name="proportional_fee_bps",
            ),
        )

    def estimate(self, *, price: float, quantity: float) -> float:
        """Estimate non-negative fees for a filled trade."""
        price = _validate_positive_float(price, name="price")
        quantity = _validate_non_negative_float(quantity, name="quantity")
        if quantity == 0.0:
            return 0.0
        notional = price * quantity
        return self.fixed_fee_per_trade + (
            notional * self.proportional_fee_bps / 10_000.0
        )


@dataclass(frozen=True)
class SpreadCostModel:
    """Spread and simplified passive adverse-selection assumptions."""

    aggressive_convention: str = AGGRESSIVE_HALF_SPREAD
    passive_spread_cost_per_unit: float = 0.0
    passive_adverse_selection_bps: float = 0.0

    def __post_init__(self) -> None:
        if self.aggressive_convention not in SUPPORTED_AGGRESSIVE_SPREAD_CONVENTIONS:
            raise ValueError(
                "aggressive_convention must be one of "
                f"{sorted(SUPPORTED_AGGRESSIVE_SPREAD_CONVENTIONS)}"
            )
        object.__setattr__(
            self,
            "passive_spread_cost_per_unit",
            _validate_non_negative_float(
                self.passive_spread_cost_per_unit,
                name="passive_spread_cost_per_unit",
            ),
        )
        object.__setattr__(
            self,
            "passive_adverse_selection_bps",
            _validate_non_negative_float(
                self.passive_adverse_selection_bps,
                name="passive_adverse_selection_bps",
            ),
        )


@dataclass(frozen=True)
class ExecutionCostConfig:
    """Cost configuration for the simplified research simulation."""

    fee_model: FeeModel = field(default_factory=FeeModel)
    spread_model: SpreadCostModel = field(default_factory=SpreadCostModel)

    def __post_init__(self) -> None:
        if not isinstance(self.fee_model, FeeModel):
            raise TypeError("fee_model must be a FeeModel")
        if not isinstance(self.spread_model, SpreadCostModel):
            raise TypeError("spread_model must be a SpreadCostModel")


@dataclass(frozen=True)
class CostBreakdown:
    """Decomposed simulated execution costs."""

    fees: float
    spread_cost: float
    slippage: float
    adverse_selection_cost: float
    total_cost: float
    notional: float

    def __post_init__(self) -> None:
        for name in (
            "fees",
            "spread_cost",
            "slippage",
            "adverse_selection_cost",
            "total_cost",
            "notional",
        ):
            object.__setattr__(
                self,
                name,
                _validate_non_negative_float(getattr(self, name), name=name),
            )
        expected_total = (
            self.fees
            + self.spread_cost
            + self.slippage
            + self.adverse_selection_cost
        )
        if not math.isclose(
            expected_total,
            self.total_cost,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("total_cost must equal the cost component sum")

    def to_dict(self) -> dict[str, float]:
        """Return a serialisable decomposition."""
        return {
            "fees": self.fees,
            "spread_cost": self.spread_cost,
            "slippage": self.slippage,
            "adverse_selection_cost": self.adverse_selection_cost,
            "total_cost": self.total_cost,
            "notional": self.notional,
        }


def _resolve_cost_config(config: ExecutionCostConfig | None) -> ExecutionCostConfig:
    return ExecutionCostConfig() if config is None else config


def estimate_aggressive_cost(
    *,
    price: float,
    quantity: float,
    spread: float,
    config: ExecutionCostConfig | None = None,
    slippage: float = 0.0,
) -> CostBreakdown:
    """Estimate costs for crossing the spread in the research simulation.

    The spread convention is explicit because different analyses may account
    for crossing costs from a mid-price or touch-price reference. No market
    impact is modelled.
    """
    resolved = _resolve_cost_config(config)
    price = _validate_positive_float(price, name="price")
    quantity = _validate_non_negative_float(quantity, name="quantity")
    spread = _validate_non_negative_float(spread, name="spread")
    slippage = _validate_non_negative_float(slippage, name="slippage")
    notional = price * quantity
    if resolved.spread_model.aggressive_convention == AGGRESSIVE_HALF_SPREAD:
        spread_cost = 0.5 * spread * quantity
    else:
        spread_cost = spread * quantity
    fees = resolved.fee_model.estimate(price=price, quantity=quantity)
    total = fees + spread_cost + slippage
    return CostBreakdown(
        fees=fees,
        spread_cost=spread_cost,
        slippage=slippage,
        adverse_selection_cost=0.0,
        total_cost=total,
        notional=notional,
    )


def estimate_passive_cost(
    *,
    price: float,
    quantity: float,
    config: ExecutionCostConfig | None = None,
    adverse_selection: bool = False,
    slippage: float = 0.0,
) -> CostBreakdown:
    """Estimate costs for passive posting under simplified assumptions.

    Passive posting does not pay a spread-crossing cost by default. Optional
    passive spread and adverse-selection costs are explicit configuration
    terms, not a production queue-position or market-impact model.
    """
    resolved = _resolve_cost_config(config)
    price = _validate_positive_float(price, name="price")
    quantity = _validate_non_negative_float(quantity, name="quantity")
    slippage = _validate_non_negative_float(slippage, name="slippage")
    if not isinstance(adverse_selection, bool):
        raise TypeError("adverse_selection must be a bool")
    notional = price * quantity
    fees = resolved.fee_model.estimate(price=price, quantity=quantity)
    spread_cost = resolved.spread_model.passive_spread_cost_per_unit * quantity
    adverse_selection_cost = (
        notional * resolved.spread_model.passive_adverse_selection_bps / 10_000.0
        if adverse_selection
        else 0.0
    )
    total = fees + spread_cost + slippage + adverse_selection_cost
    return CostBreakdown(
        fees=fees,
        spread_cost=spread_cost,
        slippage=slippage,
        adverse_selection_cost=adverse_selection_cost,
        total_cost=total,
        notional=notional,
    )


def estimate_total_cost(
    *,
    mode: ExecutionMode | str,
    price: float,
    quantity: float,
    spread: float,
    config: ExecutionCostConfig | None = None,
    adverse_selection: bool = False,
    slippage: float = 0.0,
) -> CostBreakdown:
    """Estimate total costs for a resolved aggressive or passive fill."""
    resolved_mode = ExecutionMode(mode)
    if resolved_mode is ExecutionMode.AGGRESSIVE:
        return estimate_aggressive_cost(
            price=price,
            quantity=quantity,
            spread=spread,
            config=config,
            slippage=slippage,
        )
    if resolved_mode is ExecutionMode.PASSIVE:
        return estimate_passive_cost(
            price=price,
            quantity=quantity,
            config=config,
            adverse_selection=adverse_selection,
            slippage=slippage,
        )
    raise ValueError("estimate_total_cost expects a resolved aggressive or passive mode")

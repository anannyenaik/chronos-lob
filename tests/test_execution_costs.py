"""Tests for simplified execution cost utilities."""

from __future__ import annotations

import pytest

from chronoslob.backtest.costs import (
    ExecutionCostConfig,
    FeeModel,
    SpreadCostModel,
    estimate_aggressive_cost,
    estimate_passive_cost,
    estimate_total_cost,
)
from chronoslob.backtest.execution import ExecutionMode


def test_fixed_fee_model_applies_per_filled_trade() -> None:
    fees = FeeModel(fixed_fee_per_trade=0.25)

    assert fees.estimate(price=100.0, quantity=2.0) == pytest.approx(0.25)
    assert fees.estimate(price=100.0, quantity=0.0) == 0.0


def test_proportional_bps_fee_model_uses_notional() -> None:
    fees = FeeModel(proportional_fee_bps=2.0)

    assert fees.estimate(price=100.0, quantity=5.0) == pytest.approx(0.1)


def test_aggressive_half_spread_cost_convention() -> None:
    config = ExecutionCostConfig(
        fee_model=FeeModel(),
        spread_model=SpreadCostModel(aggressive_convention="half_spread"),
    )

    cost = estimate_aggressive_cost(
        price=100.0,
        quantity=3.0,
        spread=0.04,
        config=config,
    )

    assert cost.spread_cost == pytest.approx(0.06)
    assert cost.total_cost == pytest.approx(0.06)


def test_aggressive_full_spread_cost_convention() -> None:
    config = ExecutionCostConfig(
        fee_model=FeeModel(),
        spread_model=SpreadCostModel(aggressive_convention="full_spread"),
    )

    cost = estimate_aggressive_cost(
        price=100.0,
        quantity=3.0,
        spread=0.04,
        config=config,
    )

    assert cost.spread_cost == pytest.approx(0.12)
    assert cost.total_cost == pytest.approx(0.12)


def test_passive_cost_excludes_crossing_spread_by_default() -> None:
    cost = estimate_passive_cost(price=100.0, quantity=2.0)

    assert cost.spread_cost == 0.0
    assert cost.total_cost == 0.0


def test_passive_cost_can_include_adverse_selection_assumption() -> None:
    config = ExecutionCostConfig(
        spread_model=SpreadCostModel(passive_adverse_selection_bps=1.0),
    )

    cost = estimate_passive_cost(
        price=50.0,
        quantity=4.0,
        config=config,
        adverse_selection=True,
    )

    assert cost.adverse_selection_cost == pytest.approx(0.02)
    assert cost.total_cost == pytest.approx(0.02)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: FeeModel(fixed_fee_per_trade=-0.01), "fixed_fee_per_trade"),
        (lambda: FeeModel(proportional_fee_bps=-1.0), "proportional_fee_bps"),
        (
            lambda: estimate_aggressive_cost(price=-1.0, quantity=1.0, spread=0.01),
            "price",
        ),
        (
            lambda: estimate_aggressive_cost(price=100.0, quantity=-1.0, spread=0.01),
            "quantity",
        ),
        (
            lambda: estimate_aggressive_cost(price=100.0, quantity=1.0, spread=-0.01),
            "spread",
        ),
    ],
)
def test_invalid_negative_cost_inputs_raise_clearly(
    factory: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()  # type: ignore[operator]


def test_total_cost_decomposition_for_aggressive_mode() -> None:
    config = ExecutionCostConfig(
        fee_model=FeeModel(fixed_fee_per_trade=0.01, proportional_fee_bps=1.0),
        spread_model=SpreadCostModel(aggressive_convention="half_spread"),
    )

    cost = estimate_total_cost(
        mode=ExecutionMode.AGGRESSIVE,
        price=100.0,
        quantity=2.0,
        spread=0.02,
        config=config,
        slippage=0.03,
    )

    assert cost.fees == pytest.approx(0.03)
    assert cost.spread_cost == pytest.approx(0.02)
    assert cost.slippage == pytest.approx(0.03)
    assert cost.total_cost == pytest.approx(0.08)
    assert cost.to_dict()["total_cost"] == pytest.approx(0.08)

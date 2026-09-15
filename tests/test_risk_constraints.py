"""Tests for deterministic execution-validation risk constraints."""

from __future__ import annotations

import pytest

from chronoslob.backtest.execution import TradeSide
from chronoslob.backtest.risk import (
    RiskConfig,
    RiskState,
    apply_drawdown_limit,
    apply_inventory_limit,
    apply_turnover_limit,
    should_abstain_for_risk,
    update_risk_state,
)


def test_inventory_cap_blocks_trades() -> None:
    check = apply_inventory_limit(
        RiskState(inventory=1.0),
        TradeSide.BUY,
        1.0,
        RiskConfig(inventory_limit=1.5),
    )

    assert check.should_abstain
    assert check.reason == "inventory_limit_exceeded"


def test_turnover_cap_blocks_trades() -> None:
    check = apply_turnover_limit(
        RiskState(turnover=4.0),
        proposed_turnover=2.0,
        config=RiskConfig(max_turnover=5.0),
    )

    assert check.should_abstain
    assert check.reason == "turnover_limit_exceeded"


def test_max_trade_count_blocks_trades() -> None:
    check = should_abstain_for_risk(
        RiskState(trade_count=2),
        TradeSide.BUY,
        quantity=1.0,
        price=100.0,
        config=RiskConfig(max_trades=2),
    )

    assert check.should_abstain
    assert check.reason == "max_trades_exceeded"


def test_drawdown_cap_blocks_when_state_already_breached() -> None:
    check = apply_drawdown_limit(
        RiskState(max_drawdown=0.6),
        RiskConfig(max_drawdown=0.5),
    )

    assert check.should_abstain
    assert check.reason == "drawdown_limit_exceeded"


def test_abstention_reason_is_clear_for_combined_risk_check() -> None:
    check = should_abstain_for_risk(
        RiskState(inventory=-2.0),
        TradeSide.SELL,
        quantity=1.0,
        price=100.0,
        config=RiskConfig(inventory_limit=2.5, max_trades=10),
    )

    assert check.should_abstain
    assert "inventory" in check.reason


def test_risk_state_updates_deterministically() -> None:
    state = update_risk_state(
        RiskState(),
        TradeSide.BUY,
        quantity=2.0,
        price=100.0,
        net_pnl=-0.25,
        config=RiskConfig(turnover_notional=True),
    )

    assert state.inventory == pytest.approx(2.0)
    assert state.trade_count == 1
    assert state.turnover == pytest.approx(200.0)
    assert state.cumulative_net_pnl == pytest.approx(-0.25)
    assert state.max_drawdown == pytest.approx(0.25)


def test_risk_check_passes_when_limits_are_not_breached() -> None:
    check = should_abstain_for_risk(
        RiskState(inventory=0.0, trade_count=1, turnover=1.0),
        TradeSide.BUY,
        quantity=1.0,
        price=100.0,
        config=RiskConfig(
            inventory_limit=5.0,
            max_trades=3,
            max_turnover=5.0,
        ),
    )

    assert not check.should_abstain
    assert check.reason == "risk_constraints_passed"

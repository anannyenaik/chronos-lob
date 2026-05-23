"""Tests for simplified execution-aware validation."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from chronoslob.backtest.costs import (
    ExecutionCostConfig,
    FeeModel,
    SpreadCostModel,
)
from chronoslob.backtest.execution import (
    ExecutionMode,
    MarketState,
    PredictionSignal,
    TradeSide,
)
from chronoslob.backtest.latency import LatencyConfig
from chronoslob.backtest.validation import (
    ExecutionValidationConfig,
    confidence_threshold_sweep,
    latency_sensitivity_analysis,
    run_execution_validation,
    run_execution_validation_smoke,
)


def _state(
    index: int,
    mid_price: float,
    *,
    fill_probability: float = 0.8,
    adverse_selection_label: bool = False,
    symbol: str = "TEST",
) -> MarketState:
    spread = 0.02
    return MarketState(
        timestamp=datetime(2024, 1, 1, 9, 30, tzinfo=UTC) + timedelta(seconds=index),
        symbol=symbol,
        mid_price=mid_price,
        best_bid=mid_price - spread / 2.0,
        best_ask=mid_price + spread / 2.0,
        spread=spread,
        bid_size=10.0,
        ask_size=11.0,
        fill_probability=fill_probability,
        adverse_selection_label=adverse_selection_label,
    )


def _states() -> list[MarketState]:
    return [
        _state(0, 100.0, fill_probability=0.9),
        _state(1, 100.2, fill_probability=0.7, adverse_selection_label=True),
        _state(2, 100.1, fill_probability=0.3),
        _state(3, 100.4, fill_probability=0.8),
        _state(4, 100.3, fill_probability=0.6),
        _state(5, 100.5, fill_probability=0.9),
    ]


def _signal(
    index: int,
    *,
    confidence: float = 0.9,
    side: TradeSide = TradeSide.BUY,
) -> PredictionSignal:
    timestamp = datetime(2024, 1, 1, 9, 30, tzinfo=UTC) + timedelta(seconds=index)
    return PredictionSignal(
        timestamp=timestamp,
        symbol="TEST",
        side=side,
        confidence=confidence,
        score=confidence,
        horizon=1,
    )


def _cost_config() -> ExecutionCostConfig:
    return ExecutionCostConfig(
        fee_model=FeeModel(fixed_fee_per_trade=0.01),
        spread_model=SpreadCostModel(
            aggressive_convention="half_spread",
            passive_adverse_selection_bps=1.0,
        ),
    )


def test_aggressive_mode_executes_and_pays_costs() -> None:
    result = run_execution_validation(
        [_signal(0)],
        _states(),
        ExecutionValidationConfig(
            mode=ExecutionMode.AGGRESSIVE,
            costs=_cost_config(),
        ),
    )

    row = result.results[0]

    assert row.decision.should_trade
    assert row.fill.filled
    assert row.fill.mode is ExecutionMode.AGGRESSIVE
    assert row.fill.spread_cost == pytest.approx(0.01)
    assert row.fill.fees == pytest.approx(0.01)
    assert result.summary.n_filled == 1
    assert result.summary.total_cost_simulated == pytest.approx(0.02)


def test_passive_mode_fill_depends_on_fill_probability() -> None:
    states = _states()
    states[0] = _state(0, 100.0, fill_probability=0.2)
    result = run_execution_validation(
        [_signal(0)],
        states,
        ExecutionValidationConfig(
            mode=ExecutionMode.PASSIVE,
            passive_fill_probability_threshold=0.5,
        ),
    )

    row = result.results[0]

    assert row.decision.should_trade
    assert not row.fill.filled
    assert row.fill.metadata["reason"] == "passive_not_filled"
    assert result.summary.fill_rate == 0.0


def test_hybrid_mode_chooses_take_post_and_abstain() -> None:
    signals = [
        _signal(0, confidence=0.9),
        _signal(1, confidence=0.7),
        _signal(2, confidence=0.6),
    ]
    result = run_execution_validation(
        signals,
        _states(),
        ExecutionValidationConfig(
            mode=ExecutionMode.HYBRID,
            high_confidence_threshold=0.85,
            passive_confidence_threshold=0.65,
            passive_fill_probability_threshold=0.5,
        ),
    )

    assert result.results[0].fill.mode is ExecutionMode.AGGRESSIVE
    assert result.results[1].fill.mode is ExecutionMode.PASSIVE
    assert not result.results[2].decision.should_trade
    assert result.results[2].decision.reason == "hybrid_confidence_below_passive_threshold"


def test_confidence_threshold_abstains_below_threshold() -> None:
    result = run_execution_validation(
        [_signal(0, confidence=0.49)],
        _states(),
        ExecutionValidationConfig(min_confidence=0.5),
    )

    assert not result.results[0].decision.should_trade
    assert result.results[0].decision.reason == "confidence_below_threshold"
    assert result.summary.n_abstained == 1


def test_latency_affects_execution_state() -> None:
    result = run_execution_validation(
        [_signal(0)],
        _states(),
        ExecutionValidationConfig(
            mode=ExecutionMode.AGGRESSIVE,
            latency=LatencyConfig(latency_steps=1),
        ),
    )

    row = result.results[0]

    assert row.fill.timestamp == _states()[1].timestamp
    assert row.price_move == pytest.approx(_states()[2].mid_price - _states()[1].mid_price)


def test_adverse_selection_rate_is_computed() -> None:
    states = _states()
    states[0] = _state(0, 100.0, fill_probability=0.8, adverse_selection_label=True)
    result = run_execution_validation(
        [_signal(0, confidence=0.8)],
        states,
        ExecutionValidationConfig(
            mode=ExecutionMode.PASSIVE,
            min_confidence=0.5,
            passive_fill_probability_threshold=0.5,
            costs=_cost_config(),
        ),
    )

    assert result.results[0].fill.adverse_selection
    assert result.summary.adverse_selection_rate == pytest.approx(1.0)
    assert result.results[0].fill.slippage > 0.0


def test_net_simulated_pnl_equals_gross_minus_costs() -> None:
    result = run_execution_validation(
        [_signal(0)],
        _states(),
        ExecutionValidationConfig(
            mode=ExecutionMode.AGGRESSIVE,
            costs=_cost_config(),
        ),
    )
    row = result.results[0]

    assert row.net_pnl == pytest.approx(row.gross_pnl - row.cost)
    assert result.summary.net_pnl_simulated == pytest.approx(
        result.summary.gross_pnl_simulated - result.summary.total_cost_simulated
    )


def test_turnover_summary_is_correct() -> None:
    result = run_execution_validation(
        [_signal(0), _signal(1)],
        _states(),
        ExecutionValidationConfig(
            mode=ExecutionMode.AGGRESSIVE,
            quantity=2.0,
        ),
    )

    assert result.summary.n_filled == 2
    assert result.summary.turnover == pytest.approx(4.0)
    assert result.summary.average_turnover == pytest.approx(2.0)


def test_confidence_threshold_sweep_works() -> None:
    rows = confidence_threshold_sweep(
        [_signal(0, confidence=0.55), _signal(1, confidence=0.85)],
        _states(),
        thresholds=[0.5, 0.8],
        config=ExecutionValidationConfig(mode=ExecutionMode.AGGRESSIVE),
    )

    assert [row["threshold"] for row in rows] == [0.5, 0.8]
    assert rows[0]["n_trades"] >= rows[1]["n_trades"]


def test_latency_sensitivity_analysis_works() -> None:
    rows = latency_sensitivity_analysis(
        [_signal(0), _signal(1)],
        _states(),
        latency_steps=[0, 1, 2],
        config=ExecutionValidationConfig(mode=ExecutionMode.AGGRESSIVE),
    )

    assert [row["latency_steps"] for row in rows] == [0, 1, 2]
    assert all("net_pnl_simulated" in row for row in rows)


def test_smoke_runner_returns_synthetic_only_payload() -> None:
    result = run_execution_validation_smoke(n_signals=12, seed=7)

    assert result["synthetic_plumbing_only"] is True
    assert result["live_trading"] is False
    assert result["write_outputs"] is False
    assert "confidence_threshold_sweep" in result
    assert "latency_sensitivity" in result


def test_no_market_performance_claim_fields_are_present() -> None:
    payload = json.dumps(run_execution_validation_smoke(n_signals=8, seed=2)).lower()

    for forbidden in ("sharpe", "guaranteed", "deployable", "live_profit"):
        assert forbidden not in payload


def test_invalid_inputs_fail_clearly() -> None:
    bad_states = _states()
    bad_states[0] = _state(0, 100.0, symbol="OTHER")

    with pytest.raises(ValueError, match="symbols must align"):
        run_execution_validation([_signal(0)], bad_states)

    with pytest.raises(ValueError, match="quantity"):
        ExecutionValidationConfig(quantity=0.0)

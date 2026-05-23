"""Execution-aware validation for cost-adjusted signal diagnostics.

This module implements a deterministic, offline simulation layer. It evaluates
whether prediction-like signals survive explicit spread costs, fees, latency,
passive fill assumptions, adverse-selection labels and simple risk constraints.
It is not a live trading system and does not model production market impact.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from itertools import pairwise

from chronoslob.backtest.costs import (
    CostBreakdown,
    ExecutionCostConfig,
    FeeModel,
    SpreadCostModel,
    estimate_total_cost,
)
from chronoslob.backtest.execution import (
    ExecutionDecision,
    ExecutionFill,
    ExecutionMode,
    ExecutionResult,
    MarketState,
    PredictionSignal,
    TradeSide,
    side_sign,
)
from chronoslob.backtest.latency import (
    DEFAULT_LATENCY_GRID,
    LatencyConfig,
    apply_latency,
)
from chronoslob.backtest.risk import (
    RiskConfig,
    RiskState,
    should_abstain_for_risk,
    update_risk_state,
)
from chronoslob.backtest.turnover import compute_turnover

__all__ = [
    "DEFAULT_CONFIDENCE_BUCKET_EDGES",
    "ExecutionValidationConfig",
    "ExecutionValidationResult",
    "ExecutionValidationSummary",
    "confidence_threshold_sweep",
    "latency_sensitivity_analysis",
    "run_execution_validation",
    "run_execution_validation_smoke",
    "summarise_execution_results",
]

DEFAULT_CONFIDENCE_BUCKET_EDGES: tuple[float, ...] = (0.0, 0.5, 0.7, 0.85, 1.0)
DEFAULT_CONFIDENCE_THRESHOLDS: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8)
SYNTHETIC_EXECUTION_WARNING = (
    "Synthetic execution-validation plumbing only; simulated PnL is not alpha, "
    "tradability evidence, benchmark performance or live trading performance."
)


def _validate_probability(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must satisfy 0 <= {name} <= 1")
    return numeric


def _validate_positive_float(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if numeric <= 0.0:
        raise ValueError(f"{name} must be positive")
    return numeric


def _validate_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_thresholds(thresholds: Sequence[float]) -> tuple[float, ...]:
    if not isinstance(thresholds, Sequence):
        raise TypeError("thresholds must be a sequence")
    cleaned = tuple(_validate_probability(value, name="threshold") for value in thresholds)
    if not cleaned:
        raise ValueError("thresholds must not be empty")
    return cleaned


def _validate_confidence_bucket_edges(edges: Sequence[float]) -> tuple[float, ...]:
    if not isinstance(edges, Sequence):
        raise TypeError("confidence_bucket_edges must be a sequence")
    cleaned = tuple(_validate_probability(value, name="confidence_bucket_edge") for value in edges)
    if len(cleaned) < 2:
        raise ValueError("confidence_bucket_edges must contain at least two values")
    if any(left >= right for left, right in pairwise(cleaned)):
        raise ValueError("confidence_bucket_edges must be strictly increasing")
    return cleaned


@dataclass(frozen=True)
class ExecutionValidationConfig:
    """Configuration for deterministic execution-aware validation."""

    mode: ExecutionMode = ExecutionMode.AGGRESSIVE
    quantity: float = 1.0
    min_confidence: float = 0.5
    high_confidence_threshold: float = 0.85
    passive_confidence_threshold: float = 0.65
    passive_fill_probability_threshold: float = 0.5
    realised_horizon_steps: int = 1
    latency: LatencyConfig = field(default_factory=LatencyConfig)
    costs: ExecutionCostConfig = field(default_factory=ExecutionCostConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    confidence_bucket_edges: tuple[float, ...] = DEFAULT_CONFIDENCE_BUCKET_EDGES

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", ExecutionMode(self.mode))
        object.__setattr__(
            self,
            "quantity",
            _validate_positive_float(self.quantity, name="quantity"),
        )
        object.__setattr__(
            self,
            "min_confidence",
            _validate_probability(self.min_confidence, name="min_confidence"),
        )
        object.__setattr__(
            self,
            "high_confidence_threshold",
            _validate_probability(
                self.high_confidence_threshold,
                name="high_confidence_threshold",
            ),
        )
        object.__setattr__(
            self,
            "passive_confidence_threshold",
            _validate_probability(
                self.passive_confidence_threshold,
                name="passive_confidence_threshold",
            ),
        )
        object.__setattr__(
            self,
            "passive_fill_probability_threshold",
            _validate_probability(
                self.passive_fill_probability_threshold,
                name="passive_fill_probability_threshold",
            ),
        )
        object.__setattr__(
            self,
            "realised_horizon_steps",
            _validate_positive_int(
                self.realised_horizon_steps,
                name="realised_horizon_steps",
            ),
        )
        if not isinstance(self.latency, LatencyConfig):
            raise TypeError("latency must be a LatencyConfig")
        if not isinstance(self.costs, ExecutionCostConfig):
            raise TypeError("costs must be an ExecutionCostConfig")
        if not isinstance(self.risk, RiskConfig):
            raise TypeError("risk must be a RiskConfig")
        object.__setattr__(
            self,
            "confidence_bucket_edges",
            _validate_confidence_bucket_edges(self.confidence_bucket_edges),
        )


@dataclass(frozen=True)
class ExecutionValidationSummary:
    """Summary metrics for execution-aware validation."""

    n_signals: int
    n_trades: int
    n_filled: int
    n_abstained: int
    coverage: float
    fill_rate: float
    hit_rate: float | None
    gross_pnl_simulated: float
    total_cost_simulated: float
    net_pnl_simulated: float
    average_net_pnl_simulated: float | None
    turnover: float
    average_turnover: float | None
    max_drawdown_simulated: float
    average_confidence_traded: float | None
    average_confidence_abstained: float | None
    adverse_selection_rate: float | None
    results_by_confidence_bucket: list[dict[str, float | int]]
    results_by_latency_step: list[dict[str, float | int]]

    @property
    def gross_pnl(self) -> float:
        """Backward-compatible shorthand for simulated gross PnL."""
        return self.gross_pnl_simulated

    @property
    def total_cost(self) -> float:
        """Backward-compatible shorthand for simulated total cost."""
        return self.total_cost_simulated

    @property
    def net_pnl(self) -> float:
        """Backward-compatible shorthand for simulated net PnL."""
        return self.net_pnl_simulated

    @property
    def average_net_pnl(self) -> float | None:
        """Backward-compatible shorthand for average simulated net PnL."""
        return self.average_net_pnl_simulated

    @property
    def max_drawdown(self) -> float:
        """Backward-compatible shorthand for simulated drawdown."""
        return self.max_drawdown_simulated

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable metric payload."""
        return {
            "n_signals": self.n_signals,
            "n_trades": self.n_trades,
            "n_filled": self.n_filled,
            "n_abstained": self.n_abstained,
            "coverage": self.coverage,
            "fill_rate": self.fill_rate,
            "hit_rate": self.hit_rate,
            "gross_pnl_simulated": self.gross_pnl_simulated,
            "total_cost_simulated": self.total_cost_simulated,
            "net_pnl_simulated": self.net_pnl_simulated,
            "cost_adjusted_signal_pnl": self.net_pnl_simulated,
            "average_net_pnl_simulated": self.average_net_pnl_simulated,
            "turnover": self.turnover,
            "average_turnover": self.average_turnover,
            "max_drawdown_simulated": self.max_drawdown_simulated,
            "average_confidence_traded": self.average_confidence_traded,
            "average_confidence_abstained": self.average_confidence_abstained,
            "adverse_selection_rate": self.adverse_selection_rate,
            "results_by_confidence_bucket": list(self.results_by_confidence_bucket),
            "results_by_latency_step": list(self.results_by_latency_step),
        }


@dataclass(frozen=True)
class ExecutionValidationResult:
    """Full validation output for one run."""

    config: ExecutionValidationConfig
    results: list[ExecutionResult]
    summary: ExecutionValidationSummary
    synthetic_only: bool = False
    notes: str = (
        "Execution-aware validation metrics are simplified simulation "
        "diagnostics, not live trading or tradability claims."
    )

    def to_dict(self) -> dict[str, object]:
        """Return a compact serialisable payload."""
        return {
            "synthetic_only": self.synthetic_only,
            "notes": self.notes,
            "mode": self.config.mode.value,
            "latency_steps": self.config.latency.latency_steps,
            "summary": self.summary.to_dict(),
        }


def _validate_sequences(
    signals: Sequence[PredictionSignal],
    market_states: Sequence[MarketState],
) -> tuple[list[PredictionSignal], list[MarketState]]:
    if not isinstance(signals, Sequence):
        raise TypeError("signals must be a sequence")
    if not isinstance(market_states, Sequence):
        raise TypeError("market_states must be a sequence")
    signal_list = list(signals)
    state_list = list(market_states)
    for signal in signal_list:
        if not isinstance(signal, PredictionSignal):
            raise TypeError("signals must contain PredictionSignal instances")
    for state in state_list:
        if not isinstance(state, MarketState):
            raise TypeError("market_states must contain MarketState instances")
    if signal_list and not state_list:
        raise ValueError("market_states must not be empty when signals are provided")
    if len(signal_list) > len(state_list):
        raise ValueError("market_states must contain at least as many rows as signals")
    for index, signal in enumerate(signal_list):
        if signal.symbol != state_list[index].symbol:
            raise ValueError("signal and market_state symbols must align by row")
    return signal_list, state_list


def _order_price(side: TradeSide, mode: ExecutionMode, state: MarketState) -> float:
    if mode is ExecutionMode.AGGRESSIVE:
        return state.best_ask if side is TradeSide.BUY else state.best_bid
    if mode is ExecutionMode.PASSIVE:
        return state.best_bid if side is TradeSide.BUY else state.best_ask
    raise ValueError("order price requires a resolved aggressive or passive mode")


def _abstained_result(
    signal: PredictionSignal,
    *,
    mode: ExecutionMode,
    reason: str,
    inventory_after: float,
    latency_steps: int = 0,
) -> ExecutionResult:
    decision = ExecutionDecision(
        timestamp=signal.timestamp,
        symbol=signal.symbol,
        mode=mode,
        side=signal.side,
        should_trade=False,
        reason=reason,
        confidence=signal.confidence,
        order_price=None,
        quantity=0.0,
    )
    fill = ExecutionFill(
        timestamp=signal.timestamp,
        symbol=signal.symbol,
        mode=mode,
        side=signal.side,
        filled=False,
        fill_price=None,
        quantity=0.0,
        fees=0.0,
        spread_cost=0.0,
        slippage=0.0,
        latency_steps=latency_steps,
        adverse_selection=False,
        metadata={"reason": reason},
    )
    return ExecutionResult(
        decision=decision,
        fill=fill,
        price_move=0.0,
        realised_return=0.0,
        gross_pnl=0.0,
        net_pnl=0.0,
        cost=0.0,
        inventory_after=inventory_after,
        turnover=0.0,
    )


def _resolve_mode(
    signal: PredictionSignal,
    state: MarketState,
    config: ExecutionValidationConfig,
) -> tuple[ExecutionMode, bool, str]:
    if config.mode is ExecutionMode.AGGRESSIVE:
        return ExecutionMode.AGGRESSIVE, True, "aggressive_take"
    if config.mode is ExecutionMode.PASSIVE:
        return ExecutionMode.PASSIVE, True, "passive_post"
    if signal.confidence >= config.high_confidence_threshold:
        return ExecutionMode.AGGRESSIVE, True, "hybrid_high_confidence_take"
    if signal.confidence < config.passive_confidence_threshold:
        return ExecutionMode.HYBRID, False, "hybrid_confidence_below_passive_threshold"
    if state.fill_probability is None:
        return ExecutionMode.PASSIVE, False, "hybrid_passive_fill_probability_missing"
    if state.fill_probability < config.passive_fill_probability_threshold:
        return ExecutionMode.PASSIVE, False, "hybrid_passive_fill_probability_below_threshold"
    return ExecutionMode.PASSIVE, True, "hybrid_passive_post"


def _simulate_fill(
    *,
    signal: PredictionSignal,
    execution_state: MarketState,
    mode: ExecutionMode,
    decision: ExecutionDecision,
    config: ExecutionValidationConfig,
) -> tuple[ExecutionFill, CostBreakdown | None]:
    if mode is ExecutionMode.AGGRESSIVE:
        filled = True
        fill_reason = "aggressive_immediate_fill"
    else:
        fill_probability = execution_state.fill_probability
        filled = (
            fill_probability is not None
            and fill_probability >= config.passive_fill_probability_threshold
        )
        fill_reason = "passive_fill_proxy_met" if filled else "passive_not_filled"

    if not filled:
        return (
            ExecutionFill(
                timestamp=execution_state.timestamp,
                symbol=signal.symbol,
                mode=mode,
                side=signal.side,
                filled=False,
                fill_price=None,
                quantity=0.0,
                fees=0.0,
                spread_cost=0.0,
                slippage=0.0,
                latency_steps=config.latency.latency_steps,
                adverse_selection=False,
                metadata={
                    "reason": fill_reason,
                    "fill_probability": execution_state.fill_probability,
                },
            ),
            None,
        )

    if decision.order_price is None:
        raise ValueError("filled decision requires an order_price")
    adverse_selection = bool(execution_state.adverse_selection_label)
    cost_breakdown = estimate_total_cost(
        mode=mode,
        price=decision.order_price,
        quantity=decision.quantity,
        spread=execution_state.spread,
        config=config.costs,
        adverse_selection=adverse_selection,
    )
    fill = ExecutionFill(
        timestamp=execution_state.timestamp,
        symbol=signal.symbol,
        mode=mode,
        side=signal.side,
        filled=True,
        fill_price=decision.order_price,
        quantity=decision.quantity,
        fees=cost_breakdown.fees,
        spread_cost=cost_breakdown.spread_cost,
        slippage=cost_breakdown.slippage + cost_breakdown.adverse_selection_cost,
        latency_steps=config.latency.latency_steps,
        adverse_selection=adverse_selection,
        metadata={
            "reason": fill_reason,
            "fill_probability": execution_state.fill_probability,
            "cost_breakdown": cost_breakdown.to_dict(),
        },
    )
    return fill, cost_breakdown


def _turnover_amount(fill: ExecutionFill, config: ExecutionValidationConfig) -> float:
    if not fill.filled or fill.fill_price is None:
        return 0.0
    if config.risk.turnover_notional:
        return fill.fill_price * fill.quantity
    return fill.quantity


def run_execution_validation(
    signals: Sequence[PredictionSignal],
    market_states: Sequence[MarketState],
    config: ExecutionValidationConfig | None = None,
) -> ExecutionValidationResult:
    """Run deterministic execution-aware validation over aligned rows."""
    resolved_config = ExecutionValidationConfig() if config is None else config
    if not isinstance(resolved_config, ExecutionValidationConfig):
        raise TypeError("config must be an ExecutionValidationConfig")
    signal_list, state_list = _validate_sequences(signals, market_states)
    risk_state = RiskState()
    results: list[ExecutionResult] = []

    for index, signal in enumerate(signal_list):
        current_state = state_list[index]
        if signal.confidence < resolved_config.min_confidence:
            results.append(
                _abstained_result(
                    signal,
                    mode=resolved_config.mode,
                    reason="confidence_below_threshold",
                    inventory_after=risk_state.inventory,
                )
            )
            continue

        risk_check = should_abstain_for_risk(
            risk_state,
            signal.side,
            resolved_config.quantity,
            current_state.mid_price,
            resolved_config.risk,
        )
        if risk_check.should_abstain:
            results.append(
                _abstained_result(
                    signal,
                    mode=resolved_config.mode,
                    reason=risk_check.reason,
                    inventory_after=risk_state.inventory,
                )
            )
            continue

        latency_result = apply_latency(index, state_list, resolved_config.latency)
        if not latency_result.executable or latency_result.state is None:
            results.append(
                _abstained_result(
                    signal,
                    mode=resolved_config.mode,
                    reason=latency_result.reason,
                    inventory_after=risk_state.inventory,
                    latency_steps=resolved_config.latency.latency_steps,
                )
            )
            continue

        execution_state = latency_result.state
        horizon_index = latency_result.target_index + resolved_config.realised_horizon_steps
        if horizon_index >= len(state_list):
            results.append(
                _abstained_result(
                    signal,
                    mode=resolved_config.mode,
                    reason="realisation_horizon_out_of_range",
                    inventory_after=risk_state.inventory,
                    latency_steps=resolved_config.latency.latency_steps,
                )
            )
            continue
        future_state = state_list[horizon_index]

        mode, should_trade, reason = _resolve_mode(signal, execution_state, resolved_config)
        if not should_trade:
            results.append(
                _abstained_result(
                    signal,
                    mode=mode,
                    reason=reason,
                    inventory_after=risk_state.inventory,
                    latency_steps=resolved_config.latency.latency_steps,
                )
            )
            continue

        order_price = _order_price(signal.side, mode, execution_state)
        decision = ExecutionDecision(
            timestamp=execution_state.timestamp,
            symbol=signal.symbol,
            mode=mode,
            side=signal.side,
            should_trade=True,
            reason=reason,
            confidence=signal.confidence,
            order_price=order_price,
            quantity=resolved_config.quantity,
        )
        fill, cost_breakdown = _simulate_fill(
            signal=signal,
            execution_state=execution_state,
            mode=mode,
            decision=decision,
            config=resolved_config,
        )
        if not fill.filled or cost_breakdown is None:
            results.append(
                ExecutionResult(
                    decision=decision,
                    fill=fill,
                    price_move=0.0,
                    realised_return=0.0,
                    gross_pnl=0.0,
                    net_pnl=0.0,
                    cost=0.0,
                    inventory_after=risk_state.inventory,
                    turnover=0.0,
                )
            )
            continue

        signed_price_move = side_sign(signal.side) * (
            future_state.mid_price - execution_state.mid_price
        )
        realised_return = signed_price_move / execution_state.mid_price
        gross_pnl = signed_price_move * fill.quantity
        net_pnl = gross_pnl - cost_breakdown.total_cost
        turnover_amount = _turnover_amount(fill, resolved_config)
        risk_state = update_risk_state(
            risk_state,
            signal.side,
            fill.quantity,
            fill.fill_price or execution_state.mid_price,
            net_pnl,
            resolved_config.risk,
        )
        results.append(
            ExecutionResult(
                decision=decision,
                fill=fill,
                price_move=signed_price_move,
                realised_return=realised_return,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                cost=cost_breakdown.total_cost,
                inventory_after=risk_state.inventory,
                turnover=turnover_amount,
            )
        )

    summary = summarise_execution_results(
        results,
        confidence_bucket_edges=resolved_config.confidence_bucket_edges,
    )
    return ExecutionValidationResult(
        config=resolved_config,
        results=results,
        summary=summary,
    )


def _average(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _max_drawdown(results: Sequence[ExecutionResult]) -> float:
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for result in results:
        cumulative += result.net_pnl
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
    return max_drawdown


def _confidence_bucket_rows(
    results: Sequence[ExecutionResult],
    edges: Sequence[float],
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for lower, upper in pairwise(edges):
        bucket = [
            result
            for result in results
            if result.decision.confidence >= lower
            and (
                result.decision.confidence < upper
                or (
                    math.isclose(result.decision.confidence, upper)
                    and math.isclose(upper, edges[-1])
                )
            )
        ]
        filled = [result for result in bucket if result.fill.filled]
        net_pnl = sum(result.net_pnl for result in filled)
        rows.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "n_signals": len(bucket),
                "n_trades": sum(1 for result in bucket if result.decision.should_trade),
                "n_filled": len(filled),
                "net_pnl_simulated": net_pnl,
                "average_net_pnl_simulated": (
                    net_pnl / len(filled) if filled else 0.0
                ),
            }
        )
    return rows


def _latency_bucket_rows(
    results: Sequence[ExecutionResult],
) -> list[dict[str, float | int]]:
    latency_steps = sorted({result.fill.latency_steps for result in results})
    rows: list[dict[str, float | int]] = []
    for latency_step in latency_steps:
        bucket = [
            result for result in results if result.fill.latency_steps == latency_step
        ]
        filled = [result for result in bucket if result.fill.filled]
        rows.append(
            {
                "latency_steps": latency_step,
                "n_signals": len(bucket),
                "n_filled": len(filled),
                "net_pnl_simulated": sum(result.net_pnl for result in filled),
                "total_cost_simulated": sum(result.cost for result in filled),
            }
        )
    return rows


def summarise_execution_results(
    results: Sequence[ExecutionResult],
    *,
    confidence_bucket_edges: Sequence[float] = DEFAULT_CONFIDENCE_BUCKET_EDGES,
) -> ExecutionValidationSummary:
    """Summarise execution-validation rows as simulation metrics."""
    result_list = list(results)
    for result in result_list:
        if not isinstance(result, ExecutionResult):
            raise TypeError("results must contain ExecutionResult instances")
    edges = _validate_confidence_bucket_edges(confidence_bucket_edges)
    n_signals = len(result_list)
    n_trades = sum(1 for result in result_list if result.decision.should_trade)
    filled = [result for result in result_list if result.fill.filled]
    n_filled = len(filled)
    n_abstained = n_signals - n_trades
    gross_pnl = sum(result.gross_pnl for result in filled)
    total_cost = sum(result.cost for result in filled)
    net_pnl = sum(result.net_pnl for result in filled)
    turnover_summary = compute_turnover(result_list)
    traded_confidence = [
        result.decision.confidence
        for result in result_list
        if result.decision.should_trade
    ]
    abstained_confidence = [
        result.decision.confidence
        for result in result_list
        if not result.decision.should_trade
    ]
    adverse_count = sum(1 for result in filled if result.fill.adverse_selection)
    positive_gross_count = sum(1 for result in filled if result.gross_pnl > 0.0)
    return ExecutionValidationSummary(
        n_signals=n_signals,
        n_trades=n_trades,
        n_filled=n_filled,
        n_abstained=n_abstained,
        coverage=(n_trades / n_signals if n_signals else 0.0),
        fill_rate=(n_filled / n_trades if n_trades else 0.0),
        hit_rate=(positive_gross_count / n_filled if n_filled else None),
        gross_pnl_simulated=gross_pnl,
        total_cost_simulated=total_cost,
        net_pnl_simulated=net_pnl,
        average_net_pnl_simulated=(net_pnl / n_filled if n_filled else None),
        turnover=turnover_summary.turnover,
        average_turnover=(
            turnover_summary.turnover / n_filled if n_filled else None
        ),
        max_drawdown_simulated=_max_drawdown(result_list),
        average_confidence_traded=_average(traded_confidence),
        average_confidence_abstained=_average(abstained_confidence),
        adverse_selection_rate=(adverse_count / n_filled if n_filled else None),
        results_by_confidence_bucket=_confidence_bucket_rows(result_list, edges),
        results_by_latency_step=_latency_bucket_rows(result_list),
    )


def confidence_threshold_sweep(
    signals: Sequence[PredictionSignal],
    market_states: Sequence[MarketState],
    thresholds: Sequence[float] = DEFAULT_CONFIDENCE_THRESHOLDS,
    config: ExecutionValidationConfig | None = None,
) -> list[dict[str, float | int]]:
    """Evaluate coverage versus simulated net PnL over confidence thresholds."""
    resolved_config = ExecutionValidationConfig() if config is None else config
    rows: list[dict[str, float | int]] = []
    for threshold in _validate_thresholds(thresholds):
        run_config = replace(resolved_config, min_confidence=threshold)
        result = run_execution_validation(signals, market_states, run_config)
        rows.append(
            {
                "threshold": threshold,
                "n_signals": result.summary.n_signals,
                "n_trades": result.summary.n_trades,
                "n_filled": result.summary.n_filled,
                "coverage": result.summary.coverage,
                "fill_rate": result.summary.fill_rate,
                "net_pnl_simulated": result.summary.net_pnl_simulated,
                "total_cost_simulated": result.summary.total_cost_simulated,
                "turnover": result.summary.turnover,
            }
        )
    return rows


def latency_sensitivity_analysis(
    signals: Sequence[PredictionSignal],
    market_states: Sequence[MarketState],
    latency_steps: Sequence[int] = DEFAULT_LATENCY_GRID,
    config: ExecutionValidationConfig | None = None,
) -> list[dict[str, float | int]]:
    """Evaluate simulation metrics over a deterministic latency grid."""
    resolved_config = ExecutionValidationConfig() if config is None else config
    rows: list[dict[str, float | int]] = []
    for latency_step in latency_steps:
        latency = LatencyConfig(latency_steps=latency_step)
        run_config = replace(resolved_config, latency=latency)
        result = run_execution_validation(signals, market_states, run_config)
        rows.append(
            {
                "latency_steps": latency_step,
                "n_signals": result.summary.n_signals,
                "n_trades": result.summary.n_trades,
                "n_filled": result.summary.n_filled,
                "coverage": result.summary.coverage,
                "fill_rate": result.summary.fill_rate,
                "net_pnl_simulated": result.summary.net_pnl_simulated,
                "total_cost_simulated": result.summary.total_cost_simulated,
                "turnover": result.summary.turnover,
            }
        )
    return rows


def _synthetic_market_states(
    *,
    n_rows: int,
    seed: int,
) -> list[MarketState]:
    generator = random.Random(seed)
    start = datetime(2024, 1, 2, 9, 30, tzinfo=UTC)
    states: list[MarketState] = []
    for index in range(n_rows):
        drift = 0.025 * index
        wave = 0.08 * math.sin(index / 3.0)
        noise = (generator.random() - 0.5) * 0.01
        mid = 100.0 + drift + wave + noise
        spread = 0.02 + (0.01 if index % 4 == 0 else 0.0)
        best_bid = mid - spread / 2.0
        best_ask = mid + spread / 2.0
        fill_probability = (0.35, 0.55, 0.75, 0.9)[index % 4]
        states.append(
            MarketState(
                timestamp=start + timedelta(seconds=index),
                symbol="SYNTH",
                mid_price=mid,
                best_bid=best_bid,
                best_ask=best_ask,
                spread=spread,
                bid_size=10.0 + (index % 3),
                ask_size=9.0 + (index % 4),
                volatility=0.001 + 0.0001 * (index % 5),
                fill_probability=fill_probability,
                adverse_selection_label=(index % 6 == 0),
                metadata={"synthetic_row": index},
            )
        )
    return states


def _synthetic_prediction_signals(
    states: Sequence[MarketState],
    *,
    n_signals: int,
    horizon: int,
) -> list[PredictionSignal]:
    signals: list[PredictionSignal] = []
    confidence_cycle = (0.42, 0.58, 0.69, 0.78, 0.88, 0.94)
    for index in range(n_signals):
        current = states[index]
        future = states[index + horizon]
        side = TradeSide.BUY if future.mid_price >= current.mid_price else TradeSide.SELL
        if index % 9 == 0:
            side = TradeSide.SELL if side is TradeSide.BUY else TradeSide.BUY
        confidence = confidence_cycle[index % len(confidence_cycle)]
        signals.append(
            PredictionSignal(
                timestamp=current.timestamp,
                symbol=current.symbol,
                side=side,
                confidence=confidence,
                score=confidence * side_sign(side),
                horizon=horizon,
                metadata={"synthetic_signal": index},
            )
        )
    return signals


def run_execution_validation_smoke(
    *,
    n_signals: int = 24,
    seed: int = 42,
    confidence_thresholds: Sequence[float] = DEFAULT_CONFIDENCE_THRESHOLDS,
    latency_grid: Sequence[int] = (0, 1, 2, 5),
) -> dict[str, object]:
    """Run a deterministic synthetic execution-validation smoke check."""
    _validate_positive_int(n_signals, name="n_signals")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    horizon = 2
    states = _synthetic_market_states(
        n_rows=n_signals + max(latency_grid, default=0) + horizon + 1,
        seed=seed,
    )
    signals = _synthetic_prediction_signals(
        states,
        n_signals=n_signals,
        horizon=horizon,
    )
    cost_config = ExecutionCostConfig(
        fee_model=FeeModel(fixed_fee_per_trade=0.01, proportional_fee_bps=1.0),
        spread_model=SpreadCostModel(
            aggressive_convention="half_spread",
            passive_adverse_selection_bps=0.5,
        ),
    )
    base_config = ExecutionValidationConfig(
        quantity=1.0,
        min_confidence=0.5,
        high_confidence_threshold=0.85,
        passive_confidence_threshold=0.65,
        passive_fill_probability_threshold=0.55,
        realised_horizon_steps=horizon,
        latency=LatencyConfig(latency_steps=1),
        costs=cost_config,
        risk=RiskConfig(
            inventory_limit=4.0,
            max_trades=20,
            max_turnover=20.0,
            max_drawdown=1.0,
        ),
    )
    mode_summaries: dict[str, object] = {}
    for mode in (
        ExecutionMode.AGGRESSIVE,
        ExecutionMode.PASSIVE,
        ExecutionMode.HYBRID,
    ):
        result = run_execution_validation(
            signals,
            states,
            replace(base_config, mode=mode),
        )
        mode_summaries[mode.value] = result.summary.to_dict()

    primary_config = replace(base_config, mode=ExecutionMode.HYBRID)
    primary = run_execution_validation(signals, states, primary_config)
    return {
        "synthetic_plumbing_only": True,
        "notes": SYNTHETIC_EXECUTION_WARNING,
        "seed": seed,
        "n_signals": n_signals,
        "market_state_rows": len(states),
        "execution_modes": [mode.value for mode in ExecutionMode],
        "primary_mode": ExecutionMode.HYBRID.value,
        "summary": primary.summary.to_dict(),
        "mode_summaries": mode_summaries,
        "confidence_threshold_sweep": confidence_threshold_sweep(
            signals,
            states,
            confidence_thresholds,
            primary_config,
        ),
        "latency_sensitivity": latency_sensitivity_analysis(
            signals,
            states,
            latency_grid,
            primary_config,
        ),
        "write_outputs": False,
        "network_calls": "none",
        "live_trading": False,
        "market_impact_model": "not implemented; explicit limitation",
    }

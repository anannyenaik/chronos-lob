"""Latency utilities for execution-aware validation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from chronoslob.backtest.execution import MarketState

__all__ = [
    "DEFAULT_LATENCY_GRID",
    "LatencyConfig",
    "LatencyResult",
    "apply_latency",
    "get_latency_state",
    "latency_sensitivity_grid",
]

DEFAULT_LATENCY_GRID: tuple[int, ...] = (0, 1, 2, 5, 10)


def _validate_non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True)
class LatencyConfig:
    """Latency expressed in row/event steps."""

    latency_steps: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "latency_steps",
            _validate_non_negative_int(self.latency_steps, name="latency_steps"),
        )


@dataclass(frozen=True)
class LatencyResult:
    """Market-state lookup outcome after applying latency."""

    current_index: int
    latency_steps: int
    target_index: int
    executable: bool
    state: MarketState | None
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "current_index",
            _validate_non_negative_int(self.current_index, name="current_index"),
        )
        object.__setattr__(
            self,
            "latency_steps",
            _validate_non_negative_int(self.latency_steps, name="latency_steps"),
        )
        object.__setattr__(
            self,
            "target_index",
            _validate_non_negative_int(self.target_index, name="target_index"),
        )
        if not isinstance(self.executable, bool):
            raise TypeError("executable must be a bool")
        if self.executable and self.state is None:
            raise ValueError("state is required when executable is true")
        if self.state is not None and not isinstance(self.state, MarketState):
            raise TypeError("state must be a MarketState when provided")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")


def _validate_market_states(market_states: Sequence[MarketState]) -> Sequence[MarketState]:
    if not isinstance(market_states, Sequence):
        raise TypeError("market_states must be a sequence")
    for state in market_states:
        if not isinstance(state, MarketState):
            raise TypeError("market_states must contain MarketState instances")
    return market_states


def apply_latency(
    current_index: int,
    market_states: Sequence[MarketState],
    config: LatencyConfig | None = None,
) -> LatencyResult:
    """Return the market state reached after configured latency steps."""
    current_index = _validate_non_negative_int(current_index, name="current_index")
    states = _validate_market_states(market_states)
    if current_index >= len(states):
        raise IndexError("current_index is outside the market_states sequence")
    resolved_config = LatencyConfig() if config is None else config
    if not isinstance(resolved_config, LatencyConfig):
        raise TypeError("config must be a LatencyConfig")
    target_index = current_index + resolved_config.latency_steps
    if target_index >= len(states):
        return LatencyResult(
            current_index=current_index,
            latency_steps=resolved_config.latency_steps,
            target_index=target_index,
            executable=False,
            state=None,
            reason="latency_out_of_range",
        )
    return LatencyResult(
        current_index=current_index,
        latency_steps=resolved_config.latency_steps,
        target_index=target_index,
        executable=True,
        state=states[target_index],
        reason="latency_state_available",
    )


def get_latency_state(
    current_index: int,
    market_states: Sequence[MarketState],
    config: LatencyConfig | None = None,
) -> MarketState:
    """Return the executable latency-adjusted state or raise clearly."""
    result = apply_latency(current_index, market_states, config)
    if not result.executable or result.state is None:
        raise IndexError(
            "latency_steps push beyond available market states: "
            f"current_index={result.current_index}, "
            f"latency_steps={result.latency_steps}, "
            f"target_index={result.target_index}"
        )
    return result.state


def latency_sensitivity_grid(
    latency_steps: Sequence[int] = DEFAULT_LATENCY_GRID,
) -> list[LatencyConfig]:
    """Build deterministic latency configurations for sensitivity analysis."""
    if not isinstance(latency_steps, Sequence):
        raise TypeError("latency_steps must be a sequence")
    if not latency_steps:
        raise ValueError("latency_steps must not be empty")
    return [
        LatencyConfig(latency_steps=_validate_non_negative_int(step, name="latency_step"))
        for step in latency_steps
    ]

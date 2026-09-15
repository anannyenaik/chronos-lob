"""Future mid-price labels for market-state forecasting tasks.

The functions in this module intentionally operate on explicit event
horizons. They may look forward from index ``t`` to construct labels, but
they do not write features and should not be used inside feature pipelines.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

__all__ = [
    "classify_direction",
    "compute_direction_labels",
    "compute_future_return",
    "compute_future_returns",
    "compute_return_quantile_labels",
]


def _validate_horizon(horizon: int) -> int:
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise TypeError("horizon must be an integer")
    if horizon <= 0:
        raise ValueError(f"horizon must be strictly positive; got {horizon!r}")
    return horizon


def _validate_index(index: int) -> int:
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("index must be an integer")
    if index < 0:
        raise ValueError(f"index must be non-negative; got {index!r}")
    return index


def _validate_missing(missing: str) -> str:
    if missing not in {"drop", "none"}:
        raise ValueError("missing must be one of {'drop', 'none'}")
    return missing


def _coerce_positive_finite_prices(mid_prices: Sequence[float]) -> list[float]:
    cleaned: list[float] = []
    for position, value in enumerate(mid_prices):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                "mid_prices must contain finite numeric prices; "
                f"got {type(value).__name__} at index {position}"
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"mid_prices[{position}] must be finite; got {value!r}")
        if numeric <= 0.0:
            raise ValueError(
                f"mid_prices[{position}] must be strictly positive; got {numeric!r}"
            )
        cleaned.append(numeric)
    return cleaned


def _coerce_finite_return(value: float, *, name: str = "future_return") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return numeric


def compute_future_return(
    mid_prices: Sequence[float],
    index: int,
    horizon: int,
    *,
    log_return: bool = True,
) -> float:
    """Return the future return from ``index`` to ``index + horizon``.

    ``horizon`` is an event/row horizon. Prices must be finite and strictly
    positive. ``IndexError`` is raised when the requested future observation
    is unavailable.
    """
    cleaned = _coerce_positive_finite_prices(mid_prices)
    idx = _validate_index(index)
    h = _validate_horizon(horizon)
    future_index = idx + h
    if idx >= len(cleaned):
        raise IndexError(f"index {idx} is outside mid_prices of length {len(cleaned)}")
    if future_index >= len(cleaned):
        raise IndexError(
            "insufficient future data for requested horizon: "
            f"index {idx} + horizon {h} >= length {len(cleaned)}"
        )
    current_price = cleaned[idx]
    future_price = cleaned[future_index]
    ratio = future_price / current_price
    if log_return:
        return math.log(ratio)
    return ratio - 1.0


def compute_future_returns(
    mid_prices: Sequence[float],
    horizon: int,
    *,
    log_return: bool = True,
    missing: str = "drop",
) -> list[float | None]:
    """Return future returns for all rows under an explicit missing policy.

    ``missing="drop"`` returns only rows with enough future data.
    ``missing="none"`` preserves input length and emits ``None`` for trailing
    rows whose requested horizon is unavailable.
    """
    cleaned = _coerce_positive_finite_prices(mid_prices)
    h = _validate_horizon(horizon)
    policy = _validate_missing(missing)
    output: list[float | None] = []
    for idx in range(len(cleaned)):
        if idx + h >= len(cleaned):
            if policy == "none":
                output.append(None)
            continue
        output.append(
            compute_future_return(cleaned, idx, h, log_return=log_return)
        )
    return output


def classify_direction(
    future_return: float,
    up_threshold: float,
    down_threshold: float | None = None,
) -> str:
    """Classify a future return as ``up``, ``down`` or ``stationary``."""
    ret = _coerce_finite_return(future_return)
    up = _coerce_finite_return(up_threshold, name="up_threshold")
    if up < 0.0:
        raise ValueError("up_threshold must be non-negative")
    down = up if down_threshold is None else _coerce_finite_return(
        down_threshold, name="down_threshold"
    )
    if down < 0.0:
        raise ValueError("down_threshold must be non-negative")
    if ret > up:
        return "up"
    if ret < -down:
        return "down"
    return "stationary"


def compute_direction_labels(
    mid_prices: Sequence[float],
    horizon: int,
    threshold: float,
    *,
    log_return: bool = True,
    missing: str = "drop",
) -> list[str | None]:
    """Return direction labels derived from future mid-price returns."""
    future_returns = compute_future_returns(
        mid_prices,
        horizon,
        log_return=log_return,
        missing=missing,
    )
    labels: list[str | None] = []
    for value in future_returns:
        if value is None:
            labels.append(None)
        else:
            labels.append(classify_direction(value, threshold))
    return labels


def _validate_quantiles(quantiles: tuple[float, ...]) -> tuple[float, ...]:
    previous: float | None = None
    for position, value in enumerate(quantiles):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"quantiles[{position}] must be a finite number between 0 and 1"
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"quantiles[{position}] must be finite")
        if not 0.0 < numeric < 1.0:
            raise ValueError(
                f"quantiles[{position}] must be strictly between 0 and 1; "
                f"got {numeric!r}"
            )
        if previous is not None and numeric <= previous:
            raise ValueError("quantiles must be strictly increasing")
        previous = numeric
    return tuple(float(q) for q in quantiles)


def compute_return_quantile_labels(
    returns: Sequence[float],
    quantiles: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8),
) -> list[int]:
    """Bucket supplied returns into empirical quantile labels.

    The quantile edges are fitted only from ``returns`` supplied to this
    function. Later experiment phases must fit these edges on the training
    partition and apply them to validation/test partitions to avoid split
    leakage.
    """
    cleaned = [_coerce_finite_return(value, name="returns[]") for value in returns]
    qs = _validate_quantiles(quantiles)
    if not cleaned:
        return []
    edges = np.quantile(np.asarray(cleaned, dtype=float), qs)
    return [int(np.searchsorted(edges, value, side="right")) for value in cleaned]

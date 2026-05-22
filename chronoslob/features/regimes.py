"""Simple rule-based market-regime flags.

The functions in this module assign categorical labels to single
observations given pre-computed thresholds. The thresholds may be set
manually (e.g. for unit tests) or estimated from a feature DataFrame
via :func:`compute_regime_thresholds_from_frame`. The estimates are
quantile-based and are *not* a calibrated regime model.

Regime labels here are coarse, interpretable hints — they are not
trading signals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "RegimeThresholds",
    "classify_imbalance_regime",
    "classify_liquidity_regime",
    "classify_spread_regime",
    "classify_volatility_regime",
    "compute_regime_thresholds_from_frame",
]


@dataclass(frozen=True)
class RegimeThresholds:
    """Quantile-based thresholds used to summarise market regimes.

    The quantile fields select empirical thresholds from a feature
    frame. ``imbalance_abs_threshold`` is an absolute threshold on the
    queue imbalance ``[-1, 1]``.

    All quantiles must be in ``(0, 1)``; the imbalance threshold must
    be in ``(0, 1)``.
    """

    wide_spread_quantile: float = 0.75
    high_volatility_quantile: float = 0.75
    low_liquidity_quantile: float = 0.25
    imbalance_abs_threshold: float = 0.6

    def __post_init__(self) -> None:
        for name in (
            "wide_spread_quantile",
            "high_volatility_quantile",
            "low_liquidity_quantile",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a number")
            if not 0.0 < float(value) < 1.0:
                raise ValueError(
                    f"{name} must be strictly between 0 and 1; got {value!r}"
                )
        threshold = self.imbalance_abs_threshold
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise TypeError("imbalance_abs_threshold must be a number")
        if not 0.0 < float(threshold) < 1.0:
            raise ValueError(
                "imbalance_abs_threshold must be strictly between 0 and 1; "
                f"got {threshold!r}"
            )


def classify_spread_regime(spread: float, threshold: float) -> str:
    """Return ``"wide_spread"`` when ``spread >= threshold``, else ``"normal_spread"``.

    Both arguments must be finite. Negative thresholds are accepted in
    principle (callers can use any auditable threshold) but the regime
    semantics only make sense for non-negative spreads.
    """
    if not _is_finite_number(spread) or not _is_finite_number(threshold):
        raise ValueError("spread and threshold must be finite numbers")
    return "wide_spread" if float(spread) >= float(threshold) else "normal_spread"


def classify_volatility_regime(
    volatility: float,
    low_threshold: float,
    high_threshold: float,
) -> str:
    """Return ``"low_volatility"``, ``"medium_volatility"`` or ``"high_volatility"``.

    Requires ``low_threshold <= high_threshold``. Both thresholds and
    the value must be finite numbers.
    """
    if not _is_finite_number(volatility):
        raise ValueError("volatility must be a finite number")
    if not _is_finite_number(low_threshold) or not _is_finite_number(high_threshold):
        raise ValueError("thresholds must be finite numbers")
    if float(low_threshold) > float(high_threshold):
        raise ValueError(
            "low_threshold must be <= high_threshold; "
            f"got low={low_threshold!r}, high={high_threshold!r}"
        )
    value = float(volatility)
    if value < float(low_threshold):
        return "low_volatility"
    if value >= float(high_threshold):
        return "high_volatility"
    return "medium_volatility"


def classify_liquidity_regime(depth: float, low_threshold: float) -> str:
    """Return ``"low_liquidity"`` when ``depth < low_threshold``, else ``"normal_liquidity"``."""
    if not _is_finite_number(depth) or not _is_finite_number(low_threshold):
        raise ValueError("depth and low_threshold must be finite numbers")
    return "low_liquidity" if float(depth) < float(low_threshold) else "normal_liquidity"


def classify_imbalance_regime(imbalance: float, threshold: float = 0.6) -> str:
    """Return ``"bid_heavy"``, ``"ask_heavy"`` or ``"balanced"``.

    ``imbalance`` follows the standard ``[-1, 1]`` queue-imbalance
    convention. The threshold is the absolute value at which we flip
    into a heavy regime; it must be strictly between 0 and 1.
    """
    if not _is_finite_number(imbalance):
        raise ValueError("imbalance must be a finite number")
    if not _is_finite_number(threshold):
        raise ValueError("threshold must be a finite number")
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError(
            f"threshold must be strictly between 0 and 1; got {threshold!r}"
        )
    value = float(imbalance)
    if value >= float(threshold):
        return "bid_heavy"
    if value <= -float(threshold):
        return "ask_heavy"
    return "balanced"


def compute_regime_thresholds_from_frame(
    frame: pd.DataFrame,
    thresholds: RegimeThresholds | None = None,
) -> dict[str, float]:
    """Estimate spread/volatility/liquidity thresholds from ``frame``.

    Returned keys (only those whose source column is present):

    * ``wide_spread_threshold`` from the ``spread`` column;
    * ``low_volatility_threshold`` and ``high_volatility_threshold``
      from the ``realised_volatility`` column;
    * ``low_liquidity_threshold`` from the ``total_depth`` column when
      present, otherwise from ``bid_depth_1 + ask_depth_1`` when both
      are present.

    Quantiles are taken from the configured ``thresholds`` object.
    Missing columns are silently skipped — the caller can decide which
    regimes to compute.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if thresholds is None:
        thresholds = RegimeThresholds()

    out: dict[str, float] = {}
    if "spread" in frame.columns:
        spread = frame["spread"].to_numpy(dtype=float)
        spread = spread[np.isfinite(spread)]
        if spread.size > 0:
            out["wide_spread_threshold"] = float(
                np.quantile(spread, thresholds.wide_spread_quantile)
            )

    if "realised_volatility" in frame.columns:
        vol = frame["realised_volatility"].to_numpy(dtype=float)
        vol = vol[np.isfinite(vol)]
        if vol.size > 0:
            low_q = 1.0 - thresholds.high_volatility_quantile
            high_q = thresholds.high_volatility_quantile
            out["low_volatility_threshold"] = float(np.quantile(vol, low_q))
            out["high_volatility_threshold"] = float(np.quantile(vol, high_q))

    depth_values: np.ndarray | None = None
    if "total_depth" in frame.columns:
        depth_values = frame["total_depth"].to_numpy(dtype=float)
    elif "bid_depth_1" in frame.columns and "ask_depth_1" in frame.columns:
        bid = frame["bid_depth_1"].to_numpy(dtype=float)
        ask = frame["ask_depth_1"].to_numpy(dtype=float)
        depth_values = bid + ask
    if depth_values is not None:
        depth_values = depth_values[np.isfinite(depth_values)]
        if depth_values.size > 0:
            out["low_liquidity_threshold"] = float(
                np.quantile(depth_values, thresholds.low_liquidity_quantile)
            )
    return out


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return np.isfinite(value)
    return False

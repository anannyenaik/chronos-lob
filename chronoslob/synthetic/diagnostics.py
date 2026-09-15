"""Controlled synthetic regime diagnostics.

These diagnostics use the *known* synthetic regime labels to break down feature
behaviour and simple execution-aware proxies (confidence filtering, active
fraction, turnover proxy, latency sensitivity, adverse-selection proxy) per
regime. They are controlled stress tests on synthetic data, not real-market
execution evidence, and carry no profitability or tradability claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from chronoslob.synthetic.events import REGIME_LIBRARY

__all__ = [
    "regime_execution_diagnostics",
    "regime_feature_summary",
]

_REGIME_NAME_BY_ID = {spec.regime_id: spec.name for spec in REGIME_LIBRARY.values()}


def regime_feature_summary(feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise event-level features grouped by known regime."""
    if feature_frame.empty:
        return pd.DataFrame(
            columns=[
                "regime_id",
                "regime_name",
                "row_count",
                "mean_event_intensity",
                "mean_spread",
                "mean_event_order_flow_imbalance",
                "mean_cancellation_imbalance",
                "mean_trade_imbalance",
                "mean_realised_volatility_proxy",
            ]
        )
    grouped = feature_frame.groupby("regime_id", sort=True)
    rows: list[dict[str, object]] = []
    for regime_id, group in grouped:
        rows.append(
            {
                "regime_id": int(regime_id),
                "regime_name": _REGIME_NAME_BY_ID.get(int(regime_id), "unknown"),
                "row_count": len(group),
                "mean_event_intensity": _mean(group, "event_intensity"),
                "mean_spread": _mean(group, "spread"),
                "mean_event_order_flow_imbalance": _mean(group, "event_order_flow_imbalance"),
                "mean_cancellation_imbalance": _mean(group, "cancellation_imbalance"),
                "mean_trade_imbalance": _mean(group, "trade_imbalance"),
                "mean_realised_volatility_proxy": _mean(group, "realised_volatility_proxy"),
            }
        )
    return pd.DataFrame(rows)


def regime_execution_diagnostics(
    test_frame: pd.DataFrame,
    predictions: np.ndarray,
    confidence: np.ndarray,
    *,
    confidence_threshold: float = 0.5,
    latency_steps: int = 1,
) -> pd.DataFrame:
    """Compute execution-aware proxy diagnostics per known regime.

    ``test_frame`` must carry ``regime_label``, the true ``future_mid_direction``
    and ``future_return`` columns aligned row-for-row with ``predictions`` and
    ``confidence``.
    """
    required = {"regime_label", "future_mid_direction", "future_return"}
    missing = required - set(test_frame.columns)
    if missing:
        raise ValueError(f"test_frame missing columns: {sorted(missing)}")
    if not (len(test_frame) == len(predictions) == len(confidence)):
        raise ValueError("test_frame, predictions and confidence must align in length")

    frame = test_frame.reset_index(drop=True).copy()
    frame["_prediction"] = np.asarray(predictions)
    frame["_confidence"] = np.asarray(confidence, dtype=float)
    truth = frame["future_mid_direction"].to_numpy(dtype=int)
    frame["_correct"] = (frame["_prediction"].to_numpy() == truth).astype(float)
    # Latency-shifted decision: act on the prediction from ``latency_steps`` ago.
    shifted = frame["_prediction"].shift(latency_steps)
    frame["_latency_correct"] = (shifted.to_numpy() == truth).astype(float)

    rows: list[dict[str, object]] = []
    rows.append(_diagnostic_row(frame, "all", confidence_threshold))
    for regime_id, group in frame.groupby("regime_label", sort=True):
        rows.append(
            _diagnostic_row(
                group,
                _REGIME_NAME_BY_ID.get(int(regime_id), "unknown"),
                confidence_threshold,
                regime_id=int(regime_id),
            )
        )
    return pd.DataFrame(rows)


def _diagnostic_row(
    frame: pd.DataFrame,
    regime_name: str,
    confidence_threshold: float,
    *,
    regime_id: int | None = None,
) -> dict[str, object]:
    confidence = frame["_confidence"].to_numpy(dtype=float)
    correct = frame["_correct"].to_numpy(dtype=float)
    predictions = frame["_prediction"].to_numpy()
    future_return = frame["future_return"].to_numpy(dtype=float)
    active = confidence >= confidence_threshold
    active_count = int(active.sum())
    n = len(frame)
    turnover = _turnover_proxy(predictions)
    return {
        "regime_id": "" if regime_id is None else regime_id,
        "regime_name": regime_name,
        "n_samples": n,
        "accuracy": _round(float(correct.mean()) if n else 0.0),
        "mean_confidence": _round(float(confidence.mean()) if n else 0.0),
        "active_fraction": _round(active_count / n if n else 0.0),
        "filtered_accuracy": _round(
            float(correct[active].mean()) if active_count else 0.0
        ),
        "turnover_proxy": _round(turnover),
        "latency_accuracy": _round(
            float(np.nanmean(frame["_latency_correct"].to_numpy(dtype=float)))
            if n
            else 0.0
        ),
        "adverse_selection_proxy": _round(
            float(np.abs(future_return[active]).mean()) if active_count else 0.0
        ),
    }


def _turnover_proxy(predictions: np.ndarray) -> float:
    if len(predictions) < 2:
        return 0.0
    changes = int(np.sum(predictions[1:] != predictions[:-1]))
    return changes / (len(predictions) - 1)


def _mean(group: pd.DataFrame, column: str) -> float:
    if column not in group.columns or group.empty:
        return 0.0
    return _round(float(group[column].mean()))


def _round(value: float) -> float:
    return round(float(value), 6)

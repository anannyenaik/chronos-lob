"""Small baseline diagnostics for the synthetic event-level pipeline.

These baselines validate that the synthetic event-level features and labels flow
through a standard train/validate/test protocol. They are a data-and-platform
validation exercise on controlled synthetic regimes, not a real-market result
and not evidence of tradability or profitability.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chronoslob.synthetic.features import EVENT_FEATURE_COLUMNS
from chronoslob.training.metrics import compute_classification_metrics

__all__ = [
    "BenchmarkResult",
    "DiagnosticPredictions",
    "SplitMetrics",
    "diagnostic_predictions",
    "run_synthetic_benchmark",
]


@dataclass(frozen=True)
class SplitMetrics:
    """Metrics for one model on one evaluation split."""

    model_name: str
    split: str
    n_samples: int
    accuracy: float
    macro_f1: float
    mcc: float
    brier_score: float | None
    expected_calibration_error: float | None

    def to_row(self) -> dict[str, object]:
        """Return a flat CSV-ready row."""
        return {
            "model_name": self.model_name,
            "split": self.split,
            "n_samples": self.n_samples,
            "accuracy": round(self.accuracy, 6),
            "macro_f1": round(self.macro_f1, 6),
            "mcc": round(self.mcc, 6),
            "brier_score": "" if self.brier_score is None else round(self.brier_score, 6),
            "expected_calibration_error": (
                "" if self.expected_calibration_error is None
                else round(self.expected_calibration_error, 6)
            ),
        }


@dataclass(frozen=True)
class BenchmarkResult:
    """All baseline metrics produced by a synthetic benchmark run."""

    target: str
    feature_columns: tuple[str, ...]
    class_labels: tuple[int, ...]
    chronological: list[SplitMetrics]
    regime_holdout: list[SplitMetrics]
    holdout_regimes: tuple[str, ...]
    split_sizes: dict[str, int]
    warnings: list[str] = field(default_factory=list)


def run_synthetic_benchmark(
    feature_frame: pd.DataFrame,
    label_frame: pd.DataFrame,
    *,
    target: str = "future_mid_direction",
    seed: int = 0,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    holdout_regimes: Sequence[str] = ("high_volatility", "cancellation_shock"),
) -> BenchmarkResult:
    """Run majority, logistic, ridge and gradient-boosting baselines.

    The chronological protocol splits rows by time into train/validation/test.
    The regime-holdout protocol trains on rows from regimes outside
    ``holdout_regimes`` and tests on the held-out regimes, a controlled
    synthetic stress test of regime shift.
    """
    merged = feature_frame.merge(label_frame, on="sequence_id", how="inner")
    merged = merged.sort_values("sequence_id").reset_index(drop=True)
    feature_columns = tuple(col for col in EVENT_FEATURE_COLUMNS if col in merged.columns)
    warnings: list[str] = []

    if len(merged) < 30:
        warnings.append("too few aligned rows for a meaningful split; results are indicative only")

    features = merged[list(feature_columns)].to_numpy(dtype=float)
    targets = merged[target].to_numpy(dtype=int)
    regimes = merged["regime_label"].to_numpy(dtype=int)
    class_labels = tuple(int(value) for value in sorted(np.unique(targets)))

    n = len(merged)
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + validation_fraction))
    split_sizes = {
        "train": train_end,
        "validation": max(0, val_end - train_end),
        "test": max(0, n - val_end),
    }

    chronological = _run_chronological(
        features, targets, train_end, val_end, class_labels, seed, warnings
    )
    regime_holdout = _run_regime_holdout(
        merged, features, targets, regimes, class_labels, seed, holdout_regimes, warnings
    )

    return BenchmarkResult(
        target=target,
        feature_columns=feature_columns,
        class_labels=class_labels,
        chronological=chronological,
        regime_holdout=regime_holdout,
        holdout_regimes=tuple(holdout_regimes),
        split_sizes=split_sizes,
        warnings=warnings,
    )


@dataclass(frozen=True)
class DiagnosticPredictions:
    """Test-split predictions used to drive regime diagnostics."""

    model_name: str
    test_frame: pd.DataFrame
    predictions: np.ndarray
    confidence: np.ndarray
    available: bool
    reason: str


def diagnostic_predictions(
    feature_frame: pd.DataFrame,
    label_frame: pd.DataFrame,
    *,
    target: str = "future_mid_direction",
    seed: int = 0,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    model_name: str = "logistic",
) -> DiagnosticPredictions:
    """Fit one probabilistic model on the chronological train split.

    Returns the test-split frame together with predicted classes and per-row
    maximum-class confidence so that downstream regime diagnostics can break the
    test predictions down by known regime.
    """
    merged = feature_frame.merge(label_frame, on="sequence_id", how="inner")
    merged = merged.sort_values("sequence_id").reset_index(drop=True)
    feature_columns = [col for col in EVENT_FEATURE_COLUMNS if col in merged.columns]
    empty = pd.DataFrame()
    n = len(merged)
    val_end = int(n * (train_fraction + validation_fraction))
    train_end = int(n * train_fraction)
    if train_end == 0 or val_end >= n:
        return DiagnosticPredictions(
            model_name, empty, np.empty(0), np.empty(0), False, "insufficient rows for split"
        )
    features = merged[feature_columns].to_numpy(dtype=float)
    targets = merged[target].to_numpy(dtype=int)
    x_train, y_train = features[:train_end], targets[:train_end]
    if len(np.unique(y_train)) < 2:
        return DiagnosticPredictions(
            model_name, empty, np.empty(0), np.empty(0), False, "train split lacks two classes"
        )
    class_labels = tuple(int(value) for value in sorted(np.unique(targets)))
    model = _build_models(seed)[model_name]
    model.fit(x_train, y_train)  # type: ignore[attr-defined]
    test_frame = merged.iloc[val_end:].reset_index(drop=True)
    x_test = features[val_end:]
    predictions = model.predict(x_test)  # type: ignore[attr-defined]
    probabilities = _predict_proba(model, x_test, class_labels)
    if probabilities is None:
        confidence = np.ones(len(x_test), dtype=float)
    else:
        confidence = probabilities.max(axis=1)
    return DiagnosticPredictions(
        model_name=model_name,
        test_frame=test_frame,
        predictions=np.asarray(predictions),
        confidence=np.asarray(confidence, dtype=float),
        available=len(test_frame) > 0,
        reason="ok" if len(test_frame) > 0 else "empty test split",
    )


def _build_models(seed: int) -> dict[str, object]:
    return {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=500, random_state=seed),
                ),
            ]
        ),
        "ridge": Pipeline(
            [("scaler", StandardScaler()), ("model", RidgeClassifier())]
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=60, max_depth=2, random_state=seed
        ),
    }


def _run_chronological(
    features: np.ndarray,
    targets: np.ndarray,
    train_end: int,
    val_end: int,
    class_labels: tuple[int, ...],
    seed: int,
    warnings: list[str],
) -> list[SplitMetrics]:
    x_train, y_train = features[:train_end], targets[:train_end]
    splits = {
        "validation": (features[train_end:val_end], targets[train_end:val_end]),
        "test": (features[val_end:], targets[val_end:]),
    }
    results: list[SplitMetrics] = []
    if len(x_train) == 0 or len(np.unique(y_train)) < 2:
        warnings.append("chronological train split lacks at least two classes; skipped")
        return results
    for model_name, model in _build_models(seed).items():
        fitted = _fit(model, x_train, y_train)
        for split_name, (x_eval, y_eval) in splits.items():
            if len(x_eval) == 0:
                continue
            results.append(
                _evaluate(model_name, split_name, fitted, x_eval, y_eval, class_labels)
            )
    return results


def _run_regime_holdout(
    merged: pd.DataFrame,
    features: np.ndarray,
    targets: np.ndarray,
    regimes: np.ndarray,
    class_labels: tuple[int, ...],
    seed: int,
    holdout_regimes: Sequence[str],
    warnings: list[str],
) -> list[SplitMetrics]:
    from chronoslob.synthetic.events import REGIME_LIBRARY

    holdout_ids = {
        REGIME_LIBRARY[name].regime_id for name in holdout_regimes if name in REGIME_LIBRARY
    }
    if not holdout_ids:
        warnings.append("no valid holdout regimes supplied; regime-holdout skipped")
        return []
    test_mask = np.isin(regimes, list(holdout_ids))
    train_mask = ~test_mask
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        warnings.append("regime-holdout split is empty on one side; skipped")
        return []
    x_train, y_train = features[train_mask], targets[train_mask]
    x_test, y_test = features[test_mask], targets[test_mask]
    if len(np.unique(y_train)) < 2:
        warnings.append("regime-holdout train split lacks at least two classes; skipped")
        return []
    results: list[SplitMetrics] = []
    for model_name, model in _build_models(seed).items():
        fitted = _fit(model, x_train, y_train)
        results.append(
            _evaluate(model_name, "regime_holdout_test", fitted, x_test, y_test, class_labels)
        )
    return results


def _fit(model: object, x_train: np.ndarray, y_train: np.ndarray) -> object:
    model.fit(x_train, y_train)  # type: ignore[attr-defined]
    return model


def _evaluate(
    model_name: str,
    split: str,
    model: object,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    class_labels: tuple[int, ...],
) -> SplitMetrics:
    predictions = model.predict(x_eval)  # type: ignore[attr-defined]
    probabilities = _predict_proba(model, x_eval, class_labels)
    metrics = compute_classification_metrics(
        list(y_eval),
        list(predictions),
        y_proba=probabilities,
        labels=list(class_labels),
    )
    ece = (
        _expected_calibration_error(probabilities, y_eval, class_labels)
        if probabilities is not None
        else None
    )
    return SplitMetrics(
        model_name=model_name,
        split=split,
        n_samples=len(y_eval),
        accuracy=metrics.accuracy,
        macro_f1=metrics.macro_f1,
        mcc=metrics.matthews_corrcoef,
        brier_score=metrics.brier_score,
        expected_calibration_error=ece,
    )


def _predict_proba(
    model: object,
    x_eval: np.ndarray,
    class_labels: tuple[int, ...],
) -> np.ndarray | None:
    proba_fn = getattr(model, "predict_proba", None)
    if proba_fn is None:
        return None
    probabilities = np.asarray(proba_fn(x_eval), dtype=float)
    if probabilities.shape[1] != len(class_labels):
        return None
    return probabilities


def _expected_calibration_error(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    class_labels: tuple[int, ...],
    *,
    n_bins: int = 10,
) -> float:
    confidence = probabilities.max(axis=1)
    predicted_index = probabilities.argmax(axis=1)
    label_array = np.array(class_labels)
    predicted_labels = label_array[predicted_index]
    correct = (predicted_labels == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    for lower, upper in itertools.pairwise(bins):
        in_bin = (confidence > lower) & (confidence <= upper)
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_confidence = float(confidence[in_bin].mean())
        bin_accuracy = float(correct[in_bin].mean())
        ece += (count / total) * abs(bin_accuracy - bin_confidence)
    return float(ece)

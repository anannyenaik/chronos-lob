"""Classification metrics for baseline forecasting experiments."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.metrics import (
    log_loss as sklearn_log_loss,
)

__all__ = [
    "ClassificationMetrics",
    "compute_classification_metrics",
    "confusion_matrix_as_dict",
]

_Label = str | int | float
_MetricDictValue = float | int | str | None | list[_Label]


@dataclass(frozen=True)
class ClassificationMetrics:
    """Common classification metrics for model comparison.

    F1 metrics use ``zero_division=0`` so absent predicted classes are
    reported deterministically rather than raising at evaluation time.
    """

    accuracy: float
    macro_f1: float
    weighted_f1: float
    matthews_corrcoef: float
    balanced_accuracy: float
    brier_score: float | None
    log_loss: float | None
    n_samples: int
    labels: list[_Label]

    def to_dict(self) -> dict[str, _MetricDictValue]:
        """Return a JSON-serialisable dictionary representation."""
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "matthews_corrcoef": self.matthews_corrcoef,
            "balanced_accuracy": self.balanced_accuracy,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "n_samples": self.n_samples,
            "labels": list(self.labels),
        }


def _to_python_label(value: object) -> _Label:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (str, int, float)):
        return value
    return str(value)


def _label_sort_key(value: _Label) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def _stable_sorted_labels(values: Sequence[object]) -> list[_Label]:
    unique = {_to_python_label(value) for value in values}
    ordered = list(unique)
    try:
        ordered.sort()
    except TypeError:
        ordered.sort(key=_label_sort_key)
    return ordered


def _validate_lengths(y_true: Sequence[object], y_pred: Sequence[object]) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("metrics require at least one sample")


def _normalise_probability_matrix(
    y_proba: np.ndarray | None,
    *,
    n_samples: int,
) -> np.ndarray | None:
    if y_proba is None:
        return None
    probabilities = np.asarray(y_proba, dtype=float)
    if probabilities.ndim not in {1, 2}:
        raise ValueError("y_proba must be a 1D or 2D array")
    if probabilities.shape[0] != n_samples:
        raise ValueError("y_proba row count must match y_true length")
    if bool(np.isnan(probabilities).any()):
        raise ValueError("y_proba must not contain NaN values")
    if bool(np.isinf(probabilities).any()):
        raise ValueError("y_proba must not contain infinite values")
    return probabilities


def _probabilities_are_compatible(
    probabilities: np.ndarray,
    labels: Sequence[_Label],
) -> bool:
    if probabilities.ndim == 1:
        return len(labels) == 2
    return probabilities.shape[1] == len(labels)


def _positive_class_probability(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.ndim == 1:
        return probabilities
    return probabilities[:, 1]


def _balanced_accuracy(
    y_true: Sequence[_Label],
    y_pred: Sequence[_Label],
    labels: Sequence[_Label],
) -> float:
    recalls: list[float] = []
    for label in labels:
        support = sum(1 for value in y_true if value == label)
        if support == 0:
            continue
        correct = sum(
            1
            for true_value, predicted_value in zip(y_true, y_pred, strict=True)
            if true_value == label and predicted_value == label
        )
        recalls.append(correct / support)
    if not recalls:
        return 0.0
    return float(np.mean(recalls))


def _matthews_corrcoef(
    y_true: Sequence[_Label],
    y_pred: Sequence[_Label],
) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="A single label was found in 'y_true' and 'y_pred'.*",
            category=UserWarning,
        )
        return float(matthews_corrcoef(y_true, y_pred))


def compute_classification_metrics(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    *,
    y_proba: np.ndarray | None = None,
    labels: Sequence[object] | None = None,
) -> ClassificationMetrics:
    """Compute deterministic classification metrics.

    Probability metrics are only computed when probabilities are supplied
    and their shape is compatible with the requested label set.
    """
    _validate_lengths(y_true, y_pred)
    true_labels = [_to_python_label(value) for value in y_true]
    predicted_labels = [_to_python_label(value) for value in y_pred]
    metric_labels = (
        _stable_sorted_labels([*true_labels, *predicted_labels])
        if labels is None
        else [_to_python_label(value) for value in labels]
    )
    if not metric_labels:
        raise ValueError("labels must not be empty")

    observed = set(true_labels) | set(predicted_labels)
    missing = observed - set(metric_labels)
    if missing:
        sorted_missing = sorted(missing, key=_label_sort_key)
        raise ValueError(f"labels are missing observed classes: {sorted_missing}")

    probabilities = _normalise_probability_matrix(y_proba, n_samples=len(true_labels))
    brier: float | None = None
    log_loss: float | None = None
    if probabilities is not None and _probabilities_are_compatible(
        probabilities,
        metric_labels,
    ):
        if len(metric_labels) == 2:
            brier = float(
                brier_score_loss(
                    true_labels,
                    _positive_class_probability(probabilities),
                    pos_label=metric_labels[1],
                )
            )
        log_loss = float(
            sklearn_log_loss(
                true_labels,
                probabilities,
                labels=metric_labels,
            )
        )

    return ClassificationMetrics(
        accuracy=float(accuracy_score(true_labels, predicted_labels)),
        macro_f1=float(
            f1_score(
                true_labels,
                predicted_labels,
                labels=metric_labels,
                average="macro",
                zero_division=0,
            )
        ),
        weighted_f1=float(
            f1_score(
                true_labels,
                predicted_labels,
                labels=metric_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        matthews_corrcoef=_matthews_corrcoef(true_labels, predicted_labels),
        balanced_accuracy=_balanced_accuracy(
            true_labels,
            predicted_labels,
            metric_labels,
        ),
        brier_score=brier,
        log_loss=log_loss,
        n_samples=len(true_labels),
        labels=metric_labels,
    )


def confusion_matrix_as_dict(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    labels: Sequence[object] | None = None,
) -> dict[str, Any]:
    """Return a serialisable confusion matrix and its label order."""
    _validate_lengths(y_true, y_pred)
    true_labels = [_to_python_label(value) for value in y_true]
    predicted_labels = [_to_python_label(value) for value in y_pred]
    matrix_labels = (
        _stable_sorted_labels([*true_labels, *predicted_labels])
        if labels is None
        else [_to_python_label(value) for value in labels]
    )
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=matrix_labels,
    )
    return {
        "labels": matrix_labels,
        "matrix": matrix.astype(int).tolist(),
    }

"""Evaluation helpers for fitted baseline classifiers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from chronoslob.models.baselines import BaseBaselineModel
from chronoslob.training.metrics import (
    compute_classification_metrics,
    confusion_matrix_as_dict,
)

__all__ = ["evaluate_classifier"]


def evaluate_classifier(
    model: BaseBaselineModel,
    x: np.ndarray,
    y: np.ndarray,
    *,
    labels: Sequence[object] | None = None,
) -> dict[str, Any]:
    """Evaluate a fitted classifier without writing artefacts."""
    y_pred = model.predict(x)
    y_proba = model.predict_proba(x)
    true_values = y.tolist()
    predicted_values = y_pred.tolist()
    metrics = compute_classification_metrics(
        true_values,
        predicted_values,
        y_proba=y_proba,
        labels=labels,
    )
    return {
        "metrics": metrics.to_dict(),
        "confusion_matrix": confusion_matrix_as_dict(
            true_values,
            predicted_values,
            labels=labels,
        ),
    }

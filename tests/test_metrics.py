"""Tests for classification metric helpers."""

from __future__ import annotations

import numpy as np
import pytest

from chronoslob.training.metrics import (
    compute_classification_metrics,
    confusion_matrix_as_dict,
)


def test_compute_classification_metrics_basic_multiclass() -> None:
    metrics = compute_classification_metrics(
        [0, 1, 2, 2],
        [0, 2, 2, 1],
        labels=[0, 1, 2],
    )

    assert metrics.n_samples == 4
    assert metrics.labels == [0, 1, 2]
    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.macro_f1 >= 0.0
    assert metrics.weighted_f1 >= 0.0
    assert metrics.brier_score is None


def test_binary_brier_score_when_probabilities_available() -> None:
    metrics = compute_classification_metrics(
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        y_proba=np.asarray(
            [
                [0.8, 0.2],
                [0.1, 0.9],
                [0.6, 0.4],
                [0.7, 0.3],
            ]
        ),
        labels=[0, 1],
    )

    assert metrics.brier_score is not None
    assert metrics.brier_score == pytest.approx((0.2**2 + 0.1**2 + 0.6**2 + 0.3**2) / 4)


def test_log_loss_when_probabilities_available() -> None:
    metrics = compute_classification_metrics(
        [0, 1, 1],
        [0, 1, 1],
        y_proba=np.asarray([[0.9, 0.1], [0.2, 0.8], [0.1, 0.9]]),
        labels=[0, 1],
    )

    assert metrics.log_loss is not None
    assert metrics.log_loss > 0.0


def test_absent_probabilities_leave_probability_metrics_none() -> None:
    metrics = compute_classification_metrics([0, 1], [0, 0], labels=[0, 1])

    assert metrics.brier_score is None
    assert metrics.log_loss is None


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_classification_metrics([0, 1], [0])


def test_confusion_matrix_as_dict_shape_and_labels() -> None:
    result = confusion_matrix_as_dict(
        ["down", "flat", "up", "up"],
        ["down", "up", "up", "flat"],
        labels=["down", "flat", "up"],
    )

    assert result["labels"] == ["down", "flat", "up"]
    assert result["matrix"] == [[1, 0, 0], [0, 0, 1], [0, 1, 1]]

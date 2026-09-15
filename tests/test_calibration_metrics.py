"""Tests for calibration metric utilities."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from chronoslob.models.calibration import (  # noqa: E402
    CalibrationErrorConfig,
    brier_score,
    classification_confidence,
    expected_calibration_error,
    negative_log_likelihood,
    reliability_bins,
    softmax_probabilities,
)


def test_softmax_probabilities_sum_to_one() -> None:
    logits = torch.tensor([[1.0, 2.0, 0.0], [0.5, -0.5, 1.5]])

    probabilities = softmax_probabilities(logits)

    assert probabilities.shape == logits.shape
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2))


def test_classification_confidence_returns_max_probability() -> None:
    probabilities = torch.tensor([[0.1, 0.7, 0.2], [0.45, 0.25, 0.30]])

    confidence = classification_confidence(probabilities)

    assert torch.allclose(confidence, torch.tensor([0.7, 0.45]))


def test_negative_log_likelihood_is_finite_for_valid_targets() -> None:
    logits = torch.tensor([[3.0, 0.0], [0.5, 1.5], [1.0, 2.0]])
    targets = torch.tensor([0, 1, 1], dtype=torch.long)

    loss = negative_log_likelihood(logits, targets)

    assert torch.isfinite(loss).item()
    assert loss.item() >= 0.0


def test_negative_log_likelihood_ignores_ignore_index() -> None:
    logits = torch.tensor([[3.0, 0.0], [100.0, -100.0], [0.5, 1.5]])
    targets = torch.tensor([0, -100, 1], dtype=torch.long)
    expected = negative_log_likelihood(
        torch.tensor([[3.0, 0.0], [0.5, 1.5]]),
        torch.tensor([0, 1], dtype=torch.long),
    )

    actual = negative_log_likelihood(logits, targets)

    assert torch.allclose(actual, expected)


def test_brier_score_is_finite_non_negative_and_accepts_probabilities() -> None:
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]])
    targets = torch.tensor([0, 2], dtype=torch.long)
    probabilities = softmax_probabilities(logits)

    from_logits = brier_score(logits, targets, from_logits=True)
    from_probabilities = brier_score(probabilities, targets, from_logits=False)

    assert torch.isfinite(from_logits).item()
    assert from_logits.item() >= 0.0
    assert torch.allclose(from_logits, from_probabilities)


def test_ece_is_zero_for_perfectly_confident_correct_probabilities() -> None:
    probabilities = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    targets = torch.tensor([0, 1, 2], dtype=torch.long)

    summary = expected_calibration_error(
        probabilities,
        targets,
        CalibrationErrorConfig(n_bins=5),
        from_logits=False,
    )

    assert summary.ece == pytest.approx(0.0)
    assert summary.accuracy == pytest.approx(1.0)
    assert summary.average_confidence == pytest.approx(1.0)


def test_ece_is_positive_for_miscalibrated_predictions() -> None:
    probabilities = torch.tensor(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
        ]
    )
    targets = torch.tensor([1, 1, 1], dtype=torch.long)

    summary = expected_calibration_error(
        probabilities,
        targets,
        CalibrationErrorConfig(n_bins=5),
        from_logits=False,
    )

    assert summary.ece > 0.0
    assert summary.accuracy == pytest.approx(0.0)


def test_reliability_bins_cover_valid_examples_once() -> None:
    probabilities = torch.tensor(
        [
            [0.55, 0.45],
            [0.75, 0.25],
            [0.20, 0.80],
            [0.60, 0.40],
        ]
    )
    targets = torch.tensor([0, 1, -100, 0], dtype=torch.long)

    bins = reliability_bins(
        probabilities,
        targets,
        CalibrationErrorConfig(n_bins=4),
        from_logits=False,
    )

    assert sum(item.count for item in bins) == 3
    assert len(bins) == 4


def test_empty_valid_targets_raise_clearly() -> None:
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([-100, -100], dtype=torch.long)

    with pytest.raises(ValueError, match="no valid targets"):
        expected_calibration_error(logits, targets, CalibrationErrorConfig())


def test_binary_and_multiclass_cases_work() -> None:
    binary_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    binary_targets = torch.tensor([0, 1], dtype=torch.long)
    multiclass_logits = torch.tensor([[2.0, 0.0, 1.0], [0.0, 3.0, 1.0]])
    multiclass_targets = torch.tensor([0, 1], dtype=torch.long)

    binary = expected_calibration_error(
        binary_logits,
        binary_targets,
        CalibrationErrorConfig(n_bins=3),
    )
    multiclass = expected_calibration_error(
        multiclass_logits,
        multiclass_targets,
        CalibrationErrorConfig(n_bins=3),
    )

    assert binary.n_examples == 2
    assert multiclass.n_examples == 2
    assert binary.brier_score is not None and binary.brier_score >= 0.0
    assert multiclass.brier_score is not None and multiclass.brier_score >= 0.0


def test_calibration_results_are_deterministic() -> None:
    logits = torch.tensor(
        [[2.0, 0.0, 1.0], [0.2, 1.8, 0.0], [0.0, 0.1, 2.0]]
    )
    targets = torch.tensor([0, 1, 2], dtype=torch.long)
    config = CalibrationErrorConfig(n_bins=6)

    first = expected_calibration_error(logits, targets, config)
    second = expected_calibration_error(logits, targets, config)

    assert first.to_dict() == second.to_dict()

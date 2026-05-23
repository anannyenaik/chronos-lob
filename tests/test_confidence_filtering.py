"""Tests for confidence filtering and abstention utilities."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from chronoslob.training.calibration import (  # noqa: E402
    ConfidenceFilterConfig,
    abstention_curve,
    evaluate_confidence_filter,
)


def test_threshold_filtering_reports_coverage_and_accuracy() -> None:
    probabilities = torch.tensor(
        [
            [0.90, 0.10],
            [0.80, 0.20],
            [0.55, 0.45],
            [0.40, 0.60],
        ]
    )
    targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    config = ConfidenceFilterConfig(thresholds=(0.8,))

    result = evaluate_confidence_filter(
        probabilities,
        targets,
        config,
        from_logits=False,
    )
    bucket = result.buckets[0]

    assert bucket.n_total == 4
    assert bucket.n_covered == 2
    assert bucket.coverage == pytest.approx(0.5)
    assert bucket.abstention_rate == pytest.approx(0.5)
    assert bucket.accuracy_on_covered == pytest.approx(0.5)


def test_zero_covered_examples_are_reported_clearly() -> None:
    probabilities = torch.tensor([[0.60, 0.40], [0.55, 0.45]])
    targets = torch.tensor([0, 1], dtype=torch.long)

    result = evaluate_confidence_filter(
        probabilities,
        targets,
        ConfidenceFilterConfig(thresholds=(0.99,)),
        from_logits=False,
    )
    bucket = result.buckets[0]

    assert bucket.n_covered == 0
    assert bucket.coverage == pytest.approx(0.0)
    assert bucket.abstention_rate == pytest.approx(1.0)
    assert bucket.accuracy_on_covered is None


def test_ignore_index_targets_are_excluded() -> None:
    probabilities = torch.tensor(
        [
            [0.90, 0.10],
            [0.80, 0.20],
            [0.55, 0.45],
        ]
    )
    targets = torch.tensor([0, -100, 1], dtype=torch.long)

    result = evaluate_confidence_filter(
        probabilities,
        targets,
        ConfidenceFilterConfig(thresholds=(0.5,)),
        from_logits=False,
    )

    assert result.n_total == 2
    assert result.buckets[0].n_covered == 2


def test_abstention_curve_uses_requested_coverage_levels() -> None:
    probabilities = torch.tensor(
        [
            [0.90, 0.10],
            [0.80, 0.20],
            [0.70, 0.30],
            [0.60, 0.40],
            [0.55, 0.45],
        ]
    )
    targets = torch.tensor([0, 0, 1, 1, 0], dtype=torch.long)

    points = abstention_curve(
        probabilities,
        targets,
        coverage_levels=(1.0, 0.4),
        from_logits=False,
    )

    assert [point.coverage_level for point in points] == [1.0, 0.4]
    assert points[0].n_retained == 5
    assert points[1].n_retained == 2
    assert points[1].realised_coverage == pytest.approx(0.4)


def test_abstention_curve_handles_ties_deterministically() -> None:
    probabilities = torch.tensor(
        [
            [0.60, 0.40],
            [0.60, 0.40],
            [0.40, 0.60],
            [0.40, 0.60],
        ]
    )
    targets = torch.tensor([0, 0, 0, 0], dtype=torch.long)

    first = abstention_curve(
        probabilities,
        targets,
        coverage_levels=(0.5,),
        from_logits=False,
    )
    second = abstention_curve(
        probabilities,
        targets,
        coverage_levels=(0.5,),
        from_logits=False,
    )

    assert first == second
    assert first[0].n_retained == 2
    assert first[0].accuracy == pytest.approx(1.0)


def test_invalid_thresholds_fail_clearly() -> None:
    with pytest.raises(ValueError, match="threshold"):
        ConfidenceFilterConfig(thresholds=(-0.1,))

    with pytest.raises(ValueError, match="threshold"):
        ConfidenceFilterConfig(thresholds=(1.1,))


def test_invalid_coverage_levels_fail_clearly() -> None:
    probabilities = torch.tensor([[0.60, 0.40], [0.40, 0.60]])
    targets = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="coverage_levels"):
        abstention_curve(
            probabilities,
            targets,
            coverage_levels=(0.0,),
            from_logits=False,
        )

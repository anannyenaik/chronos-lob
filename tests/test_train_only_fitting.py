"""Tests for train-only fitting utilities."""

from __future__ import annotations

import pytest

from chronoslob.training.splitters import TrainOnlyQuantileBinner


def test_train_only_quantile_binner_validates_quantiles() -> None:
    with pytest.raises(ValueError):
        TrainOnlyQuantileBinner(quantiles=(0.5, 0.5))
    with pytest.raises(ValueError):
        TrainOnlyQuantileBinner(quantiles=(0.0, 0.5))


def test_fit_stores_edges_from_training_values() -> None:
    binner = TrainOnlyQuantileBinner(quantiles=(0.25, 0.75))

    binner.fit([0.0, 10.0, 20.0, 30.0])

    assert binner.bin_edges_ == pytest.approx([7.5, 22.5])


def test_transform_before_fit_raises() -> None:
    binner = TrainOnlyQuantileBinner(quantiles=(0.5,))

    with pytest.raises(ValueError):
        binner.transform([1.0])


def test_transform_does_not_refit_on_validation_values() -> None:
    binner = TrainOnlyQuantileBinner(quantiles=(0.5,))
    binner.fit([0.0, 10.0, 20.0])
    original_edges = list(binner.bin_edges_ or [])

    transformed = binner.transform([1000.0])

    assert transformed == [1]
    assert binner.bin_edges_ == original_edges


def test_train_and_validation_values_use_train_edges() -> None:
    binner = TrainOnlyQuantileBinner(quantiles=(0.5,))

    train_bins = binner.fit_transform([0.0, 10.0, 20.0])
    validation_bins = binner.transform([5.0, 10.0, 15.0])

    assert train_bins == [0, 1, 1]
    assert validation_bins == [0, 1, 1]


def test_nan_and_inf_values_raise() -> None:
    binner = TrainOnlyQuantileBinner(quantiles=(0.5,))
    with pytest.raises(ValueError):
        binner.fit([1.0, float("nan")])
    binner.fit([0.0, 1.0])
    with pytest.raises(ValueError):
        binner.transform([float("inf")])


def test_empty_fit_raises() -> None:
    binner = TrainOnlyQuantileBinner(quantiles=(0.5,))

    with pytest.raises(ValueError):
        binner.fit([])

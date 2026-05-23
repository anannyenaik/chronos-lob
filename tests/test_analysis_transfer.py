"""Tests for ``chronoslob.analysis.transfer``."""

from __future__ import annotations

import math

import pytest

from chronoslob.analysis.transfer import (
    TransferResult,
    TransferSplit,
    build_transfer_matrix,
    compare_in_domain_vs_out_of_domain,
    summarise_transfer_results,
)


def _records() -> list[TransferResult]:
    return [
        TransferResult(
            train_scope="A",
            eval_scope="A",
            metric_name="accuracy",
            metric_value=0.6,
            is_synthetic=True,
        ),
        TransferResult(
            train_scope="A",
            eval_scope="B",
            metric_name="accuracy",
            metric_value=0.4,
            is_synthetic=True,
        ),
        TransferResult(
            train_scope="B",
            eval_scope="A",
            metric_name="accuracy",
            metric_value=0.35,
            is_synthetic=True,
        ),
        TransferResult(
            train_scope="B",
            eval_scope="B",
            metric_name="accuracy",
            metric_value=0.55,
            is_synthetic=True,
        ),
        TransferResult(
            train_scope="A",
            eval_scope="A",
            metric_name="nll",
            metric_value=0.7,
            is_synthetic=True,
        ),
    ]


def test_transfer_matrix_construction_orders_deterministically() -> None:
    matrix = build_transfer_matrix(_records(), metric_name="accuracy")
    assert matrix.train_scopes == ("A", "B")
    assert matrix.eval_scopes == ("A", "B")
    assert matrix.shape() == (2, 2)
    assert math.isclose(float(matrix.get("A", "A") or 0.0), 0.6, rel_tol=1e-9)
    assert math.isclose(float(matrix.get("A", "B") or 0.0), 0.4, rel_tol=1e-9)


def test_transfer_matrix_missing_pairs_filled_with_none() -> None:
    records = [
        TransferResult(
            train_scope="A",
            eval_scope="A",
            metric_name="accuracy",
            metric_value=0.5,
            is_synthetic=True,
        ),
        TransferResult(
            train_scope="B",
            eval_scope="B",
            metric_name="accuracy",
            metric_value=0.55,
            is_synthetic=True,
        ),
    ]
    matrix = build_transfer_matrix(records, metric_name="accuracy")
    assert matrix.get("A", "B") is None
    assert matrix.get("B", "A") is None


def test_transfer_matrix_rejects_duplicate_cells() -> None:
    duplicated = [
        *_records(),
        TransferResult(
            train_scope="A",
            eval_scope="A",
            metric_name="accuracy",
            metric_value=0.99,
            is_synthetic=True,
        ),
    ]
    with pytest.raises(ValueError):
        build_transfer_matrix(duplicated, metric_name="accuracy")


def test_in_domain_vs_out_of_domain_comparison() -> None:
    comparison = compare_in_domain_vs_out_of_domain(
        _records(), metric_name="accuracy"
    )
    assert comparison["in_domain"]["count"] == 2
    assert comparison["out_of_domain"]["count"] == 2
    assert comparison["absolute_gap_in_minus_out"] is not None
    assert comparison["is_synthetic"] is True


def test_summarise_transfer_handles_multiple_metrics_separately() -> None:
    summary = summarise_transfer_results(_records())
    assert "accuracy" in summary
    assert "nll" in summary
    assert summary["accuracy"]["n_present_cells"] == 4
    assert summary["nll"]["n_present_cells"] == 1


def test_synthetic_flag_preserved_in_matrix() -> None:
    matrix = build_transfer_matrix(_records(), metric_name="accuracy")
    assert matrix.is_synthetic is True


def test_transfer_matrix_deterministic_ordering_with_unsorted_inputs() -> None:
    rotated = list(reversed(_records()))
    matrix = build_transfer_matrix(rotated, metric_name="accuracy")
    assert matrix.train_scopes == ("A", "B")
    assert matrix.eval_scopes == ("A", "B")


def test_transfer_split_validates_non_empty_scopes() -> None:
    with pytest.raises(ValueError):
        TransferSplit(source_scope="", target_scope="A")
    with pytest.raises(ValueError):
        TransferSplit(source_scope="A", target_scope="")


def test_compare_requires_metric_name() -> None:
    with pytest.raises(ValueError):
        compare_in_domain_vs_out_of_domain(_records(), metric_name="")

"""Temporal splitting and leakage-control utilities for training workflows.

The splitters in this module operate only on row order, timestamps and label
horizon metadata. They deliberately do not inspect feature values or label
values, which keeps partition construction auditable before modelling begins.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from chronoslob.data.schemas import LabelRow

__all__ = [
    "PurgedEmbargoConfig",
    "SplitIndices",
    "TemporalSplitConfig",
    "TrainOnlyQuantileBinner",
    "WalkForwardSplitConfig",
    "apply_purge_and_embargo",
    "apply_row_embargo",
    "intervals_overlap",
    "label_horizon_end_indices_from_rows",
    "make_label_horizon_end_indices_from_frame",
    "purge_overlapping_train_indices",
    "temporal_train_validation_test_split",
    "walk_forward_splits",
]

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)
_FRACTION_TOLERANCE = 1e-9


def _validate_non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validate_positive_int(value: int, *, name: str) -> int:
    _validate_non_negative_int(value, name=name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_index_sequence(indices: Sequence[int], *, name: str) -> list[int]:
    cleaned: list[int] = []
    for position, index in enumerate(indices):
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"{name}[{position}] must be an integer")
        if index < 0:
            raise ValueError(f"{name}[{position}] must be non-negative")
        cleaned.append(index)
    if cleaned != sorted(cleaned):
        raise ValueError(f"{name} must be sorted ascending")
    if len(cleaned) != len(set(cleaned)):
        raise ValueError(f"{name} must not contain duplicate indices")
    return cleaned


class SplitIndices(BaseModel):
    """Row indices for one train/validation/test split.

    Each partition is sorted ascending, contains non-negative row indices and
    is disjoint from the other partitions.
    """

    model_config = _MODEL_CONFIG

    train: list[int] = Field(default_factory=list)
    validation: list[int] = Field(default_factory=list)
    test: list[int] = Field(default_factory=list)

    @field_validator("train", "validation", "test")
    @classmethod
    def _validate_partition(cls, value: list[int], info: Any) -> list[int]:
        return _validate_index_sequence(value, name=str(info.field_name))

    @model_validator(mode="after")
    def _validate_no_overlap(self) -> SplitIndices:
        self.assert_no_overlap()
        return self

    @property
    def n_train(self) -> int:
        """Number of training rows."""
        return len(self.train)

    @property
    def n_validation(self) -> int:
        """Number of validation rows."""
        return len(self.validation)

    @property
    def n_test(self) -> int:
        """Number of test rows."""
        return len(self.test)

    def all_indices(self) -> list[int]:
        """Return all split indices as one sorted list."""
        return sorted(chain(self.train, self.validation, self.test))

    def assert_no_overlap(self) -> None:
        """Raise ``ValueError`` if any partition overlaps another partition."""
        train = set(self.train)
        validation = set(self.validation)
        test = set(self.test)
        if train & validation:
            raise ValueError("train and validation indices overlap")
        if train & test:
            raise ValueError("train and test indices overlap")
        if validation & test:
            raise ValueError("validation and test indices overlap")

    def to_dict(self) -> dict[str, list[int]]:
        """Return a serialisable dictionary representation of the split."""
        return {
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
        }


class TemporalSplitConfig(BaseModel):
    """Configuration for a contiguous temporal train/validation/test split."""

    model_config = _MODEL_CONFIG

    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    min_train_size: int = 1
    min_validation_size: int = 1
    min_test_size: int = 0

    @field_validator("train_fraction", "validation_fraction", "test_fraction")
    @classmethod
    def _validate_fraction(cls, value: float, info: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{info.field_name} must be a finite number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{info.field_name} must be finite")
        if numeric < 0.0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return numeric

    @field_validator("min_train_size", "min_validation_size", "min_test_size")
    @classmethod
    def _validate_min_size(cls, value: int, info: Any) -> int:
        return _validate_non_negative_int(value, name=str(info.field_name))

    @model_validator(mode="after")
    def _validate_fraction_sum(self) -> TemporalSplitConfig:
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_FRACTION_TOLERANCE):
            raise ValueError("split fractions must sum to 1.0")
        return self


class WalkForwardSplitConfig(BaseModel):
    """Configuration for deterministic walk-forward validation folds."""

    model_config = _MODEL_CONFIG

    train_window: int
    validation_window: int
    test_window: int = 0
    step_size: int | None = None
    expanding_train: bool = True
    min_train_size: int | None = None

    @field_validator("train_window")
    @classmethod
    def _validate_train_window(cls, value: int) -> int:
        return _validate_positive_int(value, name="train_window")

    @field_validator("validation_window")
    @classmethod
    def _validate_validation_window(cls, value: int) -> int:
        return _validate_positive_int(value, name="validation_window")

    @field_validator("test_window")
    @classmethod
    def _validate_test_window(cls, value: int) -> int:
        return _validate_non_negative_int(value, name="test_window")

    @field_validator("step_size")
    @classmethod
    def _validate_step_size(cls, value: int | None) -> int | None:
        if value is None:
            return None
        return _validate_positive_int(value, name="step_size")

    @field_validator("min_train_size")
    @classmethod
    def _validate_min_train_size(cls, value: int | None) -> int | None:
        if value is None:
            return None
        return _validate_positive_int(value, name="min_train_size")


class PurgedEmbargoConfig(BaseModel):
    """Configuration for purging overlapping labels and row embargoes."""

    model_config = _MODEL_CONFIG

    purge: bool = True
    embargo: int = 0
    embargo_unit: str = "rows"

    @field_validator("embargo")
    @classmethod
    def _validate_embargo(cls, value: int) -> int:
        return _validate_non_negative_int(value, name="embargo")

    @field_validator("embargo_unit")
    @classmethod
    def _validate_embargo_unit(cls, value: str) -> str:
        if value != "rows":
            raise ValueError("embargo_unit currently only supports 'rows'")
        return value


DEFAULT_TEMPORAL_SPLIT_CONFIG = TemporalSplitConfig()
DEFAULT_PURGED_EMBARGO_CONFIG = PurgedEmbargoConfig()


def _validate_n_rows(n_rows: int) -> int:
    return _validate_non_negative_int(n_rows, name="n_rows")


def _allocate_counts(
    n_rows: int,
    *,
    fractions: tuple[float, float, float],
    minimums: tuple[int, int, int],
) -> tuple[int, int, int]:
    if n_rows < sum(minimums):
        raise ValueError(
            "not enough rows for requested minimum split sizes: "
            f"n_rows={n_rows}, minimum_total={sum(minimums)}"
        )

    raw_counts = [n_rows * fraction for fraction in fractions]
    counts = [math.floor(raw_count) for raw_count in raw_counts]
    remaining = n_rows - sum(counts)
    remainder_order = sorted(
        range(3),
        key=lambda index: (raw_counts[index] - counts[index], -index),
        reverse=True,
    )
    for index in remainder_order[:remaining]:
        counts[index] += 1

    for index, minimum in enumerate(minimums):
        while counts[index] < minimum:
            donors = [
                donor
                for donor in range(3)
                if counts[donor] > minimums[donor] and donor != index
            ]
            if not donors:
                raise ValueError(
                    "not enough rows for requested minimum split sizes after "
                    "fraction allocation"
                )
            donor = max(donors, key=lambda item: (counts[item] - minimums[item], item))
            counts[donor] -= 1
            counts[index] += 1

    return counts[0], counts[1], counts[2]


def temporal_train_validation_test_split(
    n_rows: int,
    config: TemporalSplitConfig = DEFAULT_TEMPORAL_SPLIT_CONFIG,
) -> SplitIndices:
    """Split ordered rows into contiguous train, validation and test partitions."""
    rows = _validate_n_rows(n_rows)
    train_count, validation_count, test_count = _allocate_counts(
        rows,
        fractions=(
            config.train_fraction,
            config.validation_fraction,
            config.test_fraction,
        ),
        minimums=(
            config.min_train_size,
            config.min_validation_size,
            config.min_test_size,
        ),
    )
    train_end = train_count
    validation_end = train_end + validation_count
    test_end = validation_end + test_count

    return SplitIndices(
        train=list(range(0, train_end)),
        validation=list(range(train_end, validation_end)),
        test=list(range(validation_end, test_end)),
    )


def walk_forward_splits(
    n_rows: int,
    config: WalkForwardSplitConfig,
) -> list[SplitIndices]:
    """Create ordered walk-forward folds, skipping incomplete final folds."""
    rows = _validate_n_rows(n_rows)
    step = config.step_size or config.validation_window
    min_train_size = config.min_train_size or config.train_window
    if not config.expanding_train and min_train_size > config.train_window:
        raise ValueError("min_train_size cannot exceed train_window for rolling folds")

    folds: list[SplitIndices] = []
    if config.expanding_train:
        train_end = max(config.train_window, min_train_size)
        while True:
            validation_start = train_end
            validation_end = validation_start + config.validation_window
            test_end = validation_end + config.test_window
            if test_end > rows:
                break
            folds.append(
                SplitIndices(
                    train=list(range(0, train_end)),
                    validation=list(range(validation_start, validation_end)),
                    test=list(range(validation_end, test_end)),
                )
            )
            train_end += step
        return folds

    train_start = 0
    while True:
        train_end = train_start + config.train_window
        validation_start = train_end
        validation_end = validation_start + config.validation_window
        test_end = validation_end + config.test_window
        if test_end > rows:
            break
        folds.append(
            SplitIndices(
                train=list(range(train_start, train_end)),
                validation=list(range(validation_start, validation_end)),
                test=list(range(validation_end, test_end)),
            )
        )
        train_start += step
    return folds


def intervals_overlap(
    start_a: int,
    end_a: int,
    start_b: int,
    end_b: int,
    *,
    closed: bool = True,
) -> bool:
    """Return whether two row-index intervals overlap.

    With ``closed=True`` the intervals are interpreted as ``[start, end]`` and
    touching endpoints overlap. With ``closed=False`` the comparison uses
    half-open-style semantics where touching endpoints do not overlap.
    """
    for name, value in (
        ("start_a", start_a),
        ("end_a", end_a),
        ("start_b", start_b),
        ("end_b", end_b),
    ):
        _validate_non_negative_int(value, name=name)
    if start_a > end_a:
        raise ValueError("start_a must be <= end_a")
    if start_b > end_b:
        raise ValueError("start_b must be <= end_b")

    if closed:
        return max(start_a, start_b) <= min(end_a, end_b)
    return max(start_a, start_b) < min(end_a, end_b)


def purge_overlapping_train_indices(
    train_indices: Sequence[int],
    evaluation_indices: Sequence[int],
    label_horizon_end_by_index: Mapping[int, int],
) -> list[int]:
    """Remove train rows whose label horizon overlaps an evaluation block."""
    train = _validate_index_sequence(train_indices, name="train_indices")
    evaluation = _validate_index_sequence(evaluation_indices, name="evaluation_indices")
    if not evaluation:
        return list(train)

    evaluation_start = min(evaluation)
    evaluation_end = max(evaluation)
    kept: list[int] = []
    for index in train:
        if index not in label_horizon_end_by_index:
            raise KeyError(f"missing label horizon metadata for train index {index}")
        horizon_end = label_horizon_end_by_index[index]
        _validate_non_negative_int(horizon_end, name=f"label_horizon_end[{index}]")
        if horizon_end < index:
            raise ValueError(
                f"label horizon end for train index {index} is before the row index"
            )
        if not intervals_overlap(index, horizon_end, evaluation_start, evaluation_end):
            kept.append(index)
    return kept


def apply_row_embargo(
    train_indices: Sequence[int],
    evaluation_indices: Sequence[int],
    embargo: int,
) -> list[int]:
    """Apply a symmetric row embargo around an evaluation block."""
    train = _validate_index_sequence(train_indices, name="train_indices")
    evaluation = _validate_index_sequence(evaluation_indices, name="evaluation_indices")
    rows = _validate_non_negative_int(embargo, name="embargo")
    if rows == 0 or not evaluation:
        return list(train)

    lower = min(evaluation) - rows
    upper = max(evaluation) + rows
    return [index for index in train if not lower <= index <= upper]


def _evaluation_indices_for_split(split: SplitIndices, purge_against: str) -> list[int]:
    if purge_against == "validation":
        return list(split.validation)
    if purge_against == "test":
        return list(split.test)
    if purge_against == "validation_test":
        return sorted([*split.validation, *split.test])
    raise ValueError(
        "purge_against must be one of 'validation', 'test' or 'validation_test'"
    )


def apply_purge_and_embargo(
    split: SplitIndices,
    label_horizon_end_by_index: Mapping[int, int],
    config: PurgedEmbargoConfig = DEFAULT_PURGED_EMBARGO_CONFIG,
    *,
    purge_against: str = "validation",
) -> SplitIndices:
    """Return a split with training rows purged and embargoed."""
    evaluation = _evaluation_indices_for_split(split, purge_against)
    train = list(split.train)
    if config.purge and evaluation:
        train = purge_overlapping_train_indices(
            train,
            evaluation,
            label_horizon_end_by_index,
        )
    if config.embargo > 0 and evaluation:
        train = apply_row_embargo(train, evaluation, config.embargo)
    return SplitIndices(
        train=train,
        validation=list(split.validation),
        test=list(split.test),
    )


def label_horizon_end_indices_from_rows(
    label_rows: Sequence[LabelRow],
    timestamp_to_index: Mapping[datetime, int] | None = None,
) -> dict[int, int]:
    """Map ``LabelRow.horizon_end`` timestamps to row-index horizon ends."""
    rows = list(label_rows)
    if timestamp_to_index is None:
        timestamp_index: dict[datetime, int] = {}
        for index, row in enumerate(rows):
            if not isinstance(row, LabelRow):
                raise TypeError(
                    "label_rows must contain LabelRow objects; "
                    f"got {type(row).__name__} at index {index}"
                )
            if row.timestamp in timestamp_index:
                raise ValueError(
                    "duplicate label row timestamps require an explicit "
                    "timestamp_to_index mapping"
                )
            timestamp_index[row.timestamp] = index
    else:
        timestamp_index = dict(timestamp_to_index)

    result: dict[int, int] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, LabelRow):
            raise TypeError(
                "label_rows must contain LabelRow objects; "
                f"got {type(row).__name__} at index {position}"
            )
        if row.timestamp not in timestamp_index:
            raise KeyError(f"timestamp {row.timestamp!r} is missing from timestamp_to_index")
        if row.horizon_end not in timestamp_index:
            raise KeyError(f"horizon_end {row.horizon_end!r} cannot be mapped to a row index")
        row_index = timestamp_index[row.timestamp]
        horizon_end_index = timestamp_index[row.horizon_end]
        _validate_non_negative_int(row_index, name="row_index")
        _validate_non_negative_int(horizon_end_index, name="horizon_end_index")
        if horizon_end_index < row_index:
            raise ValueError("horizon_end index must not be before row timestamp index")
        if row_index in result:
            raise ValueError(f"duplicate mapped row index {row_index}")
        result[row_index] = horizon_end_index
    return result


def make_label_horizon_end_indices_from_frame(
    label_frame: pd.DataFrame,
) -> dict[int, int]:
    """Build row-index horizon ends from a label frame with horizon timestamps."""
    if not isinstance(label_frame, pd.DataFrame):
        raise TypeError("label_frame must be a pandas DataFrame")
    missing = {"timestamp", "horizon_end"} - set(label_frame.columns)
    if missing:
        raise ValueError(f"label_frame is missing required columns: {sorted(missing)}")
    if label_frame.empty:
        return {}

    timestamps = pd.to_datetime(label_frame["timestamp"], utc=True, errors="raise")
    horizon_ends = pd.to_datetime(label_frame["horizon_end"], utc=True, errors="raise")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("label_frame timestamps must be sorted non-decreasing")

    ordered_timestamps = list(timestamps)
    result: dict[int, int] = {}
    for row_index, horizon_end in enumerate(horizon_ends):
        horizon_end_index = bisect_right(ordered_timestamps, horizon_end) - 1
        if horizon_end_index < 0:
            raise ValueError(
                f"horizon_end at row {row_index} is before the first timestamp"
            )
        if horizon_end_index < row_index:
            raise ValueError(
                f"horizon_end at row {row_index} maps before the row timestamp"
            )
        result[row_index] = horizon_end_index
    return result


@dataclass
class TrainOnlyQuantileBinner:
    """Quantile binner that must be fitted on the training partition only.

    The caller is responsible for calling ``fit`` with train values and then
    using the stored edges for validation/test values. ``transform`` never
    recomputes or mutates the fitted bin edges.
    """

    quantiles: tuple[float, ...]
    bin_edges_: list[float] | None = None

    def __post_init__(self) -> None:
        if not self.quantiles:
            raise ValueError("quantiles must contain at least one value")
        previous: float | None = None
        for position, quantile in enumerate(self.quantiles):
            if isinstance(quantile, bool) or not isinstance(quantile, (int, float)):
                raise TypeError(f"quantiles[{position}] must be a finite number")
            numeric = float(quantile)
            if not math.isfinite(numeric):
                raise ValueError(f"quantiles[{position}] must be finite")
            if numeric <= 0.0 or numeric >= 1.0:
                raise ValueError("quantiles must be strictly between 0 and 1")
            if previous is not None and numeric <= previous:
                raise ValueError("quantiles must be strictly increasing")
            previous = numeric
        self.quantiles = tuple(float(quantile) for quantile in self.quantiles)

    @staticmethod
    def _validate_values(values: Sequence[float], *, name: str) -> list[float]:
        cleaned: list[float] = []
        for position, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name}[{position}] must be a finite number")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"{name}[{position}] must be finite")
            cleaned.append(numeric)
        return cleaned

    def fit(self, values: Sequence[float]) -> TrainOnlyQuantileBinner:
        """Fit bin edges from finite training values."""
        cleaned = self._validate_values(values, name="values")
        if not cleaned:
            raise ValueError("fit values must be non-empty")
        edges = np.quantile(np.asarray(cleaned, dtype=float), self.quantiles)
        self.bin_edges_ = [float(edge) for edge in edges.tolist()]
        return self

    def transform(self, values: Sequence[float]) -> list[int]:
        """Bin values using previously fitted edges without refitting."""
        if self.bin_edges_ is None:
            raise ValueError("TrainOnlyQuantileBinner must be fitted before transform")
        cleaned = self._validate_values(values, name="values")
        edges = list(self.bin_edges_)
        return [
            int(np.searchsorted(edges, value, side="right"))
            for value in cleaned
        ]

    def fit_transform(self, values: Sequence[float]) -> list[int]:
        """Fit on ``values`` and return their train-edge bin assignments."""
        self.fit(values)
        return self.transform(values)

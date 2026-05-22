"""Reusable data validation helpers.

This module centralises the data-quality checks used when loading
benchmark or exchange-derived datasets. Validation results aggregate
typed :class:`chronoslob.data.schemas.DataQualityIssue` records so
callers can decide whether to surface, log or raise on failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from chronoslob.data.schemas import DataQualityIssue

if TYPE_CHECKING:
    from chronoslob.data.fi2010 import FI2010Config, FI2010Dataset

__all__ = [
    "DataValidationError",
    "DataValidationResult",
    "validate_fi2010_dataset",
    "validate_numeric_frame",
]


class DataValidationError(ValueError):
    """Raised when validation issues at the ``error`` severity are found."""

    def __init__(self, result: DataValidationResult) -> None:
        super().__init__(
            f"data validation failed with {result.error_count} error(s)"
        )
        self.result = result


@dataclass
class DataValidationResult:
    """Aggregated result of a validation pass over a tabular dataset."""

    ok: bool
    issues: list[DataQualityIssue] = field(default_factory=list)
    n_rows: int = 0
    n_columns: int = 0

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")

    def raise_if_errors(self) -> None:
        """Raise :class:`DataValidationError` if any error-severity issues exist."""
        if self.error_count > 0:
            raise DataValidationError(self)

    def summary(self) -> dict[str, int | bool]:
        return {
            "ok": self.ok,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _append_error(
    issues: list[DataQualityIssue],
    *,
    code: str,
    message: str,
    metadata: dict[str, str | int | float | bool] | None = None,
) -> None:
    issues.append(
        DataQualityIssue(
            severity="error",
            code=code,
            message=message,
            metadata=metadata or {},
        )
    )


def _append_warning(
    issues: list[DataQualityIssue],
    *,
    code: str,
    message: str,
    metadata: dict[str, str | int | float | bool] | None = None,
) -> None:
    issues.append(
        DataQualityIssue(
            severity="warning",
            code=code,
            message=message,
            metadata=metadata or {},
        )
    )


def _has_inf(series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False
    values = series.to_numpy(copy=False)
    try:
        return bool(np.isinf(values).any())
    except TypeError:
        return False


# ---------------------------------------------------------------------------
# Generic numeric-frame validator
# ---------------------------------------------------------------------------


def validate_numeric_frame(
    frame: pd.DataFrame,
    required_columns: list[str] | None = None,
    allow_nan: bool = False,
) -> DataValidationResult:
    """Validate a numeric pandas DataFrame.

    Checks: non-empty frame, presence of any required columns, no NaN
    values (unless ``allow_nan`` is set), no infinite values in numeric
    columns and no duplicate column names. The function does not raise;
    callers should inspect or :meth:`DataValidationResult.raise_if_errors`
    on the result.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    issues: list[DataQualityIssue] = []
    n_rows, n_columns = frame.shape

    if n_rows == 0:
        _append_error(
            issues,
            code="empty_frame",
            message="dataframe contains no rows",
        )

    duplicate_columns = (
        frame.columns[frame.columns.duplicated()].tolist()
    )
    if duplicate_columns:
        _append_error(
            issues,
            code="duplicate_columns",
            message=f"duplicate column names: {sorted(set(duplicate_columns))}",
        )

    missing_required: list[str] = []
    if required_columns:
        available = set(frame.columns)
        missing_required = [c for c in required_columns if c not in available]
        if missing_required:
            _append_error(
                issues,
                code="missing_required_columns",
                message=(
                    "required columns are absent: "
                    f"{sorted(set(missing_required))}"
                ),
            )

    columns_to_check = (
        required_columns
        if required_columns
        else list(frame.columns)
    )
    for column in columns_to_check:
        if column not in frame.columns:
            continue
        series = frame[column]
        if not pd.api.types.is_numeric_dtype(series):
            _append_error(
                issues,
                code="non_numeric_column",
                message=f"column {column!r} is not numeric",
                metadata={"column": column},
            )
            continue
        if not allow_nan and series.isna().any():
            _append_error(
                issues,
                code="nan_in_numeric_column",
                message=f"column {column!r} contains NaN values",
                metadata={"column": column},
            )
        if _has_inf(series):
            _append_error(
                issues,
                code="inf_in_numeric_column",
                message=f"column {column!r} contains infinite values",
                metadata={"column": column},
            )

    return DataValidationResult(
        ok=not any(issue.severity == "error" for issue in issues),
        issues=issues,
        n_rows=n_rows,
        n_columns=n_columns,
    )


# ---------------------------------------------------------------------------
# FI-2010 dataset validator
# ---------------------------------------------------------------------------


def validate_fi2010_dataset(dataset: FI2010Dataset) -> DataValidationResult:
    """Validate an FI-2010 dataset against the loader's expectations.

    Performed checks include: a non-empty frame; the presence of configured
    feature, label, split and timestamp columns; numeric, finite feature
    columns; finite numeric label columns; non-missing split values; the
    parseability of any timestamp column; and pairing of bid/ask LOB
    columns when they are present.
    """
    issues: list[DataQualityIssue] = []
    frame = dataset.frame
    config = dataset.config
    n_rows, n_columns = frame.shape

    if n_rows == 0:
        _append_error(
            issues,
            code="empty_frame",
            message="FI-2010 dataset contains no rows",
        )

    available_columns = set(frame.columns)

    for column in dataset.feature_columns:
        if column not in available_columns:
            _append_error(
                issues,
                code="missing_feature_column",
                message=f"feature column {column!r} is missing",
                metadata={"column": column},
            )
            continue
        series = frame[column]
        if not pd.api.types.is_numeric_dtype(series):
            _append_error(
                issues,
                code="non_numeric_feature",
                message=f"feature column {column!r} is not numeric",
                metadata={"column": column},
            )
            continue
        if series.isna().any():
            _append_error(
                issues,
                code="nan_in_feature",
                message=f"feature column {column!r} contains NaN",
                metadata={"column": column},
            )
        if _has_inf(series):
            _append_error(
                issues,
                code="inf_in_feature",
                message=f"feature column {column!r} contains infinite values",
                metadata={"column": column},
            )

    for column in dataset.label_columns:
        if column not in available_columns:
            _append_error(
                issues,
                code="missing_label_column",
                message=f"label column {column!r} is missing",
                metadata={"column": column},
            )
            continue
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            if series.isna().any():
                _append_error(
                    issues,
                    code="nan_in_label",
                    message=f"label column {column!r} contains NaN",
                    metadata={"column": column},
                )
            if _has_inf(series):
                _append_error(
                    issues,
                    code="inf_in_label",
                    message=(
                        f"label column {column!r} contains infinite values"
                    ),
                    metadata={"column": column},
                )

    if config.split_column is not None:
        if config.split_column not in available_columns:
            _append_error(
                issues,
                code="missing_split_column",
                message=f"split column {config.split_column!r} is missing",
                metadata={"column": config.split_column},
            )
        elif frame[config.split_column].isna().any():
            _append_error(
                issues,
                code="nan_in_split_column",
                message=(
                    f"split column {config.split_column!r} contains missing values"
                ),
                metadata={"column": config.split_column},
            )

    if config.timestamp_column is not None:
        if config.timestamp_column not in available_columns:
            _append_error(
                issues,
                code="missing_timestamp_column",
                message=(
                    f"timestamp column {config.timestamp_column!r} is missing"
                ),
                metadata={"column": config.timestamp_column},
            )
        else:
            try:
                pd.to_datetime(
                    frame[config.timestamp_column],
                    utc=True,
                    errors="raise",
                )
            except (ValueError, TypeError) as exc:
                _append_error(
                    issues,
                    code="unparseable_timestamps",
                    message=(
                        "timestamp column "
                        f"{config.timestamp_column!r} contains unparseable values: "
                        f"{exc}"
                    ),
                    metadata={"column": config.timestamp_column},
                )

    _validate_lob_pairing(issues, frame, config)

    return DataValidationResult(
        ok=not any(issue.severity == "error" for issue in issues),
        issues=issues,
        n_rows=n_rows,
        n_columns=n_columns,
    )


def _validate_lob_pairing(
    issues: list[DataQualityIssue],
    frame: pd.DataFrame,
    config: FI2010Config,
) -> None:
    """Flag inconsistent bid/ask price-quantity pairing across LOB levels.

    The check only fires for levels where at least one of the four
    expected columns is present. Fully absent levels are silently
    accepted because partial mirrors of FI-2010 are common.
    """
    bid_price_prefix = config.bid_price_prefix
    bid_qty_prefix = config.bid_quantity_prefix
    ask_price_prefix = config.ask_price_prefix
    ask_qty_prefix = config.ask_quantity_prefix
    price_level_count = config.price_level_count

    available = set(frame.columns)
    for level in range(1, price_level_count + 1):
        bp = f"{bid_price_prefix}{level}"
        bq_primary = f"{bid_qty_prefix}{level}"
        ap = f"{ask_price_prefix}{level}"
        aq_primary = f"{ask_qty_prefix}{level}"
        bq_alias = (
            f"bid_size_{level}" if bid_qty_prefix == "bid_quantity_" else None
        )
        aq_alias = (
            f"ask_size_{level}" if ask_qty_prefix == "ask_quantity_" else None
        )

        bq_present = bq_primary in available or (
            bq_alias is not None and bq_alias in available
        )
        aq_present = aq_primary in available or (
            aq_alias is not None and aq_alias in available
        )
        bp_present = bp in available
        ap_present = ap in available

        flags = [bp_present, bq_present, ap_present, aq_present]
        any_present = any(flags)
        all_present = all(flags)
        if any_present and not all_present:
            missing = [
                name
                for name, present in (
                    (bp, bp_present),
                    (bq_primary, bq_present),
                    (ap, ap_present),
                    (aq_primary, aq_present),
                )
                if not present
            ]
            _append_error(
                issues,
                code="incomplete_lob_level",
                message=(
                    f"LOB level {level} is partially specified; missing "
                    f"{sorted(missing)}"
                ),
                metadata={"level": level},
            )

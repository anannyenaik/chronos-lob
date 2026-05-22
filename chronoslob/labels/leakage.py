"""Leakage-control utilities for feature and label artefacts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from chronoslob.data.schemas import DataQualityIssue, FeatureRow, LabelRow

__all__ = [
    "LeakageCheckResult",
    "assert_feature_label_separation",
    "assert_no_future_feature_timestamps",
    "assert_temporal_label_alignment",
    "validate_no_lookahead",
]


class LeakageCheckError(ValueError):
    """Raised when leakage checks contain error-severity issues."""

    def __init__(self, result: LeakageCheckResult) -> None:
        super().__init__(f"leakage check failed with {result.error_count} error(s)")
        self.result = result


@dataclass
class LeakageCheckResult:
    """Aggregated leakage-check result."""

    ok: bool
    issues: list[DataQualityIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        """Return the number of error-severity issues."""
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        """Return the number of warning-severity issues."""
        return sum(1 for issue in self.issues if issue.severity == "warning")

    def raise_if_errors(self) -> None:
        """Raise ``LeakageCheckError`` if any error-severity issues exist."""
        if self.error_count:
            raise LeakageCheckError(self)

    def summary(self) -> dict[str, int | bool]:
        """Return a compact, serialisable summary."""
        return {
            "ok": self.ok,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
        }


_DEFAULT_SHARED_COLUMNS = {"timestamp", "symbol", "split"}
_LABEL_LIKE_PREFIXES = ("label", "y_", "future_", "target")
_IDENTIFIER_COLUMNS = {
    "timestamp",
    "symbol",
    "split",
    "horizon_start",
    "horizon_end",
    "label_source",
}


def _issue(
    *,
    severity: str,
    code: str,
    message: str,
    metadata: dict[str, str | int | float | bool] | None = None,
) -> DataQualityIssue:
    return DataQualityIssue(
        severity=severity,
        code=code,
        message=message,
        metadata=metadata or {},
    )


def _result(issues: list[DataQualityIssue]) -> LeakageCheckResult:
    return LeakageCheckResult(
        ok=not any(issue.severity == "error" for issue in issues),
        issues=issues,
    )


def _column_list(columns: set[str]) -> str:
    return ",".join(sorted(columns))


def _value_columns(frame: pd.DataFrame, allowed_shared_columns: set[str]) -> set[str]:
    excluded = allowed_shared_columns | {"horizon_start", "horizon_end", "label_source"}
    return {str(column) for column in frame.columns if str(column) not in excluded}


def _is_timezone_aware(value: object) -> bool:
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return False
        return value.tzinfo is not None and value.tzinfo.utcoffset(value) is not None
    if isinstance(value, datetime):
        return value.tzinfo is not None and value.tzinfo.utcoffset(value) is not None
    return False


def _to_datetime_or_none(value: object) -> datetime | pd.Timestamp | None:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value
    return None


def assert_feature_label_separation(
    feature_frame: pd.DataFrame,
    label_frame: pd.DataFrame,
    *,
    allowed_shared_columns: set[str] | None = None,
) -> LeakageCheckResult:
    """Check that feature and label frames do not share value columns."""
    if not isinstance(feature_frame, pd.DataFrame):
        raise TypeError("feature_frame must be a pandas DataFrame")
    if not isinstance(label_frame, pd.DataFrame):
        raise TypeError("label_frame must be a pandas DataFrame")

    shared = set(
        _DEFAULT_SHARED_COLUMNS
        if allowed_shared_columns is None
        else allowed_shared_columns
    )
    issues: list[DataQualityIssue] = []
    feature_columns = {str(column) for column in feature_frame.columns}
    label_value_columns = _value_columns(label_frame, shared)
    feature_value_columns = _value_columns(feature_frame, shared)

    label_columns_in_features = feature_columns & label_value_columns
    if label_columns_in_features:
        issues.append(
            _issue(
                severity="error",
                code="label_column_in_feature_frame",
                message="label value columns are present in the feature frame",
                metadata={"columns": _column_list(label_columns_in_features)},
            )
        )

    feature_columns_in_labels = feature_value_columns & label_value_columns
    if feature_columns_in_labels:
        issues.append(
            _issue(
                severity="warning",
                code="feature_column_in_label_frame",
                message="feature-like value columns are present in the label frame",
                metadata={"columns": _column_list(feature_columns_in_labels)},
            )
        )

    for column in sorted(feature_value_columns):
        lowered = column.lower()
        if any(lowered.startswith(prefix) for prefix in _LABEL_LIKE_PREFIXES):
            issues.append(
                _issue(
                    severity="error",
                    code="label_like_column_in_features",
                    message=(
                        f"feature column {column!r} looks like a label or target"
                    ),
                    metadata={"column": column},
                )
            )

    return _result(issues)


def assert_temporal_label_alignment(
    label_rows: Sequence[LabelRow],
) -> LeakageCheckResult:
    """Check that label horizons are explicit, ordered and timezone-aware."""
    issues: list[DataQualityIssue] = []
    for position, row in enumerate(label_rows):
        timestamp = getattr(row, "timestamp", None)
        horizon_start = getattr(row, "horizon_start", None)
        horizon_end = getattr(row, "horizon_end", None)

        if not _is_timezone_aware(timestamp):
            issues.append(
                _issue(
                    severity="error",
                    code="naive_label_timestamp",
                    message="LabelRow timestamp must be timezone-aware",
                    metadata={"position": position},
                )
            )
        if not _is_timezone_aware(horizon_start):
            issues.append(
                _issue(
                    severity="error",
                    code="naive_horizon_start",
                    message="LabelRow horizon_start must be timezone-aware",
                    metadata={"position": position},
                )
            )
        if not _is_timezone_aware(horizon_end):
            issues.append(
                _issue(
                    severity="error",
                    code="naive_horizon_end",
                    message="LabelRow horizon_end must be timezone-aware",
                    metadata={"position": position},
                )
            )

        ts_dt = _to_datetime_or_none(timestamp)
        start_dt = _to_datetime_or_none(horizon_start)
        end_dt = _to_datetime_or_none(horizon_end)
        if ts_dt is None or start_dt is None or end_dt is None:
            continue
        if (
            _is_timezone_aware(ts_dt)
            and _is_timezone_aware(start_dt)
            and start_dt < ts_dt
        ):
            issues.append(
                _issue(
                    severity="error",
                    code="horizon_start_before_timestamp",
                    message="horizon_start must be at or after timestamp",
                    metadata={"position": position},
                )
            )
        if (
            _is_timezone_aware(start_dt)
            and _is_timezone_aware(end_dt)
            and end_dt <= start_dt
        ):
            issues.append(
                _issue(
                    severity="error",
                    code="horizon_end_not_after_start",
                    message="horizon_end must be strictly after horizon_start",
                    metadata={"position": position},
                )
            )
    return _result(issues)


def assert_no_future_feature_timestamps(
    feature_rows: Sequence[FeatureRow],
) -> LeakageCheckResult:
    """Check that feature rows do not point to future origin timestamps."""
    issues: list[DataQualityIssue] = []
    for position, row in enumerate(feature_rows):
        timestamp = getattr(row, "timestamp", None)
        origin = getattr(row, "horizon_origin_timestamp", None)
        if not _is_timezone_aware(timestamp):
            issues.append(
                _issue(
                    severity="error",
                    code="naive_feature_timestamp",
                    message="FeatureRow timestamp must be timezone-aware",
                    metadata={"position": position},
                )
            )
        if origin is None:
            continue
        if not _is_timezone_aware(origin):
            issues.append(
                _issue(
                    severity="error",
                    code="naive_feature_origin_timestamp",
                    message="FeatureRow horizon_origin_timestamp must be timezone-aware",
                    metadata={"position": position},
                )
            )
            continue
        ts_dt = _to_datetime_or_none(timestamp)
        origin_dt = _to_datetime_or_none(origin)
        if ts_dt is not None and origin_dt is not None and origin_dt > ts_dt:
            issues.append(
                _issue(
                    severity="error",
                    code="feature_origin_after_timestamp",
                    message="feature horizon_origin_timestamp is after timestamp",
                    metadata={"position": position},
                )
            )
    return _result(issues)


def _check_feature_frame_origins(frame: pd.DataFrame) -> list[DataQualityIssue]:
    issues: list[DataQualityIssue] = []
    if "horizon_origin_timestamp" not in frame.columns:
        return issues
    if "timestamp" not in frame.columns:
        issues.append(
            _issue(
                severity="error",
                code="missing_feature_timestamp",
                message=(
                    "feature frame has horizon_origin_timestamp but no timestamp column"
                ),
            )
        )
        return issues
    for position, (timestamp, origin) in enumerate(
        zip(frame["timestamp"], frame["horizon_origin_timestamp"], strict=False)
    ):
        if pd.isna(origin):
            continue
        if not _is_timezone_aware(timestamp) or not _is_timezone_aware(origin):
            issues.append(
                _issue(
                    severity="error",
                    code="naive_feature_frame_timestamp",
                    message="feature frame timestamps must be timezone-aware",
                    metadata={"position": position},
                )
            )
            continue
        if origin > timestamp:
            issues.append(
                _issue(
                    severity="error",
                    code="feature_frame_origin_after_timestamp",
                    message="feature frame origin timestamp is after timestamp",
                    metadata={"position": position},
                )
            )
    return issues


def _check_label_frame_horizons(frame: pd.DataFrame) -> list[DataQualityIssue]:
    issues: list[DataQualityIssue] = []
    required = {"timestamp", "horizon_start", "horizon_end"}
    if not required.issubset(set(frame.columns)):
        return issues
    for position, row in frame.iterrows():
        timestamp = row["timestamp"]
        start = row["horizon_start"]
        end = row["horizon_end"]
        if not _is_timezone_aware(timestamp):
            issues.append(
                _issue(
                    severity="error",
                    code="naive_label_frame_timestamp",
                    message="label frame timestamp must be timezone-aware",
                    metadata={"position": int(position)},
                )
            )
        if not _is_timezone_aware(start) or not _is_timezone_aware(end):
            issues.append(
                _issue(
                    severity="error",
                    code="naive_label_frame_horizon",
                    message="label frame horizon timestamps must be timezone-aware",
                    metadata={"position": int(position)},
                )
            )
            continue
        if start < timestamp:
            issues.append(
                _issue(
                    severity="error",
                    code="label_frame_start_before_timestamp",
                    message="label frame horizon_start is before timestamp",
                    metadata={"position": int(position)},
                )
            )
        if end <= start:
            issues.append(
                _issue(
                    severity="error",
                    code="label_frame_end_not_after_start",
                    message="label frame horizon_end is not after horizon_start",
                    metadata={"position": int(position)},
                )
            )
    return issues


def validate_no_lookahead(
    feature_frame: pd.DataFrame,
    label_frame: pd.DataFrame,
    label_rows: Sequence[LabelRow] | None = None,
) -> LeakageCheckResult:
    """Run the available feature/label separation and horizon checks."""
    issues: list[DataQualityIssue] = []
    issues.extend(
        assert_feature_label_separation(feature_frame, label_frame).issues
    )
    issues.extend(_check_feature_frame_origins(feature_frame))
    issues.extend(_check_label_frame_horizons(label_frame))
    if label_rows is not None:
        issues.extend(assert_temporal_label_alignment(label_rows).issues)
    return _result(issues)

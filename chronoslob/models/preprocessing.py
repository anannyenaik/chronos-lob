"""Train-only preprocessing helpers for classical baseline experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import InitVar, dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

__all__ = [
    "FeatureMatrix",
    "TargetVector",
    "TrainOnlyStandardScaler",
    "align_feature_label_frames",
    "build_feature_matrix",
    "build_target_vector",
    "select_feature_columns",
]

_DEFAULT_EXCLUDE_COLUMNS = {
    "timestamp",
    "symbol",
    "split",
    "horizon_start",
    "horizon_end",
}

_LABEL_LIKE_PREFIXES = (
    "label",
    "y_",
    "future_",
    "target",
    "direction_",
    "return_quantile_",
    "spread_widening_",
    "bid_fill_proxy_",
    "ask_fill_proxy_",
    "bid_adverse_selection_proxy_",
    "ask_adverse_selection_proxy_",
)


def _is_label_like(column: str) -> bool:
    return column.lower().startswith(_LABEL_LIKE_PREFIXES)


def _validate_frame(frame: pd.DataFrame, *, name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")


def _validate_row_indices(
    row_indices: Sequence[int] | None,
    *,
    n_rows: int,
) -> list[int]:
    if row_indices is None:
        return list(range(n_rows))

    rows: list[int] = []
    for position, index in enumerate(row_indices):
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"row_indices[{position}] must be an integer")
        if index < 0:
            raise ValueError(f"row_indices[{position}] must be non-negative")
        if index >= n_rows:
            raise IndexError(
                f"row_indices[{position}]={index} is outside frame length {n_rows}"
            )
        rows.append(index)
    return rows


def _as_numeric_2d(x: np.ndarray, *, name: str) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D")
    try:
        return x.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values") from exc


def _contains_nan(values: np.ndarray) -> bool:
    return bool(pd.isna(values).any())


@dataclass(frozen=True)
class FeatureMatrix:
    """A validated 2D feature matrix with explicit row provenance."""

    x: np.ndarray
    feature_names: list[str]
    row_indices: list[int]
    allow_nan: InitVar[bool] = False

    def __post_init__(self, allow_nan: bool) -> None:
        """Validate matrix shape, feature names, row count and finite values."""
        numeric_x = _as_numeric_2d(self.x, name="x")
        if len(self.feature_names) != self.x.shape[1]:
            raise ValueError("feature_names length must match x.shape[1]")
        if len(self.row_indices) != self.x.shape[0]:
            raise ValueError("row_indices length must match x.shape[0]")
        if bool(np.isinf(numeric_x).any()):
            raise ValueError("x must not contain infinite values")
        if _contains_nan(numeric_x) and not allow_nan:
            raise ValueError("x contains NaN values; pass allow_nan=True explicitly")


@dataclass(frozen=True)
class TargetVector:
    """A validated 1D target vector with explicit row provenance."""

    y: np.ndarray
    target_name: str
    row_indices: list[int]
    classes: list[str | int | float] | None = None

    def __post_init__(self) -> None:
        """Validate target dimensionality and row count."""
        if not isinstance(self.y, np.ndarray):
            raise TypeError("y must be a numpy.ndarray")
        if self.y.ndim != 1:
            raise ValueError("y must be 1D")
        if not self.target_name:
            raise ValueError("target_name must be non-empty")
        if len(self.row_indices) != len(self.y):
            raise ValueError("row_indices length must match y length")
        if bool(pd.isna(self.y).any()):
            raise ValueError("y must not contain NaN values")


class TrainOnlyStandardScaler:
    """Small wrapper that makes train-only standardisation explicit."""

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Return whether the scaler has been fitted."""
        return self._is_fitted

    @property
    def mean_(self) -> np.ndarray:
        """Return a copy of the fitted training means."""
        self._raise_if_unfitted()
        return np.asarray(self._scaler.mean_, dtype=float).copy()

    @property
    def scale_(self) -> np.ndarray:
        """Return a copy of the fitted training scales."""
        self._raise_if_unfitted()
        return np.asarray(self._scaler.scale_, dtype=float).copy()

    @staticmethod
    def _validate_x(x: np.ndarray, *, name: str) -> np.ndarray:
        numeric_x = _as_numeric_2d(x, name=name)
        if _contains_nan(numeric_x):
            raise ValueError(f"{name} must not contain NaN values")
        if bool(np.isinf(numeric_x).any()):
            raise ValueError(f"{name} must not contain infinite values")
        return numeric_x

    def _raise_if_unfitted(self) -> None:
        if not self._is_fitted:
            raise ValueError("TrainOnlyStandardScaler must be fitted before transform")

    def fit(self, x: np.ndarray) -> TrainOnlyStandardScaler:
        """Fit scaling statistics on the training feature matrix only."""
        numeric_x = self._validate_x(x, name="x")
        self._scaler.fit(numeric_x)
        self._is_fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform features using previously fitted training statistics."""
        self._raise_if_unfitted()
        numeric_x = self._validate_x(x, name="x")
        return np.asarray(self._scaler.transform(numeric_x), dtype=float)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fit on training features and return their transformed values."""
        self.fit(x)
        return self.transform(x)


def select_feature_columns(
    frame: pd.DataFrame,
    *,
    exclude_columns: set[str] | None = None,
    reject_label_like: bool = True,
) -> list[str]:
    """Select numeric model feature columns from ``frame``.

    Metadata columns are excluded by default. Obvious label or target
    columns are rejected rather than silently filtered so feature leakage is
    visible during baseline preparation.
    """
    _validate_frame(frame, name="frame")
    excluded = set(_DEFAULT_EXCLUDE_COLUMNS)
    if exclude_columns is not None:
        excluded.update(exclude_columns)

    numeric_columns: list[str] = []
    for column in frame.columns:
        column_name = str(column)
        if column_name in excluded:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            numeric_columns.append(column_name)

    if reject_label_like:
        label_like_columns = [column for column in numeric_columns if _is_label_like(column)]
        if label_like_columns:
            raise ValueError(
                "candidate feature columns look like labels or targets: "
                f"{label_like_columns}"
            )
    return numeric_columns


def _validate_feature_columns(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> list[str]:
    columns = list(feature_columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"feature columns are missing from frame: {missing}")
    label_like = [column for column in columns if _is_label_like(column)]
    if label_like:
        raise ValueError(f"feature columns look like labels or targets: {label_like}")
    for column in columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise TypeError(f"feature column {column!r} is not numeric")
    return columns


def build_feature_matrix(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    row_indices: Sequence[int] | None = None,
    allow_nan: bool = False,
) -> FeatureMatrix:
    """Build a validated feature matrix from selected rows and columns."""
    _validate_frame(frame, name="frame")
    columns = (
        select_feature_columns(frame)
        if feature_columns is None
        else _validate_feature_columns(frame, feature_columns)
    )
    rows = _validate_row_indices(row_indices, n_rows=len(frame))
    values = frame.iloc[rows].loc[:, columns].to_numpy(dtype=float, copy=True)
    if bool(np.isinf(values).any()):
        raise ValueError("feature matrix contains infinite values")
    if _contains_nan(values) and not allow_nan:
        raise ValueError("feature matrix contains NaN values")
    return FeatureMatrix(
        x=values,
        feature_names=columns,
        row_indices=rows,
        allow_nan=allow_nan,
    )


def _to_python_class(value: object) -> str | int | float:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (str, int, float)):
        return value
    return str(value)


def _class_sort_key(value: str | int | float) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def _sorted_classes(values: Sequence[object]) -> list[str | int | float]:
    unique = {_to_python_class(value) for value in values}
    ordered = list(unique)
    try:
        ordered.sort()
    except TypeError:
        ordered.sort(key=_class_sort_key)
    return ordered


def build_target_vector(
    frame: pd.DataFrame,
    *,
    target_column: str,
    row_indices: Sequence[int] | None = None,
) -> TargetVector:
    """Extract a validated 1D target vector from selected rows."""
    _validate_frame(frame, name="frame")
    if target_column not in frame.columns:
        raise ValueError(f"target column is missing from frame: {target_column!r}")

    rows = _validate_row_indices(row_indices, n_rows=len(frame))
    series = frame.iloc[rows].loc[:, target_column]
    if series.isna().any():
        raise ValueError(f"target column {target_column!r} contains NaN values")
    values = series.to_numpy(copy=True)
    return TargetVector(
        y=values,
        target_name=target_column,
        row_indices=rows,
        classes=_sorted_classes(values),
    )


def align_feature_label_frames(
    feature_frame: pd.DataFrame,
    label_frame: pd.DataFrame,
    *,
    on: tuple[str, ...] = ("timestamp", "symbol"),
) -> pd.DataFrame:
    """Inner-join feature and label frames without hiding column overlap."""
    _validate_frame(feature_frame, name="feature_frame")
    _validate_frame(label_frame, name="label_frame")
    if not on:
        raise ValueError("on must contain at least one join key")

    join_keys = list(on)
    missing_feature_keys = [column for column in join_keys if column not in feature_frame.columns]
    missing_label_keys = [column for column in join_keys if column not in label_frame.columns]
    if missing_feature_keys:
        raise ValueError(f"feature_frame is missing join keys: {missing_feature_keys}")
    if missing_label_keys:
        raise ValueError(f"label_frame is missing join keys: {missing_label_keys}")

    overlapping = (set(feature_frame.columns) & set(label_frame.columns)) - set(join_keys)
    if overlapping:
        raise ValueError(
            "feature and label frames have overlapping non-key columns: "
            f"{sorted(overlapping)}"
        )
    if feature_frame.duplicated(join_keys).any():
        raise ValueError("feature_frame contains duplicate join keys")
    if label_frame.duplicated(join_keys).any():
        raise ValueError("label_frame contains duplicate join keys")

    aligned = feature_frame.merge(
        label_frame,
        on=join_keys,
        how="inner",
        sort=False,
        validate="one_to_one",
    )
    return aligned.reset_index(drop=True)

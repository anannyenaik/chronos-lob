"""FI-2010 benchmark loader.

The FI-2010 dataset is a normalised, snapshot-style benchmark matrix
published in support of public limit order book research. Different mirrors
ship slightly different layouts (CSV, plain-text matrices, train/test pre-
split or single combined files), so this module exposes a configurable
loader instead of hard-coding one fragile schema.

This module never downloads data. Users must supply a local file path and
are responsible for the licensing of any benchmark copy they obtain.
Because FI-2010 is typically pre-normalised, the snapshot conversion helper
is intended for structural interface testing rather than for recovering
true price levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from chronoslob.data.schemas import (
    MetadataDict,
    OrderBookLevel,
    OrderBookSnapshot,
)

__all__ = [
    "FI2010Config",
    "FI2010Dataset",
    "build_snapshot_from_row",
    "infer_fi2010_columns",
    "load_fi2010",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class FI2010Config(BaseModel):
    """Configuration for the FI-2010 local-file loader.

    The defaults assume a CSV-like benchmark file with a header row, ten
    book levels per side and a ``bid_price_i`` / ``ask_price_i`` /
    ``bid_quantity_i`` / ``ask_quantity_i`` naming convention. Mirrors that
    use a different layout can override the relevant fields.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=False,
        validate_assignment=True,
        arbitrary_types_allowed=False,
    )

    path: Path = Field(..., description="Local path to the FI-2010-style file.")
    feature_columns: list[str] | None = Field(
        default=None,
        description=(
            "Explicit list of feature columns. When ``None`` the loader infers "
            "features from the remaining numeric columns."
        ),
    )
    label_columns: list[str] | None = Field(
        default=None,
        description="Optional explicit list of label columns.",
    )
    timestamp_column: str | None = Field(
        default=None,
        description=(
            "Optional timestamp column. Synthetic FI-2010 mirrors and local "
            "fixtures may include this; the canonical benchmark does not."
        ),
    )
    split_column: str | None = Field(
        default=None,
        description="Optional column carrying explicit train/test split labels.",
    )
    symbol: str = Field(default="FI2010", min_length=1)
    venue: str | None = None
    delimiter: str = Field(default=",", min_length=1)
    has_header: bool = True
    normalised: bool = True
    price_level_count: int = Field(default=10, gt=0)
    bid_price_prefix: str = Field(default="bid_price_", min_length=1)
    bid_quantity_prefix: str = Field(default="bid_quantity_", min_length=1)
    ask_price_prefix: str = Field(default="ask_price_", min_length=1)
    ask_quantity_prefix: str = Field(default="ask_quantity_", min_length=1)
    allow_missing_lob_columns: bool = True

    @field_validator("path", mode="before")
    @classmethod
    def _validate_path(cls, value: object) -> Path:
        if value is None:
            raise ValueError("path must be provided")
        if isinstance(value, Path):
            if str(value) == "":
                raise ValueError("path must not be empty")
            return value
        if isinstance(value, str):
            if value == "":
                raise ValueError("path must not be empty")
            return Path(value)
        raise TypeError("path must be a string or pathlib.Path")

    @field_validator("symbol", "delimiter")
    @classmethod
    def _validate_non_empty_strings(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value

    @field_validator("venue")
    @classmethod
    def _validate_venue(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("venue must be a non-empty string when provided")
        return value

    @model_validator(mode="after")
    def _validate_distinct_columns(self) -> FI2010Config:
        if self.feature_columns is not None and self.label_columns is not None:
            overlap = set(self.feature_columns) & set(self.label_columns)
            if overlap:
                raise ValueError(
                    "feature_columns and label_columns must be disjoint; "
                    f"shared columns: {sorted(overlap)}"
                )
        if self.timestamp_column is not None and not self.timestamp_column.strip():
            raise ValueError("timestamp_column must be a non-empty string when provided")
        if self.split_column is not None and not self.split_column.strip():
            raise ValueError("split_column must be a non-empty string when provided")
        return self


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------


@dataclass
class FI2010Dataset:
    """In-memory representation of a loaded FI-2010 matrix.

    The container keeps the raw pandas DataFrame, the loader configuration
    and the resolved feature/label column lists. Downstream phases should
    use :meth:`get_features` and :meth:`get_labels` rather than reaching
    into ``frame`` directly so that future column normalisation has a
    single place to evolve.
    """

    frame: pd.DataFrame
    config: FI2010Config
    feature_columns: list[str]
    label_columns: list[str]
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    # -- Sizing ----------------------------------------------------------

    @property
    def n_rows(self) -> int:
        return len(self.frame)

    @property
    def n_features(self) -> int:
        return len(self.feature_columns)

    @property
    def n_labels(self) -> int:
        return len(self.label_columns)

    @property
    def has_labels(self) -> bool:
        return self.n_labels > 0

    # -- Accessors -------------------------------------------------------

    def get_features(self) -> pd.DataFrame:
        """Return a DataFrame view containing only the feature columns."""
        return self.frame.loc[:, self.feature_columns]

    def get_labels(self) -> pd.DataFrame:
        """Return a DataFrame view containing only the label columns."""
        return self.frame.loc[:, self.label_columns]

    def split_values(self) -> pd.Series | None:
        """Return the configured split column as a Series, or ``None``."""
        if self.config.split_column is None:
            return None
        if self.config.split_column not in self.frame.columns:
            return None
        return self.frame[self.config.split_column]

    def describe(self) -> dict[str, str | int | float | bool]:
        """Return a stable metadata dictionary describing the dataset."""
        summary: dict[str, str | int | float | bool] = {
            "symbol": self.config.symbol,
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "n_labels": self.n_labels,
            "has_labels": self.has_labels,
            "normalised": self.config.normalised,
            "has_timestamp_column": self.config.timestamp_column is not None,
            "has_split_column": self.config.split_column is not None,
        }
        if self.config.venue is not None:
            summary["venue"] = self.config.venue
        for key, value in self.metadata.items():
            summary[key] = value
        return summary

    # -- Snapshot conversion --------------------------------------------

    def to_snapshot_rows(
        self,
        max_rows: int | None = None,
    ) -> list[OrderBookSnapshot]:
        """Build :class:`OrderBookSnapshot` rows from the loaded matrix.

        Snapshots are only constructed when the configured bid/ask price
        and quantity columns are available. Because FI-2010 mirrors are
        usually normalised, the returned snapshots are intended for
        structural interface testing rather than for recovering true
        market prices.
        """
        if max_rows is not None and max_rows < 0:
            raise ValueError("max_rows must be non-negative")

        cfg = self.config
        resolved_levels = _resolve_lob_columns(self.frame, cfg)
        if not resolved_levels:
            if cfg.allow_missing_lob_columns:
                return []
            raise ValueError(
                "FI-2010 matrix is missing the configured bid/ask price and "
                "quantity columns; set allow_missing_lob_columns=True to "
                "skip snapshot construction"
            )

        use_explicit_timestamps = (
            cfg.timestamp_column is not None
            and cfg.timestamp_column in self.frame.columns
        )

        if use_explicit_timestamps:
            timestamps = pd.to_datetime(
                self.frame[cfg.timestamp_column],
                utc=True,
                errors="raise",
            )
            synthetic_time = False
        else:
            base = datetime(2000, 1, 1, tzinfo=UTC)
            timestamps = pd.Series(
                [base + timedelta(seconds=idx) for idx in range(self.n_rows)],
                index=self.frame.index,
            )
            synthetic_time = True

        limit = self.n_rows if max_rows is None else min(self.n_rows, max_rows)
        snapshots: list[OrderBookSnapshot] = []
        for position in range(limit):
            row = self.frame.iloc[position]
            ts_raw = timestamps.iloc[position]
            ts = (
                ts_raw.to_pydatetime()
                if hasattr(ts_raw, "to_pydatetime")
                else ts_raw
            )
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            snapshot = _build_snapshot(
                row=row,
                config=cfg,
                resolved_levels=resolved_levels,
                timestamp=ts,
                synthetic_time=synthetic_time,
            )
            snapshots.append(snapshot)
        return snapshots


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def infer_fi2010_columns(
    frame: pd.DataFrame,
    config: FI2010Config,
) -> tuple[list[str], list[str]]:
    """Infer feature and label columns from ``frame`` using ``config``.

    Rules:

    * If both ``feature_columns`` and ``label_columns`` are configured the
      configured lists are returned unchanged.
    * If only ``label_columns`` is configured the labels are returned as
      configured and all remaining numeric columns (excluding split and
      timestamp columns) are treated as features.
    * If only ``feature_columns`` is configured the features are returned
      as configured and labels are empty.
    * If neither is configured all numeric columns excluding split and
      timestamp are treated as features and labels are empty.
    """
    excluded = {
        name
        for name in (config.timestamp_column, config.split_column)
        if name is not None
    }
    numeric_columns = [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]

    if config.feature_columns is not None and config.label_columns is not None:
        return list(config.feature_columns), list(config.label_columns)

    if config.label_columns is not None:
        label_set = set(config.label_columns)
        feature_cols = [c for c in numeric_columns if c not in label_set]
        return feature_cols, list(config.label_columns)

    if config.feature_columns is not None:
        return list(config.feature_columns), []

    return numeric_columns, []


def load_fi2010(config: FI2010Config) -> FI2010Dataset:
    """Load an FI-2010-style local file into a validated dataset.

    The file is read with pandas using the configured delimiter and header
    handling. After reading, feature and label columns are inferred (or
    taken from the config), and :func:`validate_fi2010_dataset` is run.
    Any validation errors raise immediately so malformed files cannot be
    silently consumed by downstream phases.
    """
    path = config.path
    if not path.exists():
        raise FileNotFoundError(f"FI-2010 file not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(
            f"FI-2010 path exists but is not a regular file: {path}"
        )

    read_kwargs: dict[str, object] = {
        "sep": config.delimiter,
        "header": 0 if config.has_header else None,
    }
    frame = pd.read_csv(path, **read_kwargs)
    if not config.has_header:
        frame.columns = [f"col_{idx}" for idx in range(frame.shape[1])]

    _check_columns_exist(frame, config)

    feature_columns, label_columns = infer_fi2010_columns(frame, config)

    metadata: dict[str, str | int | float | bool] = {
        "source_path": str(path),
        "delimiter": config.delimiter,
        "has_header": config.has_header,
        "normalised": config.normalised,
        "synthetic_time": config.timestamp_column is None,
    }

    dataset = FI2010Dataset(
        frame=frame,
        config=config,
        feature_columns=feature_columns,
        label_columns=label_columns,
        metadata=metadata,
    )

    from chronoslob.data.validation import validate_fi2010_dataset

    result = validate_fi2010_dataset(dataset)
    result.raise_if_errors()
    return dataset


def build_snapshot_from_row(
    row: pd.Series,
    config: FI2010Config,
    timestamp: datetime,
) -> OrderBookSnapshot | None:
    """Build a single :class:`OrderBookSnapshot` from one DataFrame row.

    Returns ``None`` if the configured bid/ask price and quantity columns
    are not present in ``row``. The ``timestamp`` argument is used as-is;
    callers are responsible for choosing between an exchange timestamp and
    a synthetic placeholder, and for marking that choice in any downstream
    metadata.
    """
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")

    resolved_levels = _resolve_lob_columns_from_index(row.index, config)
    if not resolved_levels:
        return None

    synthetic_time = bool(
        config.timestamp_column is None
        or config.timestamp_column not in row.index
    )
    return _build_snapshot(
        row=row,
        config=config,
        resolved_levels=resolved_levels,
        timestamp=timestamp,
        synthetic_time=synthetic_time,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_BID_QUANTITY_ALIAS_PREFIX = "bid_size_"
_ASK_QUANTITY_ALIAS_PREFIX = "ask_size_"


def _check_columns_exist(frame: pd.DataFrame, config: FI2010Config) -> None:
    """Raise ``ValueError`` if explicitly configured columns are missing."""
    available = set(frame.columns)
    missing: list[str] = []
    for column in config.feature_columns or []:
        if column not in available:
            missing.append(column)
    for column in config.label_columns or []:
        if column not in available:
            missing.append(column)
    for optional_column in (config.timestamp_column, config.split_column):
        if optional_column is not None and optional_column not in available:
            missing.append(optional_column)
    if missing:
        raise ValueError(
            "FI-2010 file is missing required columns: "
            f"{sorted(set(missing))}"
        )


def _resolve_lob_columns(
    frame: pd.DataFrame,
    config: FI2010Config,
) -> list[tuple[str, str, str, str]]:
    return _resolve_lob_columns_from_index(frame.columns, config)


def _resolve_lob_columns_from_index(
    column_index: pd.Index | list[str],
    config: FI2010Config,
) -> list[tuple[str, str, str, str]]:
    """Return per-level column names ``(bid_p, bid_q, ask_p, ask_q)``.

    Quantity aliases ``bid_size_i`` / ``ask_size_i`` are accepted when the
    default ``bid_quantity_`` / ``ask_quantity_`` prefixes are configured.
    Resolution stops at the first level where any of the four columns is
    missing, so a partial book yields only the levels that are fully
    available.
    """
    available = set(column_index)
    resolved: list[tuple[str, str, str, str]] = []
    for level in range(1, config.price_level_count + 1):
        bp = f"{config.bid_price_prefix}{level}"
        ap = f"{config.ask_price_prefix}{level}"
        bq_primary = f"{config.bid_quantity_prefix}{level}"
        aq_primary = f"{config.ask_quantity_prefix}{level}"

        bq = bq_primary if bq_primary in available else None
        aq = aq_primary if aq_primary in available else None
        if bq is None and config.bid_quantity_prefix == "bid_quantity_":
            alias = f"{_BID_QUANTITY_ALIAS_PREFIX}{level}"
            if alias in available:
                bq = alias
        if aq is None and config.ask_quantity_prefix == "ask_quantity_":
            alias = f"{_ASK_QUANTITY_ALIAS_PREFIX}{level}"
            if alias in available:
                aq = alias

        if bp in available and ap in available and bq is not None and aq is not None:
            resolved.append((bp, bq, ap, aq))
        else:
            break
    return resolved


def _build_snapshot(
    row: pd.Series,
    config: FI2010Config,
    resolved_levels: list[tuple[str, str, str, str]],
    timestamp: datetime,
    synthetic_time: bool,
) -> OrderBookSnapshot:
    bids: list[OrderBookLevel] = []
    asks: list[OrderBookLevel] = []
    for bp, bq, ap, aq in resolved_levels:
        bids.append(
            OrderBookLevel(price=float(row[bp]), quantity=float(row[bq]))
        )
        asks.append(
            OrderBookLevel(price=float(row[ap]), quantity=float(row[aq]))
        )
    bids.sort(key=lambda level: level.price, reverse=True)
    asks.sort(key=lambda level: level.price)

    metadata: MetadataDict = {
        "synthetic_time": synthetic_time,
        "source": "fi2010",
        "normalised": config.normalised,
    }
    return OrderBookSnapshot(
        timestamp=timestamp,
        symbol=config.symbol,
        venue=config.venue,
        bids=bids,
        asks=asks,
        metadata=metadata,
    )

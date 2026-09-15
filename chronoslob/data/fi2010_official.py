"""Adapter for the official FI-2010 ``.txt`` matrix format.

The official FI-2010 release on Fairdata/Etsin ships limit-order-book
snapshots as whitespace-separated text matrices with 149 rows. Each
matrix column is one snapshot. Rows 1-40 carry the order-book prices
and volumes for 10 levels in the layout used by the source paper
(``P^a_t(1) V^a_t(1) P^b_t(1) V^b_t(1) ... P^a_t(10) V^a_t(10)
P^b_t(10) V^b_t(10)``). Rows 41-144 carry the 104 hand-crafted features
described in the source paper, and rows 145-149 carry the categorical
labels for prediction horizons ``k = 10, 20, 30, 50, 100`` with values
``{1, 2, 3}`` (up, stationary, down).

This module provides two narrow helpers:

* :func:`inspect_official_fi2010_file` performs a safe one-pass scan of
  a single local file and returns row count, column count, label class
  distribution, byte size and a SHA-256 checksum, without holding the
  whole matrix in memory.
* :func:`convert_official_fi2010_to_csv` converts a single official
  ``.txt`` file into a header-bearing CSV file matching the column
  convention used by :class:`chronoslob.data.fi2010.FI2010Config`, so
  the existing loader can consume it unchanged.

No network calls are performed. Output paths are caller-supplied and
the module never writes inside ``chronoslob/`` or ``tests/fixtures/``.
"""

from __future__ import annotations

import csv
import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "OFFICIAL_FI2010_HANDCRAFTED_ROW_COUNT",
    "OFFICIAL_FI2010_LABEL_HORIZONS",
    "OFFICIAL_FI2010_LEVEL_COUNT",
    "OFFICIAL_FI2010_LOB_ROW_COUNT",
    "OFFICIAL_FI2010_ROW_COUNT",
    "OfficialFI2010ConversionReport",
    "OfficialFI2010InspectionReport",
    "build_official_fi2010_column_names",
    "convert_official_fi2010_to_csv",
    "inspect_official_fi2010_file",
]

OFFICIAL_FI2010_LEVEL_COUNT = 10
OFFICIAL_FI2010_LOB_ROW_COUNT = 4 * OFFICIAL_FI2010_LEVEL_COUNT  # 40
OFFICIAL_FI2010_HANDCRAFTED_ROW_COUNT = 104
OFFICIAL_FI2010_LABEL_HORIZONS: tuple[int, ...] = (10, 20, 30, 50, 100)
OFFICIAL_FI2010_ROW_COUNT = (
    OFFICIAL_FI2010_LOB_ROW_COUNT
    + OFFICIAL_FI2010_HANDCRAFTED_ROW_COUNT
    + len(OFFICIAL_FI2010_LABEL_HORIZONS)
)  # 149

_VALID_LABEL_VALUES = frozenset({"1", "2", "3"})
_ALLOWED_SPLITS = frozenset({"train", "test"})


def _canonicalise_label_token(token: str) -> str | None:
    """Return the canonical string label for ``token`` or ``None``.

    The official FI-2010 ``.txt`` files store the categorical labels as
    floating-point literals such as ``"1.0000000e+00"``. Internally the
    loader convention uses bare integer strings ``"1"``, ``"2"`` and
    ``"3"``. This helper accepts either form (including extra
    whitespace) and returns the canonical bare integer string, or
    ``None`` when the token does not represent one of the three
    expected classes.
    """
    text = token.strip()
    if not text:
        return None
    if text in _VALID_LABEL_VALUES:
        return text
    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    integer_value = int(value)
    if integer_value != value:
        return None
    if integer_value not in (1, 2, 3):
        return None
    return str(integer_value)


@dataclass(frozen=True)
class OfficialFI2010InspectionReport:
    """Read-only summary returned by :func:`inspect_official_fi2010_file`."""

    path: Path
    byte_size: int
    sha256: str
    row_count: int
    column_count: int
    label_horizons: tuple[int, ...]
    label_class_counts: dict[str, dict[str, int]]
    issues: tuple[str, ...]

    @property
    def is_official_layout(self) -> bool:
        """Return ``True`` when row count matches the official 149-row layout."""
        return self.row_count == OFFICIAL_FI2010_ROW_COUNT


@dataclass(frozen=True)
class OfficialFI2010ConversionReport:
    """Read-only summary returned by :func:`convert_official_fi2010_to_csv`."""

    input_path: Path
    output_path: Path
    n_samples: int
    n_features: int
    n_labels: int
    label_horizons: tuple[int, ...]
    split_label: str | None
    bytes_written: int


def build_official_fi2010_column_names(
    *,
    level_count: int = OFFICIAL_FI2010_LEVEL_COUNT,
    handcrafted_count: int = OFFICIAL_FI2010_HANDCRAFTED_ROW_COUNT,
    label_horizons: Iterable[int] = OFFICIAL_FI2010_LABEL_HORIZONS,
    include_split_column: bool = False,
) -> list[str]:
    """Return CSV column names matching the existing FI-2010 loader.

    The order mirrors the loader's bid/ask price and quantity prefixes
    so a converted file can be consumed by :func:`load_fi2010` without
    any further mapping. Hand-crafted features become ``f_001 ... f_NNN``
    columns and labels become ``label_<horizon>`` columns.
    """
    if level_count <= 0:
        raise ValueError("level_count must be positive")
    if handcrafted_count < 0:
        raise ValueError("handcrafted_count must be non-negative")
    horizons = tuple(int(h) for h in label_horizons)
    if not horizons:
        raise ValueError("label_horizons must contain at least one entry")
    if any(h <= 0 for h in horizons):
        raise ValueError("label_horizons entries must be positive")
    if len(set(horizons)) != len(horizons):
        raise ValueError("label_horizons must not contain duplicates")

    columns: list[str] = []
    if include_split_column:
        columns.append("split")
    for level in range(1, level_count + 1):
        columns.append(f"bid_price_{level}")
        columns.append(f"bid_quantity_{level}")
        columns.append(f"ask_price_{level}")
        columns.append(f"ask_quantity_{level}")
    for index in range(1, handcrafted_count + 1):
        columns.append(f"f_{index:03d}")
    for horizon in horizons:
        columns.append(f"label_{horizon}")
    return columns


def _validate_input_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(
            f"official FI-2010 file not found: {candidate}",
        )
    if not candidate.is_file():
        raise FileNotFoundError(
            f"official FI-2010 path is not a regular file: {candidate}",
        )
    return candidate


def _hash_file(path: Path, *, chunk_bytes: int = 1 << 20) -> tuple[int, str]:
    """Return ``(byte_size, sha256_hex)`` for ``path`` in a streaming pass."""
    hasher = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            total += len(chunk)
            hasher.update(chunk)
    return total, hasher.hexdigest()


def _iter_row_tokens(path: Path) -> Iterable[list[str]]:
    """Yield whitespace-tokenised rows from ``path`` one at a time."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            yield line.split()


def inspect_official_fi2010_file(path: str | Path) -> OfficialFI2010InspectionReport:
    """Inspect a local FI-2010 ``.txt`` matrix safely without loading it.

    The file is scanned line-by-line. The function records the row
    count, the column count of the first row, the byte size, a SHA-256
    checksum and a per-label class distribution computed from the
    expected label rows (counted from the bottom so files with the
    expected 149 rows still report meaningful class counts).
    """
    file_path = _validate_input_path(path)
    byte_size, sha256_hex = _hash_file(file_path)

    row_count = 0
    column_count = 0
    issues: list[str] = []
    column_count_mismatch = False
    label_rows: list[list[str]] = []
    expected_label_rows = len(OFFICIAL_FI2010_LABEL_HORIZONS)

    for index, tokens in enumerate(_iter_row_tokens(file_path)):
        if row_count == 0:
            column_count = len(tokens)
        elif len(tokens) != column_count and not column_count_mismatch:
            issues.append(
                "row column count is not constant: "
                f"row {index + 1} has {len(tokens)} columns, "
                f"row 1 has {column_count} columns",
            )
            column_count_mismatch = True
        row_count += 1
        label_rows.append(tokens)
        if len(label_rows) > expected_label_rows:
            label_rows.pop(0)

    if row_count == 0:
        issues.append("file contains no non-empty rows")
    elif row_count != OFFICIAL_FI2010_ROW_COUNT:
        issues.append(
            "row count does not match the official 149-row FI-2010 layout: "
            f"observed {row_count}",
        )

    label_class_counts: dict[str, dict[str, int]] = {}
    if row_count == OFFICIAL_FI2010_ROW_COUNT and not column_count_mismatch:
        for horizon, tokens in zip(
            OFFICIAL_FI2010_LABEL_HORIZONS, label_rows, strict=True
        ):
            counts: dict[str, int] = {}
            unexpected = 0
            for token in tokens:
                canonical = _canonicalise_label_token(token)
                if canonical is not None:
                    counts[canonical] = counts.get(canonical, 0) + 1
                else:
                    unexpected += 1
            if unexpected > 0:
                issues.append(
                    f"label row for horizon {horizon} contains "
                    f"{unexpected} value(s) outside the expected "
                    "FI-2010 class set {1, 2, 3}",
                )
            label_class_counts[f"label_{horizon}"] = dict(sorted(counts.items()))

    return OfficialFI2010InspectionReport(
        path=file_path,
        byte_size=byte_size,
        sha256=sha256_hex,
        row_count=row_count,
        column_count=column_count,
        label_horizons=OFFICIAL_FI2010_LABEL_HORIZONS,
        label_class_counts=label_class_counts,
        issues=tuple(issues),
    )


def _validate_output_path(path: Path, *, overwrite: bool) -> Path:
    candidate = Path(path)
    if candidate.exists():
        if not overwrite:
            raise FileExistsError(
                f"output path already exists: {candidate}; pass overwrite=True",
            )
        if candidate.is_dir():
            raise IsADirectoryError(
                f"output path is a directory, not a file: {candidate}",
            )
    return candidate


def _validate_split_label(split_label: str | None) -> str | None:
    if split_label is None:
        return None
    if not isinstance(split_label, str):
        raise TypeError("split_label must be a string or None")
    cleaned = split_label.strip().lower()
    if cleaned not in _ALLOWED_SPLITS:
        raise ValueError(
            f"split_label must be one of {sorted(_ALLOWED_SPLITS)} or None; "
            f"got {split_label!r}",
        )
    return cleaned


def _parse_float_token(token: str) -> float:
    try:
        value = float(token)
    except ValueError as exc:
        raise ValueError(f"non-numeric value in feature row: {token!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"non-finite value in feature row: {token!r}")
    return value


def _parse_label_token(token: str) -> str:
    canonical = _canonicalise_label_token(token)
    if canonical is None:
        raise ValueError(
            f"label value outside FI-2010 class set {{1, 2, 3}}: {token!r}",
        )
    return canonical


def convert_official_fi2010_to_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    split_label: str | None = None,
    overwrite: bool = False,
) -> OfficialFI2010ConversionReport:
    """Convert one official FI-2010 ``.txt`` matrix into a header-bearing CSV.

    The CSV uses the column convention expected by
    :class:`chronoslob.data.fi2010.FI2010Config`: ``bid_price_<level>``,
    ``bid_quantity_<level>``, ``ask_price_<level>``, ``ask_quantity_<level>``
    for each of the 10 LOB levels, ``f_001 ... f_104`` for the 104 hand-crafted
    features and ``label_<horizon>`` for each of the five labels at horizons
    10, 20, 30, 50 and 100. When ``split_label`` is supplied (``"train"`` or
    ``"test"``) a ``split`` column is prepended with that value on every row;
    this lets a user concatenate converted ``train`` and ``test`` files into
    a single CSV without losing the canonical FI-2010 split.

    The official 149-row layout is enforced. Any deviation raises
    :class:`ValueError` before the output file is written.
    """
    file_path = _validate_input_path(input_path)
    resolved_output = _validate_output_path(Path(output_path), overwrite=overwrite)
    normalised_split = _validate_split_label(split_label)

    rows = list(_iter_row_tokens(file_path))
    if len(rows) != OFFICIAL_FI2010_ROW_COUNT:
        raise ValueError(
            "official FI-2010 file must have exactly "
            f"{OFFICIAL_FI2010_ROW_COUNT} non-empty rows; observed {len(rows)}",
        )
    column_counts = {len(row) for row in rows}
    if len(column_counts) != 1:
        raise ValueError(
            "official FI-2010 file rows must all have the same number of "
            f"columns; observed column counts {sorted(column_counts)}",
        )
    n_samples = column_counts.pop()
    if n_samples <= 0:
        raise ValueError("official FI-2010 file must contain at least one snapshot")

    lob_rows = rows[: OFFICIAL_FI2010_LOB_ROW_COUNT]
    handcrafted_rows = rows[
        OFFICIAL_FI2010_LOB_ROW_COUNT : OFFICIAL_FI2010_LOB_ROW_COUNT
        + OFFICIAL_FI2010_HANDCRAFTED_ROW_COUNT
    ]
    label_rows = rows[
        OFFICIAL_FI2010_LOB_ROW_COUNT + OFFICIAL_FI2010_HANDCRAFTED_ROW_COUNT :
    ]

    column_names = build_official_fi2010_column_names(
        include_split_column=normalised_split is not None,
    )

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(column_names)
        for sample_index in range(n_samples):
            row: list[str] = []
            if normalised_split is not None:
                row.append(normalised_split)
            for level in range(OFFICIAL_FI2010_LEVEL_COUNT):
                base = 4 * level
                ask_price = _parse_float_token(lob_rows[base][sample_index])
                ask_size = _parse_float_token(lob_rows[base + 1][sample_index])
                bid_price = _parse_float_token(lob_rows[base + 2][sample_index])
                bid_size = _parse_float_token(lob_rows[base + 3][sample_index])
                row.append(repr(bid_price))
                row.append(repr(bid_size))
                row.append(repr(ask_price))
                row.append(repr(ask_size))
            for feature_index in range(OFFICIAL_FI2010_HANDCRAFTED_ROW_COUNT):
                value = _parse_float_token(
                    handcrafted_rows[feature_index][sample_index],
                )
                row.append(repr(value))
            for horizon_index in range(len(OFFICIAL_FI2010_LABEL_HORIZONS)):
                row.append(
                    _parse_label_token(label_rows[horizon_index][sample_index]),
                )
            writer.writerow(row)

    bytes_written = resolved_output.stat().st_size
    return OfficialFI2010ConversionReport(
        input_path=file_path,
        output_path=resolved_output,
        n_samples=n_samples,
        n_features=OFFICIAL_FI2010_LOB_ROW_COUNT + OFFICIAL_FI2010_HANDCRAFTED_ROW_COUNT,
        n_labels=len(OFFICIAL_FI2010_LABEL_HORIZONS),
        label_horizons=OFFICIAL_FI2010_LABEL_HORIZONS,
        split_label=normalised_split,
        bytes_written=bytes_written,
    )


def summarise_inspection_report(
    report: OfficialFI2010InspectionReport,
) -> Mapping[str, object]:
    """Return a JSON-serialisable summary of an inspection report."""
    return {
        "path": str(report.path),
        "byte_size": report.byte_size,
        "sha256": report.sha256,
        "row_count": report.row_count,
        "column_count": report.column_count,
        "label_horizons": list(report.label_horizons),
        "label_class_counts": report.label_class_counts,
        "is_official_layout": report.is_official_layout,
        "issues": list(report.issues),
    }

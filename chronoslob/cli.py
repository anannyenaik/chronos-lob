"""Command-line interface for ChronosLOB."""

from __future__ import annotations

import argparse
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from chronoslob import __version__
from chronoslob.utils.paths import project_root

try:
    import typer
except ModuleNotFoundError:  # pragma: no cover - exercised by smoke command in bare envs
    typer = None  # type: ignore[assignment]

KEY_FOLDERS = (
    "configs",
    "chronoslob",
    "tests",
    "notebooks",
    "reports",
)


def _print(message: Any) -> None:
    try:
        from rich.console import Console
    except ModuleNotFoundError:
        print(message)
        return

    Console().print(message)


def _version_impl() -> None:
    _print(__version__)


def _doctor_rows() -> list[tuple[str, str]]:
    root = project_root()
    rows = [
        ("Python", platform.python_version()),
        ("Package import", f"chronoslob {__version__}"),
        ("Project root", str(root)),
    ]

    for folder in KEY_FOLDERS:
        exists = (root / folder).exists()
        rows.append((f"Folder: {folder}", "present" if exists else "missing"))

    return rows


def _doctor_impl() -> None:
    rows = _doctor_rows()

    try:
        from rich.console import Console
        from rich.table import Table
    except ModuleNotFoundError:
        print("ChronosLOB Doctor")
        for check, value in rows:
            print(f"{check}: {value}")
        return

    table = Table(title="ChronosLOB Doctor", show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Value")
    for check, value in rows:
        table.add_row(check, value)
    Console().print(table)


def _inspect_fi2010_impl(
    path: Path,
    *,
    timestamp_column: str | None = "timestamp",
    split_column: str | None = "split",
    price_level_count: int = 2,
) -> int:
    """Load an FI-2010 file and print a short data-quality summary.

    The function is intentionally read-only: it does not train, transform
    or persist anything. ``timestamp_column`` and ``split_column`` default
    to the names used by the bundled test fixture, but pass ``None`` to
    disable either when running against the canonical FI-2010 matrix.
    """
    from chronoslob.data.fi2010 import FI2010Config, load_fi2010
    from chronoslob.data.validation import validate_fi2010_dataset

    try:
        config = FI2010Config(
            path=path,
            timestamp_column=timestamp_column,
            split_column=split_column,
            price_level_count=price_level_count,
        )
        dataset = load_fi2010(config)
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except (ValueError, TypeError) as exc:
        print(f"Failed to load FI-2010 file: {exc}", file=sys.stderr)
        return 1

    summary = dataset.describe()
    validation = validate_fi2010_dataset(dataset)

    print("ChronosLOB FI-2010 inspection")
    print(f"  path:         {path}")
    print(f"  rows:         {dataset.n_rows}")
    print(f"  features:     {dataset.n_features}")
    print(f"  labels:       {dataset.n_labels}")
    print(f"  has labels:   {dataset.has_labels}")
    print(f"  has split:    {summary['has_split_column']}")
    print(f"  has ts col:   {summary['has_timestamp_column']}")
    print(f"  ok:           {validation.ok}")
    print(f"  errors:       {validation.error_count}")
    print(f"  warnings:     {validation.warning_count}")
    return 0


def _inspect_features_fi2010_impl(
    path: Path,
    *,
    timestamp_column: str | None = "timestamp",
    split_column: str | None = "split",
    label_columns: list[str] | None = None,
    price_level_count: int = 2,
    allow_synthetic_timestamps_for_time_features: bool = False,
) -> int:
    """Load an FI-2010 file, build features and print a short summary.

    The function is read-only and does not write anything. It validates
    the resulting feature frame and prints row count, feature count and
    a small list of feature columns.
    """
    from chronoslob.data.fi2010 import FI2010Config, load_fi2010
    from chronoslob.features.pipeline import (
        FeaturePipelineConfig,
        build_feature_frame_from_fi2010,
        validate_feature_frame,
    )

    resolved_labels = (
        list(label_columns)
        if label_columns is not None
        else ["label_10", "label_50", "label_100"]
    )
    try:
        config = FI2010Config(
            path=path,
            timestamp_column=timestamp_column,
            split_column=split_column,
            label_columns=resolved_labels,
            price_level_count=price_level_count,
        )
        dataset = load_fi2010(config)
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except (ValueError, TypeError) as exc:
        print(f"Failed to load FI-2010 file: {exc}", file=sys.stderr)
        return 1

    pipeline_config = FeaturePipelineConfig(
        allow_synthetic_timestamps_for_time_features=(
            allow_synthetic_timestamps_for_time_features
        )
    )
    try:
        frame = build_feature_frame_from_fi2010(dataset, pipeline_config)
    except (ValueError, TypeError) as exc:
        print(f"Failed to build feature frame: {exc}", file=sys.stderr)
        return 1

    feature_columns = [
        column
        for column in frame.columns
        if column not in {"timestamp", "symbol", "split"}
    ]
    validation = validate_feature_frame(frame)

    print("ChronosLOB FI-2010 feature inspection")
    print(f"  path:                {path}")
    print(f"  rows:                {len(frame)}")
    print(f"  feature columns:     {len(feature_columns)}")
    print(f"  synthetic_time:      {frame.attrs.get('synthetic_time', False)}")
    print(
        f"  skipped time feats:  {frame.attrs.get('skipped_time_features', False)}"
    )
    print(f"  validation ok:       {validation.ok}")
    print(f"  validation errors:   {validation.error_count}")
    print(f"  validation warnings: {validation.warning_count}")
    sample = feature_columns[:10]
    print(f"  sample columns:      {sample}")
    return 0


def _inspect_labels_fi2010_impl(
    path: Path,
    *,
    timestamp_column: str | None = "timestamp",
    split_column: str | None = "split",
    label_columns: list[str] | None = None,
    price_level_count: int = 2,
    prefer_existing_labels: bool = True,
) -> int:
    """Load an FI-2010 file, build or extract labels and print a summary."""
    from chronoslob.data.fi2010 import FI2010Config, load_fi2010
    from chronoslob.labels.pipeline import (
        build_label_frame_from_fi2010,
        validate_label_frame,
    )

    resolved_labels = (
        list(label_columns)
        if label_columns is not None
        else ["label_10", "label_50", "label_100"]
    )
    try:
        config = FI2010Config(
            path=path,
            timestamp_column=timestamp_column,
            split_column=split_column,
            label_columns=resolved_labels,
            price_level_count=price_level_count,
        )
        dataset = load_fi2010(config)
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except (ValueError, TypeError) as exc:
        print(f"Failed to load FI-2010 file: {exc}", file=sys.stderr)
        return 1

    try:
        frame = build_label_frame_from_fi2010(
            dataset,
            prefer_existing_labels=prefer_existing_labels,
        )
    except (ValueError, TypeError) as exc:
        print(f"Failed to build label frame: {exc}", file=sys.stderr)
        return 1

    non_label_columns = {
        "timestamp",
        "symbol",
        "horizon_start",
        "horizon_end",
        "split",
        "label_source",
    }
    label_cols = [column for column in frame.columns if column not in non_label_columns]
    validation = validate_label_frame(frame)

    print("ChronosLOB FI-2010 label inspection")
    print(f"  path:                {path}")
    print(f"  rows:                {len(frame)}")
    print(f"  label columns:       {len(label_cols)}")
    print(f"  validation ok:       {validation.ok}")
    print(f"  validation errors:   {validation.error_count}")
    print(f"  validation warnings: {validation.warning_count}")
    print(f"  sample columns:      {label_cols[:10]}")
    return 0


def _fallback_main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in {"-h", "--help"}:
        print(
            "Usage: python -m chronoslob.cli "
            "[version|doctor|inspect-fi2010|inspect-features-fi2010|"
            "inspect-labels-fi2010] [...]"
        )
        return 0

    command = args[0]
    if command == "version":
        _version_impl()
        return 0
    if command == "doctor":
        _doctor_impl()
        return 0
    if command == "inspect-fi2010":
        parser = argparse.ArgumentParser(
            prog="chronoslob inspect-fi2010",
            description="Load an FI-2010 file and print a data-quality summary.",
        )
        parser.add_argument("--path", type=Path, required=True)
        parser.add_argument("--timestamp-column", default="timestamp")
        parser.add_argument("--split-column", default="split")
        parser.add_argument("--price-level-count", type=int, default=2)
        parser.add_argument(
            "--no-timestamp-column",
            action="store_true",
            help="Treat the file as having no timestamp column.",
        )
        parser.add_argument(
            "--no-split-column",
            action="store_true",
            help="Treat the file as having no split column.",
        )
        parsed = parser.parse_args(args[1:])
        return _inspect_fi2010_impl(
            path=parsed.path,
            timestamp_column=(
                None if parsed.no_timestamp_column else parsed.timestamp_column
            ),
            split_column=(
                None if parsed.no_split_column else parsed.split_column
            ),
            price_level_count=parsed.price_level_count,
        )
    if command == "inspect-features-fi2010":
        parser = argparse.ArgumentParser(
            prog="chronoslob inspect-features-fi2010",
            description=(
                "Load an FI-2010 file, build microstructure features and "
                "print a short summary."
            ),
        )
        parser.add_argument("--path", type=Path, required=True)
        parser.add_argument("--timestamp-column", default="timestamp")
        parser.add_argument("--split-column", default="split")
        parser.add_argument("--price-level-count", type=int, default=2)
        parser.add_argument(
            "--no-timestamp-column",
            action="store_true",
            help="Treat the file as having no timestamp column.",
        )
        parser.add_argument(
            "--no-split-column",
            action="store_true",
            help="Treat the file as having no split column.",
        )
        parser.add_argument(
            "--allow-synthetic-time",
            action="store_true",
            help=(
                "Compute time-window features even when timestamps are "
                "synthetic. Off by default."
            ),
        )
        parsed = parser.parse_args(args[1:])
        return _inspect_features_fi2010_impl(
            path=parsed.path,
            timestamp_column=(
                None if parsed.no_timestamp_column else parsed.timestamp_column
            ),
            split_column=(
                None if parsed.no_split_column else parsed.split_column
            ),
            price_level_count=parsed.price_level_count,
            allow_synthetic_timestamps_for_time_features=(
                parsed.allow_synthetic_time
            ),
        )
    if command == "inspect-labels-fi2010":
        parser = argparse.ArgumentParser(
            prog="chronoslob inspect-labels-fi2010",
            description=(
                "Load an FI-2010 file, build or extract labels and print "
                "a short summary."
            ),
        )
        parser.add_argument("--path", type=Path, required=True)
        parser.add_argument("--timestamp-column", default="timestamp")
        parser.add_argument("--split-column", default="split")
        parser.add_argument("--price-level-count", type=int, default=2)
        parser.add_argument(
            "--no-timestamp-column",
            action="store_true",
            help="Treat the file as having no timestamp column.",
        )
        parser.add_argument(
            "--no-split-column",
            action="store_true",
            help="Treat the file as having no split column.",
        )
        parser.add_argument(
            "--generate-labels",
            action="store_true",
            help=(
                "Generate ChronosLOB labels from snapshots instead of "
                "preferring configured FI-2010 benchmark labels."
            ),
        )
        parsed = parser.parse_args(args[1:])
        return _inspect_labels_fi2010_impl(
            path=parsed.path,
            timestamp_column=(
                None if parsed.no_timestamp_column else parsed.timestamp_column
            ),
            split_column=None if parsed.no_split_column else parsed.split_column,
            price_level_count=parsed.price_level_count,
            prefer_existing_labels=not parsed.generate_labels,
        )

    print(f"Unknown command: {command}", file=sys.stderr)
    return 2


if typer is not None:
    app = typer.Typer(
        add_completion=False,
        help="ChronosLOB research-engineering utilities.",
        no_args_is_help=True,
    )


def version() -> None:
    """Print the installed ChronosLOB version."""
    _version_impl()


def doctor() -> None:
    """Print a lightweight environment and scaffold check."""
    _doctor_impl()


if typer is not None:
    _INSPECT_PATH_OPTION = typer.Option(
        ...,
        "--path",
        help="Path to the local FI-2010-style file.",
    )
    _INSPECT_TIMESTAMP_OPTION = typer.Option(
        "timestamp",
        "--timestamp-column",
        help="Name of the timestamp column, if any.",
    )
    _INSPECT_SPLIT_OPTION = typer.Option(
        "split",
        "--split-column",
        help="Name of the split column, if any.",
    )
    _INSPECT_LEVEL_COUNT_OPTION = typer.Option(
        2,
        "--price-level-count",
        help="Expected number of LOB levels per side.",
    )
    _INSPECT_NO_TIMESTAMP_OPTION = typer.Option(
        False,
        "--no-timestamp-column",
        help="Treat the file as having no timestamp column.",
    )
    _INSPECT_NO_SPLIT_OPTION = typer.Option(
        False,
        "--no-split-column",
        help="Treat the file as having no split column.",
    )

    def inspect_fi2010(
        path: Path = _INSPECT_PATH_OPTION,
        timestamp_column: str = _INSPECT_TIMESTAMP_OPTION,
        split_column: str = _INSPECT_SPLIT_OPTION,
        price_level_count: int = _INSPECT_LEVEL_COUNT_OPTION,
        no_timestamp_column: bool = _INSPECT_NO_TIMESTAMP_OPTION,
        no_split_column: bool = _INSPECT_NO_SPLIT_OPTION,
    ) -> None:
        """Load an FI-2010 file and print a data-quality summary."""
        exit_code = _inspect_fi2010_impl(
            path=path,
            timestamp_column=(
                None if no_timestamp_column else timestamp_column
            ),
            split_column=None if no_split_column else split_column,
            price_level_count=price_level_count,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    _INSPECT_FEAT_ALLOW_SYNTHETIC_OPTION = typer.Option(
        False,
        "--allow-synthetic-time",
        help=(
            "Compute time-window features even when timestamps are "
            "synthetic. Off by default."
        ),
    )
    _INSPECT_LABELS_GENERATE_OPTION = typer.Option(
        False,
        "--generate-labels",
        help=(
            "Generate ChronosLOB labels from snapshots instead of "
            "preferring configured FI-2010 benchmark labels."
        ),
    )

    def inspect_features_fi2010(
        path: Path = _INSPECT_PATH_OPTION,
        timestamp_column: str = _INSPECT_TIMESTAMP_OPTION,
        split_column: str = _INSPECT_SPLIT_OPTION,
        price_level_count: int = _INSPECT_LEVEL_COUNT_OPTION,
        no_timestamp_column: bool = _INSPECT_NO_TIMESTAMP_OPTION,
        no_split_column: bool = _INSPECT_NO_SPLIT_OPTION,
        allow_synthetic_time: bool = _INSPECT_FEAT_ALLOW_SYNTHETIC_OPTION,
    ) -> None:
        """Build microstructure features from an FI-2010 file and summarise."""
        exit_code = _inspect_features_fi2010_impl(
            path=path,
            timestamp_column=(
                None if no_timestamp_column else timestamp_column
            ),
            split_column=None if no_split_column else split_column,
            price_level_count=price_level_count,
            allow_synthetic_timestamps_for_time_features=allow_synthetic_time,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_labels_fi2010(
        path: Path = _INSPECT_PATH_OPTION,
        timestamp_column: str = _INSPECT_TIMESTAMP_OPTION,
        split_column: str = _INSPECT_SPLIT_OPTION,
        price_level_count: int = _INSPECT_LEVEL_COUNT_OPTION,
        no_timestamp_column: bool = _INSPECT_NO_TIMESTAMP_OPTION,
        no_split_column: bool = _INSPECT_NO_SPLIT_OPTION,
        generate_labels: bool = _INSPECT_LABELS_GENERATE_OPTION,
    ) -> None:
        """Build or extract FI-2010 labels and summarise."""
        exit_code = _inspect_labels_fi2010_impl(
            path=path,
            timestamp_column=(
                None if no_timestamp_column else timestamp_column
            ),
            split_column=None if no_split_column else split_column,
            price_level_count=price_level_count,
            prefer_existing_labels=not generate_labels,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

else:

    def inspect_fi2010(
        path: Path,
        timestamp_column: str = "timestamp",
        split_column: str = "split",
        price_level_count: int = 2,
        no_timestamp_column: bool = False,
        no_split_column: bool = False,
    ) -> None:
        """Load an FI-2010 file and print a data-quality summary."""
        exit_code = _inspect_fi2010_impl(
            path=path,
            timestamp_column=None if no_timestamp_column else timestamp_column,
            split_column=None if no_split_column else split_column,
            price_level_count=price_level_count,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_features_fi2010(
        path: Path,
        timestamp_column: str = "timestamp",
        split_column: str = "split",
        price_level_count: int = 2,
        no_timestamp_column: bool = False,
        no_split_column: bool = False,
        allow_synthetic_time: bool = False,
    ) -> None:
        """Build microstructure features from an FI-2010 file and summarise."""
        exit_code = _inspect_features_fi2010_impl(
            path=path,
            timestamp_column=None if no_timestamp_column else timestamp_column,
            split_column=None if no_split_column else split_column,
            price_level_count=price_level_count,
            allow_synthetic_timestamps_for_time_features=allow_synthetic_time,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_labels_fi2010(
        path: Path,
        timestamp_column: str = "timestamp",
        split_column: str = "split",
        price_level_count: int = 2,
        no_timestamp_column: bool = False,
        no_split_column: bool = False,
        generate_labels: bool = False,
    ) -> None:
        """Build or extract FI-2010 labels and summarise."""
        exit_code = _inspect_labels_fi2010_impl(
            path=path,
            timestamp_column=None if no_timestamp_column else timestamp_column,
            split_column=None if no_split_column else split_column,
            price_level_count=price_level_count,
            prefer_existing_labels=not generate_labels,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)


if typer is not None:
    app.command()(version)
    app.command()(doctor)
    app.command("inspect-fi2010")(inspect_fi2010)
    app.command("inspect-features-fi2010")(inspect_features_fi2010)
    app.command("inspect-labels-fi2010")(inspect_labels_fi2010)
else:

    def app() -> int:
        """Fallback command runner used when Typer is not installed yet."""
        return _fallback_main()


if __name__ == "__main__":
    if typer is None:
        raise SystemExit(_fallback_main())
    app()

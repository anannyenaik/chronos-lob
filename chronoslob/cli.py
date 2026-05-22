"""Command-line interface for ChronosLOB."""

from __future__ import annotations

import platform
import sys
from collections.abc import Sequence
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


def _fallback_main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in {"-h", "--help"}:
        print("Usage: python -m chronoslob.cli [version|doctor]")
        return 0

    command = args[0]
    if command == "version":
        _version_impl()
        return 0
    if command == "doctor":
        _doctor_impl()
        return 0

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
    app.command()(version)
    app.command()(doctor)
else:

    def app() -> int:
        """Fallback command runner used when Typer is not installed yet."""
        return _fallback_main()


if __name__ == "__main__":
    if typer is None:
        raise SystemExit(_fallback_main())
    app()

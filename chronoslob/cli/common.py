"""Shared command-line helpers."""

from __future__ import annotations

import platform
from typing import Any

from chronoslob import __version__
from chronoslob.utils.paths import project_root

KEY_FOLDERS = ("configs", "chronoslob", "experiments", "paper", "tests")


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


def _parse_fold_selection(value: str | None) -> list[int] | None:
    if value is None:
        return None
    text = value.strip()
    if not text or text.lower() == "all":
        return None
    folds: list[int] = []
    for token in text.split(","):
        cleaned = token.strip()
        if not cleaned:
            continue
        try:
            fold = int(cleaned)
        except ValueError as exc:
            raise ValueError(
                f"--folds must be 'all' or a comma-separated integer list; got {value!r}",
            ) from exc
        if fold <= 0:
            raise ValueError(f"--folds entries must be positive; got {fold}")
        if fold not in folds:
            folds.append(fold)
    if not folds:
        raise ValueError("--folds must contain at least one positive integer")
    return folds


def _parse_model_selection(value: str | None) -> list[str] | None:
    if value is None:
        return None
    text = value.strip()
    if not text or text.lower() == "all":
        return None
    models = [token.strip() for token in text.split(",") if token.strip()]
    if not models:
        raise ValueError("--models must contain at least one model name")
    return models


def _parse_neural_fold_selection(value: str | None) -> list[str] | None:
    if value is None:
        return None
    text = value.strip()
    if not text or text.lower() == "all":
        return None
    folds: list[str] = []
    for token in text.split(","):
        cleaned = token.strip().lower()
        if not cleaned:
            continue
        if cleaned.isdigit():
            cleaned = f"fold_{int(cleaned)}"
        if not cleaned.startswith("fold_") or not cleaned.removeprefix("fold_").isdigit():
            raise ValueError(
                f"--folds must be 'all' or a comma-separated list like fold_1,2; got {value!r}",
            )
        if int(cleaned.removeprefix("fold_")) <= 0:
            raise ValueError("--folds entries must be positive")
        if cleaned not in folds:
            folds.append(cleaned)
    if not folds:
        raise ValueError("--folds must contain at least one fold")
    return folds


def _parse_int_selection(
    value: str | None,
    *,
    option_name: str,
    positive: bool,
) -> list[int] | None:
    if value is None:
        return None
    text = value.strip()
    if not text or text.lower() == "all":
        return None
    values: list[int] = []
    for token in text.split(","):
        cleaned = token.strip()
        if not cleaned:
            continue
        try:
            number = int(cleaned)
        except ValueError as exc:
            raise ValueError(f"{option_name} entries must be integers") from exc
        if positive and number <= 0:
            raise ValueError(f"{option_name} entries must be positive")
        if not positive and number < 0:
            raise ValueError(f"{option_name} entries must be non-negative")
        if number not in values:
            values.append(number)
    if not values:
        raise ValueError(f"{option_name} must contain at least one value")
    return values

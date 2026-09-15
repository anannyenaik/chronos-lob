"""Helpers for lightweight experiment run artefacts."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = ["create_run_directory", "safe_run_name", "write_json"]

_RUN_SUBDIRECTORIES = ("configs", "artifacts", "logs", "tables")


def safe_run_name(name: str) -> str:
    """Return a readable filesystem-safe run name."""
    if not isinstance(name, str):
        raise TypeError("run name must be a string")
    stripped = name.strip().lower()
    if not stripped:
        raise ValueError("run name must be non-empty")

    cleaned = re.sub(r"[^a-z0-9._-]+", "-", stripped)
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-._")
    if not cleaned:
        raise ValueError("run name contains no filesystem-safe characters")
    return cleaned


def _unique_run_path(root: Path, base_name: str) -> Path:
    candidate = root / base_name
    if not candidate.exists():
        return candidate
    suffix = 1
    while True:
        candidate = root / f"{base_name}-{suffix:03d}"
        if not candidate.exists():
            return candidate
        suffix += 1


def create_run_directory(
    root: str | Path,
    run_name: str,
    *,
    exist_ok: bool = False,
) -> Path:
    """Create a run directory with standard subdirectories."""
    root_path = Path(root).expanduser()
    root_path.mkdir(parents=True, exist_ok=True)

    base_name = safe_run_name(run_name)
    run_path = root_path / base_name
    if run_path.exists() and not exist_ok:
        run_path = _unique_run_path(root_path, base_name)
    if run_path.exists() and not run_path.is_dir():
        raise FileExistsError(f"run path exists and is not a directory: {run_path}")

    run_path.mkdir(parents=True, exist_ok=exist_ok)
    for subdirectory in _RUN_SUBDIRECTORIES:
        (run_path / subdirectory).mkdir(parents=True, exist_ok=True)
    return run_path.resolve()


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write stable JSON and raise clearly for non-serialisable payloads."""
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    except TypeError as exc:
        raise TypeError(f"payload cannot be serialised to JSON: {exc}") from exc
    destination.write_text(text, encoding="utf-8")
    return destination.resolve()

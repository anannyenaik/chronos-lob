"""Tests for path helpers."""

from __future__ import annotations

from pathlib import Path

from chronoslob.utils.paths import ensure_dir, project_root, resolve_path


def test_project_root_returns_path() -> None:
    root = project_root()

    assert isinstance(root, Path)
    assert (root / "pyproject.toml").is_file()


def test_ensure_dir_creates_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "directory"

    resolved = ensure_dir(target)

    assert resolved == target.resolve()
    assert resolved.is_dir()


def test_resolve_path_returns_path() -> None:
    resolved = resolve_path("configs")

    assert isinstance(resolved, Path)
    assert resolved.is_absolute()

"""Path helpers for repository-local workflows."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the ChronosLOB project root."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return current.parents[2]


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root when it is not absolute."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = project_root() / candidate
    return candidate.resolve()


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return its resolved path."""
    resolved = resolve_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved

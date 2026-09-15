"""Configuration loading helpers for experiment infrastructure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from chronoslob.utils.paths import project_root

__all__ = ["load_yaml_config", "resolve_config_path"]


def resolve_config_path(path: str | Path) -> Path:
    """Resolve a config path relative to the project root when needed."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = project_root() / candidate
    return candidate.resolve(strict=False)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return a mapping."""
    resolved = resolve_config_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if payload is None:
        raise ValueError(f"YAML config is empty: {resolved}")
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must be a mapping: {resolved}")
    if not all(isinstance(key, str) for key in payload):
        raise ValueError(f"YAML config keys must be strings: {resolved}")
    return dict(payload)

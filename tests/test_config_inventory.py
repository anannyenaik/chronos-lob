"""Repository config inventory tests."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from chronoslob.utils.audit import collect_config_files
from chronoslob.utils.paths import project_root

SECRET_FIELD_PATTERN = re.compile(
    r"\b(api[_-]?key|secret|password|access[_-]?token)\s*:",
    flags=re.IGNORECASE,
)


def _load_yaml(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"{path} must parse to a mapping"
    return payload


def _iter_mapping_values(value: Any) -> Iterator[Any]:
    if isinstance(value, dict):
        for nested in value.values():
            yield from _iter_mapping_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_mapping_values(nested)
    else:
        yield value


def _iter_path_values(value: Any, key: str = "") -> Iterator[str]:
    if isinstance(value, dict):
        for nested_key, nested_value in value.items():
            if isinstance(nested_key, str):
                yield from _iter_path_values(nested_value, nested_key)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_path_values(nested, key)
    elif isinstance(value, str) and key.endswith("path"):
        yield value


def test_configs_directory_exists_and_has_yaml_files() -> None:
    root = project_root()

    assert (root / "configs").is_dir()
    assert collect_config_files(root)


def test_all_yaml_configs_parse_successfully() -> None:
    for path in collect_config_files(project_root()):
        _load_yaml(path)


def test_synthetic_smoke_configs_are_labelled() -> None:
    root = project_root()
    smoke_configs = sorted((root / "configs" / "experiments").glob("*smoke*.yaml"))

    assert smoke_configs
    for path in smoke_configs:
        text = path.read_text(encoding="utf-8").lower()
        assert "synthetic" in text, f"{path} must label smoke scope as synthetic"


def test_configs_do_not_contain_obvious_secret_fields() -> None:
    for path in collect_config_files(project_root()):
        text = path.read_text(encoding="utf-8")
        assert SECRET_FIELD_PATTERN.search(text) is None, (
            f"{path} appears to define a secret-like field"
        )


def test_configs_do_not_require_network_data_by_default() -> None:
    for path in collect_config_files(project_root()):
        payload = _load_yaml(path)
        values = [str(value).lower() for value in _iter_mapping_values(payload)]

        assert not any("http://" in value or "https://" in value for value in values)
        assert not any("ws://" in value or "wss://" in value for value in values)


def test_referenced_fixture_paths_exist_or_are_documented_examples() -> None:
    root = project_root()

    for path in collect_config_files(root):
        payload = _load_yaml(path)
        text = path.read_text(encoding="utf-8").lower()
        for raw_value in _iter_path_values(payload):
            if not ("/" in raw_value or "\\" in raw_value):
                continue
            candidate = root / raw_value
            if raw_value.startswith("tests/fixtures"):
                assert candidate.exists(), f"{path} references missing fixture {raw_value}"
            elif not candidate.exists():
                assert "documentation example" in text or "does not exist by default" in text

"""Tests for experiment config and registry helpers."""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path

import pytest

from chronoslob.training.artifacts import (
    create_run_directory,
    safe_run_name,
    write_json,
)
from chronoslob.training.config import load_yaml_config
from chronoslob.training.experiment import (
    create_experiment_metadata,
    get_git_commit,
    initialise_experiment_run,
)


def test_load_yaml_config_works_on_example_config() -> None:
    config = load_yaml_config("configs/experiments/fi2010_split_audit.yaml")

    assert config["run"]["phase"] == "phase-5"
    assert "model" not in config


def test_load_yaml_config_missing_path_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml_config("configs/experiments/missing.yaml")


def test_safe_run_name_handles_spaces_and_unsafe_characters() -> None:
    assert safe_run_name(" Split Audit / Phase 5! ") == "split-audit-phase-5"


def test_create_run_directory_creates_expected_subfolders(tmp_path: Path) -> None:
    run_path = create_run_directory(tmp_path, "Split Audit")

    assert run_path.is_dir()
    for subdirectory in ("configs", "artifacts", "logs", "tables"):
        assert (run_path / subdirectory).is_dir()


def test_write_json_writes_stable_json(tmp_path: Path) -> None:
    output = write_json(tmp_path / "metadata.json", {"b": 1, "a": 2})

    assert output.read_text(encoding="utf-8") == '{\n  "a": 2,\n  "b": 1\n}\n'
    assert json.loads(output.read_text(encoding="utf-8")) == {"a": 2, "b": 1}


def test_create_experiment_metadata_creates_timezone_aware_metadata() -> None:
    metadata = create_experiment_metadata(
        run_name="split-audit",
        phase="phase-5",
        seed=42,
        input_paths=["features.parquet"],
    )

    assert metadata.run_id.startswith("split-audit-")
    assert metadata.created_at.tzinfo is not None
    assert metadata.created_at.utcoffset() == UTC.utcoffset(metadata.created_at)
    assert metadata.seed == 42
    assert metadata.input_paths == ["features.parquet"]


def test_get_git_commit_returns_str_or_none_without_raising() -> None:
    commit = get_git_commit()

    assert commit is None or isinstance(commit, str)


def test_initialise_experiment_run_writes_metadata_json(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run:\n  name: split-audit\n", encoding="utf-8")

    metadata, run_path = initialise_experiment_run(
        root=tmp_path / "runs",
        run_name="split-audit",
        phase="phase-5",
        seed=42,
        config_path=config_path,
        input_paths=[tmp_path / "input.csv"],
        notes="metadata-only test",
    )

    metadata_path = run_path / "metadata.json"
    assert metadata_path.is_file()
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == metadata.run_id
    assert payload["phase"] == "phase-5"
    assert payload["seed"] == 42
    assert (run_path / "configs" / "config.yaml").is_file()
    assert not (run_path / "metrics.json").exists()
    assert not (run_path / "results.csv").exists()

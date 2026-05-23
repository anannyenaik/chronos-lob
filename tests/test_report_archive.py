"""Tests for technical evidence archive utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from chronoslob.utils.paths import project_root
from chronoslob.utils.report_archive import (
    EXPECTED_ARCHIVE_FILES,
    CommandSpec,
    ReportArchiveConfig,
    build_report_archive,
    capture_command,
    collect_config_inventory,
    collect_limitations_index,
    collect_module_inventory,
    collect_project_inventory,
    collect_release_history,
)


def _static_success_spec() -> CommandSpec:
    return CommandSpec(
        display_command="python -c \"print('archive-ok')\"",
        args=(sys.executable, "-c", "print('archive-ok')"),
        description="Static archive test command.",
        synthetic=True,
    )


def _phrase(*parts: str) -> str:
    return " ".join(parts)


def test_archive_config_defaults() -> None:
    config = ReportArchiveConfig()

    assert config.output_path == Path("reports/report_archive")
    assert not config.strict
    assert not config.include_smoke_training
    assert config.command_specs is None


def test_project_inventory_collection_contains_counts_and_cli() -> None:
    section = collect_project_inventory(ReportArchiveConfig(root=project_root()))

    assert "Package version" in section.content
    assert "python -m pytest" in section.content
    assert "`doctor`" in section.content
    assert "`run-project-audit`" in section.content


def test_release_history_contains_expected_milestones() -> None:
    section = collect_release_history()

    for phrase in (
        "Foundation",
        "DeepLOB-style path",
        "Self-supervision",
        "Audit and CI",
        "Evidence archive",
    ):
        assert phrase in section.content


def test_config_inventory_marks_synthetic_smoke_configs() -> None:
    section = collect_config_inventory(ReportArchiveConfig(root=project_root()))

    assert "configs/experiments/report_archive_smoke.yaml" in section.content
    assert "Evidence-archive build configuration" in section.content
    assert "Uses synthetic fixture: `yes`" in section.content


def test_module_inventory_includes_key_packages() -> None:
    section = collect_module_inventory(ReportArchiveConfig(root=project_root()))

    for module_name in (
        "chronoslob.data.fi2010",
        "chronoslob.book.reconstruction",
        "chronoslob.features.pipeline",
        "chronoslob.labels.leakage",
        "chronoslob.training.splitters",
        "chronoslob.models.transformer",
        "chronoslob.backtest.validation",
        "chronoslob.analysis.summary",
        "chronoslob.utils.audit",
    ):
        assert module_name in section.content


def test_limitations_index_references_required_caveats() -> None:
    section = collect_limitations_index()
    lowered = section.content.lower()

    for phrase in (
        "public data",
        "synthetic fixtures",
        "crypto",
        "simplified research simulation",
        "market impact model",
    ):
        assert phrase in lowered


def test_command_capture_handles_success() -> None:
    capture = capture_command(
        _static_success_spec(),
        root=project_root(),
        timeout_seconds=10,
    )

    assert capture.ok
    assert capture.exit_code == 0
    assert capture.stdout == "archive-ok"
    assert capture.synthetic


def test_command_capture_handles_optional_failure() -> None:
    capture = capture_command(
        CommandSpec(
            display_command="python -c failure",
            args=(sys.executable, "-c", "import sys; sys.exit(7)"),
            description="Expected optional failure.",
            optional=True,
        ),
        root=project_root(),
        timeout_seconds=10,
    )

    assert not capture.ok
    assert capture.exit_code == 7
    assert capture.optional


def test_build_report_archive_writes_expected_files(tmp_path: Path) -> None:
    result = build_report_archive(
        ReportArchiveConfig(
            root=project_root(),
            output_path=tmp_path / "archive",
            command_specs=(_static_success_spec(),),
        )
    )

    for relative_path in EXPECTED_ARCHIVE_FILES:
        assert (result.output_path / relative_path).is_file()
    assert len(result.files_written) == len(EXPECTED_ARCHIVE_FILES)
    assert result.commands_captured == 1


def test_mermaid_diagram_files_are_generated(tmp_path: Path) -> None:
    result = build_report_archive(
        ReportArchiveConfig(
            root=project_root(),
            output_path=tmp_path / "archive",
            command_specs=(_static_success_spec(),),
        )
    )

    for name in (
        "architecture_overview.mmd",
        "data_pipeline.mmd",
        "model_stack.mmd",
        "evaluation_stack.mmd",
    ):
        text = (result.output_path / "figures" / name).read_text(encoding="utf-8")
        assert text.startswith("flowchart")


def test_generated_archive_readme_describes_archive(tmp_path: Path) -> None:
    result = build_report_archive(
        ReportArchiveConfig(
            root=project_root(),
            output_path=tmp_path / "archive",
            command_specs=(_static_success_spec(),),
        )
    )

    readme = (result.output_path / "README.md").read_text(encoding="utf-8")
    outputs = (result.output_path / "cli_outputs.md").read_text(encoding="utf-8")

    assert "Technical Evidence Archive" in readme
    assert "synthetic fixtures" in readme.lower()
    assert "synthetic fixture" in outputs.lower()


def test_strict_mode_raises_for_failed_command(tmp_path: Path) -> None:
    config = ReportArchiveConfig(
        root=project_root(),
        output_path=tmp_path / "archive",
        strict=True,
        command_specs=(
            CommandSpec(
                display_command="python -c failure",
                args=(sys.executable, "-c", "import sys; sys.exit(5)"),
                description="Expected strict failure.",
                optional=True,
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="strict mode failed"):
        build_report_archive(config)


def test_generated_archive_avoids_unsupported_benchmark_claims(
    tmp_path: Path,
) -> None:
    result = build_report_archive(
        ReportArchiveConfig(
            root=project_root(),
            output_path=tmp_path / "archive",
            command_specs=(_static_success_spec(),),
        )
    )
    text = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in sorted(result.output_path.rglob("*"))
        if path.is_file()
    )

    for phrase in (
        _phrase("profitable", "strategy"),
        _phrase("beats", "the", "market"),
        _phrase("high", "sharpe"),
        "fake result table",
        _phrase("production", "trading", "system"),
    ):
        assert phrase not in text


def test_repeated_builds_are_deterministic_on_same_temp_root(
    tmp_path: Path,
) -> None:
    config = ReportArchiveConfig(
        root=project_root(),
        output_path=tmp_path / "archive",
        command_specs=(_static_success_spec(),),
    )

    first = build_report_archive(config)
    first_files = {
        path.relative_to(first.output_path).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(first.output_path.rglob("*"))
        if path.is_file()
    }
    second = build_report_archive(config)
    second_files = {
        path.relative_to(second.output_path).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(second.output_path.rglob("*"))
        if path.is_file()
    }

    assert first_files == second_files

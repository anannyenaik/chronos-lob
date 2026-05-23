"""Tests for local project audit utilities."""

from __future__ import annotations

from pathlib import Path

from chronoslob.utils.audit import (
    AuditIssue,
    AuditResult,
    AuditStatus,
    check_no_forbidden_claims,
    check_no_large_generated_files,
    check_required_paths,
    check_synthetic_fixture_labelling,
    run_project_audit,
)
from chronoslob.utils.paths import project_root


def test_required_paths_check_reports_missing_paths(tmp_path: Path) -> None:
    (tmp_path / "present.txt").write_text("ok", encoding="utf-8")

    result = check_required_paths(
        tmp_path,
        required_paths=("present.txt", "missing.txt"),
    )

    assert result.status == AuditStatus.FAIL
    assert result.failure_count == 1
    assert result.issues[0].path == Path("missing.txt")


def test_forbidden_claim_detection_flags_unsupported_phrase(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_text(
        "This system offers guaranteed profit.",
        encoding="utf-8",
    )

    result = check_no_forbidden_claims(tmp_path)

    assert result.status == AuditStatus.FAIL
    assert result.issues[0].matched_text == "guaranteed profit"


def test_forbidden_claim_detection_allows_avoid_context(
    tmp_path: Path,
) -> None:
    (tmp_path / "AGENTS.md").write_text(
        "Vocabulary to avoid:\n- guaranteed profit\n",
        encoding="utf-8",
    )

    result = check_no_forbidden_claims(tmp_path)

    assert result.status == AuditStatus.PASS


def test_synthetic_fixture_labelling_warns_for_unlabelled_smoke(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "configs" / "experiments"
    config_root.mkdir(parents=True)
    (config_root / "example_smoke.yaml").write_text(
        "experiment_type: example_smoke\n",
        encoding="utf-8",
    )

    result = check_synthetic_fixture_labelling(tmp_path)

    assert result.status == AuditStatus.WARNING
    assert result.warning_count == 1


def test_no_large_generated_files_check_uses_threshold(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_text("x" * 20, encoding="utf-8")

    result = check_no_large_generated_files(tmp_path, threshold_bytes=10)

    assert result.status == AuditStatus.FAIL
    assert result.failure_count == 1


def test_audit_result_severity_helpers() -> None:
    issue = AuditIssue(
        check_name="example",
        status=AuditStatus.WARNING,
        message="Example warning.",
    )
    result = AuditResult(
        name="example",
        status=AuditStatus.WARNING,
        issues=(issue,),
    )

    assert result.issue_count == 1
    assert result.warning_count == 1
    assert result.failure_count == 0
    assert not result.ok
    assert "warning" in issue.format()


def test_project_audit_smoke_on_repo_root() -> None:
    audit = run_project_audit(project_root())

    assert audit.status == AuditStatus.PASS
    assert audit.inventory.config_count > 0
    assert audit.inventory.report_count > 0
    assert audit.inventory.test_count > 0
    assert "run-project-audit" in audit.inventory.cli_commands

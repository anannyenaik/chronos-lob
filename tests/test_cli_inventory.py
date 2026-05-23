"""Tests for CLI inventory and audit command plumbing."""

from __future__ import annotations

import inspect

from chronoslob.cli import _run_project_audit_impl
from chronoslob.utils import audit as audit_module
from chronoslob.utils.audit import collect_cli_commands, run_project_audit
from chronoslob.utils.paths import project_root


def test_cli_command_collection_includes_known_commands() -> None:
    commands = collect_cli_commands(project_root())

    for command in (
        "version",
        "doctor",
        "inspect-fi2010",
        "inspect-event-log",
        "run-calibration-smoke",
        "run-execution-validation-smoke",
        "run-robustness-analysis-smoke",
        "run-project-audit",
    ):
        assert command in commands


def test_known_cli_commands_are_documented() -> None:
    root = project_root()
    docs_text = (root / "docs" / "CLI_REFERENCE.md").read_text(encoding="utf-8")

    for command in collect_cli_commands(root):
        assert f"`{command}" in docs_text


def test_run_project_audit_returns_structured_result() -> None:
    result = run_project_audit(project_root())

    assert result.ok
    assert result.inventory.cli_command_count >= 1
    assert result.issue_count == 0


def test_cli_project_audit_impl_returns_success_without_writing() -> None:
    assert _run_project_audit_impl(root=project_root(), strict=True) == 0


def test_audit_inventory_does_not_use_network_or_subprocess_calls() -> None:
    source = inspect.getsource(audit_module)

    for forbidden in ("requests", "httpx", "urllib", "socket", "subprocess"):
        assert forbidden not in source

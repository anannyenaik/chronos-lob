"""Repository report inventory and limitation tests."""

from __future__ import annotations

from chronoslob.utils.audit import (
    AuditStatus,
    check_no_forbidden_claims,
    collect_report_files,
)
from chronoslob.utils.paths import project_root

EXPECTED_REPORTS = (
    "reports/limitations.md",
    "reports/full_audit_ci_hardening.md",
    "reports/execution_aware_validation.md",
    "reports/transfer_regime_ablation_analysis.md",
    "reports/calibration_uncertainty.md",
)


def test_expected_reports_exist() -> None:
    root = project_root()

    for relative_path in EXPECTED_REPORTS:
        assert (root / relative_path).is_file(), f"Missing report {relative_path}"


def test_limitations_report_contains_required_caveats() -> None:
    text = (project_root() / "reports" / "limitations.md").read_text(
        encoding="utf-8",
    )
    lowered = text.lower()

    for phrase in (
        "public data",
        "synthetic fixture",
        "binance",
        "crypto",
        "simplified execution validation",
        "not a live trading system",
        "market impact model",
    ):
        assert phrase in lowered


def test_reports_do_not_make_forbidden_claims() -> None:
    root = project_root()
    report_paths = [path.relative_to(root) for path in collect_report_files(root)]

    result = check_no_forbidden_claims(root, scan_paths=report_paths)

    assert result.status == AuditStatus.PASS


def test_smoke_reports_and_implementation_reports_are_discoverable() -> None:
    root = project_root()
    report_names = {path.name for path in collect_report_files(root)}

    assert "execution_aware_validation.md" in report_names
    assert "transfer_regime_ablation_analysis.md" in report_names
    assert "full_audit_ci_hardening.md" in report_names
    assert len(report_names) >= 10

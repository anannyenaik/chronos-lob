"""Tests for the 10/10 research protocol layer."""

from __future__ import annotations

from pathlib import Path

import yaml  # type: ignore[import-untyped]

from chronoslob.utils.audit import (
    AuditStatus,
    check_no_forbidden_claims,
)
from chronoslob.utils.paths import project_root

PUBLIC_PROTOCOL_PATH = Path("docs") / "RESEARCH_PROTOCOL.md"
MAINTAINER_PROTOCOL_PATH = Path("reports") / "10_10_research_protocol.md"
MULTIFOLD_CONFIG_PATH = Path("configs") / "experiments" / "fi2010_multifold.yaml"

REQUIRED_TOP_LEVEL_KEYS = (
    "study_name",
    "dataset_name",
    "local_data_root_path",
    "folds",
    "official_split",
    "target",
    "models",
    "seeds",
    "metrics",
    "calibration",
    "execution_sensitivity",
    "ablations",
    "artefacts",
    "data_handling",
    "claim_boundaries",
)

PUBLIC_PROTOCOL_REQUIRED_SECTIONS = (
    "## 1. Research Question",
    "## 2. Scope",
    "## 3. Dataset Protocol",
    "## 4. FI-2010 Fold Plan",
    "## 5. Official Split Semantics",
    "## 6. Train, Validation and Test Rules",
    "## 7. Allowed Model Families",
    "## 8. Baseline Requirements",
    "## 9. Neural Benchmark Requirements",
    "## 10. Self-Supervised Claim Gate",
    "## 11. Metrics",
    "## 12. Calibration Diagnostics",
    "## 13. Execution-Aware Proxy Diagnostics",
    "## 14. Statistical Uncertainty",
    "## 15. Ablation Requirements",
    "## 16. External Benchmark Comparison",
    "## 17. Artefact Traceability",
    "## 18. Public Claim Boundaries",
    "## 19. What This Study Can Prove",
    "## 20. What This Study Cannot Prove",
)


def test_public_research_protocol_file_exists() -> None:
    assert (project_root() / PUBLIC_PROTOCOL_PATH).is_file()


def test_maintainer_research_protocol_file_exists() -> None:
    assert (project_root() / MAINTAINER_PROTOCOL_PATH).is_file()


def test_multifold_config_file_exists_and_parses() -> None:
    payload = yaml.safe_load(
        (project_root() / MULTIFOLD_CONFIG_PATH).read_text(encoding="utf-8"),
    )

    assert isinstance(payload, dict)


def test_multifold_config_contains_required_top_level_keys() -> None:
    payload = yaml.safe_load(
        (project_root() / MULTIFOLD_CONFIG_PATH).read_text(encoding="utf-8"),
    )

    assert isinstance(payload, dict)
    for key in REQUIRED_TOP_LEVEL_KEYS:
        assert key in payload, f"Multi-fold config missing required key: {key}"


def test_multifold_config_records_protocol_assumptions() -> None:
    payload = yaml.safe_load(
        (project_root() / MULTIFOLD_CONFIG_PATH).read_text(encoding="utf-8"),
    )

    assert payload["executable"] is True
    assert payload["folds"] and all(isinstance(fold, int) for fold in payload["folds"])
    assert payload["seeds"] and len(payload["seeds"]) >= 3
    assert payload["classical_runner"]["models"]
    assert payload["classical_runner"]["seeds"] == [0]
    assert payload["aggregation"]["metrics"]

    data_handling = payload["data_handling"]
    assert data_handling["download_fi2010"] is False
    assert data_handling["commit_raw_data"] is False
    assert data_handling["commit_processed_csv"] is False
    assert data_handling["commit_predictions"] is False
    assert data_handling["commit_intermediate_matrices"] is False

    boundaries = payload["claim_boundaries"]
    assert boundaries["no_profitable_strategy_claim"] is True
    assert boundaries["no_market_beating_claim"] is True
    assert boundaries["no_state_of_the_art_claim"] is True
    assert boundaries["no_foundation_model_claim"] is True
    assert boundaries["no_ssl_result_claim_until_gate_satisfied"] is True
    assert boundaries["no_live_tradability_claim"] is True


def test_public_research_protocol_has_required_sections() -> None:
    text = (project_root() / PUBLIC_PROTOCOL_PATH).read_text(encoding="utf-8")

    for heading in PUBLIC_PROTOCOL_REQUIRED_SECTIONS:
        assert heading in text, f"Public protocol missing section: {heading}"


def test_protocol_files_do_not_make_forbidden_claims() -> None:
    result = check_no_forbidden_claims(
        project_root(),
        scan_paths=(PUBLIC_PROTOCOL_PATH, MAINTAINER_PROTOCOL_PATH),
    )

    assert result.status == AuditStatus.PASS

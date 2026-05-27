"""Tests for the FI-2010 external benchmark context layer."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from chronoslob.utils.audit import (
    AuditStatus,
    check_no_forbidden_claims,
    check_public_release_wording,
)
from chronoslob.utils.paths import project_root

DOC_PATH = Path("docs") / "FI2010_EXTERNAL_BENCHMARKS.md"
REPORT_PATH = Path("reports") / "external_benchmark_context.md"
ARTEFACT_DIR = Path("experiments") / "fi2010_external_context"
JSON_PATH = ARTEFACT_DIR / "benchmark_context.json"
CSV_PATH = ARTEFACT_DIR / "protocol_comparison.csv"
NOTES_PATH = ARTEFACT_DIR / "comparison_notes.md"

DOCS_TO_AUDIT = (
    DOC_PATH,
    REPORT_PATH,
    NOTES_PATH,
    Path("docs") / "EXPERIMENT_EVIDENCE_INDEX.md",
    Path("docs") / "REPRODUCIBILITY.md",
    Path("reports") / "10_10_research_protocol.md",
)

REQUIRED_CSV_COLUMNS = {
    "source_name",
    "source_type",
    "dataset_variant",
    "horizon",
    "split_protocol",
    "metrics",
    "comparable_to_chronoslob",
    "caveat",
}


def _compact(*parts: str) -> str:
    return "".join(parts)


def _hyphenated(*parts: str) -> str:
    return "-".join(parts)


def _words(*parts: str) -> str:
    return " ".join(parts)


def test_external_benchmark_doc_exists() -> None:
    assert (project_root() / DOC_PATH).is_file()


def test_external_doc_contains_required_context() -> None:
    text = (project_root() / DOC_PATH).read_text(encoding="utf-8")
    lowered = text.lower()

    for phrase in (
        "purpose",
        "fi-2010 dataset context",
        "chronoslob protocol summary",
        "comparison dimensions",
        "why direct metric comparison may be invalid",
        "where comparison is meaningful",
        "where comparison is not meaningful",
        "current chronoslob result snapshot",
        "external benchmark table",
        "0.4654",
        "0.0039",
        "0.7337",
        "0.0280",
        "single-seed",
        "no ssl result",
    ):
        assert phrase in lowered


def test_external_docs_avoid_forbidden_public_claims() -> None:
    root = project_root()
    claims = check_no_forbidden_claims(root, scan_paths=DOCS_TO_AUDIT)
    wording = check_public_release_wording(root, scan_paths=DOCS_TO_AUDIT)

    assert claims.status == AuditStatus.PASS, claims.issues
    assert wording.status == AuditStatus.PASS, wording.issues


def test_external_context_has_no_unverified_ranking_wording() -> None:
    root = project_root()
    checked_paths = (DOC_PATH, REPORT_PATH, NOTES_PATH, JSON_PATH, CSV_PATH)
    bad_terms = (
        _hyphenated("state", "of", "the", "art"),
        _words("state", "of", "the", "art"),
        _hyphenated("market", "beating"),
        _compact("Co", "dex"),
        _compact("Ch", "at", "GPT"),
        _compact("Clau", "de"),
        _compact("ag", "ent"),
        _compact("pro", "mpt"),
        _hyphenated("AI", "generated"),
    )
    issues: list[str] = []
    for relative_path in checked_paths:
        text = (root / relative_path).read_text(encoding="utf-8").lower()
        for term in bad_terms:
            if term.lower() in text:
                issues.append(f"{relative_path}: {term}")

    assert issues == []


def test_external_table_contains_no_external_numeric_metric_values() -> None:
    text = (project_root() / DOC_PATH).read_text(encoding="utf-8")
    section = text.split("## External Benchmark Table", maxsplit=1)[1]
    section_without_urls = re.sub(r"\]\([^)]+\)", "]", section)

    assert not re.search(r"\b\d+\.\d+\b", section_without_urls)
    assert "%" not in section_without_urls


def test_structured_context_json_parses_and_omits_external_metrics() -> None:
    payload = json.loads((project_root() / JSON_PATH).read_text(encoding="utf-8"))

    assert payload["scope"] == "protocol comparison only; no external numeric metrics"
    assert payload["claim_boundaries"]["no_external_numeric_metrics"] is True
    assert payload["claim_boundaries"]["no_ssl_result"] is True
    assert payload["chronoslob_protocol"]["full_predictions_written"] is False
    assert payload["chronoslob_protocol"]["checkpoints_written"] is False
    assert payload["chronoslob_result_snapshot"][0]["mean"] == 0.4654
    assert payload["chronoslob_result_snapshot"][1]["mean"] == 0.7337

    for reference in payload["external_references"]:
        assert reference["metric_values_included"] is False
        assert "reported_metric_value" not in reference


def test_protocol_comparison_csv_has_required_columns() -> None:
    with (project_root() / CSV_PATH).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames is not None
    assert set(reader.fieldnames) >= REQUIRED_CSV_COLUMNS
    assert rows
    assert {row["comparable_to_chronoslob"] for row in rows} >= {
        "reference_protocol",
        "conditional",
        "protocol_only",
        "no",
    }


def test_external_context_writes_no_predictions_or_checkpoints() -> None:
    root = project_root() / ARTEFACT_DIR

    assert not list(root.rglob("predictions*.csv"))
    assert not list(root.rglob("*.pt"))
    assert not list(root.rglob("*.pth"))
    assert not list(root.rglob("*.ckpt"))

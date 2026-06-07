"""Cross-document consistency checks for the public evidence release.

These tests fail when the README, the final empirical report and the evidence
pack disagree about the neural / SSL evidence state. They build the final report
from the current stored artefacts so they always reflect the live generator
behaviour rather than a possibly stale committed file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chronoslob.experiments.final_report import build_final_empirical_report
from chronoslob.utils.paths import project_root

# Phrases that may only appear next to a caveat/negation marker.
_PROXY_CAVEAT_MARKERS = (
    "proxy",
    "not ",
    "no ",
    "does not",
    "do not",
    "without",
    "offline",
)


def _experiments_dir(name: str) -> Path:
    return project_root() / "experiments" / name


def _have_release_artefacts() -> bool:
    required = (
        "fi2010_multifold_classical",
        "fi2010_multifold_neural",
        "fi2010_uncertainty",
        "fi2010_neural_full_grid",
        "fi2010_neural_proper_training_subset_v2",
    )
    return all((_experiments_dir(name)).is_dir() for name in required)


@pytest.fixture(scope="module")
def generated_report_text(tmp_path_factory: pytest.TempPathFactory) -> str:
    if not _have_release_artefacts():
        pytest.skip("release artefacts are not present in this checkout")
    out = tmp_path_factory.mktemp("consistency") / "report.md"
    build_final_empirical_report(
        classical_dir=_experiments_dir("fi2010_multifold_classical"),
        neural_dir=_experiments_dir("fi2010_multifold_neural"),
        uncertainty_dir=_experiments_dir("fi2010_uncertainty"),
        ablation_dir=_experiments_dir("fi2010_brutal_ablations"),
        feature_ablation_dir=_experiments_dir("fi2010_feature_ablations"),
        execution_dir=_experiments_dir("fi2010_execution_v2"),
        execution_v3_dir=_experiments_dir("fi2010_execution_v3"),
        external_dir=_experiments_dir("fi2010_external_context"),
        neural_full_grid_dir=_experiments_dir("fi2010_neural_full_grid"),
        proper_training_dir=_experiments_dir("fi2010_neural_proper_training_subset_v2"),
        ssl_v2_analysis_dir=project_root() / "reports" / "ssl_v2_analysis",
        evidence_pack_dir=project_root() / "reports" / "evidence_pack",
        out_path=out,
        overwrite=True,
    )
    return out.read_text(encoding="utf-8")


def _readme_text() -> str:
    return (project_root() / "README.md").read_text(encoding="utf-8")


def _snapshot_block(report_text: str) -> str:
    """Return the text of the Evidence Snapshot section only."""
    start = report_text.index("### Stored Evidence Snapshot")
    end = report_text.index("### ", start + 1)
    return report_text[start:end]


def test_readme_marks_neural_full_grid_complete(generated_report_text: str) -> None:
    readme = _readme_text()
    # Precondition: README claims the neural full grid is complete_real.
    assert "Neural full grid" in readme
    assert "`complete_real`" in readme

    # The report's top-level snapshot must then describe the completed
    # matched full grid, not only reduced-scope neural evidence.
    snapshot = _snapshot_block(generated_report_text)
    assert "completed one-epoch matched comparison grid" in snapshot
    # The old conflating descriptor must not be the only neural statement.
    assert "neural_scope | reduced-scope supervised neural, single-seed" not in snapshot


def _normalised(text: str) -> str:
    """Collapse whitespace so wrapped prose phrases match as single strings."""
    return " ".join(text.lower().split())


def test_report_does_not_deny_ssl_while_matched_comparison_present(
    generated_report_text: str,
) -> None:
    # The matched SSL comparison artefacts are loaded (full grid present).
    assert "Matched SSL deltas:" in generated_report_text
    # So the report must not claim there is no SSL result at all.
    assert "No SSL result is claimed in this report." not in generated_report_text
    # It must explicitly state SSL did not improve the matched grid.
    normalised = _normalised(generated_report_text)
    assert (
        "no ssl improvement" in normalised
        or "no overall ssl improvement is supported" in normalised
    )


def test_report_separates_legacy_benchmark_from_matched_grid(
    generated_report_text: str,
) -> None:
    normalised = _normalised(generated_report_text)
    # Explicit one-epoch comparison-grid statement.
    assert "one-epoch comparison grid" in normalised
    assert "not a performance-maximising neural training result" in normalised
    # Explicit separation of the earlier reduced-scope benchmark.
    assert "reported separately and is not used as matched ssl evidence" in normalised


def test_ssl_v2_scope_matches_stored_analysis(generated_report_text: str) -> None:
    analysis_dir = project_root() / "reports" / "ssl_v2_analysis"
    summary = json.loads((analysis_dir / "summary.json").read_text(encoding="utf-8"))
    claims = json.loads(
        (analysis_dir / "ssl_v2_claim_assessment.json").read_text(encoding="utf-8")
    )
    normalised = _normalised(generated_report_text)

    assert "seeds 0, 1, 2" in normalised
    assert "seeds 1 and 2 are deferred" not in normalised
    assert "durham university hamilton" in normalised

    calibration = next(
        item for item in claims["claims"] if item["claim_id"] == "ssl_v2_calibration_improvement"
    )
    if calibration["status"] == "supported":
        assert "ssl-v2 calibration improvement is supported" in normalised
        assert "ssl-v2 calibration improvement remains unsupported" not in normalised
    assert summary["seeds"] == [0, 1, 2]


def test_readme_and_report_scope_proper_training_subset(
    generated_report_text: str,
) -> None:
    readme = _readme_text()
    assert "Proper-training neural subset" in readme
    assert "`partial_real`" in readme

    normalised = _normalised(generated_report_text)
    assert "proper_training_neural_scope" in generated_report_text
    assert "partial_real; folds 1, horizons 10, 50, seeds 0, lookbacks 50" in normalised
    assert "validation-only early stopping with best checkpoint restored before test" in normalised
    assert "the longer-training subset does not support an ssl improvement claim" in normalised


def test_execution_v3_is_described_as_proxy_not_pnl(
    generated_report_text: str,
) -> None:
    lines = generated_report_text.splitlines()
    offenders: list[str] = []
    affirmative_terms = ("pnl", "profit", "live trading", "backtest")
    for index, line in enumerate(lines):
        lowered = line.lower()
        if "execution-v3" not in lowered and "execution_v3" not in lowered:
            continue
        for term in affirmative_terms:
            if term not in lowered:
                continue
            window = " ".join(lines[max(0, index - 2) : index + 2]).lower()
            if not any(marker in window for marker in _PROXY_CAVEAT_MARKERS):
                offenders.append(f"{index + 1}: {term} in {line.strip()}")
    assert offenders == [], offenders
    # Positive caveat must be present.
    assert "offline execution-aware proxy diagnostic" in generated_report_text


def test_snapshot_order_flow_proxy_not_described_as_true_ofi(
    generated_report_text: str,
) -> None:
    if "snapshot_order_flow_proxy" not in generated_report_text:
        pytest.skip("feature ablation section not present")
    lowered = generated_report_text.lower()
    # The proxy disclaimer must be present alongside the proxy group.
    assert "not true event-level order-flow imbalance" in lowered
    # And it must never be promoted to true OFI.
    assert "true ofi on fi-2010" not in lowered

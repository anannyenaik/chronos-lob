"""Public release-readiness tests."""

from __future__ import annotations

import re
from pathlib import Path

from chronoslob.utils.paths import project_root

TEXT_SUFFIXES = {".md", ".mmd", ".py", ".toml", ".yaml", ".yml"}
ROOT_TEXT_FILES = {
    ".gitattributes",
    ".gitignore",
    "CONTRIBUTING.md",
    "LICENSE",
    "Makefile",
    "README.md",
    "ROADMAP.md",
    "pyproject.toml",
}
EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "data",
    "runs",
}
LOCAL_LINK_PATTERN = re.compile(
    r"\[[^\]]+\]\((?!https?:|mailto:|#)(?P<target>[^)#]+)(?:#[^)]+)?\)"
)
MAX_NON_CODE_LINE_LENGTH = 220


def _words(*parts: str) -> str:
    return " ".join(parts)


def _compact(*parts: str) -> str:
    return "".join(parts)


def _hyphenated(*parts: str) -> str:
    return "-".join(parts)


def _legacy_files() -> tuple[str, ...]:
    return (_compact("AG", "ENTS.md"), "PLANS.md")


def _workflow_terms() -> tuple[str, ...]:
    return (
        _compact("Co", "dex"),
        _compact("Ch", "at", "GPT"),
        _words(_compact("Mas", "ter"), _compact("Pro", "mpt")),
        _compact("pro", "mpt"),
        _compact("ch", "at"),
        _compact("ag", "ent"),
        _compact("ag", "ents"),
        _compact("AG", "ENTS"),
        _words(_compact("auto", "mated"), "contributor"),
        _hyphenated("AI", "generated"),
        _words("AI", "generated"),
        _hyphenated("AI", "coded"),
        _words("AI", "coded"),
    )


def _employment_terms() -> tuple[str, ...]:
    return (
        _hyphenated(_compact("recruit", "er"), "facing"),
        _compact("recruit", "er"),
        _compact("C", "V"),
        _compact("res", "ume"),
        _compact("car", "eer"),
        _compact("hir", "ing"),
    )


def _unsupported_claim_terms() -> tuple[str, ...]:
    return (
        _words("alpha", "validation"),
        _words("deployable", "alpha"),
        _words("AI", "trading", "bot"),
        _words("guaranteed", "alpha"),
        _words("beats", "the", "market"),
        _words("trading", "bot"),
        _words("high", "Sharpe"),
        _words("profitable", "strategy"),
        _words("deployable", "trading"),
        _words("production", "trading"),
    )


def _process_doc_terms() -> tuple[str, ...]:
    return (
        "plumbing",
        _words("fake", "metrics"),
        _words("placeholder", "metrics"),
        _words("placeholder", "results"),
        "fabricated",
        _words(_compact("mas", "ter"), _compact("pro", "mpt")),
    )


def _flag_tokens() -> tuple[str, ...]:
    """CLI flag spellings that collide with a forbidden prose term."""
    return ("--no-" + _compact("res", "ume"), "--" + _compact("res", "ume"))


def _iter_public_text_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or relative.as_posix() in ROOT_TEXT_FILES:
            files.append(path)
    return sorted(files)


def _iter_public_markdown_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.md"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        if "tests" in relative.parts:
            continue
        if relative.parts[:2] == ("reports", "report_archive"):
            continue
        files.append(path)
    return sorted(files)


def test_internal_maintenance_files_are_not_published() -> None:
    root = project_root()

    for filename in _legacy_files():
        assert not (root / filename).exists()
    assert (root / "CONTRIBUTING.md").is_file()
    assert (root / "ROADMAP.md").is_file()


def test_public_text_has_no_internal_workflow_or_positioning_terms() -> None:
    root = project_root()
    terms = _workflow_terms() + _employment_terms() + _unsupported_claim_terms()
    issues: list[str] = []

    for path in _iter_public_text_files(root):
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for token in _flag_tokens():
            text = text.replace(token, "")
        for term in terms:
            if term.lower() in text:
                issues.append(f"{path.relative_to(root)}: {term}")

    assert issues == []


def test_public_markdown_has_no_process_artifact_terms() -> None:
    root = project_root()
    issues: list[str] = []

    for path in _iter_public_text_files(root):
        if path.suffix.lower() not in {".md", ".mmd"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for term in _process_doc_terms():
            if term.lower() in text:
                issues.append(f"{path.relative_to(root)}: {term}")

    assert issues == []


def test_readme_is_polished_and_links_are_valid() -> None:
    root = project_root()
    readme_path = root / "README.md"
    text = readme_path.read_text(encoding="utf-8")
    lowered = text.lower()
    normalised = " ".join(lowered.split())

    assert text.startswith("# ChronosLOB")
    assert "research platform for limit order book" in normalised
    assert "not financial advice" in normalised
    assert "not live trading infrastructure" in normalised
    assert _words("alpha", "validation") not in normalised

    for match in LOCAL_LINK_PATTERN.finditer(text):
        target = match.group("target").strip()
        assert (root / target).exists(), f"README link target missing: {target}"


def test_roadmap_and_contributing_are_human_facing() -> None:
    root = project_root()
    roadmap = (root / "ROADMAP.md").read_text(encoding="utf-8")
    contributing = (root / "CONTRIBUTING.md").read_text(encoding="utf-8")

    assert "## Completed" in roadmap
    assert "## In Progress and Next" in roadmap
    assert "## Out of Scope" in roadmap
    assert "Validation Commands" in contributing
    assert "Pull Request Checklist" in contributing


def test_public_safety_and_limitations_remains_canonical() -> None:
    root = project_root()
    safety = (root / "docs" / "SAFETY_AND_LIMITATIONS.md").read_text(
        encoding="utf-8",
    ).lower()
    limitations = (root / "reports" / "limitations.md").read_text(
        encoding="utf-8",
    ).lower()
    archive_readme = (
        root / "reports" / "report_archive" / "README.md"
    ).read_text(encoding="utf-8").lower()

    assert "limitations" in safety
    assert "market impact" in safety
    assert "market impact" in limitations
    assert "evidence archive" in archive_readme


def test_execution_proxy_validity_boundary_is_linked_and_conservative() -> None:
    root = project_root()
    validity_path = root / "docs" / "EXECUTION_PROXY_VALIDITY.md"
    validity = validity_path.read_text(encoding="utf-8").lower()

    for phrase in (
        "confidence thresholds",
        "active fraction",
        "turnover proxy",
        "cost sensitivity",
        "latency sensitivity",
        "adverse-selection proxy",
        "normalised order-book snapshots",
        "order-add, cancel, modify and trade events",
        "deliberately not a trading claim",
    ):
        assert phrase in validity
    for forbidden_claim in (
        "pnl or realised profitability",
        "tradability",
        "live execution quality",
        "production execution simulation",
        "venue-specific queue priority",
        "market impact",
        "true fill modelling",
    ):
        assert forbidden_claim in validity

    linked_files = (
        root / "README.md",
        root / "docs" / "SAFETY_AND_LIMITATIONS.md",
        root / "docs" / "PROJECT_STATUS.md",
        root / "docs" / "FINAL_EMPIRICAL_REPORT.md",
        root / "reports" / "chronoslob_final_empirical_report.md",
    )
    for path in linked_files:
        assert "EXECUTION_PROXY_VALIDITY.md" in path.read_text(encoding="utf-8")


def test_public_markdown_files_do_not_collapse_into_single_long_lines() -> None:
    root = project_root()
    offenders: list[str] = []
    for path in _iter_public_markdown_files(root):
        text = path.read_text(encoding="utf-8")
        in_code_block = False
        for line_index, line in enumerate(text.splitlines(), start=1):
            if line.lstrip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue
            stripped = line.strip()
            if not stripped or stripped.startswith("|"):
                continue
            if stripped.startswith("http://") or stripped.startswith("https://"):
                continue
            if len(line) > MAX_NON_CODE_LINE_LENGTH:
                offenders.append(f"{path.relative_to(root)}:{line_index}: {len(line)} chars")

    assert offenders == []

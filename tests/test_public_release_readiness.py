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
        for term in terms:
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
    assert "research platform for leakage-safe limit order book" in normalised
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

    assert "Completed Components" in roadmap
    assert "Future Work" in roadmap
    assert "Validation Commands" in contributing
    assert "Pull Request Checklist" in contributing


def test_docs_and_archive_preserve_synthetic_and_limitation_caveats() -> None:
    root = project_root()
    readme = (root / "README.md").read_text(encoding="utf-8").lower()
    limitations = (root / "reports" / "limitations.md").read_text(
        encoding="utf-8"
    ).lower()
    archive_readme = (
        root / "reports" / "report_archive" / "README.md"
    ).read_text(encoding="utf-8").lower()
    safety = (root / "docs" / "SAFETY_AND_LIMITATIONS.md").read_text(
        encoding="utf-8"
    ).lower()

    assert "synthetic fixture outputs are plumbing checks only" in readme
    assert "synthetic" in archive_readme
    assert "not market evidence" in archive_readme
    assert "limitations" in safety
    assert "market impact" in limitations

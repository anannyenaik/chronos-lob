"""Local repository audit helpers for ChronosLOB.

The checks in this module are deliberately lightweight and local-only. They
inspect files under the repository root, but they do not call external services,
invoke git, download data or mutate the working tree.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from chronoslob.utils.paths import project_root

__all__ = [
    "DEFAULT_FORBIDDEN_CLAIM_PHRASES",
    "DEFAULT_LARGE_FILE_THRESHOLD_BYTES",
    "DEFAULT_PUBLIC_RELEASE_REQUIRED_FILES",
    "DEFAULT_REQUIRED_PATHS",
    "AuditIssue",
    "AuditResult",
    "AuditStatus",
    "PathInventory",
    "ProjectAuditResult",
    "check_no_forbidden_claims",
    "check_no_large_generated_files",
    "check_public_release_readme",
    "check_public_release_structure",
    "check_public_release_wording",
    "check_required_paths",
    "check_synthetic_fixture_labelling",
    "collect_cli_commands",
    "collect_config_files",
    "collect_report_files",
    "collect_test_files",
    "run_project_audit",
    "run_public_release_audit",
]


class AuditStatus(StrEnum):
    """Severity/status values used by project audit checks."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass(frozen=True)
class AuditIssue:
    """One issue found by a local audit check."""

    check_name: str
    status: AuditStatus
    message: str
    path: Path | None = None
    line_number: int | None = None
    matched_text: str | None = None

    def format(self) -> str:
        """Return a stable, human-readable issue string."""
        location = ""
        if self.path is not None:
            location = str(self.path)
            if self.line_number is not None:
                location = f"{location}:{self.line_number}"
            location = f"{location}: "
        matched = f" [{self.matched_text}]" if self.matched_text else ""
        return f"{self.status.value}: {location}{self.message}{matched}"


@dataclass(frozen=True)
class AuditResult:
    """Structured result for a single audit check."""

    name: str
    status: AuditStatus
    issues: tuple[AuditIssue, ...] = ()
    details: Mapping[str, int | str] = field(default_factory=dict)

    @property
    def issue_count(self) -> int:
        """Return the number of issues recorded by the check."""
        return len(self.issues)

    @property
    def warning_count(self) -> int:
        """Return the number of warning issues recorded by the check."""
        return sum(1 for issue in self.issues if issue.status == AuditStatus.WARNING)

    @property
    def failure_count(self) -> int:
        """Return the number of failing issues recorded by the check."""
        return sum(1 for issue in self.issues if issue.status == AuditStatus.FAIL)

    @property
    def ok(self) -> bool:
        """Return true when the check has no warnings or failures."""
        return self.status == AuditStatus.PASS


@dataclass(frozen=True)
class PathInventory:
    """Inventory of repository paths and CLI commands."""

    root: Path
    config_files: tuple[Path, ...]
    report_files: tuple[Path, ...]
    test_files: tuple[Path, ...]
    cli_commands: tuple[str, ...]

    @property
    def config_count(self) -> int:
        return len(self.config_files)

    @property
    def report_count(self) -> int:
        return len(self.report_files)

    @property
    def test_count(self) -> int:
        return len(self.test_files)

    @property
    def cli_command_count(self) -> int:
        return len(self.cli_commands)


@dataclass(frozen=True)
class ProjectAuditResult:
    """Combined result for a local ChronosLOB project audit."""

    root: Path
    inventory: PathInventory
    results: tuple[AuditResult, ...]

    @property
    def status(self) -> AuditStatus:
        if any(result.status == AuditStatus.FAIL for result in self.results):
            return AuditStatus.FAIL
        if any(result.status == AuditStatus.WARNING for result in self.results):
            return AuditStatus.WARNING
        return AuditStatus.PASS

    @property
    def issue_count(self) -> int:
        return sum(result.issue_count for result in self.results)

    @property
    def warning_count(self) -> int:
        return sum(result.warning_count for result in self.results)

    @property
    def failure_count(self) -> int:
        return sum(result.failure_count for result in self.results)

    @property
    def ok(self) -> bool:
        return self.status == AuditStatus.PASS


def _words(*parts: str) -> str:
    return " ".join(parts)


def _compact(*parts: str) -> str:
    return "".join(parts)


def _hyphenated(*parts: str) -> str:
    return "-".join(parts)


DEFAULT_FORBIDDEN_CLAIM_PHRASES: tuple[str, ...] = (
    _words("guaranteed", "alpha"),
    _words("guaranteed", "profit"),
    _words("beats", "the", "market"),
    _words("deployable", "trading", "strategy"),
    _words("deployable", "trading", "system"),
    _words("production", "trading", "system"),
    _words("live", "trading", "bot"),
    _words("trading", "bot"),
    _hyphenated("risk", "free"),
    _words("high", "Sharpe"),
    _words("profitable", "strategy"),
    _words("proven", "profitable"),
)

DEFAULT_REQUIRED_PATHS: tuple[str, ...] = (
    ".github/workflows/ci.yml",
    "CONTRIBUTING.md",
    "ROADMAP.md",
    "README.md",
    "pyproject.toml",
    "chronoslob",
    "configs",
    "docs/CLI_REFERENCE.md",
    "docs/PROJECT_STATUS.md",
    "docs/REPRODUCIBILITY.md",
    "docs/SAFETY_AND_LIMITATIONS.md",
    "reports/limitations.md",
    "reports/full_audit_ci_hardening.md",
    "reports/public_release_readiness.md",
    "tests",
)

DEFAULT_PUBLIC_RELEASE_REQUIRED_FILES: tuple[str, ...] = (
    "README.md",
    "CONTRIBUTING.md",
    "ROADMAP.md",
    "LICENSE",
    "pyproject.toml",
    ".github/workflows/ci.yml",
    "docs/CLI_REFERENCE.md",
    "docs/PROJECT_STATUS.md",
    "docs/REPRODUCIBILITY.md",
    "docs/SAFETY_AND_LIMITATIONS.md",
    "docs/REPORT_EVIDENCE_INDEX.md",
    "docs/GITHUB_POLISH_CHECKLIST.md",
    "reports/README.md",
    "reports/limitations.md",
    "reports/public_release_readiness.md",
    "reports/report_archive/README.md",
    "reports/technical_report/README.md",
)

DEFAULT_LARGE_FILE_THRESHOLD_BYTES = 1_000_000

_TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
_PUBLIC_RELEASE_TEXT_SUFFIXES = {
    ".md",
    ".mmd",
    ".py",
    ".toml",
    ".yaml",
    ".yml",
}
_AUDIT_SCAN_DIRS = (
    ".github",
    "chronoslob",
    "configs",
    "docs",
    "notebooks",
    "reports",
    "tests",
)
_ROOT_SCAN_FILES = (
    ".gitattributes",
    ".gitignore",
    "CONTRIBUTING.md",
    "LICENSE",
    "Makefile",
    "ROADMAP.md",
    "README.md",
    "pyproject.toml",
)
_LEGACY_PUBLIC_FILES = (
    _compact("AG", "ENTS.md"),
    "PLANS.md",
)
_PUBLIC_RELEASE_WORDING_GROUPS: Mapping[str, tuple[str, ...]] = {
    "internal_workflow": (
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
    ),
    "external_positioning": (
        _hyphenated(_compact("recruit", "er"), "facing"),
        _compact("recruit", "er"),
        _compact("C", "V"),
        _compact("res", "ume"),
        _compact("car", "eer"),
        _compact("hir", "ing"),
    ),
    "unsupported_claims": (
        _words("alpha", "validation"),
        _words("deployable", "alpha"),
        _words("AI", "trading", "bot"),
        *DEFAULT_FORBIDDEN_CLAIM_PHRASES,
    ),
}
_README_REQUIRED_LINK_TARGETS = (
    "docs/REPRODUCIBILITY.md",
    "docs/CLI_REFERENCE.md",
    "docs/PROJECT_STATUS.md",
    "docs/SAFETY_AND_LIMITATIONS.md",
    "docs/REPORT_EVIDENCE_INDEX.md",
    "ROADMAP.md",
)
_README_LOCAL_LINK_PATTERN = re.compile(
    r"\[[^\]]+\]\((?!https?:|mailto:|#)(?P<target>[^)#]+)(?:#[^)]+)?\)"
)
_EXCLUDED_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "data",
    "runs",
}
_CLAIM_CONTEXT_ALLOW_MARKERS = (
    "avoid",
    "caveat",
    "do not",
    "does not",
    "forbidden",
    "forbidden_claim",
    "is not",
    "must not",
    "never",
    "no ",
    "not ",
    "not be",
    "not claim",
    "should not",
    "suspicious",
    "vocabulary to avoid",
    "without",
)
_CLI_COMMAND_PATTERN = re.compile(
    r"app\.command\((?:\"(?P<name>[^\"]+)\")?\)\((?P<function>[A-Za-z_]\w*)\)"
)


def _resolve_root(root: str | Path | None = None) -> Path:
    if root is None:
        return project_root().resolve()
    return Path(root).expanduser().resolve()


def _relative_to_root(root: Path, path: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return path


def _status_from_issues(issues: Sequence[AuditIssue]) -> AuditStatus:
    if any(issue.status == AuditStatus.FAIL for issue in issues):
        return AuditStatus.FAIL
    if any(issue.status == AuditStatus.WARNING for issue in issues):
        return AuditStatus.WARNING
    return AuditStatus.PASS


def _has_excluded_part(root: Path, path: Path) -> bool:
    relative = _relative_to_root(root, path)
    return any(part in _EXCLUDED_DIR_NAMES for part in relative.parts)


def _iter_repository_files(root: Path) -> tuple[Path, ...]:
    """Return candidate repository files without walking caches or data dumps."""
    files: list[Path] = []
    existing_scan_dirs = [
        root / dirname for dirname in _AUDIT_SCAN_DIRS if (root / dirname).exists()
    ]
    if existing_scan_dirs:
        for directory in existing_scan_dirs:
            files.extend(path for path in directory.rglob("*") if path.is_file())
        files.extend(
            root / filename
            for filename in _ROOT_SCAN_FILES
            if (root / filename).is_file()
        )
    else:
        files.extend(path for path in root.rglob("*") if path.is_file())

    return tuple(
        sorted(
            {
                path.resolve()
                for path in files
                if path.is_file() and not _has_excluded_part(root, path)
            }
        )
    )


def _iter_public_release_files(root: Path) -> tuple[Path, ...]:
    files = []
    for path in _iter_repository_files(root):
        relative = _relative_to_root(root, path)
        if path.suffix.lower() in _PUBLIC_RELEASE_TEXT_SUFFIXES:
            files.append(path)
            continue
        if len(relative.parts) == 1 and relative.name in _ROOT_SCAN_FILES:
            files.append(path)
    return tuple(sorted(path.resolve() for path in files if path.is_file()))


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    except OSError:
        return None


def collect_config_files(root: str | Path | None = None) -> tuple[Path, ...]:
    """Collect YAML config files under ``configs``."""
    resolved_root = _resolve_root(root)
    config_root = resolved_root / "configs"
    if not config_root.exists():
        return ()
    files = [*config_root.rglob("*.yaml"), *config_root.rglob("*.yml")]
    return tuple(sorted(path.resolve() for path in files if path.is_file()))


def collect_report_files(root: str | Path | None = None) -> tuple[Path, ...]:
    """Collect Markdown report files under ``reports``."""
    resolved_root = _resolve_root(root)
    report_root = resolved_root / "reports"
    if not report_root.exists():
        return ()
    return tuple(sorted(path.resolve() for path in report_root.rglob("*.md") if path.is_file()))


def collect_test_files(root: str | Path | None = None) -> tuple[Path, ...]:
    """Collect pytest files under ``tests``."""
    resolved_root = _resolve_root(root)
    test_root = resolved_root / "tests"
    if not test_root.exists():
        return ()
    return tuple(sorted(path.resolve() for path in test_root.rglob("test_*.py") if path.is_file()))


def collect_cli_commands(root: str | Path | None = None) -> tuple[str, ...]:
    """Collect registered CLI command names from ``chronoslob/cli.py``."""
    resolved_root = _resolve_root(root)
    cli_path = resolved_root / "chronoslob" / "cli.py"
    text = _read_text(cli_path)
    if text is None:
        return ()

    commands: list[str] = []
    for match in _CLI_COMMAND_PATTERN.finditer(text):
        explicit_name = match.group("name")
        function_name = match.group("function")
        command = explicit_name if explicit_name else function_name.replace("_", "-")
        if command not in commands:
            commands.append(command)
    return tuple(commands)


def check_required_paths(
    root: str | Path | None = None,
    required_paths: Sequence[str | Path] = DEFAULT_REQUIRED_PATHS,
) -> AuditResult:
    """Check that required repository paths exist."""
    resolved_root = _resolve_root(root)
    issues: list[AuditIssue] = []
    for required_path in required_paths:
        candidate = resolved_root / Path(required_path)
        if not candidate.exists():
            issues.append(
                AuditIssue(
                    check_name="required_paths",
                    status=AuditStatus.FAIL,
                    message="Required path is missing.",
                    path=Path(required_path),
                )
            )
    return AuditResult(
        name="required_paths",
        status=_status_from_issues(issues),
        issues=tuple(issues),
        details={"checked": len(required_paths)},
    )


def _is_allowed_claim_context(lines: Sequence[str], line_index: int) -> bool:
    start = max(0, line_index - 12)
    end = min(len(lines), line_index + 2)
    context = " ".join(lines[start:end]).lower()
    return any(marker in context for marker in _CLAIM_CONTEXT_ALLOW_MARKERS)


def check_no_forbidden_claims(
    root: str | Path | None = None,
    forbidden_phrases: Sequence[str] = DEFAULT_FORBIDDEN_CLAIM_PHRASES,
    scan_paths: Sequence[str | Path] | None = None,
) -> AuditResult:
    """Scan text files for unsupported trading or performance claims.

    The filter is intentionally conservative but context-aware: mentions inside
    "do not claim" or "vocabulary to avoid" passages are ignored.
    """
    resolved_root = _resolve_root(root)
    if scan_paths is None:
        files = [
            path
            for path in _iter_repository_files(resolved_root)
            if path.suffix.lower() in _TEXT_SUFFIXES
            and "tests" not in _relative_to_root(resolved_root, path).parts
        ]
    else:
        files = [resolved_root / Path(path) for path in scan_paths]

    lowered_phrases = tuple((phrase, phrase.lower()) for phrase in forbidden_phrases)
    issues: list[AuditIssue] = []
    for file_path in files:
        text = _read_text(file_path)
        if text is None:
            continue
        lines = text.splitlines()
        for line_index, line in enumerate(lines):
            lowered_line = line.lower()
            for phrase, lowered_phrase in lowered_phrases:
                if lowered_phrase not in lowered_line:
                    continue
                if _is_allowed_claim_context(lines, line_index):
                    continue
                issues.append(
                    AuditIssue(
                        check_name="forbidden_claims",
                        status=AuditStatus.FAIL,
                        message="Unsupported performance or trading claim phrase.",
                        path=_relative_to_root(resolved_root, file_path),
                        line_number=line_index + 1,
                        matched_text=phrase,
                    )
                )

    return AuditResult(
        name="forbidden_claims",
        status=_status_from_issues(issues),
        issues=tuple(issues),
        details={"phrases": len(forbidden_phrases), "files_scanned": len(files)},
    )


def check_synthetic_fixture_labelling(root: str | Path | None = None) -> AuditResult:
    """Check that smoke configs and fixture READMEs are labelled synthetic."""
    resolved_root = _resolve_root(root)
    candidate_paths: list[Path] = []
    experiments_root = resolved_root / "configs" / "experiments"
    if experiments_root.exists():
        candidate_paths.extend(experiments_root.glob("*smoke*.yaml"))
        candidate_paths.extend(experiments_root.glob("*smoke*.yml"))

    fixtures_root = resolved_root / "tests" / "fixtures"
    if fixtures_root.exists():
        candidate_paths.extend(fixtures_root.rglob("README.md"))

    issues: list[AuditIssue] = []
    for path in sorted(candidate_paths):
        text = _read_text(path)
        if text is None:
            continue
        if "synthetic" not in text.lower():
            issues.append(
                AuditIssue(
                    check_name="synthetic_labelling",
                    status=AuditStatus.WARNING,
                    message="Smoke config or fixture README does not mention synthetic data.",
                    path=_relative_to_root(resolved_root, path),
                )
            )

    return AuditResult(
        name="synthetic_labelling",
        status=_status_from_issues(issues),
        issues=tuple(issues),
        details={"checked": len(candidate_paths)},
    )


def check_no_large_generated_files(
    root: str | Path | None = None,
    threshold_bytes: int = DEFAULT_LARGE_FILE_THRESHOLD_BYTES,
) -> AuditResult:
    """Check repository-facing files for unexpectedly large generated artefacts."""
    resolved_root = _resolve_root(root)
    issues: list[AuditIssue] = []
    files = _iter_repository_files(resolved_root)
    for path in files:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > threshold_bytes:
            issues.append(
                AuditIssue(
                    check_name="large_files",
                    status=AuditStatus.FAIL,
                    message=(
                        "File exceeds the local audit threshold "
                        f"({size} bytes > {threshold_bytes} bytes)."
                    ),
                    path=_relative_to_root(resolved_root, path),
                )
            )

    return AuditResult(
        name="large_files",
        status=_status_from_issues(issues),
        issues=tuple(issues),
        details={"threshold_bytes": threshold_bytes, "files_scanned": len(files)},
    )


def check_public_release_structure(
    root: str | Path | None = None,
    required_files: Sequence[str | Path] = DEFAULT_PUBLIC_RELEASE_REQUIRED_FILES,
) -> AuditResult:
    """Check that public release documentation has the expected shape."""
    resolved_root = _resolve_root(root)
    issues: list[AuditIssue] = []
    missing_count = 0

    for relative_path in required_files:
        candidate = resolved_root / Path(relative_path)
        if candidate.exists():
            continue
        missing_count += 1
        issues.append(
            AuditIssue(
                check_name="public_release_structure",
                status=AuditStatus.FAIL,
                message="Recommended public-release path is missing.",
                path=Path(relative_path),
            )
        )

    for legacy_name in _LEGACY_PUBLIC_FILES:
        if (resolved_root / legacy_name).exists():
            issues.append(
                AuditIssue(
                    check_name="public_release_structure",
                    status=AuditStatus.FAIL,
                    message="Internal-maintenance file should not be published.",
                    path=Path(legacy_name),
                )
            )

    return AuditResult(
        name="public_release_structure",
        status=_status_from_issues(issues),
        issues=tuple(issues),
        details={
            "required_files": len(required_files),
            "missing_recommended_files": missing_count,
        },
    )


def _normalise_readme_link_target(target: str) -> str:
    return target.strip().split("#", maxsplit=1)[0]


def check_public_release_readme(root: str | Path | None = None) -> AuditResult:
    """Check public README claims, links and status language."""
    resolved_root = _resolve_root(root)
    readme_path = resolved_root / "README.md"
    issues: list[AuditIssue] = []
    text = _read_text(readme_path)
    if text is None:
        issues.append(
            AuditIssue(
                check_name="public_release_readme",
                status=AuditStatus.FAIL,
                message="README.md is missing or unreadable.",
                path=Path("README.md"),
            )
        )
        return AuditResult(
            name="public_release_readme",
            status=AuditStatus.FAIL,
            issues=tuple(issues),
            details={"links_checked": 0},
        )

    required_snippets = (
        "# ChronosLOB",
        (
            "ChronosLOB is a research platform for leakage-safe limit order book "
            "representation learning, market-state forecasting, calibration and "
            "execution-aware validation."
        ),
        "not financial advice",
        "not live trading infrastructure",
        "no deployment or trading-use claims",
        "Synthetic fixture outputs are plumbing checks only.",
    )
    normalised_text = " ".join(text.split())
    for snippet in required_snippets:
        if " ".join(snippet.split()) not in normalised_text:
            issues.append(
                AuditIssue(
                    check_name="public_release_readme",
                    status=AuditStatus.FAIL,
                    message="README.md is missing required public-release wording.",
                    path=Path("README.md"),
                    matched_text=snippet,
                )
            )

    links_checked = 0
    for match in _README_LOCAL_LINK_PATTERN.finditer(text):
        target = _normalise_readme_link_target(match.group("target"))
        if not target:
            continue
        links_checked += 1
        if not (resolved_root / target).exists():
            issues.append(
                AuditIssue(
                    check_name="public_release_readme",
                    status=AuditStatus.FAIL,
                    message="README.md local link target is missing.",
                    path=Path("README.md"),
                    matched_text=target,
                )
            )

    for target in _README_REQUIRED_LINK_TARGETS:
        if f"]({target}" not in text:
            issues.append(
                AuditIssue(
                    check_name="public_release_readme",
                    status=AuditStatus.FAIL,
                    message="README.md does not link to a recommended document.",
                    path=Path("README.md"),
                    matched_text=target,
                )
            )

    return AuditResult(
        name="public_release_readme",
        status=_status_from_issues(issues),
        issues=tuple(issues),
        details={"links_checked": links_checked},
    )


def check_public_release_wording(
    root: str | Path | None = None,
    scan_paths: Sequence[str | Path] | None = None,
) -> AuditResult:
    """Scan repository text for public-release wording problems."""
    resolved_root = _resolve_root(root)
    files: Sequence[Path]
    if scan_paths is None:
        files = _iter_public_release_files(resolved_root)
    else:
        files = tuple(resolved_root / Path(path) for path in scan_paths)

    phrase_entries = tuple(
        (group, phrase, phrase.lower())
        for group, phrases in _PUBLIC_RELEASE_WORDING_GROUPS.items()
        for phrase in phrases
    )
    issues: list[AuditIssue] = []
    for file_path in files:
        text = _read_text(file_path)
        if text is None:
            continue
        for line_index, line in enumerate(text.splitlines()):
            lowered_line = line.lower()
            for group, phrase, lowered_phrase in phrase_entries:
                if lowered_phrase not in lowered_line:
                    continue
                issues.append(
                    AuditIssue(
                        check_name="public_release_wording",
                        status=AuditStatus.FAIL,
                        message=f"Public-release wording issue in {group}.",
                        path=_relative_to_root(resolved_root, file_path),
                        line_number=line_index + 1,
                        matched_text=phrase,
                    )
                )

    return AuditResult(
        name="public_release_wording",
        status=_status_from_issues(issues),
        issues=tuple(issues),
        details={
            "phrase_groups": len(_PUBLIC_RELEASE_WORDING_GROUPS),
            "files_scanned": len(files),
        },
    )


def run_public_release_audit(root: str | Path | None = None) -> ProjectAuditResult:
    """Run public-release checks and return a structured result."""
    resolved_root = _resolve_root(root)
    inventory = PathInventory(
        root=resolved_root,
        config_files=collect_config_files(resolved_root),
        report_files=collect_report_files(resolved_root),
        test_files=collect_test_files(resolved_root),
        cli_commands=collect_cli_commands(resolved_root),
    )
    results = (
        check_public_release_readme(resolved_root),
        check_public_release_structure(resolved_root),
        check_public_release_wording(resolved_root),
        check_no_forbidden_claims(resolved_root),
    )
    return ProjectAuditResult(root=resolved_root, inventory=inventory, results=results)


def run_project_audit(
    root: str | Path | None = None,
    *,
    large_file_threshold_bytes: int = DEFAULT_LARGE_FILE_THRESHOLD_BYTES,
    forbidden_phrases: Sequence[str] = DEFAULT_FORBIDDEN_CLAIM_PHRASES,
    required_paths: Sequence[str | Path] = DEFAULT_REQUIRED_PATHS,
) -> ProjectAuditResult:
    """Run the local ChronosLOB project audit and return structured results."""
    resolved_root = _resolve_root(root)
    inventory = PathInventory(
        root=resolved_root,
        config_files=collect_config_files(resolved_root),
        report_files=collect_report_files(resolved_root),
        test_files=collect_test_files(resolved_root),
        cli_commands=collect_cli_commands(resolved_root),
    )
    results = (
        check_required_paths(resolved_root, required_paths=required_paths),
        check_no_forbidden_claims(
            resolved_root,
            forbidden_phrases=forbidden_phrases,
        ),
        check_synthetic_fixture_labelling(resolved_root),
        check_no_large_generated_files(
            resolved_root,
            threshold_bytes=large_file_threshold_bytes,
        ),
        check_public_release_readme(resolved_root),
        check_public_release_structure(resolved_root),
        check_public_release_wording(resolved_root),
    )
    return ProjectAuditResult(root=resolved_root, inventory=inventory, results=results)

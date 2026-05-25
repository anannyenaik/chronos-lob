"""Technical evidence archive builders for ChronosLOB.

The utilities in this module are local-only. They collect repository inventory,
curated CLI output captures and text-based Mermaid diagrams to support
reproducibility reviews. They do not download data, run heavy training by
default, call external services or generate benchmark results.
"""

from __future__ import annotations

import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from chronoslob import __version__
from chronoslob.utils.audit import (
    collect_cli_commands,
    collect_config_files,
    collect_report_files,
    collect_test_files,
)
from chronoslob.utils.paths import project_root

__all__ = [
    "EXPECTED_ARCHIVE_FILES",
    "CommandCapture",
    "CommandSpec",
    "ReportArchiveConfig",
    "ReportArchiveResult",
    "ReportArchiveSection",
    "build_report_archive",
    "capture_command",
    "collect_cli_outputs",
    "collect_config_inventory",
    "collect_limitations_index",
    "collect_module_inventory",
    "collect_project_inventory",
    "collect_release_history",
    "collect_test_inventory",
    "default_cli_command_specs",
    "inspect_report_archive",
    "write_report_archive",
]


EXPECTED_ARCHIVE_FILES: tuple[Path, ...] = (
    Path("README.md"),
    Path("project_inventory.md"),
    Path("release_history.md"),
    Path("cli_outputs.md"),
    Path("config_inventory.md"),
    Path("module_inventory.md"),
    Path("test_inventory.md"),
    Path("limitations_index.md"),
    Path("reproducibility_commands.md"),
    Path("figures/architecture_overview.mmd"),
    Path("figures/data_pipeline.mmd"),
    Path("figures/model_stack.mmd"),
    Path("figures/evaluation_stack.mmd"),
)

_CAPTURE_LIMIT_CHARS = 8_000
_PACKAGE_AREAS = (
    "data",
    "book",
    "features",
    "labels",
    "training",
    "models",
    "backtest",
    "analysis",
    "utils",
)
_VALIDATION_COMMANDS = (
    'python -c "import chronoslob; print(chronoslob.__version__)"',
    "python -m chronoslob.cli doctor",
    "python -m chronoslob.cli inspect-release-readiness",
    "python -m chronoslob.cli run-project-audit --strict",
    "python -m pytest",
    "python -m compileall -q chronoslob tests",
    "python -m ruff check .",
    "python -m mypy chronoslob",
)


@dataclass(frozen=True)
class CommandSpec:
    """A local CLI command to capture for the report evidence archive."""

    display_command: str
    args: tuple[str, ...]
    description: str
    synthetic: bool = False
    optional: bool = True
    include_when_training: bool = False


@dataclass(frozen=True)
class CommandCapture:
    """Captured stdout and stderr from one local command."""

    display_command: str
    description: str
    exit_code: int
    stdout: str
    stderr: str
    synthetic: bool = False
    optional: bool = True
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        """Return true when the command completed with exit code zero."""
        return self.exit_code == 0 and not self.timed_out


@dataclass(frozen=True)
class ReportArchiveSection:
    """A generated report archive file."""

    relative_path: Path
    title: str
    content: str
    synthetic: bool = False


@dataclass(frozen=True)
class ReportArchiveConfig:
    """Configuration for building the local report evidence archive."""

    root: Path = field(default_factory=project_root)
    output_path: Path = Path("reports/report_archive")
    strict: bool = False
    include_smoke_training: bool = False
    command_timeout_seconds: int = 60
    command_specs: tuple[CommandSpec, ...] | None = None


@dataclass(frozen=True)
class ReportArchiveResult:
    """Summary returned after building the report evidence archive."""

    output_path: Path
    files_written: tuple[Path, ...]
    sections: tuple[ReportArchiveSection, ...]
    command_captures: tuple[CommandCapture, ...]
    warnings: tuple[str, ...] = ()

    @property
    def commands_captured(self) -> int:
        """Return the number of CLI commands captured."""
        return len(self.command_captures)

    @property
    def warnings_count(self) -> int:
        """Return the number of non-fatal warnings."""
        return len(self.warnings)

    @property
    def synthetic_section_count(self) -> int:
        """Return the number of generated sections marked as synthetic-aware."""
        return sum(1 for section in self.sections if section.synthetic)


def _resolve_root(root: Path) -> Path:
    return Path(root).expanduser().resolve()


def _resolve_output_path(root: Path, output_path: Path) -> Path:
    path = Path(output_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def _relative_to_root(root: Path, path: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return path


def _normalise_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _truncate_capture(text: str) -> str:
    stripped = text.strip()
    if len(stripped) <= _CAPTURE_LIMIT_CHARS:
        return stripped
    return f"{stripped[:_CAPTURE_LIMIT_CHARS].rstrip()}\n...[truncated]..."


def _ensure_trailing_newline(text: str) -> str:
    if text.endswith("\n"):
        return text
    return f"{text}\n"


def _markdown_list(values: Iterable[str]) -> list[str]:
    return [f"- `{value}`" for value in values]


def _format_relative_paths(root: Path, paths: Sequence[Path]) -> tuple[str, ...]:
    return tuple(str(_relative_to_root(root, path)).replace("\\", "/") for path in paths)


def default_cli_command_specs(*, include_smoke_training: bool = False) -> tuple[CommandSpec, ...]:
    """Return the default local command list for archive capture."""

    python = sys.executable
    fixture_fi2010 = "tests/fixtures/fi2010/tiny_fi2010_like.csv"
    fixture_events = "tests/fixtures/event_logs/synthetic_snapshots.jsonl"
    base_specs: list[CommandSpec] = [
        CommandSpec(
            display_command="python -m chronoslob.cli doctor",
            args=(python, "-m", "chronoslob.cli", "doctor"),
            description="Environment and package smoke check.",
            optional=False,
        ),
        CommandSpec(
            display_command="python -m chronoslob.cli run-project-audit",
            args=(python, "-m", "chronoslob.cli", "run-project-audit"),
            description="Local repository audit summary.",
            optional=False,
        ),
        CommandSpec(
            display_command=(
                "python -m chronoslob.cli inspect-fi2010 "
                f"--path {fixture_fi2010}"
            ),
            args=(
                python,
                "-m",
                "chronoslob.cli",
                "inspect-fi2010",
                "--path",
                fixture_fi2010,
            ),
            description="FI-2010-style loader inspection on a synthetic fixture.",
            synthetic=True,
        ),
        CommandSpec(
            display_command=(
                "python -m chronoslob.cli inspect-features-fi2010 "
                f"--path {fixture_fi2010}"
            ),
            args=(
                python,
                "-m",
                "chronoslob.cli",
                "inspect-features-fi2010",
                "--path",
                fixture_fi2010,
            ),
            description="Feature-pipeline inspection on a synthetic FI-2010-style fixture.",
            synthetic=True,
        ),
        CommandSpec(
            display_command=(
                "python -m chronoslob.cli inspect-labels-fi2010 "
                f"--path {fixture_fi2010}"
            ),
            args=(
                python,
                "-m",
                "chronoslob.cli",
                "inspect-labels-fi2010",
                "--path",
                fixture_fi2010,
            ),
            description="Label-pipeline inspection on a synthetic FI-2010-style fixture.",
            synthetic=True,
        ),
        CommandSpec(
            display_command=(
                "python -m chronoslob.cli inspect-event-log "
                f"--path {fixture_events}"
            ),
            args=(
                python,
                "-m",
                "chronoslob.cli",
                "inspect-event-log",
                "--path",
                fixture_events,
            ),
            description="Canonical event-log inspection on a synthetic fixture.",
            synthetic=True,
        ),
        CommandSpec(
            display_command=(
                "python -m chronoslob.cli inspect-event-tokens "
                f"--path {fixture_events}"
            ),
            args=(
                python,
                "-m",
                "chronoslob.cli",
                "inspect-event-tokens",
                "--path",
                fixture_events,
            ),
            description="Event-token inspection on a synthetic event-log fixture.",
            synthetic=True,
        ),
        CommandSpec(
            display_command="python -m chronoslob.cli inspect-transformer",
            args=(python, "-m", "chronoslob.cli", "inspect-transformer"),
            description="Transformer architecture support summary.",
        ),
        CommandSpec(
            display_command="python -m chronoslob.cli inspect-ssl",
            args=(python, "-m", "chronoslob.cli", "inspect-ssl"),
            description="Self-supervised objective support summary.",
        ),
        CommandSpec(
            display_command="python -m chronoslob.cli inspect-multitask",
            args=(python, "-m", "chronoslob.cli", "inspect-multitask"),
            description="Multi-task fine-tuning support summary.",
        ),
        CommandSpec(
            display_command="python -m chronoslob.cli inspect-calibration",
            args=(python, "-m", "chronoslob.cli", "inspect-calibration"),
            description="Calibration and uncertainty support summary.",
        ),
        CommandSpec(
            display_command="python -m chronoslob.cli inspect-execution-validation",
            args=(python, "-m", "chronoslob.cli", "inspect-execution-validation"),
            description="Execution-aware validation support summary.",
        ),
        CommandSpec(
            display_command="python -m chronoslob.cli inspect-analysis",
            args=(python, "-m", "chronoslob.cli", "inspect-analysis"),
            description="Transfer, regime, ablation and sensitivity support summary.",
        ),
    ]
    training_specs = [
        CommandSpec(
            display_command=(
                "python -m chronoslob.cli run-baseline-smoke "
                f"--path {fixture_fi2010}"
            ),
            args=(
                python,
                "-m",
                "chronoslob.cli",
                "run-baseline-smoke",
                "--path",
                fixture_fi2010,
            ),
            description="Synthetic classical-baseline smoke run.",
            synthetic=True,
            include_when_training=True,
        ),
        CommandSpec(
            display_command=(
                "python -m chronoslob.cli run-deeplob-smoke "
                f"--path {fixture_fi2010}"
            ),
            args=(
                python,
                "-m",
                "chronoslob.cli",
                "run-deeplob-smoke",
                "--path",
                fixture_fi2010,
            ),
            description="Synthetic DeepLOB-style smoke run.",
            synthetic=True,
            include_when_training=True,
        ),
        CommandSpec(
            display_command=(
                "python -m chronoslob.cli run-transformer-smoke "
                f"--path {fixture_events}"
            ),
            args=(
                python,
                "-m",
                "chronoslob.cli",
                "run-transformer-smoke",
                "--path",
                fixture_events,
            ),
            description="Synthetic transformer smoke run.",
            synthetic=True,
            include_when_training=True,
        ),
        CommandSpec(
            display_command=(
                "python -m chronoslob.cli run-ssl-smoke "
                f"--path {fixture_events}"
            ),
            args=(
                python,
                "-m",
                "chronoslob.cli",
                "run-ssl-smoke",
                "--path",
                fixture_events,
            ),
            description="Synthetic self-supervised smoke run.",
            synthetic=True,
            include_when_training=True,
        ),
        CommandSpec(
            display_command=(
                "python -m chronoslob.cli run-multitask-smoke "
                f"--path {fixture_events}"
            ),
            args=(
                python,
                "-m",
                "chronoslob.cli",
                "run-multitask-smoke",
                "--path",
                fixture_events,
            ),
            description="Synthetic multi-task smoke run.",
            synthetic=True,
            include_when_training=True,
        ),
        CommandSpec(
            display_command="python -m chronoslob.cli run-calibration-smoke",
            args=(python, "-m", "chronoslob.cli", "run-calibration-smoke"),
            description="Synthetic calibration diagnostics smoke run.",
            synthetic=True,
            include_when_training=True,
        ),
        CommandSpec(
            display_command="python -m chronoslob.cli run-execution-validation-smoke",
            args=(
                python,
                "-m",
                "chronoslob.cli",
                "run-execution-validation-smoke",
            ),
            description="Synthetic execution-validation smoke run.",
            synthetic=True,
            include_when_training=True,
        ),
        CommandSpec(
            display_command="python -m chronoslob.cli run-robustness-analysis-smoke",
            args=(
                python,
                "-m",
                "chronoslob.cli",
                "run-robustness-analysis-smoke",
            ),
            description="Synthetic robustness-analysis smoke run.",
            synthetic=True,
            include_when_training=True,
        ),
    ]
    if include_smoke_training:
        base_specs.extend(training_specs)
    return tuple(base_specs)


def capture_command(
    spec: CommandSpec,
    *,
    root: Path,
    timeout_seconds: int,
) -> CommandCapture:
    """Capture one local command without shell expansion or network access."""

    resolved_root = _resolve_root(root)
    try:
        completed = subprocess.run(
            spec.args,
            cwd=resolved_root,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _truncate_capture(_normalise_text(exc.stdout))
        stderr = _truncate_capture(_normalise_text(exc.stderr))
        timeout_note = f"Command timed out after {timeout_seconds} seconds."
        stderr = f"{stderr}\n{timeout_note}".strip()
        return CommandCapture(
            display_command=spec.display_command,
            description=spec.description,
            exit_code=-1,
            stdout=stdout,
            stderr=stderr,
            synthetic=spec.synthetic,
            optional=spec.optional,
            timed_out=True,
        )

    return CommandCapture(
        display_command=spec.display_command,
        description=spec.description,
        exit_code=completed.returncode,
        stdout=_truncate_capture(_normalise_text(completed.stdout)),
        stderr=_truncate_capture(_normalise_text(completed.stderr)),
        synthetic=spec.synthetic,
        optional=spec.optional,
    )


def collect_cli_outputs(config: ReportArchiveConfig) -> tuple[CommandCapture, ...]:
    """Capture curated lightweight CLI outputs for the technical evidence archive."""

    specs = config.command_specs
    if specs is None:
        specs = default_cli_command_specs(
            include_smoke_training=config.include_smoke_training,
        )
    root = _resolve_root(config.root)
    captures = [
        capture_command(
            spec,
            root=root,
            timeout_seconds=config.command_timeout_seconds,
        )
        for spec in specs
    ]
    return tuple(captures)


def _collect_report_files_for_inventory(root: Path, output_path: Path) -> tuple[Path, ...]:
    report_files = collect_report_files(root)
    excluded_roots = {output_path.resolve()}
    excluded_files = {(root / "reports" / "report_archive_build.md").resolve()}
    retained: list[Path] = []
    for path in report_files:
        resolved = path.resolve()
        if resolved in excluded_files:
            continue
        if any(resolved.is_relative_to(excluded_root) for excluded_root in excluded_roots):
            continue
        retained.append(resolved)
    return tuple(sorted(retained))


def collect_project_inventory(config: ReportArchiveConfig) -> ReportArchiveSection:
    """Build the project-level inventory section."""

    root = _resolve_root(config.root)
    output_path = _resolve_output_path(root, config.output_path)
    config_files = collect_config_files(root)
    report_files = _collect_report_files_for_inventory(root, output_path)
    test_files = collect_test_files(root)
    cli_commands = collect_cli_commands(root)

    package_area_lines = _markdown_list(f"chronoslob.{area}" for area in _PACKAGE_AREAS)
    cli_lines = _markdown_list(cli_commands)
    validation_lines = [f"- `{command}`" for command in _VALIDATION_COMMANDS]
    content = "\n".join(
        [
            "# Project Inventory",
            "",
            "Snapshot of repository structure for reproducibility and review.",
            "",
            f"- Package version: `{__version__}`",
            f"- Config files: `{len(config_files)}`",
            f"- Report files, excluding this generated archive: `{len(report_files)}`",
            f"- Test files: `{len(test_files)}`",
            f"- CLI commands: `{len(cli_commands)}`",
            "",
            "## Major Package Areas",
            "",
            *package_area_lines,
            "",
            "## Current CLI Commands",
            "",
            *cli_lines,
            "",
            "## Validation Command List",
            "",
            *validation_lines,
        ]
    )
    return ReportArchiveSection(
        relative_path=Path("project_inventory.md"),
        title="Project Inventory",
        content=content,
    )


def collect_release_history() -> ReportArchiveSection:
    """Build the implemented release-history summary."""

    history_rows = (
        ("Foundation", "repository scaffold, tooling and documentation conventions"),
        ("Data contracts", "schemas for events, order books, features, labels and quality issues"),
        ("Local loading", "FI-2010-style loading and validation"),
        ("Feature engine", "past-only microstructure feature generation"),
        ("Label engine", "future-window labels and no-look-ahead leakage checks"),
        ("Validation protocols", "temporal, walk-forward and purged or embargoed splitters"),
        ("Classical baselines", "baseline interfaces, metrics and train-only preprocessing"),
        ("Torch data layer", "PyTorch sequence-window datasets and loaders"),
        ("DeepLOB-style path", "supervised CNN-LSTM baseline path"),
        ("Book reconstruction", "offline Binance-style order book reconstruction"),
        ("Event logs", "canonical JSONL storage and replay-to-feature/label integration"),
        ("Tokenisation", "deterministic event tokenisation and transformer inputs"),
        ("Transformer modelling", "supervised transformer encoder architecture"),
        ("Self-supervision", "masked-field and next-field objectives"),
        ("Multi-task modelling", "fine-tuning infrastructure"),
        ("Calibration", "uncertainty and confidence-filtering diagnostics"),
        ("Execution-aware validation", "explicit simplified assumptions for costs and latency"),
        ("Robustness analysis", "transfer, regime, ablation and sensitivity summaries"),
        ("Audit and CI", "local audit utilities, CI hardening and reproducibility documentation"),
        ("Evidence archive", "technical evidence archive and public documentation polish"),
    )
    table = ["| Milestone | Implemented scope |", "|---|---|"]
    table.extend(f"| {name} | {scope}. |" for name, scope in history_rows)
    content = "\n".join(
        [
            "# Release History",
            "",
            "Summary of implementation milestones in this repository.",
            "",
            *table,
        ]
    )
    return ReportArchiveSection(
        relative_path=Path("release_history.md"),
        title="Release History",
        content=content,
    )


def _render_stream_block(label: str, value: str) -> list[str]:
    rendered = value if value else "<empty>"
    return [f"{label}:", "", "```text", rendered, "```"]


def _render_cli_outputs(
    captures: Sequence[CommandCapture],
    *,
    include_smoke_training: bool,
) -> ReportArchiveSection:
    lines = [
        "# CLI Outputs",
        "",
        "Local CLI captures included for reproducibility review. Commands run "
        "against bundled synthetic fixtures are labelled accordingly; see "
        "[../../docs/SAFETY_AND_LIMITATIONS.md] for what synthetic outputs "
        "do and do not represent.",
        "",
        f"- Optional training-style commands included: `{include_smoke_training}`",
        f"- Commands captured: `{len(captures)}`",
        "",
    ]
    for index, capture in enumerate(captures, start=1):
        lines.extend(
            [
                f"## {index}. {capture.description}",
                "",
                f"- Uses synthetic fixture: `{capture.synthetic}`",
                f"- Optional command: `{capture.optional}`",
                f"- Exit code: `{capture.exit_code}`",
                f"- Timed out: `{capture.timed_out}`",
                "",
                "Command:",
                "",
                "```bash",
                capture.display_command,
                "```",
                "",
            ]
        )
        lines.extend(_render_stream_block("Stdout", capture.stdout))
        lines.append("")
        lines.extend(_render_stream_block("Stderr", capture.stderr))
        lines.append("")

    return ReportArchiveSection(
        relative_path=Path("cli_outputs.md"),
        title="CLI Outputs",
        content="\n".join(lines).rstrip(),
        synthetic=True,
    )


def _infer_config_purpose(relative_path: Path) -> str:
    name = relative_path.name.lower()
    parent = relative_path.parent.name.lower()
    if name == "report_archive_smoke.yaml":
        return "Evidence-archive build configuration."
    if "smoke" in name:
        return "Synthetic-fixture configuration."
    if parent == "data":
        return "Local data loading or replay configuration."
    if parent == "models":
        return "Model architecture or baseline configuration."
    if parent == "experiments":
        return "Experiment or audit configuration."
    return "Repository configuration."


def collect_config_inventory(config: ReportArchiveConfig) -> ReportArchiveSection:
    """Build a deterministic config inventory grouped by directory."""

    root = _resolve_root(config.root)
    grouped: defaultdict[str, list[Path]] = defaultdict(list)
    for path in collect_config_files(root):
        relative = _relative_to_root(root, path)
        grouped[str(relative.parent).replace("\\", "/")].append(relative)

    lines = [
        "# Config Inventory",
        "",
        "Local YAML configs grouped by directory. Files containing `smoke` in "
        "the name use bundled synthetic fixtures.",
        "",
    ]
    for directory in sorted(grouped):
        lines.extend([f"## `{directory}`", ""])
        for relative in sorted(grouped[directory]):
            synthetic = "yes" if "smoke" in relative.name.lower() else "no"
            lines.append(
                f"- `{relative.as_posix()}` - {_infer_config_purpose(relative)} "
                f"Uses synthetic fixture: `{synthetic}`."
            )
        lines.append("")

    return ReportArchiveSection(
        relative_path=Path("config_inventory.md"),
        title="Config Inventory",
        content="\n".join(lines).rstrip(),
        synthetic=True,
    )


def collect_module_inventory(config: ReportArchiveConfig) -> ReportArchiveSection:
    """Build a public module inventory grouped by package area."""

    root = _resolve_root(config.root)
    lines = [
        "# Module Inventory",
        "",
        "Public modules grouped by package area.",
        "",
    ]
    for area in _PACKAGE_AREAS:
        area_root = root / "chronoslob" / area
        lines.extend([f"## `{area}`", ""])
        if not area_root.exists():
            lines.extend(["- Not present.", ""])
            continue
        modules = []
        for path in sorted(area_root.glob("*.py")):
            if path.name == "__init__.py":
                continue
            module_name = f"chronoslob.{area}.{path.stem}"
            modules.append(module_name)
        lines.extend(_markdown_list(modules))
        lines.append("")

    return ReportArchiveSection(
        relative_path=Path("module_inventory.md"),
        title="Module Inventory",
        content="\n".join(lines).rstrip(),
    )


def _infer_test_area(path: Path) -> str:
    stem = path.stem.lower()
    area_markers: Mapping[str, str] = {
        "feature": "features",
        "label": "labels",
        "split": "training",
        "baseline": "models/training",
        "torch": "training",
        "deeplob": "models",
        "transformer": "models/training",
        "ssl": "models/training",
        "multitask": "models/training",
        "calibration": "calibration",
        "execution": "backtest",
        "analysis": "analysis",
        "audit": "utils",
        "config": "configs",
        "report": "reports",
        "event": "data/book",
        "binance": "data/book",
        "fi2010": "data",
    }
    for marker, area in area_markers.items():
        if marker in stem:
            return area
    return "general"


def collect_test_inventory(config: ReportArchiveConfig) -> ReportArchiveSection:
    """Build a concise test inventory by filename and inferred area."""

    root = _resolve_root(config.root)
    grouped: defaultdict[str, list[Path]] = defaultdict(list)
    for path in collect_test_files(root):
        grouped[_infer_test_area(path)].append(_relative_to_root(root, path))

    lines = [
        "# Test Inventory",
        "",
        "Pytest files grouped by inferred area. Test source contents are not "
        "included here.",
        "",
    ]
    for area in sorted(grouped):
        lines.extend([f"## {area}", ""])
        for relative in sorted(grouped[area]):
            lines.append(f"- `{relative.as_posix()}`")
        lines.append("")

    return ReportArchiveSection(
        relative_path=Path("test_inventory.md"),
        title="Test Inventory",
        content="\n".join(lines).rstrip(),
    )


def collect_limitations_index() -> ReportArchiveSection:
    """Build a limitations and caveats index keyed to the public docs."""

    lines = [
        "# Limitations Index",
        "",
        "Pointer index to the canonical scope and limitation documents.",
        "",
        "## Primary References",
        "",
        "- `../limitations.md`: technical caveats for extending the platform.",
        "- `../../docs/SAFETY_AND_LIMITATIONS.md`: canonical scope statement.",
        "- `../../docs/REPRODUCIBILITY.md`: validation and reproducibility path.",
        "- `../../docs/PROJECT_STATUS.md`: implemented and current limitations.",
        "",
        "## Implementation Reports With Limitation Context",
        "",
        "- `../data_quality.md`",
        "- `../feature_engine.md`",
        "- `../label_engine.md`",
        "- `../leakage_controls.md`",
        "- `../validation_protocol.md`",
        "- `../calibration_uncertainty.md`",
        "- `../execution_aware_validation.md`",
        "- `../transfer_regime_ablation_analysis.md`",
        "- `../full_audit_ci_hardening.md`",
        "",
        "## Core Caveats",
        "",
        "- Public data may have coverage, preprocessing and timestamp limitations.",
        "- Synthetic fixtures exercise code paths only.",
        "- Crypto-style reconstruction examples should not be treated as equity-"
        "market evidence.",
        "- Execution-aware validation is a simplified research simulation.",
        "- Queue position, partial fills, latency realism and venue rules remain "
        "explicit assumptions.",
        "- No production market impact model is implemented.",
        "- Reported metrics must trace to versioned configs, data, seeds and "
        "stored outputs.",
    ]
    return ReportArchiveSection(
        relative_path=Path("limitations_index.md"),
        title="Limitations Index",
        content="\n".join(lines),
    )


def _collect_reproducibility_commands() -> ReportArchiveSection:
    lines = [
        "# Reproducibility Commands",
        "",
        "Canonical Python commands. Run them from the repository root.",
        "",
        "## Install",
        "",
        "```bash",
        "python -m pip install --upgrade pip",
        'python -m pip install -e ".[dev,torch]"',
        "```",
        "",
        "## Core Checks",
        "",
        "```bash",
        *(_VALIDATION_COMMANDS),
        "```",
        "",
        "## Rebuild The Evidence Archive",
        "",
        "```bash",
        "python -m chronoslob.cli build-report-archive",
        "python -m chronoslob.cli inspect-report-archive",
        "```",
        "",
        "## Lightweight CLI Commands",
        "",
        "```bash",
        (
            "python -m chronoslob.cli inspect-fi2010 --path "
            "tests/fixtures/fi2010/tiny_fi2010_like.csv"
        ),
        (
            "python -m chronoslob.cli inspect-features-fi2010 --path "
            "tests/fixtures/fi2010/tiny_fi2010_like.csv"
        ),
        (
            "python -m chronoslob.cli inspect-labels-fi2010 --path "
            "tests/fixtures/fi2010/tiny_fi2010_like.csv"
        ),
        (
            "python -m chronoslob.cli inspect-event-log --path "
            "tests/fixtures/event_logs/synthetic_snapshots.jsonl"
        ),
        (
            "python -m chronoslob.cli inspect-event-tokens --path "
            "tests/fixtures/event_logs/synthetic_snapshots.jsonl"
        ),
        "python -m chronoslob.cli inspect-transformer",
        "python -m chronoslob.cli inspect-ssl",
        "python -m chronoslob.cli inspect-multitask",
        "python -m chronoslob.cli inspect-calibration",
        "python -m chronoslob.cli inspect-execution-validation",
        "python -m chronoslob.cli inspect-analysis",
        "```",
    ]
    return ReportArchiveSection(
        relative_path=Path("reproducibility_commands.md"),
        title="Reproducibility Commands",
        content="\n".join(lines),
        synthetic=True,
    )


def _collect_report_archive_readme() -> ReportArchiveSection:
    lines = [
        "# Technical Evidence Archive",
        "",
        "Generated archive of repository inventories, release history, current "
        "CLI captures and Mermaid diagrams. Used as a reproducibility reference.",
        "",
        "Commands captured against bundled synthetic fixtures are labelled "
        "accordingly. See `../../docs/SAFETY_AND_LIMITATIONS.md` for the full "
        "scope statement.",
        "",
        "Rebuild with:",
        "",
        "```bash",
        "python -m chronoslob.cli build-report-archive",
        "```",
    ]
    return ReportArchiveSection(
        relative_path=Path("README.md"),
        title="Technical Evidence Archive",
        content="\n".join(lines),
        synthetic=True,
    )


def _collect_mermaid_diagrams() -> tuple[ReportArchiveSection, ...]:
    diagrams = {
        Path("figures/architecture_overview.mmd"): "\n".join(
            [
                "flowchart LR",
                '  data["Data layer\\nFI-2010 loader, schemas, validation"]',
                '  replay["Book and replay layer\\nLocal order book, event logs"]',
                '  features["Feature and label layer\\nPast-only features, future labels"]',
                '  splits["Split and experiment layer\\nTemporal, walk-forward, purged"]',
                '  models["Model layer\\nBaselines, DeepLOB-style, transformer"]',
                '  calibration["Calibration layer\\nUncertainty and confidence filtering"]',
                '  execution["Execution validation layer\\nCosts, latency, risk constraints"]',
                '  analysis["Analysis and report layer\\nTransfer, regime, archive"]',
                "  data --> replay --> features --> splits --> models",
                "  models --> calibration --> execution --> analysis",
                "  splits --> analysis",
                "  features --> analysis",
            ]
        ),
        Path("figures/data_pipeline.mmd"): "\n".join(
            [
                "flowchart LR",
                '  fi["FI-2010 local loader"]',
                '  binance["Offline Binance-style reconstruction"]',
                '  logs["Canonical JSONL event logs"]',
                '  snapshots["Snapshots and events"]',
                '  feats["Past-only features"]',
                '  labels["Future-window labels"]',
                '  leakage["Leakage checks"]',
                '  splits["Temporal splits"]',
                "  fi --> snapshots",
                "  binance --> logs --> snapshots",
                "  snapshots --> feats",
                "  snapshots --> labels",
                "  feats --> leakage",
                "  labels --> leakage",
                "  leakage --> splits",
            ]
        ),
        Path("figures/model_stack.mmd"): "\n".join(
            [
                "flowchart TB",
                '  inputs["Features or event tokens"]',
                '  classical["Classical baselines"]',
                '  deeplob["DeepLOB-style CNN-LSTM"]',
                '  tokenisation["Deterministic tokenisation"]',
                '  transformer["Transformer encoder"]',
                '  ssl["Self-supervised objectives"]',
                '  multitask["Multi-task heads"]',
                "  inputs --> classical",
                "  inputs --> deeplob",
                "  inputs --> tokenisation --> transformer",
                "  transformer --> ssl",
                "  transformer --> multitask",
            ]
        ),
        Path("figures/evaluation_stack.mmd"): "\n".join(
            [
                "flowchart LR",
                '  predictive["Predictive metrics"]',
                '  calibration["Calibration metrics"]',
                '  confidence["Confidence filtering"]',
                '  execution["Execution-aware validation"]',
                '  robustness["Transfer, regime, ablation, sensitivity"]',
                '  caveats["Limitations and assumptions"]',
                "  predictive --> calibration --> confidence",
                "  confidence --> execution",
                "  predictive --> robustness",
                "  execution --> robustness",
                "  robustness --> caveats",
            ]
        ),
    }
    return tuple(
        ReportArchiveSection(
            relative_path=relative_path,
            title=relative_path.stem.replace("_", " ").title(),
            content=content,
        )
        for relative_path, content in diagrams.items()
    )


def _build_warnings(captures: Sequence[CommandCapture]) -> tuple[str, ...]:
    warnings: list[str] = []
    for capture in captures:
        if capture.ok:
            continue
        severity = "optional" if capture.optional else "required"
        warnings.append(
            f"{severity} command failed: {capture.display_command} "
            f"(exit code {capture.exit_code})"
        )
    return tuple(warnings)


def _raise_for_strict_failures(captures: Sequence[CommandCapture]) -> None:
    failures = [capture for capture in captures if not capture.ok]
    if not failures:
        return
    details = "; ".join(
        f"{capture.display_command} exited {capture.exit_code}" for capture in failures
    )
    raise RuntimeError(f"Report archive strict mode failed: {details}")


def _all_archive_sections(
    config: ReportArchiveConfig,
    captures: Sequence[CommandCapture],
) -> tuple[ReportArchiveSection, ...]:
    return (
        _collect_report_archive_readme(),
        collect_project_inventory(config),
        collect_release_history(),
        _render_cli_outputs(
            captures,
            include_smoke_training=config.include_smoke_training,
        ),
        collect_config_inventory(config),
        collect_module_inventory(config),
        collect_test_inventory(config),
        collect_limitations_index(),
        _collect_reproducibility_commands(),
        *_collect_mermaid_diagrams(),
    )


def write_report_archive(
    config: ReportArchiveConfig,
    sections: Sequence[ReportArchiveSection],
) -> tuple[Path, ...]:
    """Write generated archive sections and return absolute file paths."""

    root = _resolve_root(config.root)
    output_path = _resolve_output_path(root, config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for section in sections:
        if section.relative_path.is_absolute():
            raise ValueError("Archive section paths must be relative.")
        target = (output_path / section.relative_path).resolve()
        if not target.is_relative_to(output_path):
            raise ValueError(f"Archive section path escapes output directory: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_ensure_trailing_newline(section.content), encoding="utf-8")
        written.append(target)
    return tuple(sorted(written))


def build_report_archive(config: ReportArchiveConfig | None = None) -> ReportArchiveResult:
    """Build the local technical evidence archive."""

    resolved_config = config if config is not None else ReportArchiveConfig()
    captures = collect_cli_outputs(resolved_config)
    if resolved_config.strict:
        _raise_for_strict_failures(captures)
    sections = _all_archive_sections(resolved_config, captures)
    files_written = write_report_archive(resolved_config, sections)
    root = _resolve_root(resolved_config.root)
    output_path = _resolve_output_path(root, resolved_config.output_path)
    return ReportArchiveResult(
        output_path=output_path,
        files_written=files_written,
        sections=sections,
        command_captures=captures,
        warnings=_build_warnings(captures),
    )


def inspect_report_archive(
    output_path: Path = Path("reports/report_archive"),
    *,
    root: Path | None = None,
) -> tuple[tuple[Path, bool], ...]:
    """Return expected archive files and whether each is present."""

    resolved_root = _resolve_root(project_root() if root is None else root)
    archive_root = _resolve_output_path(resolved_root, output_path)
    return tuple(
        (relative_path, (archive_root / relative_path).is_file())
        for relative_path in EXPECTED_ARCHIVE_FILES
    )

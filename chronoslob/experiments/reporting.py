"""Build empirical Markdown reports from stored paper artefacts.

The report builder is read-only with respect to experiment inputs: it validates
the supplied paper experiment directory, reads stored JSON/CSV/Markdown
artefacts and writes a report plus a small build summary. It does not train
models, build plots or infer missing metrics.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from chronoslob.experiments.artifacts import (
    load_data_manifest,
    load_results,
    validate_experiment_directory,
)
from chronoslob.experiments.manifests import stable_json_dumps
from chronoslob.experiments.schemas import DataManifest, ExperimentResults
from chronoslob.utils.paths import project_root

__all__ = [
    "PAPER_REPORT_BUILDER_VERSION",
    "PaperReportInspection",
    "PaperReportSummary",
    "build_paper_report",
    "inspect_paper_report",
]

PAPER_REPORT_BUILDER_VERSION = "phase-j/paper-report-builder/v1"
_UNAVAILABLE = "not available"
_NO_WARNING = "none"
_SMOKE_MARKERS = ("smoke", "synthetic_fixture", "tests/fixtures", "tests\\fixtures")
_REPORT_SECTION_TITLES: tuple[str, ...] = (
    "Abstract",
    "1. Dataset and provenance",
    "2. Label construction",
    "3. Leakage controls and temporal validation",
    "4. Models",
    "5. Predictive results",
    "6. Calibration results",
    "7. Execution-aware sensitivity",
    "8. Ablations and robustness",
    "9. Systems benchmarks",
    "10. Failure cases and warnings",
    "11. Limitations",
    "12. Reproducibility commands",
)

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)


class PaperReportSummary(BaseModel):
    """Machine-readable summary of a report build."""

    model_config = _MODEL_CONFIG

    created_at: datetime
    report_path: str
    summary_path: str
    experiment_dir: str
    ablation_dir: str | None = None
    systems_dir: str | None = None
    sections_written: list[str]
    artefacts_used: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    fixture_or_smoke_run: bool
    builder_version: str

    @field_validator(
        "report_path",
        "summary_path",
        "experiment_dir",
        "builder_version",
    )
    @classmethod
    def _validate_required_string(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("paper report summary paths must be non-empty")
        return value.strip()

    @field_validator("ablation_dir", "systems_dir")
    @classmethod
    def _validate_optional_string(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("optional report summary paths must be non-empty")
        return value.strip()

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class PaperReportInspection(BaseModel):
    """Read-only inspection of a generated empirical report."""

    model_config = _MODEL_CONFIG

    report_path: str
    summary_path: str | None = None
    sections_detected: list[str] = Field(default_factory=list)
    artefacts_used_count: int = 0
    warnings_count: int = 0
    fixture_or_smoke_run: bool | None = None

    @field_validator("report_path")
    @classmethod
    def _validate_report_path(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("report_path must be non-empty")
        return value.strip()


@dataclass
class _ReportData:
    experiment_dir: Path
    report_path: Path
    summary_path: Path
    manifest: DataManifest
    results: ExperimentResults
    config: Mapping[str, Any] | None
    runner_summary: Mapping[str, Any] | None
    preparation_summary: Mapping[str, Any] | None
    label_summary: Mapping[str, Any] | None
    split_summary: Mapping[str, Any] | None
    plot_summary: Mapping[str, Any] | None
    model_card_text: str | None
    predictions_rows: list[dict[str, str]]
    calibration_rows: list[dict[str, str]]
    execution_rows: list[dict[str, str]]
    ablation_dir: Path | None
    ablation_summary: Mapping[str, Any] | None
    ablation_rows: list[dict[str, str]]
    systems_dir: Path | None
    systems_summary: Mapping[str, Any] | None
    systems_environment: Mapping[str, Any] | None
    systems_rows: list[dict[str, str]]
    validation_warnings: list[str]
    artefacts_used: list[str]
    warnings: list[str]
    fixture_or_smoke_run: bool


def build_paper_report(
    experiment_dir: Path,
    out_path: Path,
    *,
    ablation_dir: Path | None = None,
    systems_dir: Path | None = None,
    overwrite: bool = False,
) -> PaperReportSummary:
    """Build a Markdown empirical report from stored artefacts only.

    Parameters
    ----------
    experiment_dir:
        Completed paper experiment directory. Required and validated against the
        existing experiment artefact contract.
    out_path:
        Markdown report path to write.
    ablation_dir:
        Optional paper ablation output directory.
    systems_dir:
        Optional systems benchmark output directory.
    overwrite:
        When ``False``, existing report or summary paths are left untouched.
    """
    resolved_experiment = Path(experiment_dir)
    resolved_report = Path(out_path)
    summary_path = _summary_path_for(resolved_report)

    if resolved_report.exists() and resolved_report.is_dir():
        raise IsADirectoryError(f"report output path is a directory: {resolved_report}")
    if not overwrite and resolved_report.exists():
        raise FileExistsError(
            "refusing to overwrite existing paper report; "
            f"pass overwrite=True to replace it: {resolved_report}"
        )
    if not overwrite and summary_path.exists():
        raise FileExistsError(
            "refusing to overwrite existing paper report summary; "
            f"pass overwrite=True to replace it: {summary_path}"
        )

    data = _load_report_data(
        experiment_dir=resolved_experiment,
        report_path=resolved_report,
        summary_path=summary_path,
        ablation_dir=Path(ablation_dir) if ablation_dir is not None else None,
        systems_dir=Path(systems_dir) if systems_dir is not None else None,
    )
    _refuse_fixture_report_under_public_reports(data)

    markdown = _render_report(data)
    resolved_report.parent.mkdir(parents=True, exist_ok=True)
    resolved_report.write_text(markdown, encoding="utf-8")

    summary = PaperReportSummary(
        created_at=datetime.now(UTC),
        report_path=_display_path(resolved_report),
        summary_path=_display_path(summary_path),
        experiment_dir=_display_path(resolved_experiment),
        ablation_dir=(
            _display_path(data.ablation_dir) if data.ablation_dir is not None else None
        ),
        systems_dir=(
            _display_path(data.systems_dir) if data.systems_dir is not None else None
        ),
        sections_written=list(_REPORT_SECTION_TITLES),
        artefacts_used=list(data.artefacts_used),
        warnings=list(data.warnings),
        fixture_or_smoke_run=data.fixture_or_smoke_run,
        builder_version=PAPER_REPORT_BUILDER_VERSION,
    )
    summary_path.write_text(stable_json_dumps(summary), encoding="utf-8")
    return summary


def inspect_paper_report(report_path: Path) -> PaperReportInspection:
    """Inspect a generated empirical report and its optional summary JSON."""
    resolved_report = Path(report_path)
    if not resolved_report.exists():
        raise FileNotFoundError(f"paper report does not exist: {resolved_report}")
    if not resolved_report.is_file():
        raise IsADirectoryError(f"paper report path is not a file: {resolved_report}")

    text = resolved_report.read_text(encoding="utf-8")
    sections = _detect_report_sections(text)
    summary_path = _summary_path_for(resolved_report)
    summary_payload: Mapping[str, Any] | None = None
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            summary_payload = payload

    artefacts_used_count = 0
    warnings_count = 0
    fixture_or_smoke_run: bool | None = None
    if summary_payload is not None:
        artefacts = summary_payload.get("artefacts_used")
        warnings = summary_payload.get("warnings")
        raw_fixture = summary_payload.get("fixture_or_smoke_run")
        artefacts_used_count = len(artefacts) if isinstance(artefacts, list) else 0
        warnings_count = len(warnings) if isinstance(warnings, list) else 0
        if isinstance(raw_fixture, bool):
            fixture_or_smoke_run = raw_fixture

    return PaperReportInspection(
        report_path=_display_path(resolved_report),
        summary_path=_display_path(summary_path) if summary_path.is_file() else None,
        sections_detected=sections,
        artefacts_used_count=artefacts_used_count,
        warnings_count=warnings_count,
        fixture_or_smoke_run=fixture_or_smoke_run,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_report_data(
    *,
    experiment_dir: Path,
    report_path: Path,
    summary_path: Path,
    ablation_dir: Path | None,
    systems_dir: Path | None,
) -> _ReportData:
    if not experiment_dir.exists():
        raise FileNotFoundError(f"paper experiment directory does not exist: {experiment_dir}")
    if not experiment_dir.is_dir():
        raise NotADirectoryError(f"paper experiment path is not a directory: {experiment_dir}")

    validation = validate_experiment_directory(experiment_dir, include_plots=True)
    if not validation.is_valid:
        missing = ", ".join(validation.missing_required) or "invalid required artefacts"
        raise ValueError(
            "paper experiment artefacts failed validation; "
            f"missing or invalid required artefacts: {missing}"
        )

    artefacts_used: list[str] = []
    warnings = _unique_strings(validation.warnings)
    validation_warnings = list(warnings)

    manifest = load_data_manifest(experiment_dir)
    results = load_results(experiment_dir)
    _record_artefact(experiment_dir / "data_manifest.json", artefacts_used)
    _record_artefact(experiment_dir / "results.json", artefacts_used)

    config = _read_yaml_mapping(
        experiment_dir / "config.yaml",
        artefacts_used=artefacts_used,
        warnings=warnings,
    )
    runner_summary = _read_json_mapping(
        experiment_dir / "runner_summary.json",
        artefacts_used=artefacts_used,
        warnings=warnings,
    )
    preparation_dir = experiment_dir / "preparation"
    preparation_summary = _read_json_mapping(
        preparation_dir / "preparation_summary.json",
        artefacts_used=artefacts_used,
        warnings=warnings,
    )
    label_summary = _read_json_mapping(
        preparation_dir / "label_summary.json",
        artefacts_used=artefacts_used,
        warnings=warnings,
    )
    split_summary = _read_json_mapping(
        preparation_dir / "split_summary.json",
        artefacts_used=artefacts_used,
        warnings=warnings,
    )
    plot_summary = _read_json_mapping(
        experiment_dir / "plot_summary.json",
        artefacts_used=artefacts_used,
        warnings=warnings,
    )
    model_card_text = _read_text(
        experiment_dir / "model_card.md",
        artefacts_used=artefacts_used,
        warnings=warnings,
    )
    _record_optional_file(experiment_dir / "confusion_matrix.json", artefacts_used)

    predictions_rows, _ = _read_csv_rows(
        experiment_dir / "predictions.csv",
        artefacts_used=artefacts_used,
        warnings=warnings,
    )
    calibration_rows, _ = _read_csv_rows(
        experiment_dir / "calibration_bins.csv",
        artefacts_used=artefacts_used,
        warnings=warnings,
    )
    execution_rows, _ = _read_csv_rows(
        experiment_dir / "execution_sensitivity.csv",
        artefacts_used=artefacts_used,
        warnings=warnings,
    )

    ablation_summary: Mapping[str, Any] | None = None
    ablation_rows: list[dict[str, str]] = []
    if ablation_dir is not None:
        _validate_optional_dir(ablation_dir, label="paper ablation")
        ablation_summary = _read_json_mapping(
            ablation_dir / "ablation_summary.json",
            artefacts_used=artefacts_used,
            warnings=warnings,
        )
        _record_optional_file(ablation_dir / "ablation_manifest.json", artefacts_used)
        ablation_rows, _ = _read_csv_rows(
            ablation_dir / "ablation_results.csv",
            artefacts_used=artefacts_used,
            warnings=warnings,
        )
        for relative_path in _string_list(
            ablation_summary.get("reports_written") if ablation_summary else None
        ):
            _record_optional_file(ablation_dir / relative_path, artefacts_used)

    systems_summary: Mapping[str, Any] | None = None
    systems_environment: Mapping[str, Any] | None = None
    systems_rows: list[dict[str, str]] = []
    if systems_dir is not None:
        _validate_optional_dir(systems_dir, label="systems benchmark")
        systems_summary = _read_json_mapping(
            systems_dir / "system_benchmark_summary.json",
            artefacts_used=artefacts_used,
            warnings=warnings,
        )
        systems_environment = _read_json_mapping(
            systems_dir / "environment.json",
            artefacts_used=artefacts_used,
            warnings=warnings,
        )
        systems_rows, _ = _read_csv_rows(
            systems_dir / "system_benchmark_results.csv",
            artefacts_used=artefacts_used,
            warnings=warnings,
        )
        for relative_path in _string_list(
            systems_summary.get("reports_written") if systems_summary else None
        ):
            _record_optional_file(systems_dir / relative_path, artefacts_used)

    fixture_or_smoke_run = _detect_fixture_or_smoke(
        experiment_dir=experiment_dir,
        manifest=manifest,
        runner_summary=runner_summary,
        model_card_text=model_card_text,
        ablation_summary=ablation_summary,
        systems_summary=systems_summary,
        systems_environment=systems_environment,
    )
    _collect_structured_warnings(
        warnings=warnings,
        results=results,
        runner_summary=runner_summary,
        plot_summary=plot_summary,
        ablation_summary=ablation_summary,
        systems_summary=systems_summary,
        systems_rows=systems_rows,
        ablation_rows=ablation_rows,
    )

    return _ReportData(
        experiment_dir=experiment_dir,
        report_path=report_path,
        summary_path=summary_path,
        manifest=manifest,
        results=results,
        config=config,
        runner_summary=runner_summary,
        preparation_summary=preparation_summary,
        label_summary=label_summary,
        split_summary=split_summary,
        plot_summary=plot_summary,
        model_card_text=model_card_text,
        predictions_rows=predictions_rows,
        calibration_rows=calibration_rows,
        execution_rows=execution_rows,
        ablation_dir=ablation_dir,
        ablation_summary=ablation_summary,
        ablation_rows=ablation_rows,
        systems_dir=systems_dir,
        systems_summary=systems_summary,
        systems_environment=systems_environment,
        systems_rows=systems_rows,
        validation_warnings=validation_warnings,
        artefacts_used=_unique_strings(artefacts_used),
        warnings=_unique_strings(warnings),
        fixture_or_smoke_run=fixture_or_smoke_run,
    )


def _validate_optional_dir(path: Path, *, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} path is not a directory: {path}")


def _read_json_mapping(
    path: Path,
    *,
    artefacts_used: list[str],
    warnings: list[str],
) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"{_display_path(path)} could not be read as JSON: {exc}")
        return None
    if not isinstance(payload, Mapping):
        warnings.append(f"{_display_path(path)} was skipped because it is not a JSON object")
        return None
    _record_artefact(path, artefacts_used)
    return dict(payload)


def _read_yaml_mapping(
    path: Path,
    *,
    artefacts_used: list[str],
    warnings: list[str],
) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        warnings.append(f"{_display_path(path)} could not be read as YAML: {exc}")
        return None
    if not isinstance(payload, Mapping):
        warnings.append(f"{_display_path(path)} was skipped because it is not a YAML mapping")
        return None
    _record_artefact(path, artefacts_used)
    return dict(payload)


def _read_text(
    path: Path,
    *,
    artefacts_used: list[str],
    warnings: list[str],
) -> str | None:
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        warnings.append(f"{_display_path(path)} could not be read: {exc}")
        return None
    _record_artefact(path, artefacts_used)
    return text


def _read_csv_rows(
    path: Path,
    *,
    artefacts_used: list[str],
    warnings: list[str],
) -> tuple[list[dict[str, str]], int]:
    if not path.is_file():
        return [], 0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [
                {str(key): "" if value is None else str(value) for key, value in row.items()}
                for row in reader
            ]
    except (OSError, csv.Error) as exc:
        warnings.append(f"{_display_path(path)} could not be read as CSV: {exc}")
        return [], 0
    _record_artefact(path, artefacts_used)
    return rows, len(rows)


def _record_optional_file(path: Path, artefacts_used: list[str]) -> None:
    if path.is_file():
        _record_artefact(path, artefacts_used)


def _record_artefact(path: Path, artefacts_used: list[str]) -> None:
    label = _display_path(path)
    if label not in artefacts_used:
        artefacts_used.append(label)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_report(data: _ReportData) -> str:
    lines: list[str] = ["# ChronosLOB Empirical Report", ""]
    lines.extend(_render_abstract(data))
    lines.extend(_render_dataset(data))
    lines.extend(_render_labels(data))
    lines.extend(_render_validation(data))
    lines.extend(_render_models(data))
    lines.extend(_render_predictive(data))
    lines.extend(_render_calibration(data))
    lines.extend(_render_execution(data))
    lines.extend(_render_ablations(data))
    lines.extend(_render_systems(data))
    lines.extend(_render_warnings(data))
    lines.extend(_render_limitations(data))
    lines.extend(_render_reproducibility(data))
    return "\n".join(lines).rstrip() + "\n"


def _render_abstract(data: _ReportData) -> list[str]:
    results = data.results
    runner = data.runner_summary or {}
    models_run = _string_list(runner.get("models_run")) or [
        result.model_name for result in results.model_results
    ]
    lines = ["## Abstract", ""]
    lines.append(
        "This empirical report summarises stored artefacts for "
        f"`{results.experiment_name}` on task `{results.task_name}`."
    )
    lines.append(
        "It covers predictive results, calibration results, execution-aware "
        "sensitivity, ablation results and systems benchmarks where those "
        "artefacts were supplied."
    )
    lines.append(
        "The target research question is whether order-book representations "
        "can improve short-horizon market-state forecasting."
    )
    lines.append(
        "Evidence is bounded by leakage-safe validation, calibration analysis "
        "and explicit execution assumptions."
    )
    lines.append(
        "This report only records evidence present on disk and does not "
        "present trading or execution-system claims."
    )
    lines.append(
        f"Successful model entries in the main experiment: {_format_list(models_run)}."
    )
    if data.fixture_or_smoke_run:
        lines.append(
            "This is a smoke report generated from fixture or smoke artefacts; "
            "it is not benchmark evidence."
        )
    lines.append("")
    return lines


def _render_dataset(data: _ReportData) -> list[str]:
    manifest = data.manifest
    runner = data.runner_summary or {}
    environment = _mapping_value(runner.get("environment"))
    limitations = manifest.notes or _UNAVAILABLE
    lines = ["## 1. Dataset and provenance", ""]
    rows = [
        ("dataset name", manifest.dataset_name),
        ("dataset version", manifest.dataset_version),
        ("dataset variant", manifest.dataset_variant),
        ("source kind", str(manifest.source_kind.value)),
        ("source path", _safe_source_path(manifest.source_path)),
        ("source checksum", manifest.source_sha256),
        ("row count", manifest.row_count),
        ("event count", manifest.event_count),
        ("feature count", manifest.feature_count),
        ("runner data source kind", runner.get("data_source_kind")),
        ("Python", environment.get("python") if environment else None),
        ("platform", environment.get("platform") if environment else None),
        ("package version", environment.get("package_version") if environment else None),
    ]
    lines.extend(_key_value_table(rows))
    lines.append("")
    lines.append(f"Limitations recorded in provenance: {_format_text(limitations)}")
    lines.append("")
    return lines


def _render_labels(data: _ReportData) -> list[str]:
    config = data.config or {}
    label_summary = data.label_summary or {}
    prep = data.preparation_summary or {}
    manifest = data.manifest
    class_counts = _mapping_value(label_summary.get("class_counts"))
    distinct = _string_list(label_summary.get("distinct_classes"))
    label_mapping = _mapping_value(config.get("label_mapping"))
    if label_mapping is None:
        label_mapping = _mapping_value(label_summary.get("label_mapping"))

    lines = ["## 2. Label construction", ""]
    rows = [
        ("task name", prep.get("task_name") or data.results.task_name),
        ("label name", prep.get("label_name") or manifest.label_name),
        ("horizon", prep.get("horizon") or manifest.horizon),
        ("label source", label_summary.get("label_source")),
        ("label mapping", _format_mapping(label_mapping) if label_mapping else None),
        ("distinct classes", _format_list(distinct) if distinct else None),
        ("class counts", _format_mapping(class_counts) if class_counts else None),
    ]
    lines.extend(_key_value_table(rows))
    lines.append("")
    lines.append(
        "Leakage details: label construction is reported from the config snapshot "
        "and preparation artefacts. Any unavailable label detail is marked as not "
        "available rather than inferred."
    )
    lines.append("")
    return lines


def _render_validation(data: _ReportData) -> list[str]:
    runner = data.runner_summary or {}
    split = (
        data.split_summary
        or _mapping_value(runner.get("split_summary"))
        or _mapping_value(runner.get("split_counts"))
        or {}
    )
    metadata = _mapping_value(runner.get("model_metadata")) or {}
    train_only_entries = _train_only_metadata(metadata)
    split_method = split.get("split_method") or runner.get("split_method")
    lines = ["## 3. Leakage controls and temporal validation", ""]
    rows = [
        ("split design", runner.get("split_name") or data.results.config_summary.split_name),
        ("split method", split_method),
        ("split column", split.get("split_column")),
        ("official train rows", split.get("official_train_rows")),
        ("official test rows", split.get("official_test_rows")),
        (
            "official train start/end",
            _range_text(
                split,
                "official_train_start_index",
                "official_train_end_index",
            ),
        ),
        (
            "official test start/end",
            _range_text(
                split,
                "official_test_start_index",
                "official_test_end_index",
            ),
        ),
        (
            "validation fraction within official train",
            split.get("validation_fraction_within_train"),
        ),
        ("total rows", split.get("n_rows")),
        ("train rows", split.get("n_train")),
        ("validation rows", split.get("n_validation")),
        ("test rows", split.get("n_test")),
        ("train start/end", _range_text(split, "train_start_index", "train_end_index")),
        (
            "validation start/end",
            _range_text(split, "validation_start_index", "validation_end_index"),
        ),
        ("test start/end", _range_text(split, "test_start_index", "test_end_index")),
    ]
    lines.extend(_key_value_table(rows))
    lines.append("")
    if train_only_entries:
        lines.append(
            "Stored model metadata records train-only preprocessing or tokenisation for:"
        )
        for entry in train_only_entries:
            lines.append(f"- `{entry}`")
    else:
        lines.append(
            "Train-only preprocessing detail is not available in stored metadata for "
            "this report."
        )
    lines.append(
        "The experiment directory passed the required artefact validation contract "
        "before this report was written."
    )
    lines.append("")
    return lines


def _render_models(data: _ReportData) -> list[str]:
    runner = data.runner_summary or {}
    requested = _string_list(runner.get("requested_models"))
    successful = _string_list(runner.get("models_run")) or [
        result.model_name for result in data.results.model_results
    ]
    skipped = _mapping_list(runner.get("skipped_models"))
    neural_settings = _mapping_value(runner.get("neural_settings")) or {}
    planned = _string_list(
        neural_settings.get("planned_models")
    )
    lines = ["## 4. Models", ""]
    rows = [
        ("requested models", _format_list(requested) if requested else None),
        ("successful models", _format_list(successful) if successful else None),
        ("skipped models", _format_skips(skipped) if skipped else "none"),
    ]
    lines.extend(_key_value_table(rows))
    lines.append("")
    if "deeplob_style" in successful or "deeplob_style" in requested:
        lines.append(
            "`deeplob_style` is reported as a compact DeepLOB-style supervised "
            "baseline in the stored runner metadata, not as an exact external-paper "
            "reproduction."
        )
    if (
        "transformer" in successful
        or "transformer" in requested
        or "matrix_transformer" in successful
        or "matrix_transformer" in requested
    ):
        lines.append(
            "`transformer` and `matrix_transformer` are supervised transformer "
            "baselines over the normalised FI-2010 matrix path; raw order-book "
            "schemas remain strict and are not used to coerce z-score rows."
        )
    if "ssl_transformer" in requested or "ssl_transformer" in planned:
        has_ssl_result = any(
            result.model_name == "ssl_transformer"
            for result in data.results.model_results
        )
        if has_ssl_result:
            lines.append(
                "`ssl_transformer` appears in stored model results; interpret it "
                "only through those stored metrics."
            )
        else:
            lines.append(
                "`ssl_transformer` is present only as planned or skipped metadata; "
                "no SSL model result is reported here."
            )
    lines.append("")
    return lines


def _render_predictive(data: _ReportData) -> list[str]:
    rows = []
    for result in data.results.model_results:
        metrics = result.metrics
        rows.append(
            [
                result.model_name,
                result.split,
                result.horizon,
                metrics.get("accuracy"),
                metrics.get("macro_f1"),
                metrics.get("n_samples"),
                metrics.get("class_count_test"),
                _format_list(result.warnings) if result.warnings else "none",
            ]
        )
    lines = ["## 5. Predictive results", ""]
    lines.append(
        "The table below is populated only from `results.json`; missing metrics are "
        "marked as not available."
    )
    lines.append("")
    lines.extend(
        _markdown_table(
            [
                "model",
                "split",
                "horizon",
                "accuracy",
                "macro F1",
                "test count",
                "class count test",
                "warnings",
            ],
            rows,
        )
    )
    lines.append("")
    if (data.experiment_dir / "plots" / "confusion_matrix.png").is_file():
        lines.append(
            _plot_markdown(
                data,
                data.experiment_dir / "plots" / "confusion_matrix.png",
                alt="Confusion matrix",
            )
        )
        lines.append("")
    elif (data.experiment_dir / "confusion_matrix.json").is_file():
        lines.append(
            "`confusion_matrix.json` is present; `plots/confusion_matrix.png` is "
            "not available."
        )
        lines.append("")
    return lines


def _render_calibration(data: _ReportData) -> list[str]:
    metric_rows = []
    bin_summary = _calibration_bin_summary(data.calibration_rows)
    for result in data.results.model_results:
        metrics = result.metrics
        model_summary = bin_summary.get(result.model_name, {})
        metric_rows.append(
            [
                result.model_name,
                result.split,
                metrics.get("expected_calibration_error"),
                metrics.get("brier_score"),
                metrics.get("mean_confidence"),
                model_summary.get("rows"),
                model_summary.get("positive_bins"),
            ]
        )

    lines = ["## 6. Calibration results", ""]
    if data.calibration_rows:
        lines.append(
            f"`calibration_bins.csv` is present with {len(data.calibration_rows)} rows."
        )
    else:
        lines.append("`calibration_bins.csv` is not available for this report.")
    lines.append("")
    lines.extend(
        _markdown_table(
            [
                "model",
                "split",
                "ECE",
                "Brier score",
                "mean confidence",
                "calibration rows",
                "positive bins",
            ],
            metric_rows,
        )
    )
    lines.append("")
    if (data.experiment_dir / "plots" / "reliability_curve.png").is_file():
        lines.append(
            _plot_markdown(
                data,
                data.experiment_dir / "plots" / "reliability_curve.png",
                alt="Reliability curve",
            )
        )
        lines.append("")
    else:
        lines.append("Reliability plot: not available.")
        lines.append("")
    return lines


def _render_execution(data: _ReportData) -> list[str]:
    summary_rows = _execution_summary_rows(data.execution_rows)
    runner = data.runner_summary or {}
    evidence = _mapping_value(runner.get("evidence")) or {}
    execution_config = _mapping_value(evidence.get("execution_sensitivity_config")) or {}
    lines = ["## 7. Execution-aware sensitivity", ""]
    if data.execution_rows:
        lines.append(
            f"`execution_sensitivity.csv` is present with {len(data.execution_rows)} rows."
        )
    else:
        lines.append("`execution_sensitivity.csv` is not available for this report.")
    lines.append("")
    config_rows = [
        (
            "confidence thresholds",
            _format_list(execution_config.get("confidence_thresholds")),
        ),
        ("cost assumptions", _format_list(execution_config.get("cost_bps"))),
        ("latency assumptions", _format_list(execution_config.get("latency_steps"))),
        (
            "return proxy",
            _format_mapping(_mapping_value(execution_config.get("return_proxy"))),
        ),
    ]
    lines.extend(_key_value_table(config_rows))
    lines.append("")
    if summary_rows:
        lines.extend(
            _markdown_table(
                [
                    "model",
                    "rows",
                    "thresholds",
                    "cost bps",
                    "latency steps",
                    "max eligible",
                    "net proxy min",
                    "net proxy max",
                ],
                summary_rows,
            )
        )
        lines.append("")
    else:
        lines.append("No execution-aware sensitivity rows are available.")
        lines.append("")
    if (data.experiment_dir / "plots" / "cost_sensitivity.png").is_file():
        lines.append(
            _plot_markdown(
                data,
                data.experiment_dir / "plots" / "cost_sensitivity.png",
                alt="Cost sensitivity",
            )
        )
        lines.append("")
    else:
        lines.append("Cost-sensitivity plot: not available.")
        lines.append("")
    lines.append(
        "These rows are proxy sensitivity under explicit assumptions, not a "
        "live or deployment-ready execution system."
    )
    lines.append("")
    return lines


def _render_ablations(data: _ReportData) -> list[str]:
    lines = ["## 8. Ablations and robustness", ""]
    if data.ablation_dir is None:
        lines.append("Ablation directory: not supplied.")
        lines.append("")
        return lines
    summary = data.ablation_summary or {}
    lines.extend(
        _key_value_table(
            [
                ("ablation set", summary.get("ablation_set")),
                ("ablations run", _format_list(summary.get("ablations_run"))),
                ("ablations skipped", _format_list(summary.get("ablations_skipped"))),
                ("models requested", _format_list(summary.get("models_requested"))),
                ("fixture run", summary.get("is_fixture")),
            ]
        )
    )
    report_paths = _string_list(summary.get("reports_written"))
    lines.append("")
    if report_paths:
        lines.append("Ablation report paths:")
        for relative_path in report_paths:
            lines.append(f"- `{_md_escape(relative_path)}`")
        lines.append("")

    key_rows = _select_ablation_rows(data.ablation_rows)
    if key_rows:
        lines.extend(
            _markdown_table(
                ["ablation", "status", "model", "metric", "value", "source or warning"],
                key_rows,
            )
        )
        lines.append("")
    else:
        lines.append("Ablation result rows: not available.")
        lines.append("")
    ssl_rows = [
        item
        for item in _mapping_list(summary.get("results"))
        if "ssl" in str(item.get("name", "")).lower()
        or "ssl" in str(item.get("ablation_type", "")).lower()
    ]
    if ssl_rows:
        lines.append("SSL pretraining ablation status:")
        for row in ssl_rows:
            status = _format_text(row.get("status"))
            reason = _format_text(row.get("reason"))
            lines.append(f"- `{_md_escape(str(row.get('name', 'ssl')))}`: {status}; {reason}")
        lines.append("")
    return lines


def _render_systems(data: _ReportData) -> list[str]:
    lines = ["## 9. Systems benchmarks", ""]
    if data.systems_dir is None:
        lines.append("Systems benchmark directory: not supplied.")
        lines.append("")
        return lines
    summary = data.systems_summary or {}
    environment = data.systems_environment or {}
    lines.extend(
        _key_value_table(
            [
                ("benchmark set", summary.get("benchmark_set")),
                ("benchmarks run", _format_list(summary.get("benchmarks_run"))),
                ("benchmarks skipped", _format_list(summary.get("benchmarks_skipped"))),
                ("models requested", _format_list(summary.get("models_requested"))),
                ("data source kind", environment.get("data_source_kind")),
                ("data row count", environment.get("data_row_count")),
                ("platform", environment.get("platform")),
            ]
        )
    )
    lines.append("")
    key_rows = _select_system_rows(data.systems_rows)
    if key_rows:
        lines.extend(
            _markdown_table(
                ["benchmark", "status", "metric", "value", "unit", "rows", "warning"],
                key_rows,
            )
        )
        lines.append("")
    else:
        lines.append("Systems benchmark result rows: not available.")
        lines.append("")
    if _detect_systems_smoke(summary, environment):
        lines.append(
            "Systems measurements supplied here are smoke measurements; fixture "
            "timings are not benchmark evidence."
        )
        lines.append("")
    return lines


def _render_warnings(data: _ReportData) -> list[str]:
    lines = ["## 10. Failure cases and warnings", ""]
    if not data.warnings:
        lines.append("- none")
        lines.append("")
        return lines

    grouped = _group_warnings(data.warnings)
    lines.append("Warning summary:")
    lines.append("")
    lines.extend(
        _markdown_table(
            ["warning group", "occurrences"],
            [[group["label"], group["count"]] for group in grouped],
        )
    )
    lines.append("")
    lines.append("Detailed warning appendix:")
    lines.append("")
    for group in grouped:
        count = cast(int, group["count"])
        label = str(group["label"])
        examples = [str(item) for item in cast(list[str], group["examples"])]
        if count == 1 and examples:
            lines.append(f"- {_format_text(examples[0])}")
            continue
        lines.append(f"- {label}: {count} occurrence(s).")
        if bool(group["show_examples"]) and examples:
            lines.append(f"  Representative detail: {_format_text(examples[0])}")
    lines.append("")
    return lines


def _render_limitations(data: _ReportData) -> list[str]:
    lines = ["## 11. Limitations", ""]
    limitations = [
        "Real benchmark evidence depends on a user-supplied local FI-2010-style file.",
        "FI-2010 is a fixed historical benchmark and may not represent other "
        "instruments, venues or regimes.",
        "Execution-aware sensitivity is a simplified proxy analysis with explicit "
        "costs and latency assumptions.",
        "There is no broker integration or order placement in this report.",
        "There is no production market impact model.",
        "SSL results are absent unless a stored SSL model result is genuinely "
        "present in the supplied artefacts.",
    ]
    if data.fixture_or_smoke_run:
        limitations.append(
            "Fixture smoke runs are infrastructure checks only and are not evidence."
        )
    for limitation in limitations:
        lines.append(f"- {limitation}")
    lines.append("")
    return lines


def _render_reproducibility(data: _ReportData) -> list[str]:
    runner = data.runner_summary or {}
    experiment = _display_cli_path(data.experiment_dir)
    config = _display_cli_path(data.experiment_dir / "config.yaml")
    data_path = _display_cli_path(Path(str(runner.get("data_path", data.manifest.source_path))))
    models = _string_list(runner.get("requested_models"))
    report = _display_cli_path(data.report_path)

    lines = ["## 12. Reproducibility commands", ""]
    commands = [
        [
            "python",
            "-m",
            "chronoslob.cli",
            "prepare-fi2010-benchmark",
            "--config",
            config,
            "--data-path",
            data_path,
            "--out",
            _display_cli_path(data.experiment_dir / "preparation"),
        ],
        [
            "python",
            "-m",
            "chronoslob.cli",
            "run-paper-experiment",
            "--config",
            config,
            "--data-path",
            data_path,
            "--out",
            experiment,
            "--models",
            ",".join(models) if models else "majority",
            "--overwrite",
        ],
        [
            "python",
            "-m",
            "chronoslob.cli",
            "build-paper-plots",
            "--experiment",
            experiment,
            "--overwrite",
        ],
        [
            "python",
            "-m",
            "chronoslob.cli",
            "inspect-paper-experiment",
            "--experiment",
            experiment,
        ],
    ]
    if data.ablation_dir is not None and data.ablation_summary is not None:
        commands.append(_ablation_command(data))
    if data.systems_dir is not None and data.systems_summary is not None:
        commands.append(_systems_command(data))
    build_command = [
        "python",
        "-m",
        "chronoslob.cli",
        "build-paper-report",
        "--experiment",
        experiment,
    ]
    if data.ablation_dir is not None:
        build_command.extend(["--ablations", _display_cli_path(data.ablation_dir)])
    if data.systems_dir is not None:
        build_command.extend(["--systems", _display_cli_path(data.systems_dir)])
    build_command.extend(["--out", report, "--overwrite"])
    commands.append(build_command)

    for command in commands:
        lines.extend(["```bash", _shell_join(command), "```", ""])
    return lines


def _ablation_command(data: _ReportData) -> list[str]:
    summary = data.ablation_summary or {}
    models = _string_list(summary.get("models_requested"))
    return [
        "python",
        "-m",
        "chronoslob.cli",
        "run-paper-ablations",
        "--config",
        _display_cli_path(
            Path(str(summary.get("base_config", data.experiment_dir / "config.yaml")))
        ),
        "--data-path",
        _display_cli_path(Path(str(summary.get("data_path", data.manifest.source_path)))),
        "--out",
        _display_cli_path(data.ablation_dir or Path("runs/paper_ablation_smoke")),
        "--models",
        ",".join(models) if models else "majority,logistic",
        "--ablation-set",
        str(summary.get("ablation_set", "smoke")),
        "--overwrite",
    ]


def _systems_command(data: _ReportData) -> list[str]:
    summary = data.systems_summary or {}
    models = _string_list(summary.get("models_requested"))
    return [
        "python",
        "-m",
        "chronoslob.cli",
        "run-system-benchmarks",
        "--config",
        _display_cli_path(
            Path(str(summary.get("config_path", data.experiment_dir / "config.yaml")))
        ),
        "--data-path",
        _display_cli_path(Path(str(summary.get("data_path", data.manifest.source_path)))),
        "--out",
        _display_cli_path(data.systems_dir or Path("runs/system_benchmark_smoke")),
        "--benchmark-set",
        str(summary.get("benchmark_set", "smoke")),
        "--models",
        ",".join(models) if models else "majority,logistic",
        "--overwrite",
    ]


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def _calibration_bin_summary(rows: Sequence[Mapping[str, str]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for row in rows:
        model = row.get("model_name", _UNAVAILABLE)
        item = summary.setdefault(model, {"rows": 0, "positive_bins": 0})
        item["rows"] += 1
        count = _safe_float(row.get("count"))
        if count is not None and count > 0:
            item["positive_bins"] += 1
    return summary


def _execution_summary_rows(rows: Sequence[Mapping[str, str]]) -> list[list[Any]]:
    by_model: dict[str, dict[str, Any]] = {}
    for row in rows:
        model = row.get("model_name", _UNAVAILABLE)
        item = by_model.setdefault(
            model,
            {
                "rows": 0,
                "thresholds": set(),
                "costs": set(),
                "latencies": set(),
                "eligible": [],
                "net": [],
            },
        )
        item["rows"] += 1
        _add_float_to_set(item["thresholds"], row.get("confidence_threshold"))
        _add_float_to_set(item["costs"], row.get("cost_bps"))
        _add_int_to_set(item["latencies"], row.get("latency_steps"))
        eligible = _safe_float(row.get("eligible_predictions"))
        if eligible is not None:
            item["eligible"].append(eligible)
        net = _safe_float(row.get("net_signal_return_proxy"))
        if net is not None:
            item["net"].append(net)

    rendered: list[list[Any]] = []
    for model in sorted(by_model):
        item = by_model[model]
        net_values = item["net"]
        eligible_values = item["eligible"]
        rendered.append(
            [
                model,
                item["rows"],
                _format_list(sorted(item["thresholds"])),
                _format_list(sorted(item["costs"])),
                _format_list(sorted(item["latencies"])),
                max(eligible_values) if eligible_values else None,
                min(net_values) if net_values else None,
                max(net_values) if net_values else None,
            ]
        )
    return rendered


def _select_ablation_rows(rows: Sequence[Mapping[str, str]]) -> list[list[Any]]:
    selected_metric_names = {
        "accuracy",
        "macro_f1",
        "expected_calibration_error",
        "net_signal_return_proxy",
        "status",
    }
    selected: list[list[Any]] = []
    for row in rows:
        metric = row.get("metric_name", "")
        status = row.get("status", "")
        if metric not in selected_metric_names and status != "skipped":
            continue
        selected.append(
            [
                row.get("ablation_name"),
                status,
                row.get("model_name"),
                metric,
                row.get("metric_value"),
                _shorten(
                    row.get("warning")
                    if status == "skipped"
                    else row.get("source_experiment")
                ),
            ]
        )
        if len(selected) >= 16:
            break
    return selected


def _select_system_rows(rows: Sequence[Mapping[str, str]]) -> list[list[Any]]:
    selected_metric_names = {
        "rows_per_second",
        "features_per_second",
        "elapsed_seconds",
        "prediction_rows",
        "artefact_count",
        "latency_ms_per_window",
        "peak_memory_mb",
        "status",
    }
    selected: list[list[Any]] = []
    for row in rows:
        metric = row.get("metric_name", "")
        status = row.get("status", "")
        if metric not in selected_metric_names and status != "skipped":
            continue
        selected.append(
            [
                row.get("benchmark_name"),
                status,
                metric,
                row.get("metric_value"),
                row.get("metric_unit"),
                row.get("rows"),
                "" if row.get("warning") == _NO_WARNING else _shorten(row.get("warning")),
            ]
        )
        if len(selected) >= 16:
            break
    return selected


def _train_only_metadata(metadata: Mapping[str, Any]) -> list[str]:
    entries: list[str] = []
    for model_name, raw in metadata.items():
        model_payload = _mapping_value(raw)
        if model_payload is None:
            continue
        standardisation = _mapping_value(model_payload.get("standardisation"))
        if standardisation is not None and standardisation.get("fit_split") == "train":
            entries.append(f"{model_name} standardisation")
        if model_payload.get("tokenisation_fit_split") == "train":
            entries.append(f"{model_name} tokenisation")
        window_policy = _mapping_value(model_payload.get("window_policy"))
        if window_policy is not None and window_policy.get("windows_stay_inside_split") is True:
            entries.append(f"{model_name} split-contained windows")
    return entries


def _collect_structured_warnings(
    *,
    warnings: list[str],
    results: ExperimentResults,
    runner_summary: Mapping[str, Any] | None,
    plot_summary: Mapping[str, Any] | None,
    ablation_summary: Mapping[str, Any] | None,
    systems_summary: Mapping[str, Any] | None,
    systems_rows: Sequence[Mapping[str, str]],
    ablation_rows: Sequence[Mapping[str, str]],
) -> None:
    for result in results.model_results:
        for warning in result.warnings:
            warnings.append(f"model {result.model_name}: {warning}")
    if runner_summary is not None:
        for skip in _mapping_list(runner_summary.get("skipped_models")):
            model = skip.get("model_name", "unknown")
            reason = skip.get("reason", _UNAVAILABLE)
            warnings.append(f"requested model {model!r} was skipped: {reason}")
        evidence = _mapping_value(runner_summary.get("evidence")) or {}
        for key in ("calibration_warning", "execution_warning"):
            value = evidence.get(key)
            if isinstance(value, str) and value.strip():
                warnings.append(value.strip())
        plots = _mapping_value(runner_summary.get("plots")) or {}
        warnings.extend(_string_list(plots.get("warnings")))
    if plot_summary is not None:
        warnings.extend(_string_list(plot_summary.get("warnings")))
    if ablation_summary is not None:
        warnings.extend(_string_list(ablation_summary.get("warnings")))
        for ablation_result in _mapping_list(ablation_summary.get("results")):
            name = ablation_result.get("name", "unknown")
            reason = ablation_result.get("reason")
            if isinstance(reason, str) and reason.strip():
                warnings.append(f"ablation {name!r}: {reason.strip()}")
            for warning in _string_list(ablation_result.get("warnings")):
                warnings.append(f"ablation {name!r}: {warning}")
    for row in ablation_rows:
        warning = row.get("warning", "")
        if warning and warning != _NO_WARNING:
            warnings.append(f"ablation row {row.get('ablation_name', 'unknown')}: {warning}")
    if systems_summary is not None:
        warnings.extend(_string_list(systems_summary.get("warnings")))
    for row in systems_rows:
        warning = row.get("warning", "")
        if warning and warning != _NO_WARNING:
            warnings.append(
                f"systems row {row.get('benchmark_name', 'unknown')}: {warning}"
            )


def _warning_group_label(warning: str) -> tuple[str, bool]:
    lowered = warning.casefold()
    if (
        "feature_generation_speed measured normalised fi-2010 matrix feature throughput"
        in lowered
        and "raw order-book snapshot reconstruction was not used" in lowered
    ):
        return (
            "feature_generation_speed measured normalised FI-2010 matrix "
            "feature throughput; raw order-book snapshot reconstruction was "
            "not used",
            False,
        )
    if "optional artefact missing: plots/" in lowered:
        marker = "optional artefact missing: "
        start = lowered.find(marker)
        if start >= 0:
            path = warning[start + len(marker) :].split()[0]
            return f"optional plot artefact missing: {path}", False
        return "optional plot artefacts missing", False
    if "orderbooklevel" in lowered and "quantity must be non-negative" in lowered:
        return (
            "raw OrderBookLevel conversion rejected normalised negative quantities",
            True,
        )
    if "no traceable runner support for ssl" in lowered:
        return "SSL pretraining remains unsupported in the paper runner", False
    if "feature_patterns produced no matching feature columns" in lowered:
        return "feature-pattern ablation matched no columns", True
    if "regime breakdown skipped" in lowered:
        return "regime breakdown plot skipped because no genuine regime data exists", False
    return _shorten(warning, limit=180), True


def _group_warnings(warnings: Sequence[str]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    order: list[str] = []
    for raw in warnings:
        text = _format_text(raw)
        if not text or text == _UNAVAILABLE:
            continue
        label, show_examples = _warning_group_label(text)
        if label not in grouped:
            grouped[label] = {
                "label": label,
                "count": 0,
                "examples": [],
                "show_examples": show_examples,
            }
            order.append(label)
        item = grouped[label]
        item["count"] = int(cast(int, item["count"])) + 1
        examples = cast(list[str], item["examples"])
        if text not in examples:
            examples.append(text)
    return [grouped[label] for label in order]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _key_value_table(rows: Sequence[tuple[str, Any]]) -> list[str]:
    return _markdown_table(
        ["field", "value"],
        [[key, _format_text(value)] for key, value in rows],
    )


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(_md_escape(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    if not rows:
        lines.append(
            "| "
            + " | ".join(_UNAVAILABLE if index == 0 else "" for index, _ in enumerate(headers))
            + " |"
        )
        return lines
    for row in rows:
        padded = list(row) + [None] * max(0, len(headers) - len(row))
        lines.append(
            "| "
            + " | ".join(_md_escape(_format_text(value)) for value in padded[: len(headers)])
            + " |"
        )
    return lines


def _format_text(value: Any) -> str:
    if value is None:
        return _UNAVAILABLE
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if not math.isfinite(numeric):
            return _UNAVAILABLE
        if numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:.6g}"
    if isinstance(value, (list, tuple, set)):
        return _format_list(list(value))
    if isinstance(value, Mapping):
        return _format_mapping(value)
    text = str(value).strip()
    return text if text else _UNAVAILABLE


def _format_list(value: Any) -> str:
    if value is None:
        return _UNAVAILABLE
    if isinstance(value, str):
        return value if value else _UNAVAILABLE
    if not isinstance(value, Sequence):
        return _format_text(value)
    items = [_format_text(item) for item in value]
    return ", ".join(items) if items else _UNAVAILABLE


def _format_mapping(value: Mapping[str, Any] | None) -> str:
    if not value:
        return _UNAVAILABLE
    parts = [f"{key}={_format_text(value[key])}" for key in sorted(value)]
    return ", ".join(parts) if parts else _UNAVAILABLE


def _format_skips(skipped: Sequence[Mapping[str, Any]]) -> str:
    parts = []
    for item in skipped:
        model = _format_text(item.get("model_name"))
        reason = _format_text(item.get("reason"))
        parts.append(f"{model}: {reason}")
    return "; ".join(parts) if parts else "none"


def _md_escape(value: str) -> str:
    return value.replace("\n", " ").replace("|", "\\|")


def _shorten(value: Any, *, limit: int = 140) -> str:
    text = _format_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _add_float_to_set(target: set[float], value: Any) -> None:
    numeric = _safe_float(value)
    if numeric is not None:
        target.add(numeric)


def _add_int_to_set(target: set[int], value: Any) -> None:
    numeric = _safe_float(value)
    if numeric is not None:
        target.add(int(numeric))


def _mapping_value(
    value: Any,
    default: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    return default


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _range_text(payload: Mapping[str, Any], start_key: str, end_key: str) -> str:
    start = payload.get(start_key)
    end = payload.get(end_key)
    if start is None or end is None:
        return _UNAVAILABLE
    return f"{_format_text(start)} to {_format_text(end)}"


def _unique_strings(values: Sequence[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = str(raw).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _plot_markdown(data: _ReportData, path: Path, *, alt: str) -> str:
    _record_artefact(path, data.artefacts_used)
    relative = _relative_markdown_path(path, base=data.report_path.parent)
    return f"![{_md_escape(alt)}]({_md_escape(relative)})"


def _relative_markdown_path(path: Path, *, base: Path) -> str:
    try:
        relative = os.path.relpath(path.resolve(), start=base.resolve())
    except (OSError, ValueError):
        return _display_path(path)
    return Path(relative).as_posix()


def _safe_source_path(value: str) -> str:
    if not value:
        return _UNAVAILABLE
    return _display_path(Path(value))


def _display_path(path: Path) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except (OSError, ValueError):
        return candidate.as_posix()


def _display_cli_path(path: Path) -> str:
    return _display_path(Path(path))


def _shell_join(parts: Sequence[str]) -> str:
    escaped: list[str] = []
    for part in parts:
        text = str(part)
        if not text:
            escaped.append('""')
        elif any(char.isspace() for char in text):
            escaped.append('"' + text.replace('"', '\\"') + '"')
        else:
            escaped.append(text)
    return " ".join(escaped)


def _summary_path_for(report_path: Path) -> Path:
    return report_path.with_name(f"{report_path.stem}_summary.json")


# ---------------------------------------------------------------------------
# Detection and safety
# ---------------------------------------------------------------------------


def _detect_fixture_or_smoke(
    *,
    experiment_dir: Path,
    manifest: DataManifest,
    runner_summary: Mapping[str, Any] | None,
    model_card_text: str | None,
    ablation_summary: Mapping[str, Any] | None,
    systems_summary: Mapping[str, Any] | None,
    systems_environment: Mapping[str, Any] | None,
) -> bool:
    candidates: list[str] = [
        _display_path(experiment_dir),
        str(manifest.source_kind.value),
        manifest.source_path,
    ]
    if runner_summary is not None:
        candidates.extend(
            [
                str(runner_summary.get("is_fixture", "")),
                str(runner_summary.get("data_source_kind", "")),
                str(runner_summary.get("data_path", "")),
                str(runner_summary.get("output_dir", "")),
            ]
        )
    if model_card_text is not None:
        candidates.append(model_card_text)
    if ablation_summary is not None:
        candidates.extend(
            [
                str(ablation_summary.get("is_fixture", "")),
                str(ablation_summary.get("data_source_kind", "")),
                str(ablation_summary.get("ablation_set", "")),
                " ".join(_string_list(ablation_summary.get("warnings"))),
            ]
        )
    if systems_summary is not None:
        candidates.extend(
            [
                str(systems_summary.get("benchmark_set", "")),
                " ".join(_string_list(systems_summary.get("warnings"))),
            ]
        )
    if systems_environment is not None:
        candidates.extend(
            [
                str(systems_environment.get("data_source_kind", "")),
                str(systems_environment.get("data_source_path", "")),
            ]
        )
    lowered = "\n".join(candidates).lower().replace("\\", "/")
    return any(marker in lowered for marker in _SMOKE_MARKERS)


def _detect_systems_smoke(
    summary: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> bool:
    candidates = [
        str(summary.get("benchmark_set", "")),
        " ".join(_string_list(summary.get("warnings"))),
        str(environment.get("data_source_kind", "")),
        str(environment.get("data_source_path", "")),
    ]
    lowered = "\n".join(candidates).lower().replace("\\", "/")
    return any(marker in lowered for marker in _SMOKE_MARKERS)


def _refuse_fixture_report_under_public_reports(data: _ReportData) -> None:
    if not data.fixture_or_smoke_run:
        return
    reports_dir = project_root() / "reports"
    try:
        data.report_path.resolve().relative_to(reports_dir.resolve())
    except ValueError:
        return
    raise ValueError(
        "refusing to write a fixture or smoke empirical report under reports/; "
        "use an ignored output path such as runs/"
    )


def _detect_report_sections(text: str) -> list[str]:
    sections: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            sections.append(line.removeprefix("## ").strip())
    return sections

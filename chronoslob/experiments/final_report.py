"""Build the final FI-2010 empirical report from stored artefacts only."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from chronoslob.experiments.manifests import sha256_file, stable_json_dumps
from chronoslob.utils.paths import project_root

__all__ = [
    "FINAL_EMPIRICAL_REPORT_BUILDER_VERSION",
    "FinalEmpiricalReportSummary",
    "build_final_empirical_report",
]

FINAL_EMPIRICAL_REPORT_BUILDER_VERSION = "phase-k/final-empirical-report/v1"

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)

_SECTION_TITLES: tuple[str, ...] = (
    "Evidence Snapshot",
    "Research Question",
    "Dataset And Split Protocol",
    "Model Families",
    "Main Result Table",
    "Uncertainty Summary",
    "Ablation Summary",
    "Execution-Aware Proxy Summary",
    "External Benchmark Context",
    "What This Proves",
    "What This Does Not Prove",
    "Limitations",
    "Artefact Traceability",
    "Reproduction Commands",
)

_REQUIRED_CLASSICAL_FILES = ("summary.json", "results_summary.csv")
_REQUIRED_NEURAL_FILES = ("summary.json", "results_summary.csv")
_REQUIRED_UNCERTAINTY_FILES = (
    "summary.json",
    "metric_confidence_intervals.csv",
    "model_ranking.csv",
)
_OPTIONAL_ABLATION_FILES = (
    "summary.json",
    "model_class_ablation.csv",
    "feature_group_ablation.csv",
    "horizon_ablation.csv",
    "calibration_threshold_ablation.csv",
    "execution_cost_latency_ablation.csv",
    "skipped_ablations.json",
)
_OPTIONAL_EXECUTION_FILES = (
    "summary.json",
    "degradation_summary.csv",
    "confidence_threshold_summary.csv",
    "turnover_summary.csv",
    "fill_assumption_summary.csv",
    "adverse_selection_summary.csv",
    "skipped_diagnostics.json",
)
_OPTIONAL_EXTERNAL_FILES = (
    "benchmark_context.json",
    "protocol_comparison.csv",
)


class FinalEmpiricalReportSummary(BaseModel):
    """Machine-readable summary of a final empirical report build."""

    model_config = _MODEL_CONFIG

    generated_at: datetime
    git_commit: str | None = None
    report_path: str
    summary_path: str
    input_artefact_paths: dict[str, str]
    input_file_hashes: dict[str, str]
    headline_metrics: dict[str, Any]
    skipped_sections: list[str] = Field(default_factory=list)
    missing_sections: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    sections_written: list[str]
    builder_version: str

    @field_validator("generated_at")
    @classmethod
    def _validate_generated_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware")
        return value

    @field_validator("report_path", "summary_path", "builder_version")
    @classmethod
    def _validate_non_empty_string(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("final report summary fields must be non-empty strings")
        return value.strip()


@dataclass
class _OptionalArtefacts:
    directory: Path | None = None
    summary: dict[str, Any] | None = None
    json_payloads: dict[str, dict[str, Any]] = field(default_factory=dict)
    csv_rows: dict[str, list[dict[str, str]]] = field(default_factory=dict)


@dataclass
class _FinalReportData:
    classical_dir: Path
    neural_dir: Path
    uncertainty_dir: Path
    ablation: _OptionalArtefacts
    execution: _OptionalArtefacts
    external: _OptionalArtefacts
    report_path: Path
    summary_path: Path
    generated_at: datetime
    git_commit: str | None
    classical_summary: dict[str, Any]
    classical_results: list[dict[str, str]]
    neural_summary: dict[str, Any]
    neural_results: list[dict[str, str]]
    uncertainty_summary: dict[str, Any]
    uncertainty_intervals: list[dict[str, str]]
    uncertainty_ranking: list[dict[str, str]]
    input_artefact_paths: dict[str, str]
    input_file_hashes: dict[str, str]
    warnings: list[str]
    skipped_sections: list[str]
    missing_sections: list[str]
    recorded_skips: list[str]


def build_final_empirical_report(
    *,
    classical_dir: Path,
    neural_dir: Path,
    uncertainty_dir: Path,
    out_path: Path,
    ablation_dir: Path | None = None,
    execution_dir: Path | None = None,
    external_dir: Path | None = None,
    overwrite: bool = False,
) -> FinalEmpiricalReportSummary:
    """Build the final Markdown empirical report and companion JSON summary."""
    report_path = Path(out_path)
    summary_path = _summary_path_for(report_path)
    if report_path.exists() and report_path.is_dir():
        raise IsADirectoryError(f"final report output path is a directory: {report_path}")
    if not overwrite and report_path.exists():
        raise FileExistsError(
            "refusing to overwrite existing final empirical report; "
            f"pass overwrite=True to replace it: {report_path}"
        )
    if not overwrite and summary_path.exists():
        raise FileExistsError(
            "refusing to overwrite existing final empirical report summary; "
            f"pass overwrite=True to replace it: {summary_path}"
        )

    data = _load_final_report_data(
        classical_dir=Path(classical_dir),
        neural_dir=Path(neural_dir),
        uncertainty_dir=Path(uncertainty_dir),
        ablation_dir=Path(ablation_dir) if ablation_dir is not None else None,
        execution_dir=Path(execution_dir) if execution_dir is not None else None,
        external_dir=Path(external_dir) if external_dir is not None else None,
        report_path=report_path,
        summary_path=summary_path,
    )
    headline_metrics = _headline_metrics(data)
    markdown = _render_report(data, headline_metrics)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown, encoding="utf-8")

    summary = FinalEmpiricalReportSummary(
        generated_at=data.generated_at,
        git_commit=data.git_commit,
        report_path=_display_path(report_path),
        summary_path=_display_path(summary_path),
        input_artefact_paths=dict(sorted(data.input_artefact_paths.items())),
        input_file_hashes=dict(sorted(data.input_file_hashes.items())),
        headline_metrics=headline_metrics,
        skipped_sections=list(data.skipped_sections),
        missing_sections=list(data.missing_sections),
        warnings=list(data.warnings),
        sections_written=list(_SECTION_TITLES),
        builder_version=FINAL_EMPIRICAL_REPORT_BUILDER_VERSION,
    )
    summary_path.write_text(stable_json_dumps(summary), encoding="utf-8")
    return summary


def _load_final_report_data(
    *,
    classical_dir: Path,
    neural_dir: Path,
    uncertainty_dir: Path,
    ablation_dir: Path | None,
    execution_dir: Path | None,
    external_dir: Path | None,
    report_path: Path,
    summary_path: Path,
) -> _FinalReportData:
    input_paths: dict[str, str] = {}
    file_hashes: dict[str, str] = {}
    warnings: list[str] = []
    skipped_sections: list[str] = []
    missing_sections: list[str] = []
    recorded_skips: list[str] = []

    classical = _require_directory(classical_dir, "classical")
    neural = _require_directory(neural_dir, "neural")
    uncertainty = _require_directory(uncertainty_dir, "uncertainty")
    input_paths["classical_dir"] = _display_path(classical)
    input_paths["neural_dir"] = _display_path(neural)
    input_paths["uncertainty_dir"] = _display_path(uncertainty)

    _require_files(classical, _REQUIRED_CLASSICAL_FILES, "classical")
    _require_files(neural, _REQUIRED_NEURAL_FILES, "neural")
    _require_files(uncertainty, _REQUIRED_UNCERTAINTY_FILES, "uncertainty")

    classical_summary = _read_json(
        classical / "summary.json",
        "classical_summary",
        input_paths,
        file_hashes,
    )
    classical_results = _read_csv(
        classical / "results_summary.csv",
        "classical_results_summary",
        input_paths,
        file_hashes,
    )
    neural_summary = _read_json(
        neural / "summary.json",
        "neural_summary",
        input_paths,
        file_hashes,
    )
    neural_results = _read_csv(
        neural / "results_summary.csv",
        "neural_results_summary",
        input_paths,
        file_hashes,
    )
    uncertainty_summary = _read_json(
        uncertainty / "summary.json",
        "uncertainty_summary",
        input_paths,
        file_hashes,
    )
    uncertainty_intervals = _read_csv(
        uncertainty / "metric_confidence_intervals.csv",
        "uncertainty_metric_confidence_intervals",
        input_paths,
        file_hashes,
    )
    uncertainty_ranking = _read_csv(
        uncertainty / "model_ranking.csv",
        "uncertainty_model_ranking",
        input_paths,
        file_hashes,
    )

    ablation = _load_optional_artefacts(
        directory=ablation_dir,
        label="ablations",
        section_title="Ablation Summary",
        expected_files=_OPTIONAL_ABLATION_FILES,
        input_paths=input_paths,
        file_hashes=file_hashes,
        warnings=warnings,
        skipped_sections=skipped_sections,
        missing_sections=missing_sections,
    )
    execution = _load_optional_artefacts(
        directory=execution_dir,
        label="execution",
        section_title="Execution-Aware Proxy Summary",
        expected_files=_OPTIONAL_EXECUTION_FILES,
        input_paths=input_paths,
        file_hashes=file_hashes,
        warnings=warnings,
        skipped_sections=skipped_sections,
        missing_sections=missing_sections,
    )
    external = _load_optional_artefacts(
        directory=external_dir,
        label="external",
        section_title="External Benchmark Context",
        expected_files=_OPTIONAL_EXTERNAL_FILES,
        input_paths=input_paths,
        file_hashes=file_hashes,
        warnings=warnings,
        skipped_sections=skipped_sections,
        missing_sections=missing_sections,
    )
    recorded_skips.extend(_extract_recorded_skips(ablation, "ablation"))
    recorded_skips.extend(_extract_recorded_skips(execution, "execution"))

    return _FinalReportData(
        classical_dir=classical,
        neural_dir=neural,
        uncertainty_dir=uncertainty,
        ablation=ablation,
        execution=execution,
        external=external,
        report_path=report_path,
        summary_path=summary_path,
        generated_at=datetime.now(UTC),
        git_commit=_current_git_commit(),
        classical_summary=classical_summary,
        classical_results=classical_results,
        neural_summary=neural_summary,
        neural_results=neural_results,
        uncertainty_summary=uncertainty_summary,
        uncertainty_intervals=uncertainty_intervals,
        uncertainty_ranking=uncertainty_ranking,
        input_artefact_paths=input_paths,
        input_file_hashes=file_hashes,
        warnings=warnings,
        skipped_sections=skipped_sections + recorded_skips,
        missing_sections=missing_sections,
        recorded_skips=recorded_skips,
    )


def _load_optional_artefacts(
    *,
    directory: Path | None,
    label: str,
    section_title: str,
    expected_files: Sequence[str],
    input_paths: dict[str, str],
    file_hashes: dict[str, str],
    warnings: list[str],
    skipped_sections: list[str],
    missing_sections: list[str],
) -> _OptionalArtefacts:
    if directory is None:
        message = f"{label} input not supplied"
        warnings.append(f"{message}; {section_title} will be marked skipped.")
        skipped_sections.append(section_title)
        missing_sections.append(message)
        return _OptionalArtefacts()
    candidate = Path(directory)
    if not candidate.exists():
        message = f"{label} artefact directory missing: {_display_path(candidate)}"
        warnings.append(f"{message}; {section_title} will be marked skipped.")
        skipped_sections.append(section_title)
        missing_sections.append(message)
        return _OptionalArtefacts()
    if not candidate.is_dir():
        message = f"{label} artefact path is not a directory: {_display_path(candidate)}"
        warnings.append(f"{message}; {section_title} will be marked skipped.")
        skipped_sections.append(section_title)
        missing_sections.append(message)
        return _OptionalArtefacts()

    input_paths[f"{label}_dir"] = _display_path(candidate)
    artefacts = _OptionalArtefacts(directory=candidate)
    for filename in expected_files:
        path = candidate / filename
        key = f"{label}_{Path(filename).stem}"
        if not path.is_file():
            warnings.append(f"Optional artefact missing: {_display_path(path)}")
            missing_sections.append(f"{section_title}: missing {_display_path(path)}")
            continue
        if path.suffix.lower() == ".json":
            payload = _read_json(path, key, input_paths, file_hashes)
            artefacts.json_payloads[filename] = payload
            if filename in {"summary.json", "benchmark_context.json"}:
                artefacts.summary = payload
        elif path.suffix.lower() == ".csv":
            artefacts.csv_rows[filename] = _read_csv(path, key, input_paths, file_hashes)
        else:
            _record_file(path, key, input_paths, file_hashes)
    return artefacts


def _render_report(data: _FinalReportData, headline_metrics: dict[str, Any]) -> str:
    lines: list[str] = [
        "# ChronosLOB Final Empirical Report",
        "",
        "Generated from stored FI-2010 artefacts only. No model training is run by this builder.",
        "",
    ]
    lines.extend(_section("Evidence Snapshot", _render_evidence_snapshot(data, headline_metrics)))
    lines.extend(_section("Research Question", _render_research_question()))
    lines.extend(_section("Dataset And Split Protocol", _render_dataset_protocol(data)))
    lines.extend(_section("Model Families", _render_model_families(data)))
    lines.extend(_section("Main Result Table", _render_main_results(data)))
    lines.extend(_section("Uncertainty Summary", _render_uncertainty(data)))
    lines.extend(_section("Ablation Summary", _render_ablations(data)))
    lines.extend(_section("Execution-Aware Proxy Summary", _render_execution(data)))
    lines.extend(_section("External Benchmark Context", _render_external(data)))
    lines.extend(_section("What This Proves", _render_what_this_proves()))
    lines.extend(_section("What This Does Not Prove", _render_what_this_does_not_prove()))
    lines.extend(_section("Limitations", _render_limitations(data)))
    lines.extend(_section("Artefact Traceability", _render_traceability(data)))
    lines.extend(_section("Reproduction Commands", _render_reproduction_commands()))
    return "\n".join(lines).rstrip() + "\n"


def _section(title: str, body: Sequence[str]) -> list[str]:
    return [f"## {title}", "", *body, ""]


def _render_evidence_snapshot(
    data: _FinalReportData,
    headline_metrics: Mapping[str, Any],
) -> list[str]:
    classical = cast(dict[str, Any], headline_metrics.get("best_classical", {}))
    neural = cast(dict[str, Any], headline_metrics.get("best_neural", {}))
    execution_status = (
        "proxy diagnostics loaded"
        if data.execution.directory is not None and data.execution.summary is not None
        else "skipped"
    )
    external_status = (
        "protocol context loaded"
        if data.external.directory is not None and data.external.summary is not None
        else "skipped"
    )
    rows = [
        ("generated_at", data.generated_at.isoformat()),
        ("git_commit", data.git_commit or "not available"),
        ("classical_scope", "multi-fold classical results"),
        (
            "best_classical_test_macro_f1",
            _metric_snapshot(classical, include_lookback=False),
        ),
        ("neural_scope", "reduced-scope supervised neural, single-seed"),
        ("best_neural_test_macro_f1", _metric_snapshot(neural, include_lookback=True)),
        ("execution_scope", f"{execution_status}; metrics are proxy diagnostics"),
        (
            "external_scope",
            f"{external_status}; protocol context only, not ranking claims",
        ),
        ("report_path", _display_path(data.report_path)),
        ("summary_path", _display_path(data.summary_path)),
    ]
    return _markdown_table(("field", "value"), rows)


def _render_research_question() -> list[str]:
    return [
        "Can stored FI-2010 artefacts support a traceable assessment of predictive "
        "mid-price direction performance, uncertainty, robustness, execution-aware "
        "proxy diagnostics and external protocol context?",
    ]


def _render_dataset_protocol(data: _FinalReportData) -> list[str]:
    external_protocol = _nested_mapping(
        data.external.summary,
        "chronoslob_protocol",
    )
    split_protocol = _mapping_str(
        external_protocol,
        "split_protocol",
        default="official split column with validation carved from train only",
    )
    variant = _mapping_str(
        external_protocol,
        "variant",
        default="NoAuction ZScore",
    )
    rows = [
        ("dataset", _mapping_str(data.classical_summary, "dataset_name", default="FI-2010")),
        ("variant", variant),
        ("task", _mapping_str(data.classical_summary, "task_name", default="midprice_direction")),
        ("target_horizon", _mapping_str(data.classical_summary, "target_horizon")),
        ("split_protocol", split_protocol),
        (
            "folds",
            _join_values(
                _mapping_list(data.classical_summary, "folds_completed")
                or _mapping_list(external_protocol, "folds"),
            ),
        ),
        (
            "classical_protocol",
            "multi-fold; one stored classical seed across completed folds",
        ),
        (
            "neural_protocol",
            "reduced-scope supervised neural; one seed and one lookback in stored artefacts",
        ),
    ]
    return _markdown_table(("field", "value"), rows)


def _render_model_families(data: _FinalReportData) -> list[str]:
    rows = [
        (
            "classical",
            _join_values(_mapping_list(data.classical_summary, "models_requested")),
            "multi-fold stored fold summaries",
        ),
        (
            "neural",
            _join_values(_mapping_list(data.neural_summary, "models_requested")),
            "reduced-scope, single-seed, lookback "
            + _join_values(_mapping_list(data.neural_summary, "lookbacks")),
        ),
    ]
    return _markdown_table(("family", "models", "scope"), rows)


def _render_main_results(data: _FinalReportData) -> list[str]:
    rows: list[tuple[str, ...]] = []
    for row in _sorted_test_rows(data.classical_results, lookback_column=None):
        rows.append(
            (
                row.get("model_name", ""),
                "classical",
                "multi-fold",
                _format_metric_pair(
                    _row_float(row, "macro_f1_mean"),
                    _row_float(row, "macro_f1_std"),
                ),
                _format_float(_row_float(row, "accuracy_mean")),
                _format_float(_row_float(row, "mcc_mean")),
                row.get("fold_count", ""),
                _empty_to_na(row.get("run_count", "")),
                "n/a",
            )
        )
    for row in _sorted_test_rows(data.neural_results, lookback_column="lookback"):
        rows.append(
            (
                row.get("model_name", ""),
                "supervised neural",
                "reduced-scope, single-seed",
                _format_metric_pair(
                    _row_float(row, "macro_f1_mean"),
                    _row_float(row, "macro_f1_std"),
                ),
                _format_float(_row_float(row, "accuracy_mean")),
                _format_float(_row_float(row, "mcc_mean")),
                row.get("fold_count", ""),
                _empty_to_na(row.get("seed_count", "")),
                _empty_to_na(row.get("lookback", "")),
            )
        )
    intro = [
        "Classical rows are multi-fold. Neural rows are reduced-scope, single-seed "
        "supervised results and are not used here to assert superiority over the "
        "classical family.",
        "",
    ]
    table = _markdown_table(
        (
            "model",
            "family",
            "scope",
            "test macro-F1",
            "accuracy",
            "MCC",
            "folds",
            "seeds/runs",
            "lookback",
        ),
        rows,
    )
    return intro + table


def _render_uncertainty(data: _FinalReportData) -> list[str]:
    best_classical = _best_metric_row(data.classical_results, lookback_column=None)
    best_neural = _best_metric_row(data.neural_results, lookback_column="lookback")
    interval_rows = []
    for source, best in (("classical", best_classical), ("neural", best_neural)):
        if best is None:
            continue
        interval = _find_interval(
            data.uncertainty_intervals,
            source=source,
            model_name=best.get("model_name", ""),
            lookback=best.get("lookback", ""),
            split="test",
            metric="macro_f1",
        )
        if interval is None:
            continue
        interval_rows.append(
            (
                source,
                best.get("model_name", ""),
                _empty_to_na(best.get("lookback", "")),
                _empty_to_na(interval.get("n_folds", "")),
                _empty_to_na(interval.get("n_seeds", "")),
                _format_float(_row_float(interval, "mean")),
                _format_float(_row_float(interval, "ci_lower")),
                _format_float(_row_float(interval, "ci_upper")),
                _format_float(_row_float(interval, "bootstrap_lower")),
                _format_float(_row_float(interval, "bootstrap_upper")),
            )
        )
    rows = _markdown_table(
        (
            "source",
            "model",
            "lookback",
            "folds",
            "seeds",
            "mean",
            "CI lower",
            "CI upper",
            "bootstrap lower",
            "bootstrap upper",
        ),
        interval_rows,
    )
    notes = [
        "Seed variance is not available in the stored evidence; intervals are "
        "fold-level diagnostics.",
        "",
    ]
    return notes + rows


def _render_ablations(data: _FinalReportData) -> list[str]:
    if data.ablation.directory is None or data.ablation.summary is None:
        return ["Skipped: ablation artefacts were not supplied or were unavailable."]

    summary = data.ablation.summary
    family_rows = [
        (
            "feature_groups",
            str(len(data.ablation.csv_rows.get("feature_group_ablation.csv", []))),
            _delta_range(data.ablation.csv_rows.get("feature_group_ablation.csv", [])),
        ),
        (
            "model_class",
            str(len(data.ablation.csv_rows.get("model_class_ablation.csv", []))),
            _delta_range(data.ablation.csv_rows.get("model_class_ablation.csv", [])),
        ),
        (
            "horizon",
            str(len(data.ablation.csv_rows.get("horizon_ablation.csv", []))),
            _delta_range(data.ablation.csv_rows.get("horizon_ablation.csv", [])),
        ),
        (
            "calibration",
            str(len(data.ablation.csv_rows.get("calibration_threshold_ablation.csv", []))),
            "ECE diagnostics",
        ),
        (
            "execution",
            str(len(data.ablation.csv_rows.get("execution_cost_latency_ablation.csv", []))),
            "cost/latency proxy diagnostics",
        ),
    ]
    lines = [
        "Stored ablations are diagnostic stress checks; skipped ablations remain explicit.",
        "",
    ]
    lines.extend(
        _markdown_table(
            ("family", "stored rows", "summary"),
            family_rows,
        )
    )
    lines.append("")
    rows = [
        ("families_run", _join_values(_mapping_list(summary, "families_run"))),
        ("families_skipped", _join_values(_mapping_list(summary, "families_skipped"))),
        ("checkpoints_written", str(summary.get("checkpoints_written", "not available"))),
    ]
    lines.extend(_markdown_table(("field", "value"), rows))
    if data.recorded_skips:
        lines.append("")
        lines.extend(_markdown_table(("recorded skip",), [(skip,) for skip in data.recorded_skips]))
    return lines


def _render_execution(data: _FinalReportData) -> list[str]:
    if data.execution.directory is None or data.execution.summary is None:
        return ["Skipped: execution artefacts were not supplied or were unavailable."]

    rows = []
    for row in data.execution.csv_rows.get("degradation_summary.csv", []):
        rows.append(
            (
                row.get("model_name", ""),
                row.get("source", ""),
                row.get("status", ""),
                _format_float(_row_float(row, "statistical_value")),
                _format_float(_row_float(row, "base_proxy_value")),
                _format_float(_row_float(row, "exec_proxy_value")),
                _format_float(_row_float(row, "relative_degradation_proxy")),
            )
        )
    lines = [
        "Execution-aware metrics are proxy diagnostics only. They are not a backtest "
        "and make no profitability or tradability claim.",
        "",
    ]
    lines.extend(
        _markdown_table(
            (
                "model",
                "source",
                "status",
                "test macro-F1",
                "base proxy",
                "stress proxy",
                "relative degradation",
            ),
            rows,
        )
    )
    lines.append("")
    summary = data.execution.summary
    assumption_rows = [
        ("proxy_diagnostics", str(summary.get("proxy_diagnostics", "not available"))),
        ("fill_assumption", _mapping_str(summary, "fill_assumption")),
        ("checkpoints_required", str(summary.get("checkpoints_required", "not available"))),
        (
            "full_predictions_required",
            str(summary.get("full_predictions_required", "not available")),
        ),
    ]
    lines.extend(_markdown_table(("field", "value"), assumption_rows))
    return lines


def _render_external(data: _FinalReportData) -> list[str]:
    if data.external.directory is None or data.external.summary is None:
        return ["Skipped: external context artefacts were not supplied or were unavailable."]

    summary = data.external.summary
    references = summary.get("external_references")
    rows = []
    if isinstance(references, list):
        for item in references:
            if not isinstance(item, Mapping):
                continue
            rows.append(
                (
                    str(item.get("source_name", "")),
                    str(item.get("source_type", "")),
                    str(item.get("metric_values_included", "not available")),
                )
            )
    lines = [
        "External comparisons are protocol context, not ranking claims. No external "
        "numeric metrics are imported into this report.",
        "",
    ]
    lines.extend(_markdown_table(("source", "type", "numeric metrics included"), rows))
    return lines


def _render_what_this_proves() -> list[str]:
    return [
        "- The committed artefacts support a traceable multi-fold classical FI-2010 result.",
        "- The committed artefacts support reduced-scope, single-seed supervised neural evidence.",
        "- The uncertainty, ablation and proxy-diagnostic layers are generated from stored tables.",
        "- External references are used only to document protocol context.",
    ]


def _render_what_this_does_not_prove() -> list[str]:
    return [
        "- Profitability or tradability in deployed markets.",
        "- Production execution quality or market-impact realism.",
        "- Foundation-model status.",
        "- SSL or SOTA performance.",
        "- Neural superiority over the classical family.",
    ]


def _render_limitations(data: _FinalReportData) -> list[str]:
    classical_seed_count = len(_mapping_list(data.classical_summary, "seeds"))
    neural_seed_count = len(_mapping_list(data.neural_summary, "seeds"))
    rows = [
        ("classical_seed_count", str(classical_seed_count or "not available")),
        ("neural_seed_count", str(neural_seed_count or "not available")),
        (
            "neural_scope",
            "single seed and single lookback in stored reduced-scope artefacts",
        ),
        (
            "execution_scope",
            "proxy diagnostics only; queue, impact and venue mechanics are not modelled",
        ),
        (
            "external_scope",
            "protocol context only; no external numeric metrics are copied",
        ),
        (
            "prediction_checkpoint_policy",
            "full predictions and checkpoints are not required by this report builder",
        ),
    ]
    return _markdown_table(("limitation", "detail"), rows)


def _render_traceability(data: _FinalReportData) -> list[str]:
    rows = [
        (label, path, data.input_file_hashes.get(path, "directory"))
        for label, path in sorted(data.input_artefact_paths.items())
    ]
    return _markdown_table(("artefact", "path", "sha256"), rows)


def _render_reproduction_commands() -> list[str]:
    return [
        "```bash",
        "python -m chronoslob.cli build-final-empirical-report \\",
        "  --classical experiments/fi2010_multifold_classical \\",
        "  --neural experiments/fi2010_multifold_neural \\",
        "  --uncertainty experiments/fi2010_uncertainty \\",
        "  --ablations experiments/fi2010_brutal_ablations \\",
        "  --execution experiments/fi2010_execution_v2 \\",
        "  --external experiments/fi2010_external_context \\",
        "  --out reports/chronoslob_final_empirical_report.md \\",
        "  --overwrite",
        "",
        "python -m chronoslob.cli doctor",
        "python -m chronoslob.cli inspect-release-readiness",
        "python -m chronoslob.cli run-project-audit --strict",
        "python -m pytest",
        "python -m compileall -q chronoslob tests",
        "python -m ruff check .",
        "python -m mypy chronoslob",
        "```",
    ]


def _headline_metrics(data: _FinalReportData) -> dict[str, Any]:
    best_classical = _best_metric_row(data.classical_results, lookback_column=None)
    best_neural = _best_metric_row(data.neural_results, lookback_column="lookback")
    metrics: dict[str, Any] = {}
    if best_classical is not None:
        metrics["best_classical"] = _headline_from_row(
            best_classical,
            scope="multi-fold classical",
        )
    if best_neural is not None:
        metrics["best_neural"] = _headline_from_row(
            best_neural,
            scope="reduced-scope supervised neural",
        )
    degradation = _first_status_row(
        data.execution.csv_rows.get("degradation_summary.csv", []),
        status="ok",
        model_name=str(metrics.get("best_classical", {}).get("model_name", "")),
    )
    if degradation is not None:
        metrics["execution_proxy"] = {
            "model_name": degradation.get("model_name", ""),
            "base_proxy_metric": degradation.get("base_proxy_metric", ""),
            "base_proxy_value": _row_float(degradation, "base_proxy_value"),
            "exec_proxy_metric": degradation.get("exec_proxy_metric", ""),
            "exec_proxy_value": _row_float(degradation, "exec_proxy_value"),
            "relative_degradation_proxy": _row_float(
                degradation,
                "relative_degradation_proxy",
            ),
            "scope": "proxy diagnostics",
        }
    return metrics


def _headline_from_row(row: Mapping[str, str], *, scope: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_name": row.get("model_name", ""),
        "split": row.get("split", ""),
        "scope": scope,
        "fold_count": _row_int(row, "fold_count"),
        "run_count": _row_int(row, "run_count"),
        "accuracy_mean": _row_float(row, "accuracy_mean"),
        "accuracy_std": _row_float(row, "accuracy_std"),
        "macro_f1_mean": _row_float(row, "macro_f1_mean"),
        "macro_f1_std": _row_float(row, "macro_f1_std"),
        "mcc_mean": _row_float(row, "mcc_mean"),
        "mcc_std": _row_float(row, "mcc_std"),
    }
    if row.get("lookback"):
        payload["lookback"] = _row_int(row, "lookback")
    if row.get("seed_count"):
        payload["seed_count"] = _row_int(row, "seed_count")
    return payload


def _metric_snapshot(row: Mapping[str, Any], *, include_lookback: bool) -> str:
    model = str(row.get("model_name", "not available"))
    mean = _any_float(row.get("macro_f1_mean"))
    std = _any_float(row.get("macro_f1_std"))
    metric = _format_metric_pair(mean, std)
    extra = ""
    if include_lookback and row.get("lookback") is not None:
        extra = f", lookback {row.get('lookback')}"
    return f"{model}: {metric}{extra}"


def _best_metric_row(
    rows: Sequence[Mapping[str, str]],
    *,
    lookback_column: str | None,
) -> dict[str, str] | None:
    candidates = _sorted_test_rows(rows, lookback_column=lookback_column)
    return candidates[0] if candidates else None


def _sorted_test_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    lookback_column: str | None,
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for row in rows:
        if row.get("split") != "test":
            continue
        if _row_float(row, "macro_f1_mean") is None:
            continue
        copy = dict(row)
        if lookback_column is not None and lookback_column not in copy:
            copy[lookback_column] = ""
        candidates.append(copy)
    return sorted(
        candidates,
        key=lambda item: _row_float(item, "macro_f1_mean") or -math.inf,
        reverse=True,
    )


def _find_interval(
    rows: Sequence[Mapping[str, str]],
    *,
    source: str,
    model_name: str,
    lookback: str,
    split: str,
    metric: str,
) -> dict[str, str] | None:
    lookback_text = str(lookback).strip()
    for row in rows:
        if row.get("source") != source:
            continue
        if row.get("model_name") != model_name:
            continue
        if row.get("split") != split or row.get("metric") != metric:
            continue
        row_lookback = str(row.get("lookback", "")).strip()
        if lookback_text and row_lookback not in {lookback_text, f"{lookback_text}.0"}:
            continue
        if not lookback_text and row_lookback:
            continue
        return dict(row)
    return None


def _first_status_row(
    rows: Sequence[Mapping[str, str]],
    *,
    status: str,
    model_name: str,
) -> dict[str, str] | None:
    for row in rows:
        if row.get("status") != status:
            continue
        if model_name and row.get("model_name") != model_name:
            continue
        return dict(row)
    for row in rows:
        if row.get("status") == status:
            return dict(row)
    return None


def _delta_range(rows: Sequence[Mapping[str, str]]) -> str:
    values = [
        value
        for row in rows
        if row.get("split") == "test"
        and row.get("metric_name") == "macro_f1"
        and row.get("status") == "ok"
        for value in [_row_float(row, "delta_vs_baseline")]
        if value is not None
    ]
    if not values:
        return "not available"
    return f"{min(values):.4f} to {max(values):.4f} test macro-F1 delta"


def _extract_recorded_skips(artefacts: _OptionalArtefacts, prefix: str) -> list[str]:
    skips: list[str] = []
    for payload in artefacts.json_payloads.values():
        raw_skips = payload.get("skipped")
        if not isinstance(raw_skips, list):
            continue
        for item in raw_skips:
            if not isinstance(item, Mapping):
                continue
            name = (
                item.get("ablation_name")
                or item.get("diagnostic")
                or item.get("scope")
                or "unknown"
            )
            reason = item.get("skip_reason", "reason not recorded")
            skips.append(f"{prefix} {name}: {reason}")
    return skips


def _require_directory(path: Path, label: str) -> Path:
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"{label} artefact directory does not exist: {candidate}")
    if not candidate.is_dir():
        raise NotADirectoryError(f"{label} artefact path is not a directory: {candidate}")
    return candidate


def _require_files(directory: Path, filenames: Sequence[str], label: str) -> None:
    for filename in filenames:
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"{label} required artefact is missing: {path}"
            )


def _read_json(
    path: Path,
    key: str,
    input_paths: dict[str, str],
    file_hashes: dict[str, str],
) -> dict[str, Any]:
    _record_file(path, key, input_paths, file_hashes)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artefact must contain an object: {path}")
    return cast(dict[str, Any], payload)


def _read_csv(
    path: Path,
    key: str,
    input_paths: dict[str, str],
    file_hashes: dict[str, str],
) -> list[dict[str, str]]:
    _record_file(path, key, input_paths, file_hashes)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                str(column): "" if value is None else str(value)
                for column, value in row.items()
                if column is not None
            }
            for row in reader
        ]


def _record_file(
    path: Path,
    key: str,
    input_paths: dict[str, str],
    file_hashes: dict[str, str],
) -> None:
    display = _display_path(path)
    input_paths[key] = display
    file_hashes[display] = sha256_file(path)


def _summary_path_for(report_path: Path) -> Path:
    return report_path.with_name(f"{report_path.stem}_summary.json")


def _current_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root(),
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def _display_path(path: Path) -> str:
    candidate = Path(path)
    try:
        resolved = candidate.resolve(strict=False)
        root = project_root().resolve(strict=False)
        return resolved.relative_to(root).as_posix()
    except (OSError, ValueError):
        return candidate.as_posix()


def _markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> list[str]:
    rendered = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        rendered.append("| " + " | ".join(_escape_table_cell(cell) for cell in row) + " |")
    return rendered


def _escape_table_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _mapping_str(
    payload: Mapping[str, Any] | None,
    key: str,
    *,
    default: str = "not available",
) -> str:
    if payload is None:
        return default
    value = payload.get(key)
    if value is None:
        return default
    if isinstance(value, str):
        return value if value else default
    return str(value)


def _mapping_list(payload: Mapping[str, Any] | None, key: str) -> list[Any]:
    if payload is None:
        return []
    value = payload.get(key)
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return []


def _nested_mapping(
    payload: Mapping[str, Any] | None,
    key: str,
) -> Mapping[str, Any] | None:
    if payload is None:
        return None
    value = payload.get(key)
    if isinstance(value, Mapping):
        return value
    return None


def _join_values(values: Sequence[Any]) -> str:
    if not values:
        return "not available"
    return ", ".join(str(value) for value in values)


def _row_float(row: Mapping[str, str], key: str) -> float | None:
    return _any_float(row.get(key))


def _any_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _row_int(row: Mapping[str, str], key: str) -> int | None:
    value = _row_float(row, key)
    if value is None:
        return None
    return int(value)


def _format_float(value: float | None, *, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _format_metric_pair(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "n/a"
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {std:.4f}"


def _empty_to_na(value: str | None) -> str:
    if value is None:
        return "n/a"
    text = str(value).strip()
    return text if text else "n/a"

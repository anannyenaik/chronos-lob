"""Build the final FI-2010 empirical report from stored artefacts only."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import textwrap
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
    "Evidence Status Summary",
    "Evidence Pack Audit",
    "Research Question",
    "Dataset And Split Protocol",
    "Model Families",
    "Main Result Table",
    "Self-Supervised Pretraining",
    "Full Neural Grid",
    "Proper-Training Neural Subset",
    "Legacy Reduced-Scope Benchmark",
    "SSL Interpretation",
    "SSL Failure Analysis",
    "Second-Generation SSL Objective",
    "Figure Index",
    "Uncertainty Summary",
    "Ablation Summary",
    "Feature Ablation and Stability Analysis",
    "Execution-Aware Proxy Summary",
    "Forecasting versus Signal-Quality Gap",
    "External Benchmark Context",
    "Synthetic Event-Level Extension",
    "Real Event-Level L2 Replay Extension",
    "What This Supports",
    "What This Does Not Claim",
    "Limitations",
    "Artefact Traceability",
    "Reproduction Commands",
)

# SSL rows are admitted only when these artefacts exist and the recorded
# encoder-checkpoint SHA256 matches the checkpoint actually on disk.
_REQUIRED_SSL_FILES = (
    "summary.json",
    "results_summary.csv",
    "comparison_summary.csv",
)
_REQUIRED_FULL_GRID_FILES = (
    "summary.json",
    "results_summary.csv",
    "aggregate_summary.csv",
    "aggregate_summary.json",
    "ssl_comparison.csv",
    "failures.csv",
)
_REQUIRED_PROPER_TRAINING_FILES = (
    "summary.json",
    "config_snapshot.json",
    "results_summary.csv",
    "aggregate_summary.csv",
    "aggregate_summary.json",
    "training_curves_summary.csv",
    "ssl_comparison.csv",
    "failures.csv",
    "sha256_manifest.json",
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
_OPTIONAL_FEATURE_ABLATION_FILES = (
    "summary.json",
    "results_summary.csv",
    "aggregate_summary.csv",
    "feature_delta_summary.csv",
    "ablation_manifest.json",
    "failures.json",
)
_OPTIONAL_FEATURE_ABLATION_ANALYSIS_FILES = (
    "summary.json",
    "feature_ablation_analysis.md",
    "feature_delta_by_horizon.csv",
    "feature_delta_by_model.csv",
    "feature_delta_by_fold.csv",
    "feature_delta_by_seed.csv",
    "feature_group_stability.csv",
    "snapshot_order_flow_proxy_scope.csv",
    "feature_claim_assessment.json",
    "figure_manifest.json",
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
_OPTIONAL_EXECUTION_V3_FILES = (
    "summary.json",
    "execution_v3_manifest.json",
    "confidence_threshold_summary.csv",
    "confidence_threshold_aggregate.csv",
    "cost_sensitivity_summary.csv",
    "latency_sensitivity_summary.csv",
    "fill_assumption_summary.csv",
    "adverse_selection_summary.csv",
    "regime_execution_summary.csv",
    "skipped_diagnostics.json",
)
_OPTIONAL_EXECUTION_CENTREPIECE_FILES = (
    "execution_centrepiece.md",
    "centrepiece_summary.json",
    "forecasting_vs_signal_quality.csv",
    "confidence_threshold_tradeoff.csv",
    "metric_to_proxy_gap.csv",
    "latency_cost_gap.csv",
    "adverse_selection_by_confidence.csv",
    "execution_centrepiece_claim_assessment.json",
    "figure_manifest.json",
)
_OPTIONAL_EXTERNAL_FILES = (
    "benchmark_context.json",
    "protocol_comparison.csv",
)
_OPTIONAL_SYNTHETIC_FILES = (
    "summary.json",
    "synthetic_replay_quality.json",
    "synthetic_benchmark_summary.csv",
    "synthetic_regime_diagnostics.csv",
    "synthetic_claim_assessment.json",
)
_OPTIONAL_BINANCE_L2_FILES = (
    "summary.json",
    "replay_quality.json",
    "feature_summary.csv",
    "update_continuity_summary.csv",
    "binance_claim_assessment.json",
)
_OPTIONAL_SSL_V2_ANALYSIS_FILES = (
    "summary.json",
    "ssl_v2_analysis.md",
    "ssl_v2_metric_summary.csv",
    "ssl_v2_delta_by_horizon.csv",
    "ssl_v2_delta_by_fold.csv",
    "ssl_v2_loss_components.csv",
    "ssl_v2_claim_assessment.json",
    "figure_manifest.json",
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
class _SSLArtefacts:
    """Validated self-supervised pretraining artefacts.

    ``admitted`` is ``True`` only when the SSL directory, its summary, results
    and comparison tables all exist and at least one pretrained encoder
    checkpoint's recorded SHA256 matches the checkpoint on disk. The report
    refuses to render SSL rows unless ``admitted`` is ``True``.
    """

    admitted: bool = False
    reason: str = "SSL input not supplied"
    directory: Path | None = None
    summary: dict[str, Any] | None = None
    results_rows: list[dict[str, str]] = field(default_factory=list)
    comparison_rows: list[dict[str, str]] = field(default_factory=list)
    verified_runs: list[str] = field(default_factory=list)


@dataclass
class _FullGridArtefacts:
    """Validated full supervised-vs-SSL neural grid aggregate artefacts."""

    available: bool = False
    evidence_level: str = "missing"
    reason: str = "full neural grid input not supplied"
    directory: Path | None = None
    summary: dict[str, Any] | None = None
    results_rows: list[dict[str, str]] = field(default_factory=list)
    aggregate_rows: list[dict[str, str]] = field(default_factory=list)
    comparison_rows: list[dict[str, str]] = field(default_factory=list)
    failure_rows: list[dict[str, str]] = field(default_factory=list)


@dataclass
class _ProperTrainingArtefacts:
    """Validated longer-training neural subset aggregate artefacts.

    Reported separately from the one-epoch matched full grid. ``available`` is
    ``True`` once the required aggregate tables load; ``evidence_level`` records
    whether the stored scope is ``complete_real``, ``partial_real`` or
    ``smoke_test_only`` so the report never overstates the subset.
    """

    available: bool = False
    evidence_level: str = "missing"
    reason: str = "proper-training subset input not supplied"
    directory: Path | None = None
    summary: dict[str, Any] | None = None
    aggregate_rows: list[dict[str, str]] = field(default_factory=list)
    comparison_rows: list[dict[str, str]] = field(default_factory=list)
    training_rows: list[dict[str, str]] = field(default_factory=list)
    failure_rows: list[dict[str, str]] = field(default_factory=list)


@dataclass
class _FigureArtefacts:
    """Optional generated figure manifest loaded for the final report."""

    available: bool = False
    reason: str = "figure manifest not supplied"
    directory: Path | None = None
    manifest_path: Path | None = None
    manifest: dict[str, Any] | None = None
    entries: list[dict[str, Any]] = field(default_factory=list)
    smoke_test_only: bool = False


@dataclass
class _EvidencePackArtefacts:
    """Optional release evidence-pack artefacts loaded for report caveats."""

    available: bool = False
    reason: str = "evidence pack not supplied"
    directory: Path | None = None
    manifest: dict[str, Any] | None = None
    claim_audit: dict[str, Any] | None = None
    supported_claims: str | None = None
    unsupported_claims: str | None = None


@dataclass
class _FinalReportData:
    classical_dir: Path
    neural_dir: Path
    uncertainty_dir: Path
    ablation: _OptionalArtefacts
    feature_ablation: _OptionalArtefacts
    feature_ablation_analysis: _OptionalArtefacts
    execution: _OptionalArtefacts
    execution_v3: _OptionalArtefacts
    execution_centrepiece: _OptionalArtefacts
    external: _OptionalArtefacts
    synthetic: _OptionalArtefacts
    binance_l2: _OptionalArtefacts
    ssl_v2_analysis: _OptionalArtefacts
    ssl: _SSLArtefacts
    full_grid: _FullGridArtefacts
    proper_training: _ProperTrainingArtefacts
    figures: _FigureArtefacts
    evidence_pack: _EvidencePackArtefacts
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
    feature_ablation_dir: Path | None = None,
    feature_ablation_analysis_dir: Path | None = None,
    execution_dir: Path | None = None,
    execution_v3_dir: Path | None = None,
    execution_centrepiece_dir: Path | None = None,
    external_dir: Path | None = None,
    synthetic_lob_dir: Path | None = None,
    binance_l2_dir: Path | None = None,
    ssl_dir: Path | None = None,
    neural_full_grid_dir: Path | None = None,
    proper_training_dir: Path | None = None,
    ssl_v2_analysis_dir: Path | None = None,
    evidence_pack_dir: Path | None = None,
    overwrite: bool = False,
) -> FinalEmpiricalReportSummary:
    """Build the final Markdown empirical report and companion JSON summary.

    When ``ssl_dir`` is supplied and contains valid, SHA256-verified
    self-supervised pretraining artefacts, an SSL comparison row is admitted.
    Otherwise the SSL section is marked skipped and no SSL result is claimed.
    """
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
        feature_ablation_dir=(
            Path(feature_ablation_dir) if feature_ablation_dir is not None else None
        ),
        feature_ablation_analysis_dir=(
            Path(feature_ablation_analysis_dir)
            if feature_ablation_analysis_dir is not None
            else None
        ),
        execution_dir=Path(execution_dir) if execution_dir is not None else None,
        execution_v3_dir=(Path(execution_v3_dir) if execution_v3_dir is not None else None),
        execution_centrepiece_dir=(
            Path(execution_centrepiece_dir)
            if execution_centrepiece_dir is not None
            else None
        ),
        external_dir=Path(external_dir) if external_dir is not None else None,
        synthetic_lob_dir=(Path(synthetic_lob_dir) if synthetic_lob_dir is not None else None),
        binance_l2_dir=(Path(binance_l2_dir) if binance_l2_dir is not None else None),
        ssl_dir=Path(ssl_dir) if ssl_dir is not None else None,
        neural_full_grid_dir=(
            Path(neural_full_grid_dir) if neural_full_grid_dir is not None else None
        ),
        proper_training_dir=(
            Path(proper_training_dir) if proper_training_dir is not None else None
        ),
        ssl_v2_analysis_dir=(
            Path(ssl_v2_analysis_dir) if ssl_v2_analysis_dir is not None else None
        ),
        evidence_pack_dir=(Path(evidence_pack_dir) if evidence_pack_dir is not None else None),
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
    feature_ablation_dir: Path | None,
    feature_ablation_analysis_dir: Path | None,
    execution_dir: Path | None,
    execution_v3_dir: Path | None,
    execution_centrepiece_dir: Path | None,
    external_dir: Path | None,
    synthetic_lob_dir: Path | None,
    binance_l2_dir: Path | None,
    ssl_dir: Path | None,
    neural_full_grid_dir: Path | None,
    proper_training_dir: Path | None,
    ssl_v2_analysis_dir: Path | None,
    evidence_pack_dir: Path | None,
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
    feature_ablation = _load_optional_artefacts(
        directory=feature_ablation_dir,
        label="feature_ablations",
        section_title="Feature Ablation and Stability Analysis",
        expected_files=_OPTIONAL_FEATURE_ABLATION_FILES,
        input_paths=input_paths,
        file_hashes=file_hashes,
        warnings=warnings,
        skipped_sections=skipped_sections,
        missing_sections=missing_sections,
    )
    feature_ablation_analysis = (
        _load_optional_artefacts(
            directory=feature_ablation_analysis_dir,
            label="feature_ablation_analysis",
            section_title="Feature Ablation and Stability Analysis",
            expected_files=_OPTIONAL_FEATURE_ABLATION_ANALYSIS_FILES,
            input_paths=input_paths,
            file_hashes=file_hashes,
            warnings=warnings,
            skipped_sections=skipped_sections,
            missing_sections=missing_sections,
        )
        if feature_ablation_analysis_dir is not None
        else _OptionalArtefacts()
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
    execution_v3 = _load_optional_artefacts(
        directory=execution_v3_dir,
        label="execution_v3",
        section_title="Execution-Aware Proxy Summary",
        expected_files=_OPTIONAL_EXECUTION_V3_FILES,
        input_paths=input_paths,
        file_hashes=file_hashes,
        warnings=warnings,
        skipped_sections=skipped_sections,
        missing_sections=missing_sections,
    )
    execution_centrepiece = _load_optional_artefacts(
        directory=execution_centrepiece_dir,
        label="execution_centrepiece",
        section_title="Forecasting versus Signal-Quality Gap",
        expected_files=_OPTIONAL_EXECUTION_CENTREPIECE_FILES,
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
    synthetic = _load_optional_artefacts(
        directory=synthetic_lob_dir,
        label="synthetic_lob",
        section_title="Synthetic Event-Level Extension",
        expected_files=_OPTIONAL_SYNTHETIC_FILES,
        input_paths=input_paths,
        file_hashes=file_hashes,
        warnings=warnings,
        skipped_sections=skipped_sections,
        missing_sections=missing_sections,
    )
    binance_l2 = _load_optional_artefacts(
        directory=binance_l2_dir,
        label="binance_l2",
        section_title="Real Event-Level L2 Replay Extension",
        expected_files=_OPTIONAL_BINANCE_L2_FILES,
        input_paths=input_paths,
        file_hashes=file_hashes,
        warnings=warnings,
        skipped_sections=skipped_sections,
        missing_sections=missing_sections,
    )
    ssl_v2_analysis = (
        _load_optional_artefacts(
            directory=ssl_v2_analysis_dir,
            label="ssl_v2_analysis",
            section_title="Second-Generation SSL Objective",
            expected_files=_OPTIONAL_SSL_V2_ANALYSIS_FILES,
            input_paths=input_paths,
            file_hashes=file_hashes,
            warnings=warnings,
            skipped_sections=skipped_sections,
            missing_sections=missing_sections,
        )
        if ssl_v2_analysis_dir is not None
        else _OptionalArtefacts()
    )
    recorded_skips.extend(_extract_recorded_skips(ablation, "ablation"))
    recorded_skips.extend(_extract_recorded_skips(feature_ablation, "feature_ablation"))
    recorded_skips.extend(
        _extract_recorded_skips(feature_ablation_analysis, "feature_ablation_analysis")
    )
    recorded_skips.extend(_extract_recorded_skips(execution, "execution"))
    recorded_skips.extend(_extract_recorded_skips(execution_v3, "execution_v3"))

    ssl = _load_ssl_artefacts(
        directory=ssl_dir,
        input_paths=input_paths,
        file_hashes=file_hashes,
        warnings=warnings,
        skipped_sections=skipped_sections,
        missing_sections=missing_sections,
    )
    full_grid = _load_full_grid_artefacts(
        directory=neural_full_grid_dir,
        input_paths=input_paths,
        file_hashes=file_hashes,
        warnings=warnings,
        skipped_sections=skipped_sections,
        missing_sections=missing_sections,
    )
    proper_training = _load_proper_training_artefacts(
        directory=proper_training_dir,
        input_paths=input_paths,
        file_hashes=file_hashes,
        warnings=warnings,
        skipped_sections=skipped_sections,
        missing_sections=missing_sections,
    )
    figures = _load_figure_artefacts(
        neural_full_grid_dir=neural_full_grid_dir,
        input_paths=input_paths,
        file_hashes=file_hashes,
        warnings=warnings,
    )
    evidence_pack = _load_evidence_pack_artefacts(
        directory=evidence_pack_dir,
        input_paths=input_paths,
        file_hashes=file_hashes,
        warnings=warnings,
        skipped_sections=skipped_sections,
        missing_sections=missing_sections,
    )

    return _FinalReportData(
        classical_dir=classical,
        neural_dir=neural,
        uncertainty_dir=uncertainty,
        ablation=ablation,
        feature_ablation=feature_ablation,
        feature_ablation_analysis=feature_ablation_analysis,
        execution=execution,
        execution_v3=execution_v3,
        execution_centrepiece=execution_centrepiece,
        external=external,
        synthetic=synthetic,
        binance_l2=binance_l2,
        ssl_v2_analysis=ssl_v2_analysis,
        ssl=ssl,
        full_grid=full_grid,
        proper_training=proper_training,
        figures=figures,
        evidence_pack=evidence_pack,
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
            if filename in {"summary.json", "benchmark_context.json"} or Path(
                filename
            ).stem.endswith("summary"):
                artefacts.summary = payload
        elif path.suffix.lower() == ".csv":
            artefacts.csv_rows[filename] = _read_csv(path, key, input_paths, file_hashes)
        else:
            _record_file(path, key, input_paths, file_hashes)
    return artefacts


_SSL_SECTION_TITLE = "Self-Supervised Pretraining"


def _verify_ssl_checkpoints(directory: Path) -> list[str]:
    """Return run ids whose encoder checkpoint SHA256 matches its manifest.

    A run is verified only when ``pretrain/artefact_manifest.json`` records a
    SHA256 for ``pretrained_encoder.pt`` and the checkpoint on disk hashes to
    exactly that value. Missing files, unreadable manifests or hash mismatches
    leave the run unverified.
    """
    runs_dir = directory / "runs"
    if not runs_dir.is_dir():
        return []
    verified: list[str] = []
    for run_dir in sorted(runs_dir.iterdir()):
        manifest_path = run_dir / "pretrain" / "artefact_manifest.json"
        encoder_path = run_dir / "pretrain" / "pretrained_encoder.pt"
        if not manifest_path.is_file() or not encoder_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(manifest, dict):
            continue
        recorded = manifest.get("sha256")
        if not isinstance(recorded, dict):
            continue
        expected = recorded.get("pretrained_encoder.pt")
        if not isinstance(expected, str) or not expected:
            continue
        if sha256_file(encoder_path) == expected:
            verified.append(run_dir.name)
    return verified


def _load_ssl_artefacts(
    *,
    directory: Path | None,
    input_paths: dict[str, str],
    file_hashes: dict[str, str],
    warnings: list[str],
    skipped_sections: list[str],
    missing_sections: list[str],
) -> _SSLArtefacts:
    """Strictly validate SSL artefacts; refuse to admit rows unless valid."""

    def _refuse(reason: str) -> _SSLArtefacts:
        warnings.append(f"{reason}; {_SSL_SECTION_TITLE} will be marked skipped.")
        skipped_sections.append(_SSL_SECTION_TITLE)
        missing_sections.append(reason)
        return _SSLArtefacts(admitted=False, reason=reason)

    if directory is None:
        return _refuse("SSL input not supplied")
    candidate = Path(directory)
    if not candidate.exists() or not candidate.is_dir():
        return _refuse(f"SSL artefact directory missing: {_display_path(candidate)}")
    for filename in _REQUIRED_SSL_FILES:
        if not (candidate / filename).is_file():
            return _refuse(
                f"SSL artefacts incomplete: missing {_display_path(candidate / filename)}"
            )

    verified_runs = _verify_ssl_checkpoints(candidate)
    if not verified_runs:
        return _refuse(
            "SSL artefacts present but no pretrained encoder checkpoint could be "
            "SHA256-verified against its manifest"
        )

    summary = _read_json(
        candidate / "summary.json",
        "ssl_summary",
        input_paths,
        file_hashes,
    )
    results_rows = _read_csv(
        candidate / "results_summary.csv",
        "ssl_results_summary",
        input_paths,
        file_hashes,
    )
    comparison_rows = _read_csv(
        candidate / "comparison_summary.csv",
        "ssl_comparison_summary",
        input_paths,
        file_hashes,
    )
    has_ssl_row = any(
        str(row.get("model_name", "")).strip() == "ssl_transformer" for row in results_rows
    )
    if not has_ssl_row:
        return _refuse("SSL results_summary.csv does not contain an ssl_transformer row")

    input_paths["ssl_dir"] = _display_path(candidate)
    return _SSLArtefacts(
        admitted=True,
        reason="verified",
        directory=candidate,
        summary=summary,
        results_rows=results_rows,
        comparison_rows=comparison_rows,
        verified_runs=verified_runs,
    )


_FULL_GRID_SECTION_TITLE = "Full Neural Grid"


def _load_full_grid_artefacts(
    *,
    directory: Path | None,
    input_paths: dict[str, str],
    file_hashes: dict[str, str],
    warnings: list[str],
    skipped_sections: list[str],
    missing_sections: list[str],
) -> _FullGridArtefacts:
    """Load full neural grid aggregates without promoting unsupported claims."""

    def _refuse(reason: str) -> _FullGridArtefacts:
        warnings.append(f"{reason}; {_FULL_GRID_SECTION_TITLE} will make no claim.")
        skipped_sections.append(_FULL_GRID_SECTION_TITLE)
        missing_sections.append(reason)
        return _FullGridArtefacts(available=False, reason=reason)

    if directory is None:
        return _refuse("full neural grid input not supplied")
    candidate = Path(directory)
    if not candidate.exists() or not candidate.is_dir():
        return _refuse(f"full neural grid artefact directory missing: {_display_path(candidate)}")
    for filename in _REQUIRED_FULL_GRID_FILES:
        if not (candidate / filename).is_file():
            return _refuse(
                "full neural grid artefacts incomplete: missing "
                f"{_display_path(candidate / filename)}"
            )

    summary = _read_json(
        candidate / "summary.json",
        "neural_full_grid_summary",
        input_paths,
        file_hashes,
    )
    results_rows = _read_csv(
        candidate / "results_summary.csv",
        "neural_full_grid_results_summary",
        input_paths,
        file_hashes,
    )
    aggregate_rows = _read_csv(
        candidate / "aggregate_summary.csv",
        "neural_full_grid_aggregate_summary",
        input_paths,
        file_hashes,
    )
    _read_json(
        candidate / "aggregate_summary.json",
        "neural_full_grid_aggregate_summary_json",
        input_paths,
        file_hashes,
    )
    comparison_rows = _read_csv(
        candidate / "ssl_comparison.csv",
        "neural_full_grid_ssl_comparison",
        input_paths,
        file_hashes,
    )
    failure_rows = _read_csv(
        candidate / "failures.csv",
        "neural_full_grid_failures",
        input_paths,
        file_hashes,
    )
    input_paths["neural_full_grid_dir"] = _display_path(candidate)

    smoke = (
        bool(summary.get("smoke_test")) or str(summary.get("execution_mode", "")).lower() == "smoke"
    )
    if smoke:
        warnings.append(
            "full neural grid artefacts are marked smoke-test-only; no empirical "
            "full-grid result is claimed."
        )
        skipped_sections.append(_FULL_GRID_SECTION_TITLE)
        evidence_level = "smoke_test_only"
    else:
        evidence_level = str(summary.get("evidence_level", "real_grid"))
    return _FullGridArtefacts(
        available=True,
        evidence_level=evidence_level,
        reason="loaded",
        directory=candidate,
        summary=summary,
        results_rows=results_rows,
        aggregate_rows=aggregate_rows,
        comparison_rows=comparison_rows,
        failure_rows=failure_rows,
    )


_PROPER_TRAINING_SECTION_TITLE = "Proper-Training Neural Subset"


def _load_proper_training_artefacts(
    *,
    directory: Path | None,
    input_paths: dict[str, str],
    file_hashes: dict[str, str],
    warnings: list[str],
    skipped_sections: list[str],
    missing_sections: list[str],
) -> _ProperTrainingArtefacts:
    """Load the longer-training neural subset without overstating its scope."""

    def _refuse(reason: str) -> _ProperTrainingArtefacts:
        warnings.append(f"{reason}; {_PROPER_TRAINING_SECTION_TITLE} will make no claim.")
        skipped_sections.append(_PROPER_TRAINING_SECTION_TITLE)
        missing_sections.append(reason)
        return _ProperTrainingArtefacts(available=False, reason=reason)

    if directory is None:
        return _refuse("proper-training subset input not supplied")
    candidate = Path(directory)
    if not candidate.exists() or not candidate.is_dir():
        return _refuse(
            f"proper-training subset artefact directory missing: {_display_path(candidate)}"
        )
    for filename in _REQUIRED_PROPER_TRAINING_FILES:
        if not (candidate / filename).is_file():
            return _refuse(
                "proper-training subset artefacts incomplete: missing "
                f"{_display_path(candidate / filename)}"
            )

    summary = _read_json(
        candidate / "summary.json",
        "proper_training_summary",
        input_paths,
        file_hashes,
    )
    _read_json(
        candidate / "config_snapshot.json",
        "proper_training_config_snapshot",
        input_paths,
        file_hashes,
    )
    aggregate_rows = _read_csv(
        candidate / "aggregate_summary.csv",
        "proper_training_aggregate_summary",
        input_paths,
        file_hashes,
    )
    comparison_rows = _read_csv(
        candidate / "ssl_comparison.csv",
        "proper_training_ssl_comparison",
        input_paths,
        file_hashes,
    )
    training_rows = _read_csv(
        candidate / "training_curves_summary.csv",
        "proper_training_curves_summary",
        input_paths,
        file_hashes,
    )
    failure_rows = _read_csv(
        candidate / "failures.csv",
        "proper_training_failures",
        input_paths,
        file_hashes,
    )
    _read_json(
        candidate / "sha256_manifest.json",
        "proper_training_sha256_manifest",
        input_paths,
        file_hashes,
    )
    input_paths["proper_training_dir"] = _display_path(candidate)

    smoke = (
        bool(summary.get("smoke_test")) or str(summary.get("execution_mode", "")).lower() == "smoke"
    )
    if smoke:
        warnings.append(
            "proper-training subset artefacts are marked smoke-test-only; no "
            "empirical longer-training result is claimed."
        )
        skipped_sections.append(_PROPER_TRAINING_SECTION_TITLE)
        evidence_level = "smoke_test_only"
    else:
        evidence_level = str(summary.get("evidence_level", "partial_real"))
    return _ProperTrainingArtefacts(
        available=True,
        evidence_level=evidence_level,
        reason="loaded",
        directory=candidate,
        summary=summary,
        aggregate_rows=aggregate_rows,
        comparison_rows=comparison_rows,
        training_rows=training_rows,
        failure_rows=failure_rows,
    )


def _load_figure_artefacts(
    *,
    neural_full_grid_dir: Path | None,
    input_paths: dict[str, str],
    file_hashes: dict[str, str],
    warnings: list[str],
) -> _FigureArtefacts:
    """Load an optional FI-2010 figure manifest when one exists."""
    candidates: list[Path] = []
    if neural_full_grid_dir is not None:
        grid_dir = Path(neural_full_grid_dir)
        candidates.extend(
            [
                grid_dir / "figure_manifest.json",
                grid_dir / "figures" / "figure_manifest.json",
            ]
        )
    candidates.append(
        project_root() / "reports" / "figures" / "fi2010_neural_full_grid" / "figure_manifest.json"
    )
    manifest_path = next((path for path in candidates if path.is_file()), None)
    if manifest_path is None:
        return _FigureArtefacts(reason="figure manifest not found")
    try:
        manifest = _read_json(
            manifest_path,
            "fi2010_figure_manifest",
            input_paths,
            file_hashes,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        warnings.append(f"figure manifest could not be loaded: {exc}")
        return _FigureArtefacts(reason="figure manifest invalid")
    raw_entries = manifest.get("figures")
    entries: list[dict[str, Any]] = []
    if isinstance(raw_entries, list):
        entries = [
            cast(dict[str, Any], dict(entry)) for entry in raw_entries if isinstance(entry, Mapping)
        ]
    smoke = bool(manifest.get("smoke_test")) or (
        bool(entries) and all(bool(entry.get("smoke_test")) for entry in entries)
    )
    input_paths["fi2010_figure_dir"] = _display_path(manifest_path.parent)
    return _FigureArtefacts(
        available=True,
        reason="loaded",
        directory=manifest_path.parent,
        manifest_path=manifest_path,
        manifest=manifest,
        entries=entries,
        smoke_test_only=smoke,
    )


def _load_evidence_pack_artefacts(
    *,
    directory: Path | None,
    input_paths: dict[str, str],
    file_hashes: dict[str, str],
    warnings: list[str],
    skipped_sections: list[str],
    missing_sections: list[str],
) -> _EvidencePackArtefacts:
    if directory is None:
        skipped_sections.append("Evidence Pack Audit")
        missing_sections.append("evidence pack input not supplied")
        return _EvidencePackArtefacts()
    candidate = Path(directory)
    if not candidate.exists():
        warnings.append(f"evidence pack directory missing: {_display_path(candidate)}")
        skipped_sections.append("Evidence Pack Audit")
        missing_sections.append(f"evidence pack directory missing: {_display_path(candidate)}")
        return _EvidencePackArtefacts(reason="evidence pack directory missing")
    if not candidate.is_dir():
        warnings.append(f"evidence pack path is not a directory: {_display_path(candidate)}")
        skipped_sections.append("Evidence Pack Audit")
        missing_sections.append(
            f"evidence pack path is not a directory: {_display_path(candidate)}"
        )
        return _EvidencePackArtefacts(reason="evidence pack path is not a directory")

    manifest_path = candidate / "evidence_pack_manifest.json"
    claim_audit_path = candidate / "claim_audit.json"
    if not manifest_path.is_file() or not claim_audit_path.is_file():
        warnings.append("evidence pack is missing manifest or claim audit JSON")
        skipped_sections.append("Evidence Pack Audit")
        missing_sections.append("evidence pack manifest or claim audit missing")
        return _EvidencePackArtefacts(directory=candidate, reason="required files missing")

    # The evidence pack is a regenerated sibling release bundle, not a source
    # input. Recording the hashes of its volatile outputs (which carry a fresh
    # generated_at every build) would make the report look stale whenever the
    # pack is rebuilt, so read these files for content without hashing them.
    volatile_hashes: dict[str, str] = {}
    try:
        manifest = _read_json(
            manifest_path,
            "evidence_pack_manifest",
            input_paths,
            volatile_hashes,
        )
        claim_audit = _read_json(
            claim_audit_path,
            "evidence_pack_claim_audit",
            input_paths,
            volatile_hashes,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        warnings.append(f"evidence pack could not be loaded: {exc}")
        skipped_sections.append("Evidence Pack Audit")
        missing_sections.append("evidence pack JSON invalid")
        return _EvidencePackArtefacts(directory=candidate, reason="invalid JSON")

    supported = _read_optional_text(
        candidate / "supported_claims.md",
        "evidence_pack_supported_claims",
        input_paths,
        volatile_hashes,
    )
    unsupported = _read_optional_text(
        candidate / "unsupported_claims.md",
        "evidence_pack_unsupported_claims",
        input_paths,
        volatile_hashes,
    )
    input_paths["evidence_pack_dir"] = _display_path(candidate)
    return _EvidencePackArtefacts(
        available=True,
        reason="loaded",
        directory=candidate,
        manifest=manifest,
        claim_audit=claim_audit,
        supported_claims=supported,
        unsupported_claims=unsupported,
    )


def _render_report(data: _FinalReportData, headline_metrics: dict[str, Any]) -> str:
    lines: list[str] = [
        "# ChronosLOB Final Empirical Report",
        "",
        "Generated from stored FI-2010 artefacts only. No model training is run by this builder.",
        "",
    ]
    lines.extend(_section("Evidence Snapshot", _render_evidence_snapshot(data, headline_metrics)))
    lines.extend(_section("Evidence Status Summary", _render_evidence_status_summary(data)))
    lines.extend(_section("Evidence Pack Audit", _render_evidence_pack_audit(data)))
    lines.extend(_section("Research Question", _render_research_question()))
    lines.extend(_section("Dataset And Split Protocol", _render_dataset_protocol(data)))
    lines.extend(_section("Model Families", _render_model_families(data)))
    lines.extend(_section("Main Result Table", _render_main_results(data)))
    lines.extend(_section(_SSL_SECTION_TITLE, _render_ssl(data)))
    lines.extend(_section(_FULL_GRID_SECTION_TITLE, _render_full_grid(data)))
    lines.extend(_section(_PROPER_TRAINING_SECTION_TITLE, _render_proper_training(data)))
    lines.extend(_section("Legacy Reduced-Scope Benchmark", _render_legacy_neural_benchmark(data)))
    lines.extend(_section("SSL Interpretation", _render_ssl_interpretation(data)))
    lines.extend(_section("SSL Failure Analysis", _render_ssl_failure_analysis(data)))
    lines.extend(
        _section("Second-Generation SSL Objective", _render_ssl_v2_analysis(data))
    )
    lines.extend(_section("Figure Index", _render_figure_index(data)))
    lines.extend(_section("Uncertainty Summary", _render_uncertainty(data)))
    lines.extend(_section("Ablation Summary", _render_ablations(data)))
    lines.extend(
        _section("Feature Ablation and Stability Analysis", _render_feature_ablations(data))
    )
    lines.extend(_section("Execution-Aware Proxy Summary", _render_execution(data)))
    lines.extend(
        _section(
            "Forecasting versus Signal-Quality Gap",
            _render_forecasting_signal_gap(data),
        )
    )
    lines.extend(_section("External Benchmark Context", _render_external(data)))
    lines.extend(
        _section("Synthetic Event-Level Extension", _render_synthetic_extension(data))
    )
    lines.extend(
        _section(
            "Real Event-Level L2 Replay Extension", _render_binance_l2_extension(data)
        )
    )
    lines.extend(_section("What This Supports", _render_what_this_proves(data)))
    lines.extend(_section("What This Does Not Claim", _render_what_this_does_not_prove(data)))
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
    execution_v3_status = _execution_v3_status(data)
    external_status = (
        "protocol context loaded"
        if data.external.directory is not None and data.external.summary is not None
        else "skipped"
    )
    full_grid_status, matched_count = _full_grid_snapshot(data)
    ssl_comparison_status = _ssl_comparison_snapshot(data, matched_count)
    proper_training_status = _proper_training_snapshot(data)
    legacy_epochs = _mapping_str(data.neural_summary, "max_epochs", default="prior")
    rows = [
        ("generated_at", data.generated_at.isoformat()),
        ("git_commit", data.git_commit or "not available"),
        ("classical_scope", "multi-fold classical results"),
        (
            "best_classical_test_macro_f1",
            _metric_snapshot(classical, include_lookback=False),
        ),
        ("neural_full_grid_scope", full_grid_status),
        ("proper_training_neural_scope", proper_training_status),
        ("ssl_comparison_scope", ssl_comparison_status),
        (
            "legacy_reduced_scope_neural_scope",
            f"separate earlier {legacy_epochs}-epoch reduced-scope supervised "
            "benchmark, single-seed, lookback 20; reported separately, not used "
            "as matched-grid or SSL evidence",
        ),
        (
            "best_legacy_reduced_scope_neural_test_macro_f1",
            f"{_metric_snapshot(neural, include_lookback=True)} "
            f"(separate {legacy_epochs}-epoch reduced-scope benchmark)",
        ),
        ("execution_scope", f"{execution_status}; metrics are proxy diagnostics"),
        ("execution_v3_scope", execution_v3_status),
        ("execution_centrepiece_scope", _execution_centrepiece_status(data)),
        (
            "external_scope",
            f"{external_status}; protocol context only, not ranking claims",
        ),
        ("report_path", _display_path(data.report_path)),
        ("summary_path", _display_path(data.summary_path)),
    ]
    return _markdown_table(("field", "value"), rows)


def _full_grid_snapshot(data: _FinalReportData) -> tuple[str, int]:
    """Return the matched full-grid snapshot text and matched comparison count."""
    grid = data.full_grid
    if not grid.available:
        return ("not supplied; no matched full-grid neural result claimed", 0)
    if grid.evidence_level == "smoke_test_only":
        return ("smoke-test-only; not empirical evidence", 0)
    summary = grid.summary or {}
    matched_count = sum(1 for row in grid.comparison_rows if row.get("status") == "matched")
    pretrain = _mapping_str(summary, "pretrain_epochs", default="1")
    fine_tune = _mapping_str(summary, "max_epochs", default="1")
    descriptor = (
        "completed one-epoch matched comparison grid; "
        f"folds {_join_values(_mapping_list(summary, 'folds'))}, "
        f"horizons {_join_values(_mapping_list(summary, 'horizons'))}, "
        f"seeds {_join_values(_mapping_list(summary, 'seeds'))}, "
        f"objectives {_join_values(_mapping_list(summary, 'objectives'))}; "
        f"pretrain_epochs {pretrain}, fine_tune_epochs {fine_tune}; "
        f"{summary.get('completed_run_count', 'not available')} completed, "
        f"{summary.get('failed_run_count', 'not available')} failed; "
        "matched comparison and pipeline evidence, not a "
        "performance-maximising neural benchmark"
    )
    return (descriptor, matched_count)


def _ssl_comparison_snapshot(data: _FinalReportData, matched_count: int) -> str:
    """Describe the SSL comparison state for the evidence snapshot."""
    if matched_count > 0:
        return (
            "matched supervised-vs-SSL comparison present in the full grid "
            "(masked_reconstruction, next_field objectives); no SSL improvement "
            "is supported"
        )
    if data.ssl.admitted:
        return (
            "standalone SSL runner artefacts verified; like-for-like comparison "
            "only, no SSL improvement is supported"
        )
    return "no matched SSL comparison available; no SSL result claimed"


def _proper_training_snapshot(data: _FinalReportData) -> str:
    """Describe the longer-training subset state without upgrading its scope."""
    subset = data.proper_training
    if not subset.available:
        return "not supplied; no longer-training neural result claimed"
    if subset.evidence_level == "smoke_test_only":
        return "smoke-test-only; code-path check, not empirical evidence"
    summary = subset.summary or {}
    return (
        f"{subset.evidence_level}; folds "
        f"{_join_values(_mapping_list(summary, 'folds'))}, horizons "
        f"{_join_values(_mapping_list(summary, 'horizons'))}, seeds "
        f"{_join_values(_mapping_list(summary, 'seeds'))}, lookbacks "
        f"{_join_values(_mapping_list(summary, 'lookbacks'))}, objectives "
        f"{_join_values(_mapping_list(summary, 'objectives'))}; "
        f"max_epochs {_mapping_str(summary, 'max_epochs')}, "
        f"patience {_mapping_str(summary, 'early_stopping_patience')}; "
        "validation-only early stopping with best checkpoint restored before test"
    )


def _render_evidence_status_summary(data: _FinalReportData) -> list[str]:
    """Render a conservative, plain-language status summary near the top.

    This section keeps the README, evidence pack and this report aligned on the
    same neural and SSL status, and separates the completed one-epoch matched
    full grid from the earlier reduced-scope supervised benchmark.
    """
    grid = data.full_grid
    full_grid_complete = grid.available and grid.evidence_level != "smoke_test_only"
    matched_count = sum(1 for row in grid.comparison_rows if row.get("status") == "matched")
    legacy_epochs = _mapping_str(data.neural_summary, "max_epochs", default="prior")

    complete_bullets = [
        "- Multi-fold classical FI-2010 benchmark across the stored folds.",
    ]
    if full_grid_complete:
        complete_bullets.append(
            "- One-epoch matched neural full grid across folds "
            f"{_join_values(_mapping_list(grid.summary, 'folds'))}, horizons "
            f"{_join_values(_mapping_list(grid.summary, 'horizons'))}, seeds "
            f"{_join_values(_mapping_list(grid.summary, 'seeds'))} and objectives "
            f"{_join_values(_mapping_list(grid.summary, 'objectives'))}."
        )
        if matched_count > 0:
            complete_bullets.append("- A matched supervised-vs-SSL comparison inside that grid.")
    if (
        data.execution_v3.directory is not None
        and data.execution_v3.summary is not None
        and not bool(data.execution_v3.summary.get("smoke_test"))
    ):
        complete_bullets.append("- Execution-v3 offline cost-adjusted proxy diagnostics.")
    proper = data.proper_training
    if proper.available and proper.evidence_level == "complete_real":
        complete_bullets.append(
            "- Proper-training neural subset with validation-only early stopping "
            "and best-checkpoint restoration, reported separately from the "
            "one-epoch matched grid."
        )

    partial_bullets = []
    if proper.available and proper.evidence_level == "partial_real":
        partial_bullets.append(
            "- Proper-training neural subset: documented partial longer-training "
            "modelling evidence with validation-only early stopping; exact folds, "
            "horizons, seeds, lookbacks and objectives are listed in its section."
        )
    elif proper.available and proper.evidence_level == "smoke_test_only":
        partial_bullets.append(
            "- Proper-training neural subset: smoke-test-only code-path artefacts; "
            "not empirical longer-training evidence."
        )
    if data.feature_ablation.directory is not None and data.feature_ablation.summary is not None:
        summary = data.feature_ablation.summary
        analysis_summary = data.feature_ablation_analysis.summary or {}
        status = _mapping_str(analysis_summary, "evidence_status", default="partial_real")
        partial_bullets.append(
            "- FI-2010 snapshot feature-ablation evidence: "
            f"{status}; folds {_join_values(_mapping_list(summary, 'folds'))}, horizons "
            f"{_join_values(_mapping_list(summary, 'horizons'))}, seeds "
            f"{_join_values(_mapping_list(summary, 'seeds'))}, models "
            f"{_join_values(_mapping_list(summary, 'models'))}."
        )

    not_claimed_bullets = [
        "- No SSL improvement: SSL was implemented and tested under matched "
        "settings, but no SSL improvement is supported.",
        "- No profitability, tradability, live-trading, PnL, SOTA, "
        "foundation-model or production-execution-simulator claim.",
        "- No true event-level order flow or queue position is observed from FI-2010 snapshots.",
    ]
    legacy_bullet = (
        f"- The earlier {legacy_epochs}-epoch reduced-scope supervised "
        "matrix-transformer benchmark (single seed, lookback 20) is reported "
        "separately and is not used as matched SSL evidence."
    )

    lines: list[str] = [
        *_wrap_prose(
            "This summary uses the same status language as the README and the evidence pack."
        ),
        "",
        "What is complete (`complete_real`):",
        "",
        *_wrap_bullets(complete_bullets),
    ]
    if partial_bullets:
        lines.extend(["", "What is partial (`partial_real`):", "", *_wrap_bullets(partial_bullets)])
    lines.extend(
        [
            "",
            "What is separate legacy / reduced-scope evidence:",
            "",
            *_wrap_bullet(legacy_bullet),
            "",
            "What is not claimed:",
            "",
            *_wrap_bullets(not_claimed_bullets),
            "",
        ]
    )
    if full_grid_complete:
        lines.extend(
            _wrap_prose(
                "The completed matched full grid is a one-epoch comparison grid. "
                "It is useful for controlled supervised-vs-SSL comparison and "
                "pipeline validation, but it is not a performance-maximising "
                "neural training result."
            )
        )
        lines.append("")
    lines.extend(
        _wrap_prose(
            f"The earlier {legacy_epochs}-epoch reduced-scope supervised "
            "matrix-transformer result is reported separately and is not used as "
            "matched SSL evidence."
        )
    )
    return lines


def _render_evidence_pack_audit(data: _FinalReportData) -> list[str]:
    pack = data.evidence_pack
    if not pack.available or pack.manifest is None or pack.claim_audit is None:
        return [
            "Skipped: evidence-pack artefacts were not supplied or were unavailable.",
            "Release claim status should be checked with `build-evidence-pack` before public use.",
        ]
    artefact_counts = _nested_mapping(pack.manifest, "artefact_status_counts") or {}
    claim_counts = _nested_mapping(pack.claim_audit, "claim_status_counts") or {}
    rows = [
        ("evidence_pack_status", pack.reason),
        ("evidence_pack_dir", _display_path(pack.directory) if pack.directory else "n/a"),
        ("artefact_status_counts", _format_mapping_counts(artefact_counts)),
        ("claim_status_counts", _format_mapping_counts(claim_counts)),
        (
            "supported_claims",
            _first_markdown_bullets(pack.supported_claims, fallback="none fully supported"),
        ),
        (
            "unsupported_or_limited_claims",
            _first_markdown_bullets(pack.unsupported_claims, fallback="see claim audit"),
        ),
    ]
    caveats = [
        "",
        "Release caveats from the evidence pack:",
        "",
        "- Smoke diagnostics are not empirical evidence.",
        "- SSL improvement language requires real aggregate comparison artefacts.",
        "- Execution-v3 metrics remain offline proxy diagnostics.",
        "- FI-2010 snapshot features do not expose event-level order flow or queue position.",
    ]
    return [*_markdown_table(("field", "value"), rows), *caveats]


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
            _neural_protocol_text(data),
        ),
    ]
    return _markdown_table(("field", "value"), rows)


def _neural_protocol_text(data: _FinalReportData) -> str:
    """Describe both the matched full grid and the separate legacy benchmark."""
    grid = data.full_grid
    legacy = (
        "separate earlier "
        f"{_mapping_str(data.neural_summary, 'max_epochs', default='prior')}-epoch "
        "reduced-scope supervised benchmark (single seed, lookback 20) reported "
        "separately"
    )
    if grid.available and grid.evidence_level != "smoke_test_only":
        summary = grid.summary or {}
        return (
            "matched one-epoch full grid over folds "
            f"{_join_values(_mapping_list(summary, 'folds'))}, horizons "
            f"{_join_values(_mapping_list(summary, 'horizons'))}, seeds "
            f"{_join_values(_mapping_list(summary, 'seeds'))} and objectives "
            f"{_join_values(_mapping_list(summary, 'objectives'))}; "
            f"{legacy}"
        )
    return f"{legacy}; no matched full-grid artefacts supplied"


def _render_model_families(data: _FinalReportData) -> list[str]:
    rows = [
        (
            "classical",
            _join_values(_mapping_list(data.classical_summary, "models_requested")),
            "multi-fold stored fold summaries",
        )
    ]
    if data.full_grid.available and data.full_grid.evidence_level != "smoke_test_only":
        summary = data.full_grid.summary or {}
        rows.append(
            (
                "neural matched grid",
                "matrix_transformer",
                "one-epoch matched supervised/SSL grid over folds "
                f"{_join_values(_mapping_list(summary, 'folds'))}, horizons "
                f"{_join_values(_mapping_list(summary, 'horizons'))}, seeds "
                f"{_join_values(_mapping_list(summary, 'seeds'))}; comparison evidence",
            )
        )
    if data.proper_training.available and data.proper_training.evidence_level != "smoke_test_only":
        summary = data.proper_training.summary or {}
        rows.append(
            (
                "neural proper-training subset",
                "matrix_transformer",
                f"{data.proper_training.evidence_level}; folds "
                f"{_join_values(_mapping_list(summary, 'folds'))}, horizons "
                f"{_join_values(_mapping_list(summary, 'horizons'))}, seeds "
                f"{_join_values(_mapping_list(summary, 'seeds'))}, lookbacks "
                f"{_join_values(_mapping_list(summary, 'lookbacks'))}; "
                "validation-only early stopping",
            )
        )
    rows.append(
        (
            "neural legacy supervised",
            _join_values(_mapping_list(data.neural_summary, "models_requested")),
            "separate earlier reduced-scope, single-seed, lookback "
            + _join_values(_mapping_list(data.neural_summary, "lookbacks")),
        )
    )
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


def _render_ssl(data: _FinalReportData) -> list[str]:
    ssl = data.ssl
    if not ssl.admitted:
        grid = data.full_grid
        matched_count = sum(1 for row in grid.comparison_rows if row.get("status") == "matched")
        full_grid_ssl = (
            grid.available and grid.evidence_level != "smoke_test_only" and matched_count > 0
        )
        if full_grid_ssl:
            return [
                *_wrap_prose(
                    "The standalone SSL runner artefact is not supplied, so no "
                    "standalone `ssl_transformer` row is admitted here."
                ),
                "",
                *_wrap_prose(
                    "However, the matched supervised-vs-SSL comparison is not "
                    "absent from this report: it is reported in the Full Neural "
                    "Grid section below, where masked_reconstruction and "
                    "next_field objectives are compared against the supervised "
                    "baseline under identical fold, horizon, seed, lookback, "
                    "architecture and preprocessing settings."
                ),
                "",
                *_wrap_prose(
                    "That matched comparison is a one-epoch grid. No SSL "
                    "improvement over the matched supervised baseline is "
                    "supported; deltas are reported metric-by-metric in the Full "
                    "Neural Grid section."
                ),
            ]
        return [
            f"Skipped: {ssl.reason}.",
            "",
            "The final report builder refuses to admit self-supervised rows "
            "unless a pretrained encoder checkpoint is SHA256-verified against "
            "its manifest. No SSL result is claimed in this report.",
        ]

    rows: list[tuple[str, ...]] = []
    for row in ssl.comparison_rows:
        if str(row.get("status", "")).strip() not in ("", "ok"):
            continue
        rows.append(
            (
                row.get("fold_id", ""),
                _empty_to_na(row.get("seed", "")),
                _empty_to_na(row.get("lookback", "")),
                _format_float(_row_float(row, "ssl_macro_f1")),
                _format_float(_row_float(row, "supervised_macro_f1")),
                _format_float(_row_float(row, "macro_f1_delta")),
            )
        )
    intro = [
        "Self-supervised pretraining used official training rows only; the "
        "validation pretraining loss is carved from training rows and the "
        "official split-aware test evaluation is preserved. The fine-tuned "
        "ssl_transformer and the supervised baseline share one architecture, "
        "set of folds, horizons, seeds and preprocessing.",
        "",
        f"Verified pretrained encoder checkpoints: {len(ssl.verified_runs)}.",
        "",
        "These are reduced-scope SSL artefacts. The macro-F1 deltas below "
        "report a like-for-like comparison and are not a self-supervised "
        "effectiveness or SOTA claim.",
        "",
    ]
    table = _markdown_table(
        (
            "fold",
            "seed",
            "lookback",
            "ssl_transformer test macro-F1",
            "supervised test macro-F1",
            "macro-F1 delta",
        ),
        rows,
    )
    return intro + table


def _render_full_grid(data: _FinalReportData) -> list[str]:
    grid = data.full_grid
    if not grid.available:
        return [
            f"Status: unavailable. {grid.reason}.",
            "",
            "No full-grid neural result is claimed.",
        ]
    summary = grid.summary or {}
    smoke = grid.evidence_level == "smoke_test_only"
    status_lines = [
        f"Status: {'smoke-test-only' if smoke else 'loaded'}.",
        (
            "These artefacts are code-path smoke outputs and are not empirical evidence."
            if smoke
            else "These artefacts are loaded as aggregate full-grid evidence, "
            "subject to the failures table below."
        ),
        "",
    ]
    status_rows = [
        ("execution_mode", _mapping_str(summary, "execution_mode")),
        ("folds", _join_values(_mapping_list(summary, "folds"))),
        ("horizons", _join_values(_mapping_list(summary, "horizons"))),
        ("seeds", _join_values(_mapping_list(summary, "seeds"))),
        ("lookbacks", _join_values(_mapping_list(summary, "lookbacks"))),
        ("objectives", _join_values(_mapping_list(summary, "objectives"))),
        ("pretrain_epochs", _mapping_str(summary, "pretrain_epochs")),
        ("fine_tune_epochs", _mapping_str(summary, "max_epochs")),
        ("completed_runs", str(summary.get("completed_run_count", "not available"))),
        ("failed_runs", str(summary.get("failed_run_count", "not available"))),
        ("core_grid_complete", str(summary.get("core_grid_complete", "not available"))),
    ]
    lines = status_lines + _markdown_table(("field", "value"), status_rows)
    lines.extend(["", "Aggregate supervised-vs-SSL rows:", ""])
    aggregate_rows = []
    for row in grid.aggregate_rows:
        aggregate_rows.append(
            (
                row.get("horizon", ""),
                row.get("lookback", ""),
                row.get("model_family", ""),
                row.get("pretraining_objective", ""),
                row.get("completed_run_count", ""),
                row.get("failed_run_count", ""),
                _format_float(_row_float(row, "mean_macro_f1")),
                _format_float(_row_float(row, "std_macro_f1")),
                _format_float(_row_float(row, "mean_mcc")),
                _format_float(_row_float(row, "mean_ece")),
            )
        )
    lines.extend(
        _markdown_table(
            (
                "horizon",
                "lookback",
                "model",
                "pretraining",
                "completed",
                "failed",
                "mean macro-F1",
                "std macro-F1",
                "mean MCC",
                "mean ECE",
            ),
            aggregate_rows,
        )
    )
    lines.extend(["", "Matched SSL deltas:", ""])
    matched_rows = [row for row in grid.comparison_rows if row.get("status") == "matched"]
    delta_rows = []
    for row in matched_rows:
        delta_rows.append(
            (
                row.get("horizon", ""),
                row.get("lookback", ""),
                row.get("ssl_objective", ""),
                _format_float(_row_float(row, "delta_macro_f1")),
                _format_float(_row_float(row, "delta_mcc")),
                _format_float(_row_float(row, "delta_ece")),
                row.get("macro_f1_outcome", ""),
                row.get("mcc_outcome", ""),
                row.get("ece_outcome", ""),
            )
        )
    lines.extend(
        _markdown_table(
            (
                "horizon",
                "lookback",
                "SSL objective",
                "delta macro-F1",
                "delta MCC",
                "delta ECE",
                "macro-F1",
                "MCC",
                "ECE",
            ),
            delta_rows,
        )
    )
    lines.extend(["", "Failure summary:", ""])
    invalid_failures = [
        row
        for row in grid.failure_rows
        if str(row.get("invalidates_aggregate_claims", "")).lower() == "true"
    ]
    failure_rows = [
        (
            row.get("fold", ""),
            row.get("horizon", ""),
            row.get("seed", ""),
            row.get("objective", ""),
            row.get("status", ""),
            _empty_to_na(row.get("reason", "")),
        )
        for row in grid.failure_rows[:10]
    ]
    if failure_rows:
        lines.extend(
            _markdown_table(
                ("fold", "horizon", "seed", "objective", "status", "reason"),
                failure_rows,
            )
        )
        if len(grid.failure_rows) > 10:
            lines.append(f"{len(grid.failure_rows) - 10} additional rows are in failures.csv.")
    else:
        lines.append("No failed or reused-existing runs are recorded.")
    lines.extend(["", *_full_grid_interpretation(grid, matched_rows, invalid_failures)])
    return lines


_PROPER_TRAINING_FRAMING = (
    "The one-epoch full grid is retained as matched comparison evidence. The "
    "proper-training subset is used to assess whether the neural models remain "
    "credible under a more realistic training budget."
)


def _render_proper_training(data: _FinalReportData) -> list[str]:
    subset = data.proper_training
    framing = _wrap_prose(_PROPER_TRAINING_FRAMING)
    if not subset.available:
        return [
            *framing,
            "",
            f"Status: unavailable. {subset.reason}.",
            "",
            "No longer-training neural result is claimed.",
        ]
    summary = subset.summary or {}
    smoke = subset.evidence_level == "smoke_test_only"
    lines = [*framing, ""]
    lines.append(f"Status: {'smoke-test-only' if smoke else 'loaded'}.")
    if smoke:
        lines.extend(
            [
                "These artefacts are code-path smoke outputs and are not empirical evidence.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "These artefacts are loaded as longer-training modelling evidence, "
                "subject to the scope and failure rows below.",
                "",
            ]
        )
    status_rows = [
        ("subset_kind", _mapping_str(summary, "subset_kind", default="proper_training_subset")),
        ("evidence_level", subset.evidence_level),
        ("scope_label", _mapping_str(summary, "scope_label")),
        ("execution_mode", _mapping_str(summary, "execution_mode")),
        ("folds", _join_values(_mapping_list(summary, "folds"))),
        ("horizons", _join_values(_mapping_list(summary, "horizons"))),
        ("seeds", _join_values(_mapping_list(summary, "seeds"))),
        ("lookbacks", _join_values(_mapping_list(summary, "lookbacks"))),
        ("objectives", _join_values(_mapping_list(summary, "objectives"))),
        ("max_epochs", _mapping_str(summary, "max_epochs")),
        ("early_stopping_metric", _mapping_str(summary, "early_stopping_metric")),
        ("early_stopping_patience", _mapping_str(summary, "early_stopping_patience")),
        ("pretrain_epochs", _mapping_str(summary, "pretrain_epochs")),
        ("completed_runs", str(summary.get("completed_run_count", "not available"))),
        ("failed_runs", str(summary.get("failed_run_count", "not available"))),
        (
            "planned_scope_complete",
            str(summary.get("planned_scope_complete", "not available")),
        ),
        (
            "target_scope_complete",
            str(summary.get("target_scope_complete", "not available")),
        ),
        (
            "model_selection",
            "validation-only early stopping; best checkpoint restored before test",
        ),
    ]
    lines.extend(_markdown_table(("field", "value"), status_rows))

    if not smoke and not bool(summary.get("target_scope_complete")):
        lines.extend(
            [
                "",
                *_wrap_prose(
                    "Scope note: this is a documented partial subset "
                    f"(`{subset.evidence_level}`). It does not cover the full "
                    "primary proper-training target (folds 1-5, horizons 10 and "
                    "50, seed 0, all three objectives, lookback 50, max_epochs 25, "
                    "patience 5); the table above states exactly what was run."
                ),
            ]
        )

    lines.extend(["", "Training / early-stopping summary:", ""])
    lines.extend(_markdown_table(("field", "value"), _proper_training_curve_rows(subset)))

    lines.extend(["", "Aggregate test metrics by objective:", ""])
    aggregate_rows = [
        (
            row.get("horizon", ""),
            row.get("lookback", ""),
            row.get("pretraining_objective", ""),
            row.get("completed_run_count", ""),
            _format_float(_row_float(row, "mean_macro_f1")),
            _format_float(_row_float(row, "std_macro_f1")),
            _format_float(_row_float(row, "mean_mcc")),
            _format_float(_row_float(row, "mean_ece")),
        )
        for row in subset.aggregate_rows
    ]
    lines.extend(
        _markdown_table(
            (
                "horizon",
                "lookback",
                "pretraining",
                "completed",
                "mean macro-F1",
                "std macro-F1",
                "mean MCC",
                "mean ECE",
            ),
            aggregate_rows,
        )
    )

    matched_rows = [row for row in subset.comparison_rows if row.get("status") == "matched"]
    lines.extend(["", "Matched SSL deltas (longer training):", ""])
    delta_rows = [
        (
            row.get("horizon", ""),
            row.get("seed", ""),
            row.get("ssl_objective", ""),
            _format_float(_row_float(row, "delta_macro_f1")),
            _format_float(_row_float(row, "delta_mcc")),
            _format_float(_row_float(row, "delta_ece")),
            row.get("macro_f1_outcome", ""),
            row.get("mcc_outcome", ""),
            row.get("ece_outcome", ""),
        )
        for row in matched_rows
    ]
    lines.extend(
        _markdown_table(
            (
                "horizon",
                "seed",
                "SSL objective",
                "delta macro-F1",
                "delta MCC",
                "delta ECE",
                "macro-F1",
                "MCC",
                "ECE",
            ),
            delta_rows,
        )
    )
    lines.extend(["", *_proper_training_interpretation(subset, matched_rows)])
    return lines


def _proper_training_curve_rows(
    subset: _ProperTrainingArtefacts,
) -> list[tuple[str, str]]:
    rows = subset.training_rows
    completed = [row for row in rows if row.get("status") in ("", "completed", "skipped_existing")]
    best_epochs = [
        value for row in completed for value in [_row_int(row, "best_epoch")] if value is not None
    ]
    epochs_ran = [
        value for row in completed for value in [_row_int(row, "epochs_ran")] if value is not None
    ]
    early_stopped = sum(
        1 for row in completed if str(row.get("early_stopped", "")).strip().lower() == "true"
    )
    best_epoch_text = (
        f"{min(best_epochs)} to {max(best_epochs)} (mean {sum(best_epochs) / len(best_epochs):.1f})"
        if best_epochs
        else "not available"
    )
    epochs_ran_text = f"{min(epochs_ran)} to {max(epochs_ran)}" if epochs_ran else "not available"
    return [
        ("runs_with_curves", str(len(completed))),
        ("best_epoch_range", best_epoch_text),
        ("epochs_ran_range", epochs_ran_text),
        ("runs_early_stopped", f"{early_stopped} of {len(completed)}"),
        (
            "curve_files",
            "per-run runs/**/curves.csv and curves.json (train/validation loss, "
            "validation macro-F1, accuracy, MCC)",
        ),
    ]


def _proper_training_interpretation(
    subset: _ProperTrainingArtefacts,
    matched_rows: Sequence[Mapping[str, str]],
) -> list[str]:
    if subset.evidence_level == "smoke_test_only":
        return [
            "Interpretation: smoke-test-only; no empirical longer-training or SSL claim is made."
        ]
    invalid_failures = [
        row
        for row in subset.failure_rows
        if str(row.get("invalidates_aggregate_claims", "")).lower() == "true"
    ]
    if invalid_failures:
        return [
            "Interpretation: failed runs are present, so longer-training claims are "
            "limited to completed runs and are not described as a complete subset."
        ]
    if not matched_rows:
        return [
            "Interpretation: no matched supervised-vs-SSL pairs are available in the "
            "subset, so no SSL delta claim is made."
        ]
    lines = ["Interpretation:"]
    for objective in ("masked_reconstruction", "next_field"):
        rows = [row for row in matched_rows if row.get("ssl_objective") == objective]
        if not rows:
            lines.append(f"- {objective}: no matched rows.")
            continue
        macro = _outcome_counts(rows, "macro_f1_outcome")
        mcc = _outcome_counts(rows, "mcc_outcome")
        ece = _outcome_counts(rows, "ece_outcome")
        macro_mean = _mean_delta(rows, "delta_macro_f1")
        mcc_mean = _mean_delta(rows, "delta_mcc")
        ece_mean = _mean_delta(rows, "delta_ece")
        lines.append(
            f"- {objective}: mean deltas macro-F1 {_format_float(macro_mean)}, "
            f"MCC {_format_float(mcc_mean)}, ECE {_format_float(ece_mean)}; "
            f"outcomes macro-F1 {macro}, MCC {mcc}, ECE {ece}."
        )
    lines.append(
        "  Under this longer-training budget no broad SSL improvement is claimed; "
        "deltas are reported metric-by-metric, fold-by-fold and seed-by-seed. Any "
        "improvement is scoped to exactly the rows above."
    )
    return lines


def _render_legacy_neural_benchmark(data: _FinalReportData) -> list[str]:
    legacy_epochs = _mapping_str(data.neural_summary, "max_epochs", default="prior")
    lookbacks = _join_values(_mapping_list(data.neural_summary, "lookbacks"))
    seeds = _join_values(_mapping_list(data.neural_summary, "seeds"))
    rows = [
        (
            row.get("model_name", ""),
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
        for row in _sorted_test_rows(data.neural_results, lookback_column="lookback")
    ]
    lines = [
        *_wrap_prose(
            f"This is the earlier {legacy_epochs}-epoch reduced-scope supervised "
            "neural benchmark. It is reported separately from the one-epoch "
            "matched full grid and from the proper-training supervised-vs-SSL "
            "subset, and it is not used as matched SSL evidence."
        ),
        "",
        (
            f"Stored scope: seeds {seeds or 'not available'}, "
            f"lookbacks {lookbacks or 'not available'}."
        ),
        "",
    ]
    lines.extend(
        _markdown_table(
            (
                "model",
                "test macro-F1",
                "accuracy",
                "MCC",
                "folds",
                "seeds",
                "lookback",
            ),
            rows,
        )
    )
    return lines


def _render_ssl_interpretation(data: _FinalReportData) -> list[str]:
    full_grid_rows = [
        row for row in data.full_grid.comparison_rows if row.get("status") == "matched"
    ]
    proper_rows = [
        row for row in data.proper_training.comparison_rows if row.get("status") == "matched"
    ]
    lines = [
        *_wrap_prose(
            "SSL evidence is interpreted only through matched supervised-vs-SSL "
            "rows. The one-epoch full grid remains comparison and infrastructure "
            "evidence; the proper-training subset is longer-training modelling "
            "evidence at its exact stored scope."
        ),
        "",
    ]
    rows: list[tuple[str, str, str, str, str, str]] = []
    for source, matched in (
        ("one-epoch full grid", full_grid_rows),
        ("proper-training subset", proper_rows),
    ):
        for objective in ("masked_reconstruction", "next_field"):
            objective_rows = [row for row in matched if row.get("ssl_objective") == objective]
            if not objective_rows:
                continue
            rows.append(
                (
                    source,
                    objective,
                    _format_float(_mean_delta(objective_rows, "delta_macro_f1")),
                    _format_float(_mean_delta(objective_rows, "delta_mcc")),
                    _format_float(_mean_delta(objective_rows, "delta_ece")),
                    _outcome_counts(objective_rows, "macro_f1_outcome"),
                )
            )
    if rows:
        lines.extend(
            _markdown_table(
                (
                    "source",
                    "SSL objective",
                    "mean delta macro-F1",
                    "mean delta MCC",
                    "mean delta ECE",
                    "macro-F1 outcomes",
                ),
                rows,
            )
        )
        lines.append("")
    if proper_rows and not _all_rows_improve(proper_rows, "delta_macro_f1"):
        lines.append("The longer-training subset does not support an SSL improvement claim.")
    else:
        lines.append(
            "No broad SSL improvement claim is made unless all matched aggregate "
            "deltas support it without metric-specific degradation."
        )
    return lines


def _render_ssl_failure_analysis(data: _FinalReportData) -> list[str]:
    """Reference the dedicated SSL failure-analysis report and its conclusions.

    The three bodies of SSL evidence are kept distinct: the completed one-epoch
    matched full grid, the longer-training proper-training subset v2 and the
    older reduced-scope supervised benchmark used only for context.
    """
    full_grid_rows = [
        row for row in data.full_grid.comparison_rows if row.get("status") == "matched"
    ]
    proper_rows = [
        row for row in data.proper_training.comparison_rows if row.get("status") == "matched"
    ]
    lines = [
        *_wrap_prose(
            "A dedicated SSL failure-analysis report at "
            "reports/ssl_failure_analysis/ssl_failure_analysis.md separates three "
            "distinct bodies of evidence and never merges them: the completed "
            "one-epoch matched full grid (folds 1-5, horizons 10/20/50, seeds 0-2), "
            "the longer-training proper-training subset v2 (fold 1, horizons 10 and "
            "50, seed 0, partial_real) and a separate older reduced-scope "
            "supervised benchmark used only for context."
        ),
        "",
    ]
    if full_grid_rows:
        for objective in ("masked_reconstruction", "next_field"):
            objective_rows = [
                row for row in full_grid_rows if row.get("ssl_objective") == objective
            ]
            if not objective_rows:
                continue
            macro = _mean_delta(objective_rows, "delta_macro_f1")
            ece = _mean_delta(objective_rows, "delta_ece")
            lines.append(
                f"- Full grid {objective}: mean macro-F1 delta {_format_float(macro)}, "
                f"mean ECE delta {_format_float(ece)} (lower ECE is better)."
            )
        lines.append("")
    if proper_rows:
        h50_rows = [row for row in proper_rows if _row_float(row, "horizon") == 50.0]
        masked_h50 = [
            row for row in h50_rows if row.get("ssl_objective") == "masked_reconstruction"
        ]
        if masked_h50:
            macro = _mean_delta(masked_h50, "delta_macro_f1")
            mcc = _mean_delta(masked_h50, "delta_mcc")
            ece = _mean_delta(masked_h50, "delta_ece")
            lines.append(
                "- Proper-training masked SSL at fold 1 / horizon 50: macro-F1 delta "
                f"{_format_float(macro)}, MCC delta {_format_float(mcc)}, ECE delta "
                f"{_format_float(ece)} (calibration worsened)."
            )
            lines.append("")
    lines.extend(
        _wrap_bullets(
            [
                "- Full-grid SSL does not improve overall: matched macro-F1 deltas are "
                "neutral-to-negative and calibration does not improve uniformly.",
                "- Proper-training subset v2 shows a narrow fold-1/horizon-50 predictive "
                "gain in macro-F1 and MCC, but ECE worsened in every matched SSL row.",
                "- No broad SSL improvement and no calibration improvement is claimed.",
                "- More evidence would require broader proper-training runs and/or "
                "better SSL objective design rather than any success claim.",
            ]
        )
    )
    return lines


def _render_ssl_v2_analysis(data: _FinalReportData) -> list[str]:
    artefacts = data.ssl_v2_analysis
    if artefacts.directory is None or artefacts.summary is None:
        return [
            "Skipped: SSL-v2 analysis artefacts were not supplied.",
            "No second-generation SSL result is implied without stored analysis outputs.",
        ]
    summary = artefacts.summary
    claim_payload = artefacts.json_payloads.get("ssl_v2_claim_assessment.json", {})
    claims = claim_payload.get("claims", [])
    claim_status = {
        str(item.get("claim_id")): str(item.get("status"))
        for item in claims
        if isinstance(item, Mapping)
    }
    horizon_rows = artefacts.csv_rows.get("ssl_v2_delta_by_horizon.csv", [])
    lines = [
        *_wrap_prose(
            "A second-generation SSL objective was added after the SSL failure "
            "analysis showed that first-generation random field reconstruction and "
            "next-field prediction did not broadly improve predictive metrics or "
            "calibration. The SSL-v2 objective is market-state-aware and remains a "
            "scoped comparison, not a general representation or trading claim."
        ),
        "",
        f"- evidence level: {_mapping_str(summary, 'evidence_level')}",
        f"- scope label: {_mapping_str(summary, 'scope_label')}",
        f"- matched supervised-vs-SSL-v2 rows: {_mapping_str(summary, 'ssl_v2_matched_rows')}",
        f"- failures: {_mapping_str(summary, 'failure_count')}",
        "",
    ]
    if horizon_rows:
        rows = []
        for row in horizon_rows:
            if row.get("ssl_objective") != "market_state_multitask":
                continue
            rows.append(
                (
                    str(row.get("horizon", "")),
                    _empty_to_na(row.get("matched_run_count")),
                    _format_float(_row_float(row, "mean_delta_macro_f1")),
                    _format_float(_row_float(row, "mean_delta_mcc")),
                    _format_float(_row_float(row, "mean_delta_ece")),
                    _format_float(_row_float(row, "mean_delta_brier_score")),
                )
            )
        if rows:
            lines.extend(
                _markdown_table(
                    (
                        "horizon",
                        "matched rows",
                        "mean delta macro-F1",
                        "mean delta MCC",
                        "mean delta ECE",
                        "mean delta Brier",
                    ),
                    rows,
                )
            )
            lines.append("")
    lines.extend(
        _wrap_bullets(
            [
                "- SSL-v2 predictive improvement is reported only when matched macro-F1 "
                "and MCC deltas support it in the stored scope.",
                "- SSL-v2 calibration improvement is reported only when ECE and Brier "
                "deltas both support it.",
                "- Broad SSL improvement remains bounded by the combined SSL-v1 and "
                "SSL-v2 evidence.",
            ]
        )
    )
    if claim_status:
        lines.append("")
        lines.extend(
            _markdown_table(
                ("claim", "status"),
                [
                    (claim_id, status)
                    for claim_id, status in sorted(claim_status.items())
                ],
            )
        )
    return lines


def _all_rows_improve(rows: Sequence[Mapping[str, str]], column: str) -> bool:
    values = [value for row in rows for value in [_row_float(row, column)] if value is not None]
    return bool(values) and all(value > 0.0 for value in values)


def _render_figure_index(data: _FinalReportData) -> list[str]:
    figures = data.figures
    if not figures.available:
        return [
            f"Skipped: {figures.reason}.",
            "No diagnostic figure evidence is implied without a figure manifest.",
        ]
    completed = [entry for entry in figures.entries if str(entry.get("status", "")) == "completed"]
    skipped = [entry for entry in figures.entries if str(entry.get("status", "")) == "skipped"]
    lines: list[str] = []
    if figures.smoke_test_only:
        lines.extend(
            [
                "Figures are smoke-test diagnostics only and are not empirical evidence.",
                "",
            ]
        )
    if completed:
        rows = [
            (
                str(entry.get("figure_id", "")),
                str(entry.get("title", "")),
                _empty_to_na(_optional_entry_str(entry, "file_path")),
                _figure_description(entry),
            )
            for entry in completed
        ]
        lines.extend(_markdown_table(("figure", "title", "path", "description"), rows))
    else:
        lines.append("No completed figures are recorded in the manifest.")
    if skipped:
        lines.extend(["", "Skipped plots:", ""])
        skipped_rows = [
            (
                str(entry.get("figure_id", "")),
                _empty_to_na(_optional_entry_str(entry, "reason")),
            )
            for entry in skipped
        ]
        lines.extend(_markdown_table(("figure", "reason"), skipped_rows))
    return lines


def _optional_entry_str(entry: Mapping[str, Any], key: str) -> str:
    value = entry.get(key)
    return "" if value is None else str(value)


def _figure_description(entry: Mapping[str, Any]) -> str:
    figure_id = str(entry.get("figure_id", ""))
    descriptions = {
        "reliability_curve": "confidence calibration from stored predictions",
        "macro_f1_by_fold": "fold-level macro-F1 diagnostic",
        "macro_f1_by_horizon": "mean macro-F1 across horizons",
        "ece_by_horizon": "mean calibration error across horizons",
        "ssl_matched_delta": "matched supervised-vs-SSL deltas only",
        "confidence_threshold_eligible_fraction": "retained sample fraction by confidence",
        "confidence_threshold_macro_f1": "macro-F1 on retained high-confidence samples",
        "cost_adjusted_proxy": "proxy diagnostics only when artefacts exist",
        "regime_breakdown": "regime-labelled predictions only when present",
    }
    if figure_id.startswith("confusion_matrix_h"):
        return "canonical up/stationary/down confusion matrix"
    return descriptions.get(figure_id, "stored-artefact diagnostic figure")


def _full_grid_interpretation(
    grid: _FullGridArtefacts,
    matched_rows: Sequence[Mapping[str, str]],
    invalid_failures: Sequence[Mapping[str, str]],
) -> list[str]:
    if grid.evidence_level == "smoke_test_only":
        return ["Interpretation: smoke-test-only; no empirical SSL claim is made."]
    if invalid_failures:
        return [
            "Interpretation: failed runs are present, so aggregate claims are "
            "limited to completed runs and should not be described as a complete grid."
        ]
    if not matched_rows:
        return [
            "Interpretation: no matched supervised-vs-SSL pairs are available, "
            "so no SSL delta claim is made."
        ]
    lines = ["Interpretation:"]
    for objective in ("masked_reconstruction", "next_field"):
        rows = [row for row in matched_rows if row.get("ssl_objective") == objective]
        if not rows:
            lines.append(f"- {objective}: no matched rows.")
            continue
        macro = _outcome_counts(rows, "macro_f1_outcome")
        mcc = _outcome_counts(rows, "mcc_outcome")
        ece = _outcome_counts(rows, "ece_outcome")
        macro_mean = _mean_delta(rows, "delta_macro_f1")
        mcc_mean = _mean_delta(rows, "delta_mcc")
        ece_mean = _mean_delta(rows, "delta_ece")
        lines.append(
            f"- {objective}: mean deltas macro-F1 {_format_float(macro_mean)}, "
            f"MCC {_format_float(mcc_mean)}, ECE {_format_float(ece_mean)}; "
            f"outcomes macro-F1 {macro}, MCC {mcc}, ECE {ece}."
        )
        lines.append(
            "  No overall SSL improvement is supported; report any deltas metric-by-metric."
        )
    return lines


def _mean_delta(rows: Sequence[Mapping[str, str]], column: str) -> float | None:
    values = [value for row in rows for value in [_row_float(row, column)] if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _outcome_counts(rows: Sequence[Mapping[str, str]], column: str) -> str:
    counts = {"win": 0, "loss": 0, "tie": 0}
    for row in rows:
        value = row.get(column, "")
        if value in counts:
            counts[value] += 1
    return f"{counts['win']} win/{counts['loss']} loss/{counts['tie']} tie"


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


def _render_feature_ablations(data: _FinalReportData) -> list[str]:
    if data.feature_ablation.directory is None or data.feature_ablation.summary is None:
        return [
            "Skipped: FI-2010 microstructure feature-ablation artefacts were not "
            "supplied or were unavailable.",
        ]

    summary = data.feature_ablation.summary
    analysis_summary = data.feature_ablation_analysis.summary or {}
    claim_assessment = data.feature_ablation_analysis.json_payloads.get(
        "feature_claim_assessment.json",
        {},
    )
    claims = claim_assessment.get("claims", {}) if isinstance(claim_assessment, Mapping) else {}
    aggregate_rows = data.feature_ablation.csv_rows.get("aggregate_summary.csv", [])
    delta_rows = data.feature_ablation.csv_rows.get("feature_delta_summary.csv", [])
    stability_rows = data.feature_ablation_analysis.csv_rows.get("feature_group_stability.csv", [])
    snapshot_rows = data.feature_ablation_analysis.csv_rows.get(
        "snapshot_order_flow_proxy_scope.csv",
        [],
    )
    smoke = bool(summary.get("smoke_test"))
    evidence_status = _mapping_str(analysis_summary, "evidence_status", default="partial_real")
    scope_summary = analysis_summary if analysis_summary else summary
    horizons = _mapping_list(scope_summary, "horizons")
    models = _mapping_list(scope_summary, "models")
    folds = _mapping_list(scope_summary, "folds")
    seeds = _mapping_list(scope_summary, "seeds")
    nonlinear_models = [model for model in models if str(model) == "gradient_boosting"]
    added_horizons = sorted(str(horizon) for horizon in horizons if str(horizon) in {"20", "50"})
    broader_snapshot = (
        claims.get("broader_horizon_snapshot_proxy_importance", {})
        if isinstance(claims, Mapping)
        else {}
    )
    nonlinear_claim = (
        claims.get("nonlinear_model_feature_stability", {})
        if isinstance(claims, Mapping)
        else {}
    )
    artefact_policy = _nested_mapping(summary, "artefact_policy") or {}
    lines = [
        "Feature-ablation evidence is reported as a scoped feature-stability "
        "analysis over FI-2010 snapshot columns. Unsupported event-level groups "
        "remain explicit.",
        "`snapshot_order_flow_proxy` is a labelled snapshot proxy derived from "
        "FI-2010 matrices. It should not be interpreted as true event-level "
        "order-flow imbalance; it is not true event-level order-flow "
        "imbalance evidence.",
        "The analysis is not causal feature importance and does not establish "
        "universal feature importance across all models or horizons.",
        "",
    ]
    if smoke:
        lines.extend(
            [
                "Only smoke-test artefacts were supplied, so this section is a "
                "pipeline check rather than empirical evidence.",
                "",
            ]
        )
    status_rows = [
        ("runner_version", _mapping_str(summary, "runner_version")),
        ("evidence_status", evidence_status),
        ("smoke_test", str(smoke)),
        (
            "completed_run_count",
            _mapping_str(scope_summary, "completed_run_count", default="not available"),
        ),
        (
            "failed_run_count",
            _mapping_str(scope_summary, "failed_run_count", default="not available"),
        ),
        ("folds", _join_values(folds)),
        ("horizons", _join_values(horizons)),
        ("seeds", _join_values(seeds)),
        ("models", _join_values(models)),
        ("horizon_20_50_added", "yes" if added_horizons else "no"),
        ("non_linear_model_evidence", _join_values(nonlinear_models)),
        ("feature_groups", _join_values(_mapping_list(summary, "feature_groups"))),
        ("proxy_groups", _join_values(_mapping_list(summary, "proxy_groups"))),
        (
            "unsupported_groups",
            _join_values(_mapping_list(summary, "unsupported_groups")),
        ),
        ("raw_predictions_saved", str(bool(artefact_policy.get("raw_predictions_saved")))),
    ]
    lines.extend(_markdown_table(("field", "value"), status_rows))
    if data.feature_ablation_analysis.directory is not None:
        lines.extend(
            [
                "",
                "Feature-stability artefact:",
                "",
                f"- `{_display_path(data.feature_ablation_analysis.directory)}`",
            ]
        )
    if claims:
        lines.extend(["", "Feature-claim assessment:", ""])
        claim_rows: list[tuple[str, str, str]] = []
        for claim_id in (
            "feature_ablation_infrastructure",
            "horizon10_logistic_ridge_snapshot_proxy_importance",
            "broader_horizon_snapshot_proxy_importance",
            "nonlinear_model_feature_stability",
            "causal_feature_importance",
            "true_event_level_ofi",
        ):
            payload = claims.get(claim_id, {}) if isinstance(claims, Mapping) else {}
            if isinstance(payload, Mapping):
                claim_rows.append(
                    (
                        claim_id,
                        str(payload.get("status", "")),
                        str(payload.get("reason", "")),
                    )
                )
        lines.extend(_markdown_table(("claim", "status", "reason"), claim_rows))
    if snapshot_rows:
        degraded = sum(
            1
            for row in snapshot_rows
            if str(row.get("macro_f1_degraded_when_removed", "")).lower() == "true"
        )
        lines.extend(
            [
                "",
                "`snapshot_order_flow_proxy` scope:",
                "",
                (
                    f"- Removing the labelled snapshot proxy degraded macro-F1 in "
                    f"{degraded}/{len(snapshot_rows)} matched rows in the analysed scope."
                ),
                (
                    "- Horizon 20/50 status: "
                    f"{broader_snapshot.get('status', 'not available')}."
                ),
                (
                    "- Non-linear model status: "
                    f"{nonlinear_claim.get('status', 'not available')}."
                ),
            ]
        )
    if stability_rows:
        lines.extend(["", "Feature-group stability rows:", ""])
        stability_table_rows: list[tuple[str, str, str, str, str]] = []
        for row in stability_rows[:10]:
            stability_table_rows.append(
                (
                    row.get("feature_group", ""),
                    _format_float(_row_float(row, "mean_delta_macro_f1")),
                    _format_float(_row_float(row, "mean_delta_mcc")),
                    _format_float(_row_float(row, "macro_f1_degradation_fraction")),
                    _format_float(_row_float(row, "stability_score")),
                )
            )
        lines.extend(
            _markdown_table(
                (
                    "feature group",
                    "mean delta macro-F1",
                    "mean delta MCC",
                    "degradation fraction",
                    "stability score",
                ),
                stability_table_rows,
            )
        )
    if not bool(artefact_policy.get("raw_predictions_saved")):
        lines.extend(
            [
                "",
                "Execution-aware ablation diagnostics require retained prediction-level "
                "outputs or a targeted rerun.",
            ]
        )
    if aggregate_rows:
        lines.extend(["", "Aggregate rows:", ""])
        aggregate_table_rows: list[tuple[str, str, str, str, str, str, str]] = []
        for row in aggregate_rows[:10]:
            aggregate_table_rows.append(
                (
                    row.get("horizon", ""),
                    row.get("model", ""),
                    row.get("ablation_mode", ""),
                    row.get("feature_group", ""),
                    row.get("completed_runs", ""),
                    _format_float(_row_float(row, "mean_macro_f1")),
                    _format_float(_row_float(row, "mean_mcc")),
                )
            )
        lines.extend(
            _markdown_table(
                (
                    "horizon",
                    "model",
                    "mode",
                    "feature group",
                    "completed",
                    "mean macro-F1",
                    "mean MCC",
                ),
                aggregate_table_rows,
            )
        )
    if delta_rows:
        lines.extend(["", "Matched deltas versus all-features baseline:", ""])
        delta_table_rows: list[tuple[str, str, str, str, str, str, str]] = []
        for row in delta_rows[:10]:
            delta_table_rows.append(
                (
                    row.get("horizon", ""),
                    row.get("model", ""),
                    row.get("ablation_mode", ""),
                    row.get("feature_group", ""),
                    _format_float(_row_float(row, "delta_macro_f1")),
                    _format_float(_row_float(row, "delta_mcc")),
                    row.get("interpretation", ""),
                )
            )
        lines.extend(
            _markdown_table(
                (
                    "horizon",
                    "model",
                    "mode",
                    "feature group",
                    "delta macro-F1",
                    "delta MCC",
                    "interpretation",
                ),
                delta_table_rows,
            )
        )
    return lines


def _render_execution(data: _FinalReportData) -> list[str]:
    lines = [
        "Execution-aware sections are offline execution-aware proxy diagnostics only.",
        (
            "They separate classification performance from confidence-filtered signal "
            "quality and cost-adjusted proxy diagnostics, and they do not support "
            "live-trading, profitability or PnL claims."
        ),
        "",
        "Execution-v3 is offline proxy analysis. To be explicit about its scope:",
        "",
        "- it is not PnL;",
        "- it is not live-trading evidence;",
        "- it is not a production execution simulation;",
        (
            "- the confidence, cost, latency, fill and adverse-selection diagnostics "
            "test whether predictive metrics survive simple execution-like frictions."
        ),
        "",
        (
            "A richer reviewer-facing breakdown of active fraction, turnover proxy, "
            "latency sensitivity, cost sensitivity, fill-assumption sensitivity and "
            "the adverse-selection proxy is generated by `analyse-fi2010-execution-v3`."
        ),
        (
            "It explicitly skips regime diagnostics and is stored under "
            "`reports/execution_v3_analysis/execution_v3_analysis.md`."
        ),
        "",
    ]
    lines.extend(["Execution-v3 status:", ""])
    lines.extend(_markdown_table(("field", "value"), _execution_v3_status_rows(data)))
    if data.execution_v3.directory is not None and data.execution_v3.summary is not None:
        lines.extend(["", "Confidence filtering:", ""])
        lines.extend(
            _markdown_table(
                (
                    "model",
                    "objective",
                    "horizon",
                    "threshold",
                    "active fraction",
                    "macro-F1",
                    "cost-adjusted proxy",
                    "status",
                ),
                _execution_v3_confidence_rows(data.execution_v3),
            )
        )
        lines.extend(["", "Cost sensitivity:", ""])
        lines.extend(
            _markdown_table(
                (
                    "model",
                    "objective",
                    "horizon",
                    "fee bps",
                    "spread x",
                    "gross proxy",
                    "cost-adjusted proxy",
                    "cost mode",
                ),
                _execution_v3_cost_rows(data.execution_v3),
            )
        )
        latency_rows = _execution_v3_latency_rows(data.execution_v3)
        if latency_rows:
            lines.extend(["", "Latency sensitivity:", ""])
            lines.extend(
                _markdown_table(
                    (
                        "model",
                        "objective",
                        "horizon",
                        "latency",
                        "active trades",
                        "hit rate",
                        "cost-adjusted proxy",
                        "status",
                    ),
                    latency_rows,
                )
            )
        lines.extend(["", "Fill assumptions:", ""])
        lines.extend(
            _markdown_table(
                (
                    "model",
                    "objective",
                    "fill mode",
                    "filled",
                    "fill fraction",
                    "hit rate",
                    "cost-adjusted proxy",
                    "status",
                ),
                _execution_v3_fill_rows(data.execution_v3),
            )
        )
        adverse_rows = _execution_v3_adverse_rows(data.execution_v3)
        if adverse_rows:
            lines.extend(["", "Adverse-selection proxy:", ""])
            lines.extend(
                _markdown_table(
                    (
                        "model",
                        "objective",
                        "horizon",
                        "confidence bucket",
                        "fill assumption",
                        "adverse fraction",
                        "mode",
                    ),
                    adverse_rows,
                )
            )
    elif data.execution_v3.directory is None:
        lines.append("")
        lines.append("Execution-v3 artefacts were not supplied, so no execution-v3 claim is made.")

    if data.execution.directory is not None and data.execution.summary is not None:
        lines.extend(["", "Legacy execution-v2 proxy snapshot:", ""])
        lines.extend(_render_execution_v2_table(data))
    skipped_rows = _execution_v3_skipped_rows(data.execution_v3)
    if skipped_rows:
        lines.extend(["", "Skipped diagnostics:", ""])
        lines.extend(_markdown_table(("diagnostic", "scope", "reason"), skipped_rows))
    lines.extend(
        [
            "",
            "Conservative interpretation: execution-v3 can show how stored FI-2010 "
            "signals respond to confidence filters, costs, latency and fill proxy "
            "assumptions. It does not establish deployable execution quality.",
        ]
    )
    return lines


def _render_execution_v2_table(data: _FinalReportData) -> list[str]:
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
    return _markdown_table(
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


def _execution_v3_status(data: _FinalReportData) -> str:
    artefacts = data.execution_v3
    if artefacts.directory is None or artefacts.summary is None:
        return "not supplied; no execution-v3 claim is made"
    summary = artefacts.summary
    if bool(summary.get("smoke_test")):
        return "smoke-test-only; not empirical evidence"
    return (
        "offline execution-aware proxy diagnostic loaded; "
        f"payoff_mode={_mapping_str(summary, 'payoff_mode')}, "
        f"cost_mode={_mapping_str(summary, 'cost_mode')}"
    )


def _execution_centrepiece_status(data: _FinalReportData) -> str:
    artefacts = data.execution_centrepiece
    if artefacts.directory is None or artefacts.summary is None:
        return "not supplied; no forecasting-versus-signal-quality centrepiece claim is made"
    summary = artefacts.summary
    if bool(summary.get("smoke_test")):
        return "smoke-test-only; not empirical evidence"
    return (
        "forecasting-versus-signal-quality gap centrepiece loaded; "
        "offline diagnostic only"
    )


def _execution_v3_status_rows(data: _FinalReportData) -> list[tuple[str, str]]:
    artefacts = data.execution_v3
    if artefacts.directory is None or artefacts.summary is None:
        return [
            ("status", "missing"),
            ("claim", "no execution-v3 claim is made"),
            ("reason", "execution-v3 artefacts were not supplied or were unavailable"),
        ]
    summary = artefacts.summary
    return [
        ("status", _execution_v3_status(data)),
        ("payoff_mode", _mapping_str(summary, "payoff_mode")),
        ("cost_mode", _mapping_str(summary, "cost_mode")),
        ("smoke_test_status", _mapping_str(summary, "smoke_test_status")),
        (
            "diagnostics_produced",
            _join_values(_mapping_list(summary, "diagnostics_produced")),
        ),
        (
            "diagnostics_skipped",
            _join_values(_mapping_list(summary, "diagnostics_skipped")),
        ),
    ]


def _execution_v3_confidence_rows(artefacts: _OptionalArtefacts) -> list[tuple[str, ...]]:
    rows = artefacts.csv_rows.get("confidence_threshold_aggregate.csv", [])
    rendered: list[tuple[str, ...]] = []
    for row in rows[:10]:
        rendered.append(
            (
                row.get("model_family", ""),
                row.get("pretraining_objective", ""),
                row.get("horizon", ""),
                row.get("threshold", ""),
                _format_float(_row_float(row, "mean_active_trade_fraction")),
                _format_float(_row_float(row, "mean_macro_f1")),
                _format_float(_row_float(row, "mean_net_cost_adjusted_proxy")),
                row.get("status", ""),
            )
        )
    return rendered


def _execution_v3_cost_rows(artefacts: _OptionalArtefacts) -> list[tuple[str, ...]]:
    rows = artefacts.csv_rows.get("cost_sensitivity_summary.csv", [])
    rendered: list[tuple[str, ...]] = []
    for row in rows[:10]:
        rendered.append(
            (
                row.get("model_family", ""),
                row.get("pretraining_objective", ""),
                row.get("horizon", ""),
                row.get("fee_bps", ""),
                row.get("spread_multiplier", ""),
                _format_float(_row_float(row, "gross_proxy")),
                _format_float(_row_float(row, "net_proxy")),
                row.get("cost_mode", ""),
            )
        )
    return rendered


def _execution_v3_latency_rows(artefacts: _OptionalArtefacts) -> list[tuple[str, ...]]:
    rows = artefacts.csv_rows.get("latency_sensitivity_summary.csv", [])
    rendered: list[tuple[str, ...]] = []
    for row in rows[:10]:
        rendered.append(
            (
                row.get("model_family", ""),
                row.get("pretraining_objective", ""),
                row.get("horizon", ""),
                row.get("latency_step", ""),
                row.get("active_trades", ""),
                _format_float(_row_float(row, "directional_hit_rate")),
                _format_float(_row_float(row, "net_proxy")),
                row.get("status", ""),
            )
        )
    return rendered


def _execution_v3_fill_rows(artefacts: _OptionalArtefacts) -> list[tuple[str, ...]]:
    rows = artefacts.csv_rows.get("fill_assumption_summary.csv", [])
    rendered: list[tuple[str, ...]] = []
    for row in rows[:10]:
        rendered.append(
            (
                row.get("model_family", ""),
                row.get("pretraining_objective", ""),
                row.get("fill_mode", ""),
                row.get("filled_count", ""),
                _format_float(_row_float(row, "fill_fraction")),
                _format_float(_row_float(row, "directional_hit_rate_on_filled_trades")),
                _format_float(_row_float(row, "net_proxy")),
                row.get("status", ""),
            )
        )
    return rendered


def _execution_v3_adverse_rows(artefacts: _OptionalArtefacts) -> list[tuple[str, ...]]:
    rows = artefacts.csv_rows.get("adverse_selection_summary.csv", [])
    rendered: list[tuple[str, ...]] = []
    for row in rows[:10]:
        if row.get("status") != "ok":
            continue
        rendered.append(
            (
                row.get("model_family", ""),
                row.get("pretraining_objective", ""),
                row.get("horizon", ""),
                row.get("confidence_bucket", ""),
                row.get("fill_assumption", ""),
                _format_float(_row_float(row, "adverse_fraction")),
                row.get("adverse_selection_mode", ""),
            )
        )
    return rendered


def _execution_v3_skipped_rows(artefacts: _OptionalArtefacts) -> list[tuple[str, str, str]]:
    payload = artefacts.json_payloads.get("skipped_diagnostics.json")
    if not payload:
        return []
    raw = payload.get("skipped")
    if not isinstance(raw, list):
        return []
    rows: list[tuple[str, str, str]] = []
    for item in raw[:20]:
        if not isinstance(item, Mapping):
            continue
        rows.append(
            (
                str(item.get("diagnostic", "")),
                str(item.get("scope", "")),
                str(item.get("skip_reason", "")),
            )
        )
    return rows


def _render_forecasting_signal_gap(data: _FinalReportData) -> list[str]:
    artefacts = data.execution_centrepiece
    if artefacts.directory is None or artefacts.summary is None:
        return [
            "Skipped: execution centrepiece artefacts were not supplied or were unavailable.",
            "The final report therefore relies on the execution-aware proxy summary above.",
        ]
    summary = artefacts.summary
    figure_manifest = artefacts.json_payloads.get("figure_manifest.json", {})
    claim_statuses = summary.get("claim_statuses")
    unavailable = summary.get("unavailable_fields")
    rows = [
        ("centrepiece_report", _display_path(artefacts.directory / "execution_centrepiece.md")),
        (
            "central_figure",
            _centrepiece_figure_path(figure_manifest)
            or _display_path(artefacts.directory / "forecasting_vs_signal_quality.png"),
        ),
        ("raw_predictions_required", _mapping_str(summary, "raw_predictions_required")),
        ("payoff_mode", _mapping_str(summary, "payoff_mode")),
        ("cost_mode", _mapping_str(summary, "cost_mode")),
        (
            "claim_statuses",
            _status_mapping_text(claim_statuses) if isinstance(claim_statuses, Mapping) else "",
        ),
    ]
    lines = [
        "The execution centrepiece is the compact reviewer-facing bridge between "
        "forecast metrics and execution-aware signal-quality proxy diagnostics.",
        (
            "It uses retained execution-v3 analysis tables, retained full-grid "
            "predictive/calibration summaries and no deleted raw predictions."
        ),
        "",
        *_markdown_table(("field", "value"), rows),
        "",
        "Representative metric-to-proxy rows:",
        "",
    ]
    lines.extend(_table_from_rows(artefacts.csv_rows.get("metric_to_proxy_gap.csv", []), limit=6))
    if isinstance(unavailable, Mapping) and unavailable:
        lines.extend(["", "Explicitly unavailable fields:", ""])
        lines.extend(
            _markdown_table(
                ("field", "reason"),
                [(str(key), str(value)) for key, value in sorted(unavailable.items())],
            )
        )
    lines.extend(
        [
            "",
            "Conservative interpretation: the forecasting-versus-signal-quality gap "
            "does not establish profitability or tradability.",
            (
                "It shows why macro-F1 and calibration must be read alongside "
                "confidence filtering, active fraction, turnover proxy, "
                "cost-adjusted proxy, latency sensitivity and adverse-selection "
                "proxy diagnostics."
            ),
        ]
    )
    return lines


def _centrepiece_figure_path(manifest: Mapping[str, Any]) -> str:
    figures = manifest.get("figures")
    if not isinstance(figures, list):
        return ""
    for entry in figures:
        if not isinstance(entry, Mapping):
            continue
        if entry.get("figure_id") == "forecasting_vs_signal_quality":
            return str(entry.get("file_path", ""))
    return ""


def _status_mapping_text(value: Mapping[Any, Any]) -> str:
    return ", ".join(f"{key}={status}" for key, status in sorted(value.items()))


def _table_from_rows(rows: Sequence[Mapping[str, str]], *, limit: int) -> list[str]:
    if not rows:
        return ["No retained metric-to-proxy rows were available."]
    selected = list(rows[:limit])
    preferred = [
        "pretraining_objective",
        "horizon",
        "predictive_macro_f1",
        "predictive_ece",
        "active_fraction_at_0_70",
        "turnover_proxy_at_0_70",
        "cost_adjusted_proxy_at_0_70",
        "latency_degradation_vs_lag0",
        "high_confidence_adverse_selection_proxy",
    ]
    headers = [header for header in preferred if any(header in row for row in selected)]
    rendered_rows = [
        tuple(row.get(header, "unavailable") for header in headers) for row in selected
    ]
    return _markdown_table(headers, rendered_rows)


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


def _render_synthetic_extension(data: _FinalReportData) -> list[str]:
    """Render the synthetic event-level extension section conservatively."""
    boundary = [
        "",
        *_wrap_prose(
            "This extension demonstrates event-level pipeline support under controlled "
            "synthetic regimes. It does not provide real-market evidence. It does not "
            "change FI-2010 limitations. It enables event-level feature validation "
            "that FI-2010 cannot support."
        ),
    ]
    if data.synthetic.directory is None or data.synthetic.summary is None:
        return [
            "Skipped: synthetic event-level extension artefacts were not supplied or "
            "were unavailable.",
            *boundary,
        ]
    summary = data.synthetic.summary
    quality = data.synthetic.json_payloads.get("synthetic_replay_quality.json", {})
    rows = [
        ("evidence_level", _mapping_str(summary, "evidence_level", default="n/a")),
        ("event_count", _mapping_str(summary, "event_count")),
        ("snapshot_count", _mapping_str(summary, "snapshot_count")),
        ("regimes", _join_values(_mapping_list(summary, "regimes"))),
        ("replay_ok", _mapping_str(summary, "replay_ok")),
        ("no_lookahead_ok", _mapping_str(summary, "leakage_ok")),
        ("target", _mapping_str(summary, "target")),
        (
            "crossed_snapshots",
            str(quality.get("crossed_snapshot_count", "n/a"))
            if isinstance(quality, Mapping)
            else "n/a",
        ),
    ]
    lines = _markdown_table(("field", "value"), rows)
    return [*lines, *boundary]


def _render_binance_l2_extension(data: _FinalReportData) -> list[str]:
    """Render the real event-level L2 replay extension section conservatively."""
    boundary = [
        "",
        "Binance L2 replay is scoped to real event-level aggregated depth-stream "
        "ingestion and replay when a local Binance capture is supplied.",
        "Fixture runs are engineering checks. It is crypto-market engineering "
        "evidence, not equity-market evidence.",
        "Binance diff-depth updates are aggregated level updates, not individual "
        "order-event data.",
        "It does not provide profitability, tradability or predictive-success evidence.",
        "It is not live-trading evidence. It complements the FI-2010 and synthetic "
        "evidence.",
    ]
    if data.binance_l2.directory is None or data.binance_l2.summary is None:
        return [
            "Skipped: real event-level L2 replay extension artefacts were not supplied "
            "or were unavailable.",
            *boundary,
        ]
    summary = data.binance_l2.summary
    quality = data.binance_l2.json_payloads.get("replay_quality.json", {})
    crossed = (
        str(quality.get("crossed_count", "n/a")) if isinstance(quality, Mapping) else "n/a"
    )
    rows = [
        ("venue", _mapping_str(summary, "venue", default="binance")),
        ("symbol", _mapping_str(summary, "symbol", default="n/a")),
        ("evidence_level", _mapping_str(summary, "evidence_level", default="n/a")),
        ("diff_event_count", _mapping_str(summary, "diff_event_count")),
        ("applied_event_count", _mapping_str(summary, "applied_event_count")),
        ("snapshot_count", _mapping_str(summary, "snapshot_count")),
        ("feature_row_count", _mapping_str(summary, "feature_row_count")),
        ("replay_ok", _mapping_str(summary, "replay_ok")),
        ("crossed_count", crossed),
    ]
    lines = _markdown_table(("field", "value"), rows)
    return [*lines, *boundary]


def _render_what_this_proves(data: _FinalReportData) -> list[str]:
    legacy_epochs = _mapping_str(data.neural_summary, "max_epochs", default="prior")
    lines = [
        "- The committed artefacts support a traceable multi-fold classical FI-2010 result.",
        f"- A separate, earlier {legacy_epochs}-epoch reduced-scope, single-seed "
        "supervised neural benchmark is reported on its own terms and is not "
        "used as matched-grid or SSL evidence.",
        "- The uncertainty, ablation and proxy-diagnostic layers are generated from stored tables.",
        "- External references are used only to document protocol context.",
    ]
    if data.ssl.admitted:
        lines.append(
            "- A leakage-safe self-supervised pretraining and fine-tuning path "
            "produced SHA256-verified encoder checkpoints and a like-for-like "
            "ssl_transformer vs supervised comparison.",
        )
    if data.full_grid.available and data.full_grid.evidence_level != "smoke_test_only":
        lines.extend(
            _wrap_bullet(
                "- The one-epoch full neural grid artefacts compare supervised "
                "and SSL matrix-transformer variants under matched fold, horizon, "
                "seed, lookback, architecture and preprocessing keys; this is "
                "matched comparison evidence and supports no SSL improvement claim."
            )
        )
    if (
        data.execution_v3.directory is not None
        and data.execution_v3.summary is not None
        and not bool(data.execution_v3.summary.get("smoke_test"))
    ):
        lines.append(
            "- Execution-v3 artefacts support an offline execution-aware proxy "
            "diagnostic over stored FI-2010 full-grid predictions.",
        )
    if (
        data.execution_centrepiece.directory is not None
        and data.execution_centrepiece.summary is not None
        and not bool(data.execution_centrepiece.summary.get("smoke_test"))
    ):
        lines.append(
            "- The execution centrepiece supports a forecasting-versus-signal-quality "
            "gap analysis using retained predictive, calibration and proxy "
            "diagnostic tables.",
        )
    if (
        data.feature_ablation.directory is not None
        and data.feature_ablation.summary is not None
        and not bool(data.feature_ablation.summary.get("smoke_test"))
    ):
        lines.append(
            "- Feature-ablation artefacts support leakage-safe FI-2010 snapshot "
            "feature-family diagnostics with proxy features labelled as proxies.",
        )
    return lines


def _render_what_this_does_not_prove(data: _FinalReportData) -> list[str]:
    ssl_line = (
        "- Self-supervised superiority or SOTA status; the SSL section "
        "reports a like-for-like delta only."
        if data.ssl.admitted
        else "- SSL improvement or SOTA status."
    )
    return [
        "- Profitability or tradability in deployed markets.",
        "- Production execution quality or market-impact realism.",
        "- Unsupported live-trading claims from execution-v3 proxy diagnostics.",
        "- Foundation-model status.",
        ssl_line,
        "- A full-grid SSL improvement claim when the full-grid directory is missing, "
        "smoke-only or contains failed matched runs.",
        "- True order-flow, cancellation, trade-imbalance or queue-position claims "
        "from FI-2010 feature ablations; absent event-level fields remain unsupported.",
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
            "offline execution-aware proxy diagnostics only; queue, impact and "
            "venue mechanics are not modelled",
        ),
        (
            "execution_centrepiece_scope",
            "forecasting-versus-signal-quality gap analysis over retained proxy "
            "tables; no raw predictions or realised execution outcomes are read",
        ),
        (
            "external_scope",
            "protocol context only; no external numeric metrics are copied",
        ),
        (
            "prediction_checkpoint_policy",
            "full predictions and checkpoints are not required by this report builder",
        ),
        (
            "full_neural_grid_scope",
            "reported only when aggregate artefacts are supplied; smoke artefacts "
            "are not empirical evidence",
        ),
        (
            "feature_ablation_scope",
            "snapshot-derived FI-2010 diagnostics only; snapshot-flow columns are "
            "proxies and unsupported event-level groups remain unavailable",
        ),
    ]
    return _markdown_table(("limitation", "detail"), rows)


def _render_traceability(data: _FinalReportData) -> list[str]:
    rows = [
        (label, path, data.input_file_hashes.get(path, _unhashed_traceability_label(path)))
        for label, path in sorted(data.input_artefact_paths.items())
    ]
    return _markdown_table(("artefact", "path", "sha256"), rows)


def _unhashed_traceability_label(path: str) -> str:
    return "not_hashed" if Path(path).suffix else "directory"


def _render_reproduction_commands() -> list[str]:
    return [
        "```bash",
        "python -m chronoslob.cli build-final-empirical-report \\",
        "  --classical experiments/fi2010_multifold_classical \\",
        "  --neural experiments/fi2010_multifold_neural \\",
        "  --uncertainty experiments/fi2010_uncertainty \\",
        "  --ablations experiments/fi2010_brutal_ablations \\",
        "  --execution experiments/fi2010_execution_v2 \\",
        "  --execution-v3 experiments/fi2010_execution_v3 \\",
        "  --external experiments/fi2010_external_context \\",
        "  --neural-full-grid experiments/fi2010_neural_full_grid \\",
        "  --feature-ablations experiments/fi2010_feature_ablations \\",
        "  --feature-ablation-analysis reports/feature_ablation_analysis \\",
        "  --execution-centrepiece reports/execution_centrepiece \\",
        "  --out reports/chronoslob_final_empirical_report.md \\",
        "  --overwrite",
        "",
        "python -m chronoslob.cli build-execution-centrepiece \\",
        "  --execution-analysis reports/execution_v3_analysis \\",
        "  --execution-v3 experiments/fi2010_execution_v3 \\",
        "  --neural-full-grid experiments/fi2010_neural_full_grid \\",
        "  --out reports/execution_centrepiece \\",
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
            raise FileNotFoundError(f"{label} required artefact is missing: {path}")


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


def _read_optional_text(
    path: Path,
    key: str,
    input_paths: dict[str, str],
    file_hashes: dict[str, str],
) -> str | None:
    if not path.is_file():
        return None
    _record_file(path, key, input_paths, file_hashes)
    return path.read_text(encoding="utf-8")


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


def _wrap_prose(text: str, *, width: int = 200) -> list[str]:
    """Wrap a prose paragraph so each physical line stays within ``width``."""
    wrapped = textwrap.wrap(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return wrapped or [text]


def _wrap_bullets(bullets: Sequence[str], *, width: int = 200) -> list[str]:
    """Wrap a sequence of ``- `` bullets, flattening into output lines."""
    lines: list[str] = []
    for bullet in bullets:
        lines.extend(_wrap_bullet(bullet, width=width))
    return lines


def _wrap_bullet(text: str, *, width: int = 200) -> list[str]:
    """Wrap a ``- `` bullet, indenting continuation lines to stay in the list."""
    prefix = "- "
    body = text[len(prefix) :] if text.startswith(prefix) else text
    wrapped = textwrap.wrap(
        body,
        width=width - len(prefix),
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not wrapped:
        return [text]
    return [prefix + wrapped[0], *(f"  {part}" for part in wrapped[1:])]


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


def _format_mapping_counts(payload: Mapping[str, Any]) -> str:
    if not payload:
        return "not available"
    return ", ".join(f"{key}={value}" for key, value in sorted(payload.items()))


def _first_markdown_bullets(text: str | None, *, fallback: str) -> str:
    if not text:
        return fallback
    bullets = [
        line.removeprefix("- ").strip() for line in text.splitlines() if line.startswith("- ")
    ]
    if not bullets:
        return fallback
    return "; ".join(bullets[:3])


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

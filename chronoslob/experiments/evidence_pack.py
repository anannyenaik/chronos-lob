"""Release-grade evidence-pack orchestration for ChronosLOB.

The evidence pack is intentionally conservative. It inventories stored
artefacts, separates smoke diagnostics from empirical outputs, audits common
public claims against those artefacts and writes public summaries that avoid
stronger language than the stored evidence supports.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from chronoslob import __version__
from chronoslob.experiments.manifests import sha256_file, stable_json_dumps
from chronoslob.utils.paths import project_root

__all__ = [
    "ARTEFACT_INVENTORY_COLUMNS",
    "EVIDENCE_PACK_VERSION",
    "ArtefactInventoryRow",
    "ClaimAuditEntry",
    "EvidencePackBuildResult",
    "EvidencePackConfig",
    "EvidencePackError",
    "audit_claims",
    "build_evidence_pack",
    "discover_artefacts",
]

EVIDENCE_PACK_VERSION = "release/evidence-pack/v1"

ArtefactStatus = Literal[
    "missing",
    "smoke_test_only",
    "partial_real",
    "complete_real",
    "invalid",
    "stale",
    "unsupported",
    "unknown_staleness",
]
ClaimStatus = Literal[
    "supported",
    "partially_supported",
    "unsupported",
    "forbidden",
    "needs_real_evidence",
    "smoke_only",
]
StalenessState = Literal["fresh", "stale", "unknown"]

ARTEFACT_INVENTORY_COLUMNS: tuple[str, ...] = (
    "artefact_name",
    "artefact_type",
    "path",
    "exists",
    "status",
    "smoke_test",
    "completed_runs",
    "failed_runs",
    "skipped_runs",
    "input_hashes_present",
    "output_hashes_present",
    "generated_at",
    "git_commit",
    "supports_claims",
    "limitations",
    "notes",
)

_PUBLIC_BULLET_CONSERVATIVE = "public_bullets_conservative.md"
_PUBLIC_BULLET_STRONG = "public_bullets_strong_if_supported.md"

_OUTPUT_FILENAMES: tuple[str, ...] = (
    "evidence_pack_manifest.json",
    "evidence_pack_summary.md",
    "artefact_inventory.csv",
    "claim_audit.json",
    "claim_audit.md",
    "supported_claims.md",
    "unsupported_claims.md",
    _PUBLIC_BULLET_CONSERVATIVE,
    _PUBLIC_BULLET_STRONG,
    "reproduction_commands.md",
    "release_checklist.md",
    "readme_result_snapshot.md",
)

_REAL_STATUSES: set[ArtefactStatus] = {"complete_real"}
_PARTIAL_STATUSES: set[ArtefactStatus] = {"partial_real", "stale", "unknown_staleness"}
_SMOKE_STATUSES: set[ArtefactStatus] = {"smoke_test_only"}
_BAD_STATUSES: set[ArtefactStatus] = {"missing", "invalid", "unsupported"}

_EXPANDED_FEATURE_ABLATION_FOLDS = {"fold_1", "fold_2", "fold_3", "fold_4", "fold_5"}
_EXPANDED_FEATURE_ABLATION_HORIZONS = {10, 20, 50}
_EXPANDED_FEATURE_ABLATION_SEEDS = {0, 1, 2}
_EXPANDED_FEATURE_ABLATION_MODELS = {
    "logistic",
    "ridge",
    "elastic_net",
    "gradient_boosting",
}
_EXPANDED_FEATURE_ABLATION_RUNS = 5040

_FORBIDDEN_CLAIMS: tuple[str, ...] = (
    "profitable trading strategy",
    "live trading system",
    "production execution simulator",
    "state-of-the-art",
    "foundation model",
    "true OFI on FI-2010",
    "queue-position modelling on FI-2010",
    "tradable alpha",
    "PnL",
)

_PUBLIC_CLAIM_ALLOW_MARKERS: tuple[str, ...] = (
    "avoid",
    "blocked",
    "caveat",
    "do not",
    "does not",
    "forbidden",
    "is not",
    "must not",
    "no ",
    "not ",
    "not a",
    "not an",
    "unsupported",
    "without",
)


class EvidencePackError(RuntimeError):
    """Raised when strict evidence-pack validation fails."""


@dataclass(frozen=True)
class EvidencePackConfig:
    """Configuration for building a release evidence pack."""

    out_dir: Path = Path("reports/evidence_pack")
    neural_full_grid_dir: Path = Path("experiments/fi2010_neural_full_grid")
    proper_training_dir: Path = Path("experiments/fi2010_neural_proper_training_subset_v2")
    ssl_analysis_dir: Path = Path("reports/ssl_failure_analysis")
    figures_dir: Path = Path("reports/figures/fi2010_neural_full_grid")
    execution_v3_dir: Path = Path("experiments/fi2010_execution_v3")
    execution_v3_analysis_dir: Path = Path("reports/execution_v3_analysis")
    feature_ablations_dir: Path = Path("experiments/fi2010_feature_ablations")
    feature_ablation_analysis_dir: Path = Path("reports/feature_ablation_analysis")
    ablation_figures_dir: Path = Path("reports/figures/fi2010_feature_ablations")
    final_report_path: Path = Path("reports/chronoslob_final_empirical_report.md")
    synthetic_lob_dir: Path = Path("reports/synthetic_lob_extension")
    classical_dir: Path = Path("experiments/fi2010_multifold_classical")
    ssl_dir: Path = Path("experiments/fi2010_ssl")
    feature_audit_dir: Path | None = Path("reports/feature_audit")
    project_audit_dir: Path | None = Path("reports/report_archive")
    strict: bool = True
    allow_smoke_test: bool = False
    overwrite: bool = False


@dataclass(frozen=True)
class _ArtefactSpec:
    name: str
    artefact_type: str
    path: Path | None
    required_files: tuple[str, ...]
    metadata_files: tuple[str, ...]
    allow_file: bool = False
    limitations: str = ""
    notes: str = ""


@dataclass(frozen=True)
class _HashReference:
    path: Path
    expected_sha256: str
    kind: Literal["input", "output"]


@dataclass(frozen=True)
class _LoadedArtefact:
    payloads: dict[str, dict[str, Any]]
    csv_rows: dict[str, list[dict[str, str]]]
    text_files: dict[str, str]
    invalid_reasons: list[str]


@dataclass(frozen=True)
class _StalenessResult:
    state: StalenessState
    reason: str


@dataclass
class ArtefactInventoryRow:
    """One row in the evidence artefact inventory."""

    artefact_name: str
    artefact_type: str
    path: str
    exists: bool
    status: ArtefactStatus
    smoke_test: bool
    completed_runs: int | None
    failed_runs: int | None
    skipped_runs: int | None
    input_hashes_present: bool
    output_hashes_present: bool
    generated_at: str | None
    git_commit: str | None
    limitations: str
    notes: str
    required_files_missing: list[str] = field(default_factory=list)
    supporting_files: list[str] = field(default_factory=list)
    supports_claims: list[str] = field(default_factory=list)
    payloads: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    csv_rows: dict[str, list[dict[str, str]]] = field(default_factory=dict, repr=False)
    text_files: dict[str, str] = field(default_factory=dict, repr=False)

    def to_csv_row(self) -> dict[str, str]:
        """Return the public CSV representation."""
        return {
            "artefact_name": self.artefact_name,
            "artefact_type": self.artefact_type,
            "path": self.path,
            "exists": "true" if self.exists else "false",
            "status": self.status,
            "smoke_test": "true" if self.smoke_test else "false",
            "completed_runs": _optional_int_text(self.completed_runs),
            "failed_runs": _optional_int_text(self.failed_runs),
            "skipped_runs": _optional_int_text(self.skipped_runs),
            "input_hashes_present": "true" if self.input_hashes_present else "false",
            "output_hashes_present": "true" if self.output_hashes_present else "false",
            "generated_at": self.generated_at or "",
            "git_commit": self.git_commit or "",
            "supports_claims": "; ".join(sorted(set(self.supports_claims))),
            "limitations": self.limitations,
            "notes": self.notes,
        }

    def to_manifest_payload(self) -> dict[str, Any]:
        """Return the machine-readable manifest representation."""
        return {
            **self.to_csv_row(),
            "exists": self.exists,
            "smoke_test": self.smoke_test,
            "completed_runs": self.completed_runs,
            "failed_runs": self.failed_runs,
            "skipped_runs": self.skipped_runs,
            "input_hashes_present": self.input_hashes_present,
            "output_hashes_present": self.output_hashes_present,
            "required_files_missing": list(self.required_files_missing),
            "supporting_files": list(self.supporting_files),
            "supports_claims": sorted(set(self.supports_claims)),
        }


@dataclass
class ClaimAuditEntry:
    """One audited public claim."""

    claim_id: str
    claim_text: str
    status: ClaimStatus
    supporting_artefacts: list[str]
    required_artefacts: list[str]
    reason: str
    safe_rewording: str
    category: str

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serialisable payload."""
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "status": self.status,
            "supporting_artefacts": list(self.supporting_artefacts),
            "required_artefacts": list(self.required_artefacts),
            "reason": self.reason,
            "safe_rewording": self.safe_rewording,
            "category": self.category,
        }


@dataclass(frozen=True)
class EvidencePackBuildResult:
    """Summary returned by the evidence-pack builder."""

    out_dir: Path
    manifest_path: Path
    files_written: tuple[Path, ...]
    inventory: tuple[ArtefactInventoryRow, ...]
    claim_audit: tuple[ClaimAuditEntry, ...]
    warnings: tuple[str, ...]
    strict_violations: tuple[str, ...]


def build_evidence_pack(config: EvidencePackConfig) -> EvidencePackBuildResult:
    """Build an evidence pack from stored artefacts."""
    out_dir = Path(config.out_dir)
    _prepare_output_dir(out_dir, overwrite=config.overwrite)

    inventory = discover_artefacts(config)
    claim_audit = audit_claims(inventory)
    _attach_claim_support(inventory, claim_audit)

    public_claim_violations = _scan_public_claim_violations(config.final_report_path)
    warnings = _build_warnings(
        inventory,
        public_claim_violations=public_claim_violations,
        allow_smoke_test=config.allow_smoke_test,
    )
    strict_violations = _strict_violations(
        inventory,
        public_claim_violations=public_claim_violations,
        allow_smoke_test=config.allow_smoke_test,
    )
    if config.strict and strict_violations:
        raise EvidencePackError("; ".join(strict_violations))

    files_written = _write_pack_outputs(
        out_dir=out_dir,
        config=config,
        inventory=inventory,
        claim_audit=claim_audit,
        warnings=warnings,
        strict_violations=strict_violations,
    )
    return EvidencePackBuildResult(
        out_dir=out_dir,
        manifest_path=out_dir / "evidence_pack_manifest.json",
        files_written=tuple(files_written),
        inventory=tuple(inventory),
        claim_audit=tuple(claim_audit),
        warnings=tuple(warnings),
        strict_violations=tuple(strict_violations),
    )


def discover_artefacts(config: EvidencePackConfig) -> list[ArtefactInventoryRow]:
    """Discover and classify all evidence-pack artefact groups."""
    current_commit = _current_git_commit()
    records: list[ArtefactInventoryRow] = []
    for spec in _artefact_specs(config):
        records.append(_classify_artefact(spec, current_commit=current_commit))
    return records


def audit_claims(records: Sequence[ArtefactInventoryRow]) -> list[ClaimAuditEntry]:
    """Audit common public claims against the supplied artefact inventory."""
    by_name = {record.artefact_name: record for record in records}
    entries: list[ClaimAuditEntry] = [
        _audit_infrastructure_claim(
            "general.reproducible_platform",
            "ChronosLOB is a reproducible LOB research platform",
            ["project_audit_archive"],
            by_name,
            safe="ChronosLOB includes local tooling and artefact inventories for "
            "reproducible LOB research workflows.",
            reason_supported="Project-audit archive artefacts are present.",
        ),
        _audit_infrastructure_claim(
            "general.leakage_safe_fi2010",
            "ChronosLOB uses leakage-safe FI-2010 evaluation",
            ["fi2010_classical_benchmarks"],
            by_name,
            safe="ChronosLOB stores FI-2010 evaluation artefacts with train/test "
            "protocol metadata and leakage-control documentation.",
            reason_supported="Stored FI-2010 benchmark artefacts are present.",
        ),
        _audit_infrastructure_claim(
            "general.train_only_ssl",
            "ChronosLOB includes train-only SSL pretraining",
            ["fi2010_ssl_runner_outputs", "fi2010_neural_full_grid"],
            by_name,
            safe="ChronosLOB includes code paths for train-only SSL pretraining; "
            "empirical claims require real non-smoke SSL artefacts.",
            reason_supported="Real SSL artefacts or full-grid SSL objectives are present.",
            any_required=True,
        ),
        _audit_infrastructure_claim(
            "general.supervised_vs_ssl_transformers",
            "ChronosLOB compares supervised and SSL transformers",
            ["fi2010_neural_full_grid"],
            by_name,
            safe="ChronosLOB includes infrastructure to compare supervised and SSL "
            "transformers under matched FI-2010 settings.",
            reason_supported="Full-grid aggregate and SSL comparison artefacts are present.",
        ),
        _audit_infrastructure_claim(
            "general.execution_proxy_diagnostics",
            "ChronosLOB includes execution-aware proxy diagnostics",
            ["execution_v3_outputs"],
            by_name,
            safe="ChronosLOB includes offline execution-aware proxy diagnostics with "
            "explicit limitations.",
            reason_supported="Execution-v3 artefacts are present.",
        ),
        _audit_infrastructure_claim(
            "general.execution_proxy_analysis",
            "ChronosLOB includes a richer execution-aware proxy analysis report",
            ["execution_v3_analysis_report"],
            by_name,
            safe="ChronosLOB includes a richer offline execution-aware proxy analysis "
            "covering confidence, turnover, cost, latency, fill and adverse-selection "
            "proxies; regime diagnostics are explicitly skipped.",
            reason_supported="Execution-v3 analysis report artefacts are present.",
        ),
        _audit_infrastructure_claim(
            "general.feature_ablations",
            "ChronosLOB includes microstructure feature ablations",
            ["feature_ablation_outputs"],
            by_name,
            safe="ChronosLOB includes FI-2010 snapshot-feature ablation diagnostics "
            "with proxy and unsupported groups labelled.",
            reason_supported="Feature-ablation artefacts are present.",
        ),
        _audit_feature_ablation_infrastructure_claim(by_name),
    ]

    entries.extend(
        [
            _audit_model_metric_claim(by_name),
            _audit_ssl_implementation_claim(by_name),
            _audit_ssl_delta_claim(by_name, metric="macro_f1"),
            _audit_ssl_delta_claim(by_name, metric="calibration"),
            _audit_ssl_proper_training_claim(by_name),
            _audit_ssl_execution_claim(by_name),
            _audit_gradient_boosting_claim(by_name),
            _audit_feature_group_improvement_claim(by_name),
            _audit_snapshot_proxy_claim(
                by_name,
                claim_id="horizon10_logistic_ridge_snapshot_proxy_importance",
                claim_text=(
                    "snapshot_order_flow_proxy remains important at horizon 10 "
                    "for logistic/ridge"
                ),
                analysis_key="horizon10_logistic_ridge_snapshot_proxy_importance",
                required=["feature_ablation_outputs", "feature_delta_summary.csv"],
                fallback_horizon10=True,
            ),
            _audit_snapshot_proxy_claim(
                by_name,
                claim_id="broader_horizon_snapshot_proxy_importance",
                claim_text="snapshot_order_flow_proxy importance survives horizons 20 and 50",
                analysis_key="broader_horizon_snapshot_proxy_importance",
                required=[
                    "feature_ablation_outputs",
                    "feature_ablation_analysis_report",
                    "snapshot_order_flow_proxy_scope.csv",
                ],
                fallback_horizon10=False,
            ),
            _audit_snapshot_proxy_claim(
                by_name,
                claim_id="nonlinear_model_feature_stability",
                claim_text="Feature-ablation effects appear in a non-linear model slice",
                analysis_key="nonlinear_model_feature_stability",
                required=[
                    "feature_ablation_outputs",
                    "feature_ablation_analysis_report",
                    "feature_claim_assessment.json",
                ],
                fallback_horizon10=False,
            ),
            _audit_confidence_filtering_claim(by_name),
            _forbidden_named_claim(
                "causal_feature_importance",
                "Feature ablations prove causal feature importance",
                "Feature ablations are scoped diagnostics and do not identify causal effects.",
            ),
            _forbidden_named_claim(
                "true_event_level_ofi",
                "snapshot_order_flow_proxy is true event-level OFI",
                (
                    "snapshot_order_flow_proxy is a labelled snapshot proxy derived from "
                    "FI-2010 matrices and is not true event-level order-flow imbalance."
                ),
            ),
        ]
    )
    entries.extend(_audit_synthetic_claims(by_name))
    entries.extend(_forbidden_claim_entries())
    return entries


def _artefact_specs(config: EvidencePackConfig) -> tuple[_ArtefactSpec, ...]:
    return (
        _ArtefactSpec(
            name="fi2010_classical_benchmarks",
            artefact_type="fi2010_supervised_classical",
            path=config.classical_dir,
            required_files=("summary.json", "results_summary.csv"),
            metadata_files=("summary.json", "model_failures.json"),
            limitations="Classical benchmark artefacts do not establish live tradability.",
        ),
        _ArtefactSpec(
            name="fi2010_ssl_runner_outputs",
            artefact_type="fi2010_ssl_runner",
            path=config.ssl_dir,
            required_files=("summary.json", "results_summary.csv", "comparison_summary.csv"),
            metadata_files=("summary.json", "model_failures.json"),
            limitations="SSL rows are evidence only when non-smoke and checkpoint hashes verify.",
        ),
        _ArtefactSpec(
            name="fi2010_neural_full_grid",
            artefact_type="fi2010_neural_full_grid",
            path=config.neural_full_grid_dir,
            required_files=(
                "summary.json",
                "results_summary.csv",
                "aggregate_summary.csv",
                "aggregate_summary.json",
                "ssl_comparison.csv",
                "failures.csv",
            ),
            metadata_files=("summary.json", "aggregate_summary.json"),
            limitations="Full-grid claims require non-smoke aggregate artefacts.",
        ),
        _ArtefactSpec(
            name="fi2010_neural_proper_training_subset",
            artefact_type="fi2010_neural_proper_training",
            path=config.proper_training_dir,
            required_files=(
                "summary.json",
                "config_snapshot.json",
                "results_summary.csv",
                "aggregate_summary.csv",
                "aggregate_summary.json",
                "training_curves_summary.csv",
                "ssl_comparison.csv",
                "failures.csv",
                "sha256_manifest.json",
            ),
            metadata_files=("summary.json", "aggregate_summary.json"),
            limitations=(
                "Proper-training subset claims are limited to the exact folds, "
                "horizons, seeds, lookbacks and objectives stored in the artefacts."
            ),
        ),
        _ArtefactSpec(
            name="ssl_failure_analysis_report",
            artefact_type="ssl_failure_analysis",
            path=config.ssl_analysis_dir,
            required_files=(
                "ssl_failure_analysis.md",
                "ssl_claim_assessment.json",
                "ssl_metric_summary.csv",
                "ssl_delta_by_objective.csv",
            ),
            metadata_files=("summary.json", "ssl_claim_assessment.json"),
            limitations=(
                "The SSL failure analysis reads retained lightweight comparison "
                "tables only; it does not establish a broad SSL improvement or any "
                "calibration improvement."
            ),
        ),
        _ArtefactSpec(
            name="fi2010_figures",
            artefact_type="fi2010_figures",
            path=config.figures_dir,
            required_files=("figure_manifest.json",),
            metadata_files=("figure_manifest.json", "label_mapping_audit.json"),
            limitations="Figures summarise stored artefacts; they are stale if inputs changed.",
        ),
        _ArtefactSpec(
            name="execution_v3_outputs",
            artefact_type="execution_v3",
            path=config.execution_v3_dir,
            required_files=("summary.json", "execution_v3_manifest.json"),
            metadata_files=(
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
            ),
            limitations="Offline proxy diagnostics only; not execution simulation or trading PnL.",
        ),
        _ArtefactSpec(
            name="execution_v3_analysis_report",
            artefact_type="execution_v3_analysis",
            path=config.execution_v3_analysis_dir,
            required_files=(
                "execution_v3_analysis.md",
                "execution_claim_assessment.json",
                "confidence_filtering_summary.csv",
                "cost_sensitivity_summary.csv",
            ),
            metadata_files=(
                "summary.json",
                "execution_claim_assessment.json",
                "confidence_filtering_summary.csv",
                "turnover_proxy_summary.csv",
                "latency_sensitivity_summary.csv",
                "cost_sensitivity_summary.csv",
                "fill_assumption_summary.csv",
                "adverse_selection_proxy_summary.csv",
                "skipped_regime_diagnostics.json",
            ),
            limitations=(
                "Richer offline execution-aware proxy analysis over retained tables; "
                "not execution simulation, not live trading and not realised PnL."
            ),
        ),
        _ArtefactSpec(
            name="feature_registry_audit_outputs",
            artefact_type="feature_registry_audit",
            path=config.feature_audit_dir,
            required_files=("summary.json",),
            metadata_files=("summary.json", "feature_audit.json"),
            limitations="Feature audit outputs document registry checks, not performance.",
            notes="The CLI feature audit is read-only unless a caller stores its output.",
        ),
        _ArtefactSpec(
            name="feature_ablation_outputs",
            artefact_type="feature_ablations",
            path=config.feature_ablations_dir,
            required_files=(
                "summary.json",
                "results_summary.csv",
                "aggregate_summary.csv",
                "feature_delta_summary.csv",
                "ablation_manifest.json",
                "failures.json",
            ),
            metadata_files=("summary.json", "ablation_manifest.json", "failures.json"),
            limitations=(
                "FI-2010 snapshot features cannot prove true event-level OFI, "
                "trade imbalance, cancellation imbalance or queue position."
            ),
        ),
        _ArtefactSpec(
            name="feature_ablation_analysis_report",
            artefact_type="feature_ablation_analysis",
            path=config.feature_ablation_analysis_dir,
            required_files=(
                "summary.json",
                "feature_ablation_analysis.md",
                "feature_delta_by_horizon.csv",
                "feature_delta_by_model.csv",
                "feature_delta_by_fold.csv",
                "feature_delta_by_seed.csv",
                "feature_group_stability.csv",
                "snapshot_order_flow_proxy_scope.csv",
                "feature_claim_assessment.json",
            ),
            metadata_files=(
                "summary.json",
                "feature_claim_assessment.json",
                "figure_manifest.json",
            ),
            limitations=(
                "Scoped feature-stability analysis over retained feature-ablation tables; "
                "not causal feature importance and not event-level OFI evidence."
            ),
        ),
        _ArtefactSpec(
            name="ablation_figures",
            artefact_type="ablation_figures",
            path=config.ablation_figures_dir,
            required_files=("figure_manifest.json",),
            metadata_files=("figure_manifest.json",),
            limitations="Ablation figures visualise stored ablation tables only.",
        ),
        _ArtefactSpec(
            name="final_empirical_report",
            artefact_type="final_report",
            path=config.final_report_path,
            required_files=(),
            metadata_files=(_summary_filename(config.final_report_path),),
            allow_file=True,
            limitations="The generated report is a stored-artefact summary, not a manual paper.",
        ),
        _ArtefactSpec(
            name="synthetic_lob_extension_report",
            artefact_type="synthetic_lob_extension",
            path=config.synthetic_lob_dir,
            required_files=(
                "synthetic_lob_report.md",
                "summary.json",
                "synthetic_replay_quality.json",
                "synthetic_benchmark_summary.csv",
                "synthetic_regime_diagnostics.csv",
                "synthetic_claim_assessment.json",
            ),
            metadata_files=(
                "summary.json",
                "synthetic_claim_assessment.json",
                "synthetic_replay_quality.json",
                "synthetic_feature_summary.csv",
                "synthetic_benchmark_summary.csv",
                "figure_manifest.json",
            ),
            limitations=(
                "Synthetic event-level extension only; controlled stress-test data, "
                "not real-market evidence. Does not change FI-2010 limitations and "
                "does not imply tradability, returns or real-market generalisation."
            ),
        ),
        _ArtefactSpec(
            name="project_audit_archive",
            artefact_type="project_audit",
            path=config.project_audit_dir,
            required_files=(
                "README.md",
                "project_inventory.md",
                "module_inventory.md",
                "test_inventory.md",
                "reproducibility_commands.md",
            ),
            metadata_files=(),
            limitations="Archive files need review when repository state changes.",
        ),
    )


def _classify_artefact(
    spec: _ArtefactSpec,
    *,
    current_commit: str | None,
) -> ArtefactInventoryRow:
    if spec.path is None:
        return ArtefactInventoryRow(
            artefact_name=spec.name,
            artefact_type=spec.artefact_type,
            path="not configured",
            exists=False,
            status="unsupported",
            smoke_test=False,
            completed_runs=None,
            failed_runs=None,
            skipped_runs=None,
            input_hashes_present=False,
            output_hashes_present=False,
            generated_at=None,
            git_commit=None,
            limitations=spec.limitations,
            notes=_join_notes(spec.notes, "No artefact path was configured."),
        )

    path = Path(spec.path)
    exists = path.exists()
    display = _display_path(path)
    if not exists:
        return ArtefactInventoryRow(
            artefact_name=spec.name,
            artefact_type=spec.artefact_type,
            path=display,
            exists=False,
            status="missing",
            smoke_test=False,
            completed_runs=None,
            failed_runs=None,
            skipped_runs=None,
            input_hashes_present=False,
            output_hashes_present=False,
            generated_at=None,
            git_commit=None,
            limitations=spec.limitations,
            notes=_join_notes(spec.notes, "Artefact path is missing."),
        )
    if path.is_file() and not spec.allow_file:
        return ArtefactInventoryRow(
            artefact_name=spec.name,
            artefact_type=spec.artefact_type,
            path=display,
            exists=True,
            status="invalid",
            smoke_test=False,
            completed_runs=None,
            failed_runs=None,
            skipped_runs=None,
            input_hashes_present=False,
            output_hashes_present=False,
            generated_at=None,
            git_commit=None,
            limitations=spec.limitations,
            notes=_join_notes(spec.notes, "Expected an artefact directory, found a file."),
        )

    missing = _missing_required_files(spec, path)
    loaded = _load_artefact_files(spec, path)
    if loaded.invalid_reasons:
        return _record_from_loaded(
            spec,
            path,
            exists=True,
            status="invalid",
            missing=missing,
            loaded=loaded,
            notes=_join_notes(spec.notes, "; ".join(loaded.invalid_reasons)),
        )

    structured_metadata_present = bool(loaded.payloads) or bool(loaded.csv_rows)
    metadata_present = structured_metadata_present or bool(loaded.text_files)
    if not metadata_present and missing:
        return _record_from_loaded(
            spec,
            path,
            exists=True,
            status="invalid",
            missing=missing,
            loaded=loaded,
            notes=_join_notes(spec.notes, "No recognised metadata or required outputs found."),
        )
    if missing and not structured_metadata_present and not spec.allow_file:
        return _record_from_loaded(
            spec,
            path,
            exists=True,
            status="missing",
            missing=missing,
            loaded=loaded,
            notes=_join_notes(spec.notes, "Required evidence files are missing."),
        )

    smoke = _payloads_mark_smoke(loaded.payloads)
    if smoke:
        status: ArtefactStatus = "smoke_test_only"
    elif missing:
        status = "partial_real" if structured_metadata_present or spec.allow_file else "missing"
    else:
        status = _completion_status(spec, loaded)

    input_refs = _hash_references(loaded.payloads, base_path=path, kind="input")
    output_refs = _hash_references(loaded.payloads, base_path=path, kind="output")
    input_hashes_present = bool(input_refs)
    output_hashes_present = bool(output_refs)
    staleness = _assess_staleness(
        loaded=loaded,
        input_refs=input_refs,
        output_refs=output_refs,
        artefact_path=path,
        current_commit=current_commit,
    )
    if status not in {"invalid", "missing", "unsupported", "smoke_test_only"}:
        if staleness.state == "stale":
            status = "stale"
        elif staleness.state == "unknown":
            status = "unknown_staleness"

    notes = _join_notes(spec.notes, staleness.reason)
    return ArtefactInventoryRow(
        artefact_name=spec.name,
        artefact_type=spec.artefact_type,
        path=display,
        exists=True,
        status=status,
        smoke_test=smoke,
        completed_runs=_completed_runs(loaded),
        failed_runs=_failed_runs(loaded),
        skipped_runs=_skipped_runs(loaded),
        input_hashes_present=input_hashes_present,
        output_hashes_present=output_hashes_present,
        generated_at=_generated_at(loaded.payloads),
        git_commit=_git_commit(loaded.payloads),
        limitations=spec.limitations,
        notes=notes,
        required_files_missing=missing,
        supporting_files=_supporting_files(spec, path),
        payloads=loaded.payloads,
        csv_rows=loaded.csv_rows,
        text_files=loaded.text_files,
    )


def _record_from_loaded(
    spec: _ArtefactSpec,
    path: Path,
    *,
    exists: bool,
    status: ArtefactStatus,
    missing: Sequence[str],
    loaded: _LoadedArtefact,
    notes: str,
) -> ArtefactInventoryRow:
    return ArtefactInventoryRow(
        artefact_name=spec.name,
        artefact_type=spec.artefact_type,
        path=_display_path(path),
        exists=exists,
        status=status,
        smoke_test=_payloads_mark_smoke(loaded.payloads),
        completed_runs=_completed_runs(loaded),
        failed_runs=_failed_runs(loaded),
        skipped_runs=_skipped_runs(loaded),
        input_hashes_present=bool(_hash_references(loaded.payloads, base_path=path, kind="input")),
        output_hashes_present=bool(
            _hash_references(loaded.payloads, base_path=path, kind="output")
        ),
        generated_at=_generated_at(loaded.payloads),
        git_commit=_git_commit(loaded.payloads),
        limitations=spec.limitations,
        notes=notes,
        required_files_missing=list(missing),
        supporting_files=_supporting_files(spec, path),
        payloads=loaded.payloads,
        csv_rows=loaded.csv_rows,
        text_files=loaded.text_files,
    )


def _missing_required_files(spec: _ArtefactSpec, path: Path) -> list[str]:
    if path.is_file():
        if spec.allow_file:
            summary_path = path.with_name(_summary_filename(path))
            return [] if summary_path.is_file() else [_display_path(summary_path)]
        return []
    missing: list[str] = []
    for relative in spec.required_files:
        if not (path / relative).is_file():
            missing.append(relative)
    return missing


def _load_artefact_files(spec: _ArtefactSpec, path: Path) -> _LoadedArtefact:
    payloads: dict[str, dict[str, Any]] = {}
    csv_rows: dict[str, list[dict[str, str]]] = {}
    text_files: dict[str, str] = {}
    invalid: list[str] = []

    candidates = _candidate_files(spec, path)
    for label, file_path in candidates:
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        try:
            if suffix == ".json":
                payloads[label] = _read_json_object(file_path)
            elif suffix == ".csv":
                csv_rows[label] = _read_csv(file_path)
            elif suffix in {".md", ".txt"}:
                text_files[label] = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            invalid.append(f"{_display_path(file_path)}: {exc}")
    return _LoadedArtefact(
        payloads=payloads,
        csv_rows=csv_rows,
        text_files=text_files,
        invalid_reasons=invalid,
    )


def _candidate_files(spec: _ArtefactSpec, path: Path) -> list[tuple[str, Path]]:
    if path.is_file():
        candidates = [("report", path), ("summary", path.with_name(_summary_filename(path)))]
    else:
        names = list(dict.fromkeys([*spec.required_files, *spec.metadata_files]))
        candidates = [(Path(name).stem, path / name) for name in names]
        for extra in ("README.md", "summary.json", "figure_manifest.json"):
            extra_path = path / extra
            if extra_path.is_file() and all(existing != extra_path for _, existing in candidates):
                candidates.append((Path(extra).stem, extra_path))
        for manifest_path in sorted(path.rglob("*manifest*.json"))[:50]:
            if all(existing != manifest_path for _, existing in candidates):
                candidates.append((manifest_path.stem, manifest_path))
    return candidates


def _completion_status(spec: _ArtefactSpec, loaded: _LoadedArtefact) -> ArtefactStatus:
    failed = _failed_runs(loaded)
    completed = _completed_runs(loaded)
    skipped = _skipped_runs(loaded)
    if spec.artefact_type in {"fi2010_figures", "ablation_figures"}:
        if failed and failed > 0:
            return "partial_real"
        if skipped and skipped > 0:
            return "partial_real"
        return "complete_real" if completed and completed > 0 else "partial_real"
    if spec.artefact_type == "final_report":
        return "complete_real" if loaded.payloads else "partial_real"
    if spec.artefact_type == "project_audit":
        return "complete_real"
    if failed is not None and failed > 0:
        return "partial_real" if completed and completed > 0 else "invalid"
    if completed is not None and completed <= 0:
        return "partial_real"
    if spec.artefact_type == "fi2010_neural_full_grid":
        summary = loaded.payloads.get("summary", {})
        if bool(summary.get("core_grid_complete")):
            return "complete_real"
        evidence_level = str(summary.get("evidence_level", "")).lower()
        if evidence_level in {"full_grid_complete", "complete_real"}:
            return "complete_real"
        return "partial_real"
    if spec.artefact_type == "fi2010_neural_proper_training":
        summary = loaded.payloads.get("summary", {})
        evidence_level = str(summary.get("evidence_level", "")).lower()
        if evidence_level == "complete_real" or bool(summary.get("target_scope_complete")):
            return "complete_real"
        return "partial_real"
    if spec.artefact_type == "feature_ablations":
        summary = loaded.payloads.get("summary", {})
        return "complete_real" if _feature_ablation_scope_complete(summary) else "partial_real"
    if spec.artefact_type == "synthetic_lob_extension":
        summary = loaded.payloads.get("summary", {})
        if bool(summary.get("smoke_test")):
            return "smoke_test_only"
        if not bool(summary.get("replay_ok", True)) or not bool(summary.get("leakage_ok", True)):
            return "partial_real"
        return "complete_real"
    return "complete_real"


def _feature_ablation_scope_complete(summary: Mapping[str, Any]) -> bool:
    folds = {str(item) for item in _list_payload(summary.get("folds"))}
    horizons = {_int_payload(item) for item in _list_payload(summary.get("horizons"))}
    seeds = {_int_payload(item) for item in _list_payload(summary.get("seeds"))}
    models = {str(item) for item in _list_payload(summary.get("models"))}
    completed = _int_payload(summary.get("completed_run_count"))
    failed = _int_payload(summary.get("failed_run_count")) or 0
    present_horizons = {item for item in horizons if item is not None}
    present_seeds = {item for item in seeds if item is not None}
    return (
        folds >= _EXPANDED_FEATURE_ABLATION_FOLDS
        and present_horizons >= _EXPANDED_FEATURE_ABLATION_HORIZONS
        and present_seeds >= _EXPANDED_FEATURE_ABLATION_SEEDS
        and models >= _EXPANDED_FEATURE_ABLATION_MODELS
        and completed is not None
        and completed >= _EXPANDED_FEATURE_ABLATION_RUNS
        and failed == 0
    )


def _list_payload(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _int_payload(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _assess_staleness(
    *,
    loaded: _LoadedArtefact,
    input_refs: Sequence[_HashReference],
    output_refs: Sequence[_HashReference],
    artefact_path: Path,
    current_commit: str | None,
) -> _StalenessResult:
    reasons: list[str] = []
    checked = False

    recorded_commit = _git_commit(loaded.payloads)
    if recorded_commit and current_commit:
        checked = True
        if recorded_commit != current_commit:
            return _StalenessResult(
                state="stale",
                reason="Recorded git commit differs from the current repository commit.",
            )

    for reference in [*input_refs, *output_refs]:
        if not reference.path.is_file():
            return _StalenessResult(
                state="stale",
                reason=f"Recorded {reference.kind} hash path is missing: "
                f"{_display_path(reference.path)}.",
            )
        checked = True
        actual = sha256_file(reference.path)
        if actual != reference.expected_sha256:
            return _StalenessResult(
                state="stale",
                reason=f"Recorded {reference.kind} hash no longer matches: "
                f"{_display_path(reference.path)}.",
            )

    output_mtime = _artefact_output_mtime(artefact_path, loaded)
    if input_refs and output_mtime is not None:
        checked = True
        for reference in input_refs:
            if reference.path.is_file() and reference.path.stat().st_mtime > output_mtime:
                return _StalenessResult(
                    state="stale",
                    reason=f"Input is newer than derived output: {_display_path(reference.path)}.",
                )

    if _generated_at(loaded.payloads) is not None:
        checked = True
        reasons.append("Generated timestamp is present.")

    if checked:
        suffix = "; ".join(reasons) if reasons else "Hash/commit staleness checks passed."
        return _StalenessResult(state="fresh", reason=suffix)
    return _StalenessResult(
        state="unknown",
        reason="No commit, input-hash or timestamp evidence was available for staleness.",
    )


def _artefact_output_mtime(artefact_path: Path, loaded: _LoadedArtefact) -> float | None:
    candidates: list[Path] = []
    if artefact_path.is_file():
        candidates.append(artefact_path)
    else:
        for filename in ("summary.json", "figure_manifest.json", "execution_v3_manifest.json"):
            candidate = artefact_path / filename
            if candidate.is_file():
                candidates.append(candidate)
    for payload in loaded.payloads.values():
        for key in ("report_path", "summary_path", "manifest_path"):
            value = payload.get(key)
            if isinstance(value, str):
                candidate = _resolve_recorded_path(value, base_path=artefact_path)
                if candidate.is_file():
                    candidates.append(candidate)
    if not candidates:
        return None
    return max(candidate.stat().st_mtime for candidate in candidates)


def _hash_references(
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    base_path: Path,
    kind: Literal["input", "output"],
) -> list[_HashReference]:
    references: list[_HashReference] = []
    for payload in payloads.values():
        references.extend(_hash_refs_from_payload(payload, base_path=base_path, kind=kind))
    deduped: dict[tuple[str, str, str], _HashReference] = {}
    for reference in references:
        key = (str(reference.path), reference.expected_sha256, reference.kind)
        deduped[key] = reference
    return list(deduped.values())


def _hash_refs_from_payload(
    payload: Mapping[str, Any],
    *,
    base_path: Path,
    kind: Literal["input", "output"],
) -> list[_HashReference]:
    refs: list[_HashReference] = []
    hash_key = "input_file_hashes" if kind == "input" else "output_file_hashes"
    file_key = "input_artefact_paths" if kind == "input" else "output_files"
    raw_hashes = payload.get(hash_key)
    raw_paths = payload.get(file_key)
    if isinstance(raw_hashes, Mapping):
        for key, value in raw_hashes.items():
            if not _is_sha256(value):
                continue
            raw_path = None
            if isinstance(raw_paths, Mapping):
                raw_path = raw_paths.get(key)
            if raw_path is None:
                raw_path = key
            if _path_like(raw_path):
                refs.append(
                    _HashReference(
                        path=_resolve_recorded_path(str(raw_path), base_path=base_path),
                        expected_sha256=str(value),
                        kind=kind,
                    )
                )

    if kind == "input":
        refs.extend(_fold_input_refs(payload, base_path=base_path))
        config_sha = payload.get("config_sha256")
        config_path = payload.get("config_path")
        if _is_sha256(config_sha) and _path_like(config_path):
            refs.append(
                _HashReference(
                    path=_resolve_recorded_path(str(config_path), base_path=base_path),
                    expected_sha256=str(config_sha),
                    kind="input",
                )
            )
    return refs


def _fold_input_refs(payload: Mapping[str, Any], *, base_path: Path) -> list[_HashReference]:
    refs: list[_HashReference] = []
    raw = payload.get("fold_inputs")
    if not isinstance(raw, list):
        return refs
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        path = item.get("path")
        sha = item.get("sha256")
        if _path_like(path) and _is_sha256(sha):
            refs.append(
                _HashReference(
                    path=_resolve_recorded_path(str(path), base_path=base_path),
                    expected_sha256=str(sha),
                    kind="input",
                )
            )
    return refs


def _payloads_mark_smoke(payloads: Mapping[str, Mapping[str, Any]]) -> bool:
    return any(_payload_marks_smoke(payload) for payload in payloads.values())


def _payload_marks_smoke(payload: Mapping[str, Any]) -> bool:
    if bool(payload.get("smoke_test")) or bool(payload.get("fixture_or_smoke_run")):
        return True
    if bool(payload.get("is_fixture")):
        return True
    for key in ("execution_mode", "config_mode", "evidence_level", "smoke_test_status"):
        value = payload.get(key)
        if not isinstance(value, str):
            continue
        lowered = value.lower()
        if "smoke" not in lowered:
            continue
        # Negated labels such as "not smoke-test" or "non-smoke" echo a
        # non-smoke status and must not be read as a smoke marker.
        if "not smoke" in lowered or "non-smoke" in lowered or "non smoke" in lowered:
            continue
        return True
    return False


def _completed_runs(loaded: _LoadedArtefact) -> int | None:
    value = _first_int_value(
        loaded.payloads,
        "completed_run_count",
        "completed_runs",
        "ok_row_count",
        "result_rows",
        "prediction_row_count",
    )
    if value is not None:
        return value
    figures = _figure_entries(loaded)
    if figures:
        return sum(1 for entry in figures if entry.get("status") == "completed")
    return _count_csv_status(loaded, {"completed", "ok"})


def _failed_runs(loaded: _LoadedArtefact) -> int | None:
    value = _first_int_value(
        loaded.payloads,
        "failed_run_count",
        "failure_count",
        "failed_runs",
        "failed_row_count",
    )
    if value is not None:
        return value
    failures = loaded.payloads.get("failures")
    if failures:
        raw = failures.get("failure_count")
        parsed = _int_or_none(raw)
        if parsed is not None:
            return parsed
    return _count_csv_status(loaded, {"failed", "error"})


def _skipped_runs(loaded: _LoadedArtefact) -> int | None:
    value = _first_int_value(
        loaded.payloads,
        "skipped_existing_count",
        "skipped_runs",
        "skipped_row_count",
        "skipped_count",
        "missing_pair_count",
    )
    if value is not None:
        return value
    figures = _figure_entries(loaded)
    if figures:
        return sum(1 for entry in figures if entry.get("status") == "skipped")
    return _count_csv_status(loaded, {"skipped", "skipped_existing"})


def _first_int_value(
    payloads: Mapping[str, Mapping[str, Any]],
    *keys: str,
) -> int | None:
    for payload in payloads.values():
        for key in keys:
            parsed = _int_or_none(payload.get(key))
            if parsed is not None:
                return parsed
    return None


def _figure_entries(loaded: _LoadedArtefact) -> list[Mapping[str, Any]]:
    for payload in loaded.payloads.values():
        raw = payload.get("figures")
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, Mapping)]
    return []


def _count_csv_status(loaded: _LoadedArtefact, statuses: set[str]) -> int | None:
    total = 0
    seen_status_column = False
    for rows in loaded.csv_rows.values():
        for row in rows:
            status = row.get("status")
            if status is None:
                continue
            seen_status_column = True
            if status.strip().lower() in statuses:
                total += 1
    return total if seen_status_column else None


def _generated_at(payloads: Mapping[str, Mapping[str, Any]]) -> str | None:
    for payload in payloads.values():
        for key in ("generated_at", "created_at"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _git_commit(payloads: Mapping[str, Mapping[str, Any]]) -> str | None:
    for payload in payloads.values():
        value = payload.get("git_commit")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _supporting_files(spec: _ArtefactSpec, path: Path) -> list[str]:
    files: list[str] = []
    if path.is_file():
        files.append(_display_path(path))
        summary = path.with_name(_summary_filename(path))
        if summary.is_file():
            files.append(_display_path(summary))
        return files
    for relative in spec.required_files:
        candidate = path / relative
        if candidate.is_file():
            files.append(_display_path(candidate))
    for relative in spec.metadata_files:
        candidate = path / relative
        display = _display_path(candidate)
        if candidate.is_file() and display not in files:
            files.append(display)
    return files


def _audit_infrastructure_claim(
    claim_id: str,
    claim_text: str,
    required: Sequence[str],
    by_name: Mapping[str, ArtefactInventoryRow],
    *,
    safe: str,
    reason_supported: str,
    any_required: bool = False,
) -> ClaimAuditEntry:
    records = [by_name[name] for name in required if name in by_name]
    if any_required:
        best = _best_support_status(records)
    else:
        best = _all_required_support_status(records, required)
    supporting = [
        record.artefact_name
        for record in records
        if record.status in _REAL_STATUSES | _PARTIAL_STATUSES
    ]
    if best == "supported":
        reason = reason_supported
    elif best == "smoke_only":
        reason = "Only smoke-test diagnostics are available for this claim."
    elif best == "partially_supported":
        reason = (
            "Some artefacts exist, but the evidence is partial, stale or lacks clean staleness."
        )
    else:
        reason = "Required non-smoke artefacts are missing or invalid."
    return ClaimAuditEntry(
        claim_id=claim_id,
        claim_text=claim_text,
        status=best,
        supporting_artefacts=supporting,
        required_artefacts=list(required),
        reason=reason,
        safe_rewording=safe,
        category="general infrastructure",
    )


def _all_required_support_status(
    records: Sequence[ArtefactInventoryRow],
    required: Sequence[str],
) -> ClaimStatus:
    if len(records) != len(required):
        return "unsupported"
    statuses = {record.status for record in records}
    if statuses <= _REAL_STATUSES:
        return "supported"
    if statuses and statuses <= _SMOKE_STATUSES:
        return "smoke_only"
    if statuses & _REAL_STATUSES and not statuses & _BAD_STATUSES:
        return "partially_supported"
    if statuses & _PARTIAL_STATUSES:
        return "partially_supported"
    if statuses & _SMOKE_STATUSES:
        return "smoke_only"
    return "unsupported"


def _best_support_status(records: Sequence[ArtefactInventoryRow]) -> ClaimStatus:
    statuses = {record.status for record in records}
    if statuses & _REAL_STATUSES:
        return "supported"
    if statuses & _PARTIAL_STATUSES:
        return "partially_supported"
    if statuses and statuses <= _SMOKE_STATUSES:
        return "smoke_only"
    if statuses & _SMOKE_STATUSES:
        return "smoke_only"
    return "unsupported"


def _audit_model_metric_claim(
    by_name: Mapping[str, ArtefactInventoryRow],
) -> ClaimAuditEntry:
    required = ["fi2010_classical_benchmarks", "fi2010_neural_full_grid"]
    records = [record for name in required if (record := by_name.get(name)) is not None]
    has_real_table = any(
        record.status in _REAL_STATUSES | _PARTIAL_STATUSES and _best_macro_f1(record) is not None
        for record in records
    )
    has_smoke_table = any(record.status == "smoke_test_only" for record in records)
    if has_real_table:
        status: ClaimStatus = "partially_supported"
        reason = (
            "Stored result tables contain macro-F1 values, but this template claim "
            "requires the exact model, metric value, scope and artefact path."
        )
    elif has_smoke_table:
        status = "smoke_only"
        reason = "Macro-F1 values found only in smoke diagnostics are not empirical evidence."
    else:
        status = "needs_real_evidence"
        reason = "No real result table with macro-F1 evidence was found."
    return ClaimAuditEntry(
        claim_id="empirical.model_macro_f1",
        claim_text="Model X achieved macro-F1 Y",
        status=status,
        supporting_artefacts=[
            record.artefact_name for record in records if _best_macro_f1(record) is not None
        ],
        required_artefacts=required,
        reason=reason,
        safe_rewording=(
            "Quote model, split, fold/seed scope and macro-F1 directly from "
            "artefact_inventory.csv and the referenced result table."
        ),
        category="empirical result",
    )


def _audit_ssl_implementation_claim(
    by_name: Mapping[str, ArtefactInventoryRow],
) -> ClaimAuditEntry:
    """SSL was implemented and evaluated under matched settings (implementation claim)."""
    claim_id = "empirical.ssl_implemented_evaluated"
    claim_text = "SSL was implemented and evaluated under matched FI-2010 settings."
    required = ["fi2010_neural_full_grid", "ssl_failure_analysis_report"]
    grid = by_name.get("fi2010_neural_full_grid")
    report = by_name.get("ssl_failure_analysis_report")
    if grid is None or report is None:
        return _needs_real_evidence(claim_id, claim_text, required)
    if grid.status in _BAD_STATUSES or report.status in _BAD_STATUSES:
        return _needs_real_evidence(claim_id, claim_text, required)
    if grid.status == "smoke_test_only":
        return _smoke_only_claim(claim_id, claim_text, required, ["fi2010_neural_full_grid"])
    rows = _comparison_rows(grid)
    if not rows:
        return _needs_real_evidence(
            claim_id,
            claim_text,
            required,
            reason="No matched SSL comparison rows were found.",
        )
    return ClaimAuditEntry(
        claim_id=claim_id,
        claim_text=claim_text,
        status="supported",
        supporting_artefacts=["fi2010_neural_full_grid", "ssl_failure_analysis_report"],
        required_artefacts=required,
        reason=(
            "Matched supervised-vs-SSL comparison rows and the SSL failure-analysis "
            "report are present; this is an implementation-and-evaluation claim, not "
            "a result claim."
        ),
        safe_rewording=(
            "SSL objectives were implemented and evaluated under matched settings."
        ),
        category="empirical result",
    )


def _audit_ssl_proper_training_claim(
    by_name: Mapping[str, ArtefactInventoryRow],
) -> ClaimAuditEntry:
    """Narrow fold-1/horizon-50 proper-training predictive-metric improvement."""
    claim_id = "empirical.ssl_proper_training_h50"
    claim_text = (
        "Masked SSL improved fold-1/horizon-50 predictive metrics in the "
        "proper-training subset."
    )
    required = ["fi2010_neural_proper_training_subset", "ssl_failure_analysis_report"]
    subset = by_name.get("fi2010_neural_proper_training_subset")
    report = by_name.get("ssl_failure_analysis_report")
    if subset is None or report is None:
        return _needs_real_evidence(claim_id, claim_text, required)
    if subset.status in _BAD_STATUSES or report.status in _BAD_STATUSES:
        return _needs_real_evidence(claim_id, claim_text, required)
    if subset.status == "smoke_test_only":
        return _smoke_only_claim(
            claim_id, claim_text, required, ["fi2010_neural_proper_training_subset"]
        )
    rows = _comparison_rows(subset)
    h50_masked = [
        row
        for row in rows
        if _row_float(row, "horizon") == 50.0
        and str(row.get("ssl_objective", "")) == "masked_reconstruction"
    ]
    macro = [_row_float(row, "delta_macro_f1") for row in h50_masked]
    mcc = [_row_float(row, "delta_mcc") for row in h50_masked]
    predictive = (
        bool(h50_masked)
        and all(value is not None and value > 0.0 for value in macro)
        and all(value is not None and value > 0.0 for value in mcc)
    )
    if not predictive:
        return _needs_real_evidence(
            claim_id,
            claim_text,
            required,
            reason="Stored rows do not show a fold-1/horizon-50 predictive gain.",
        )
    ece_worsened = any((_row_float(row, "delta_ece") or 0.0) > 0.0 for row in h50_masked)
    reason = (
        "Masked SSL improved macro-F1 and MCC at fold 1 / horizon 50 of the "
        "partial_real proper-training subset"
    )
    reason += (
        ", but calibration worsened, so the scope is too small for a broad claim."
        if ece_worsened
        else ", but the scope is a single partial_real slice."
    )
    return ClaimAuditEntry(
        claim_id=claim_id,
        claim_text=claim_text,
        status="partially_supported",
        supporting_artefacts=[
            "fi2010_neural_proper_training_subset",
            "ssl_failure_analysis_report",
        ],
        required_artefacts=required,
        reason=reason,
        safe_rewording=(
            "The proper-training subset shows a narrow fold-1/horizon-50 "
            "predictive-metric improvement, but calibration worsened and the scope "
            "is partial_real."
        ),
        category="empirical result",
    )


def _audit_ssl_delta_claim(
    by_name: Mapping[str, ArtefactInventoryRow],
    *,
    metric: Literal["macro_f1", "calibration"],
) -> ClaimAuditEntry:
    record = by_name.get("fi2010_neural_full_grid")
    claim_text = "SSL improved macro-F1" if metric == "macro_f1" else "SSL improved calibration"
    claim_id = f"empirical.ssl_improved_{metric}"
    required = ["fi2010_neural_full_grid", "ssl_comparison.csv"]
    if record is None or record.status in _BAD_STATUSES:
        return _needs_real_evidence(claim_id, claim_text, required)
    if record.status == "smoke_test_only":
        return _smoke_only_claim(claim_id, claim_text, required, ["fi2010_neural_full_grid"])
    rows = _comparison_rows(record)
    if not rows:
        return _needs_real_evidence(
            claim_id,
            claim_text,
            required,
            reason="No SSL comparison rows were found.",
        )
    if metric == "macro_f1":
        values = [_row_float(row, "delta_macro_f1") for row in rows if _row_ok(row)]
        improved = [value for value in values if value is not None and value > 0.0]
        degraded = [value for value in values if value is not None and value <= 0.0]
        metric_name = "macro-F1"
    else:
        values = [_row_float(row, "delta_ece") for row in rows if _row_ok(row)]
        improved = [value for value in values if value is not None and value < 0.0]
        degraded = [value for value in values if value is not None and value >= 0.0]
        metric_name = "ECE calibration"
    if values and improved and not degraded and record.status == "complete_real":
        status: ClaimStatus = "supported"
        reason = f"All matched non-smoke SSL comparison rows improved {metric_name}."
    elif values and improved:
        status = "unsupported"
        reason = (
            f"Some matched rows improved {metric_name}, but the broad SSL "
            "improvement claim is not supported because the stored evidence is "
            "mixed, partial, stale or not cleanly complete."
        )
    elif values:
        status = "unsupported"
        reason = f"Stored SSL comparison rows do not support a positive {metric_name} claim."
    else:
        status = "needs_real_evidence"
        reason = f"SSL comparison rows do not contain usable {metric_name} deltas."
    return ClaimAuditEntry(
        claim_id=claim_id,
        claim_text=claim_text,
        status=status,
        supporting_artefacts=["fi2010_neural_full_grid"] if values else [],
        required_artefacts=required,
        reason=reason,
        safe_rewording=(
            "Report the stored supervised-vs-SSL deltas by objective, fold, horizon "
            "and seed; state mixed or negative deltas explicitly."
        ),
        category="empirical result",
    )


def _audit_ssl_execution_claim(
    by_name: Mapping[str, ArtefactInventoryRow],
) -> ClaimAuditEntry:
    required = ["fi2010_neural_full_grid", "execution_v3_outputs"]
    grid = by_name.get("fi2010_neural_full_grid")
    execution = by_name.get("execution_v3_outputs")
    if grid is None or execution is None:
        return _needs_real_evidence(
            "empirical.ssl_execution_proxy",
            "SSL improved execution-aware proxy metrics",
            required,
        )
    if grid.status == "smoke_test_only" or execution.status == "smoke_test_only":
        return _smoke_only_claim(
            "empirical.ssl_execution_proxy",
            "SSL improved execution-aware proxy metrics",
            required,
            [record.artefact_name for record in (grid, execution)],
        )
    if grid.status in _BAD_STATUSES or execution.status in _BAD_STATUSES:
        return _needs_real_evidence(
            "empirical.ssl_execution_proxy",
            "SSL improved execution-aware proxy metrics",
            required,
        )
    return ClaimAuditEntry(
        claim_id="empirical.ssl_execution_proxy",
        claim_text="SSL improved execution-aware proxy metrics",
        status="needs_real_evidence",
        supporting_artefacts=[],
        required_artefacts=required,
        reason=(
            "Execution-v3 proxy tables are present, but no canonical aggregate "
            "SSL-vs-supervised execution improvement claim is inferred automatically."
        ),
        safe_rewording=(
            "Report execution-v3 proxy rows descriptively and avoid SSL improvement "
            "language unless a real matched aggregate table explicitly supports it."
        ),
        category="empirical result",
    )


def _audit_gradient_boosting_claim(
    by_name: Mapping[str, ArtefactInventoryRow],
) -> ClaimAuditEntry:
    record = by_name.get("fi2010_classical_benchmarks")
    required = ["fi2010_classical_benchmarks", "results_summary.csv"]
    claim_text = "Gradient boosting remained the strongest classical baseline"
    if record is None or record.status in _BAD_STATUSES:
        return _needs_real_evidence("empirical.gradient_boosting_best", claim_text, required)
    if record.status == "smoke_test_only":
        return _smoke_only_claim(
            "empirical.gradient_boosting_best",
            claim_text,
            required,
            ["fi2010_classical_benchmarks"],
        )
    best = _best_macro_f1(record)
    if best is None:
        return _needs_real_evidence(
            "empirical.gradient_boosting_best",
            claim_text,
            required,
            reason="Classical results table did not contain a usable test macro-F1 row.",
        )
    best_name = str(best.get("model_name") or best.get("model") or "")
    if best_name == "gradient_boosting" and record.status == "complete_real":
        status: ClaimStatus = "supported"
        reason = "Gradient boosting is the best stored classical test macro-F1 row."
    elif best_name == "gradient_boosting":
        status = "partially_supported"
        reason = "Gradient boosting is best in stored rows, but evidence is not clean complete."
    else:
        status = "unsupported"
        reason = f"The best stored classical macro-F1 row is {best_name or 'not available'}."
    return ClaimAuditEntry(
        claim_id="empirical.gradient_boosting_best",
        claim_text=claim_text,
        status=status,
        supporting_artefacts=["fi2010_classical_benchmarks"],
        required_artefacts=required,
        reason=reason,
        safe_rewording=(
            "Name the best classical model from the stored result table and include "
            "the metric, split and scope."
        ),
        category="empirical result",
    )


def _audit_feature_group_improvement_claim(
    by_name: Mapping[str, ArtefactInventoryRow],
) -> ClaimAuditEntry:
    record = by_name.get("feature_ablation_outputs")
    required = ["feature_ablation_outputs", "feature_delta_summary.csv"]
    claim_text = "Feature group X improved performance"
    if record is None or record.status in _BAD_STATUSES:
        return _needs_real_evidence("empirical.feature_group_improved", claim_text, required)
    if record.status == "smoke_test_only":
        return _smoke_only_claim(
            "empirical.feature_group_improved",
            claim_text,
            required,
            ["feature_ablation_outputs"],
        )
    rows = record.csv_rows.get("feature_delta_summary", [])
    values = [
        _row_float(row, "delta_macro_f1")
        for row in rows
        if _row_float(row, "delta_macro_f1") is not None
    ]
    positive = [value for value in values if value is not None and value > 0.0]
    if positive:
        status: ClaimStatus = "partially_supported"
        reason = (
            "Some feature-delta rows are positive, but the placeholder claim must "
            "name the exact group, model, horizon, split and ablation mode."
        )
    else:
        status = "unsupported" if values else "needs_real_evidence"
        reason = "No positive stored feature-group macro-F1 deltas were found."
    return ClaimAuditEntry(
        claim_id="empirical.feature_group_improved",
        claim_text=claim_text,
        status=status,
        supporting_artefacts=["feature_ablation_outputs"] if values else [],
        required_artefacts=required,
        reason=reason,
        safe_rewording=(
            "Describe feature-ablation deltas by named group and scope; do not "
            "generalise beyond the stored FI-2010 snapshot diagnostics."
        ),
        category="empirical result",
    )


def _audit_feature_ablation_infrastructure_claim(
    by_name: Mapping[str, ArtefactInventoryRow],
) -> ClaimAuditEntry:
    record = by_name.get("feature_ablation_outputs")
    required = ["feature_ablation_outputs"]
    claim_text = "Feature-ablation infrastructure is available"
    if record is None or record.status in _BAD_STATUSES:
        return _needs_real_evidence("feature_ablation_infrastructure", claim_text, required)
    status: ClaimStatus = "smoke_only" if record.status == "smoke_test_only" else "supported"
    return ClaimAuditEntry(
        claim_id="feature_ablation_infrastructure",
        claim_text=claim_text,
        status=status,
        supporting_artefacts=["feature_ablation_outputs"],
        required_artefacts=required,
        reason="Feature-ablation runner outputs and summary tables are present.",
        safe_rewording=(
            "ChronosLOB includes FI-2010 feature-ablation evidence infrastructure with "
            "proxy and unsupported groups labelled."
        ),
        category="infrastructure",
    )


def _audit_snapshot_proxy_claim(
    by_name: Mapping[str, ArtefactInventoryRow],
    *,
    claim_id: str,
    claim_text: str,
    analysis_key: str,
    required: Sequence[str],
    fallback_horizon10: bool,
) -> ClaimAuditEntry:
    analysis = by_name.get("feature_ablation_analysis_report")
    analysis_payload = (
        analysis.payloads.get("feature_claim_assessment", {})
        if analysis is not None and analysis.status not in _BAD_STATUSES
        else {}
    )
    claims_payload = analysis_payload.get("claims") if isinstance(analysis_payload, Mapping) else {}
    scoped = claims_payload.get(analysis_key) if isinstance(claims_payload, Mapping) else None
    if isinstance(scoped, Mapping):
        status = _claim_status_from_analysis(str(scoped.get("status", "needs_real_evidence")))
        return ClaimAuditEntry(
            claim_id=claim_id,
            claim_text=claim_text,
            status=status,
            supporting_artefacts=["feature_ablation_analysis_report", "feature_ablation_outputs"],
            required_artefacts=list(required),
            reason=str(scoped.get("reason", "feature-ablation analysis claim assessment loaded")),
            safe_rewording=(
                "State the exact feature group, horizons, models, folds and seeds; "
                "describe snapshot_order_flow_proxy as a labelled snapshot proxy."
            ),
            category="empirical result",
        )
    if fallback_horizon10:
        fallback = _horizon10_snapshot_proxy_fallback(by_name.get("feature_ablation_outputs"))
        if fallback is not None:
            status, reason, degraded, total = fallback
            return ClaimAuditEntry(
                claim_id=claim_id,
                claim_text=claim_text,
                status=status,
                supporting_artefacts=["feature_ablation_outputs"],
                required_artefacts=list(required),
                reason=f"{reason} ({degraded}/{total} matched rows degraded when removed).",
                safe_rewording=(
                    "In the stored horizon-10 logistic/ridge slice, removing "
                    "snapshot_order_flow_proxy degraded macro-F1; this is a "
                    "horizon/model-specific effect for a labelled snapshot proxy."
                ),
                category="empirical result",
            )
    return _needs_real_evidence(
        claim_id,
        claim_text,
        required,
        reason="Feature-ablation stability analysis artefacts were not available for this claim.",
    )


def _claim_status_from_analysis(status: str) -> ClaimStatus:
    cleaned = status.strip().lower()
    if cleaned == "supported":
        return "supported"
    if cleaned == "partially_supported":
        return "partially_supported"
    if cleaned == "unsupported":
        return "unsupported"
    if cleaned == "forbidden":
        return "forbidden"
    if cleaned == "needs_real_evidence":
        return "needs_real_evidence"
    if cleaned == "smoke_only":
        return "smoke_only"
    if cleaned == "needs_prediction_outputs":
        return "needs_real_evidence"
    return "needs_real_evidence"


def _horizon10_snapshot_proxy_fallback(
    record: ArtefactInventoryRow | None,
) -> tuple[ClaimStatus, str, int, int] | None:
    if record is None or record.status in _BAD_STATUSES or record.status == "smoke_test_only":
        return None
    rows = record.csv_rows.get("feature_delta_summary", [])
    matched = [
        row
        for row in rows
        if row.get("ablation_mode") == "remove_one_group"
        and row.get("feature_group") == "snapshot_order_flow_proxy"
        and _row_float(row, "horizon") == 10.0
        and row.get("model") in {"logistic", "ridge"}
    ]
    if not matched:
        return None
    degraded = sum(
        1
        for row in matched
        if (_row_float(row, "delta_macro_f1") is not None)
        and (_row_float(row, "delta_macro_f1") or 0.0) < 0.0
    )
    total = len(matched)
    if degraded == total:
        return (
            "supported",
            "Retained horizon-10 logistic/ridge tables support the snapshot proxy finding",
            degraded,
            total,
        )
    if degraded > total / 2:
        return (
            "partially_supported",
            (
                "Retained horizon-10 logistic/ridge tables partially support "
                "the snapshot proxy finding"
            ),
            degraded,
            total,
        )
    return (
        "unsupported",
        "Retained horizon-10 logistic/ridge tables do not support the snapshot proxy finding",
        degraded,
        total,
    )


def _forbidden_named_claim(
    claim_id: str,
    claim_text: str,
    reason: str,
) -> ClaimAuditEntry:
    return ClaimAuditEntry(
        claim_id=claim_id,
        claim_text=claim_text,
        status="forbidden",
        supporting_artefacts=[],
        required_artefacts=[],
        reason=reason,
        safe_rewording=(
            "Use feature-ablation evidence language with exact scope and proxy caveats."
        ),
        category="forbidden or high-risk",
    )


def _audit_confidence_filtering_claim(
    by_name: Mapping[str, ArtefactInventoryRow],
) -> ClaimAuditEntry:
    record = by_name.get("execution_v3_outputs")
    required = ["execution_v3_outputs", "confidence_threshold_aggregate.csv"]
    claim_text = "Confidence filtering improved cost-adjusted proxy"
    if record is None or record.status in _BAD_STATUSES:
        return _needs_real_evidence("empirical.confidence_filtering", claim_text, required)
    if record.status == "smoke_test_only":
        return _smoke_only_claim(
            "empirical.confidence_filtering",
            claim_text,
            required,
            ["execution_v3_outputs"],
        )
    rows = record.csv_rows.get("confidence_threshold_aggregate", [])
    improvement = _confidence_filtering_improvement(rows)
    if improvement is True and record.status == "complete_real":
        status: ClaimStatus = "supported"
        reason = "Stored threshold aggregates show higher cost-adjusted proxy at a filter."
    elif improvement is True:
        status = "partially_supported"
        reason = (
            "Stored threshold aggregates show an improvement, but evidence is not clean complete."
        )
    elif improvement is False:
        status = "unsupported"
        reason = (
            "Stored threshold aggregates do not show an improvement over the reference threshold."
        )
    else:
        status = "needs_real_evidence"
        reason = "No usable threshold aggregate rows were found."
    return ClaimAuditEntry(
        claim_id="empirical.confidence_filtering",
        claim_text=claim_text,
        status=status,
        supporting_artefacts=["execution_v3_outputs"] if improvement is not None else [],
        required_artefacts=required,
        reason=reason,
        safe_rewording=(
            "Report confidence-threshold proxy diagnostics with threshold, payoff mode, "
            "cost mode and retained-sample fraction."
        ),
        category="empirical result",
    )


def _synthetic_support_status(record: ArtefactInventoryRow | None) -> ClaimStatus:
    """Map a synthetic artefact status to an infrastructure-support claim status.

    The synthetic extension is an infrastructure-and-data artefact: as long as it
    is present and complete (even if a later commit bump marks it stale) it
    supports the synthetic-pipeline claims. Smoke-only artefacts only support a
    smoke claim; missing or invalid artefacts support nothing.
    """
    if record is None:
        return "unsupported"
    if record.status in _REAL_STATUSES | _PARTIAL_STATUSES:
        return "supported"
    if record.status == "smoke_test_only":
        return "smoke_only"
    return "unsupported"


def _audit_synthetic_claims(
    by_name: Mapping[str, ArtefactInventoryRow],
) -> list[ClaimAuditEntry]:
    """Audit the synthetic event-level extension claims separately from FI-2010."""
    record = by_name.get("synthetic_lob_extension_report")
    support = _synthetic_support_status(record)
    required = ["synthetic_lob_extension_report"]
    supporting = (
        ["synthetic_lob_extension_report"]
        if support in {"supported", "smoke_only"}
        else []
    )
    has_regime_diag = bool(
        record is not None
        and any(
            "synthetic_regime_diagnostics" in name for name in record.supporting_files
        )
    )

    def _supported_claim(
        claim_id: str,
        claim_text: str,
        safe: str,
        reason_supported: str,
    ) -> ClaimAuditEntry:
        if support == "supported":
            reason = reason_supported
        elif support == "smoke_only":
            reason = "Only synthetic smoke artefacts are present; not a full synthetic run."
        else:
            reason = "Synthetic extension artefacts are missing or invalid."
        return ClaimAuditEntry(
            claim_id=claim_id,
            claim_text=claim_text,
            status=support,
            supporting_artefacts=supporting,
            required_artefacts=required,
            reason=reason,
            safe_rewording=safe,
            category="synthetic extension",
        )

    diagnostics_status: ClaimStatus = support if has_regime_diag else "needs_real_evidence"
    entries = [
        _supported_claim(
            "synthetic.event_level_pipeline",
            "ChronosLOB supports a synthetic event-level LOB pipeline",
            "ChronosLOB includes a synthetic event-level pipeline (generation, "
            "replay, features, labels) on controlled synthetic regimes; not real-market data.",
            "Synthetic pipeline artefacts with passing replay and no-lookahead checks are present.",
        ),
        _supported_claim(
            "synthetic.event_level_features",
            "ChronosLOB computes true event-level features on synthetic event streams",
            "Event-level order-flow, cancellation and trade imbalance are computed on "
            "synthetic event streams only; FI-2010 still exposes snapshot proxies only.",
            "Synthetic event-level feature artefacts are present.",
        ),
        ClaimAuditEntry(
            claim_id="synthetic.regime_diagnostics",
            claim_text="ChronosLOB produces synthetic regime stress-test diagnostics",
            status=diagnostics_status,
            supporting_artefacts=supporting if has_regime_diag else [],
            required_artefacts=required,
            reason=(
                "Per-regime synthetic execution-aware proxy diagnostics are present."
                if has_regime_diag
                else "Synthetic regime diagnostics artefacts were not found."
            ),
            safe_rewording=(
                "Synthetic regime diagnostics are controlled stress tests on known "
                "regimes, not real-market execution evidence."
            ),
            category="synthetic extension",
        ),
        ClaimAuditEntry(
            claim_id="synthetic.real_market_event_level_generalisation",
            claim_text="The synthetic extension generalises to real-market event-level data",
            status="unsupported",
            supporting_artefacts=[],
            required_artefacts=required,
            reason="All synthetic data is controlled and does not transfer to real markets.",
            safe_rewording=(
                "State that synthetic results do not establish real-market generalisation."
            ),
            category="synthetic extension",
        ),
        ClaimAuditEntry(
            claim_id="synthetic.live_trading_or_profitability",
            claim_text="The synthetic extension shows live trading or profitability",
            status="forbidden",
            supporting_artefacts=[],
            required_artefacts=[],
            reason="This claim is blocked by release policy for ChronosLOB artefacts.",
            safe_rewording="Describe offline synthetic diagnostics only; do not imply returns.",
            category="forbidden or high-risk",
        ),
        ClaimAuditEntry(
            claim_id="synthetic.fi2010_true_event_level_ofi",
            claim_text="The synthetic extension provides true event-level OFI on FI-2010",
            status="forbidden",
            supporting_artefacts=[],
            required_artefacts=[],
            reason=(
                "FI-2010 snapshots expose only proxies; event-level order flow exists "
                "for synthetic data only."
            ),
            safe_rewording=(
                "Event-level order flow is available only on synthetic streams, not FI-2010."
            ),
            category="forbidden or high-risk",
        ),
    ]
    return entries


def _forbidden_claim_entries() -> list[ClaimAuditEntry]:
    entries: list[ClaimAuditEntry] = []
    for index, claim in enumerate(_FORBIDDEN_CLAIMS, start=1):
        entries.append(
            ClaimAuditEntry(
                claim_id=f"forbidden.{index}",
                claim_text=claim,
                status="forbidden",
                supporting_artefacts=[],
                required_artefacts=[],
                reason="This claim is blocked by release policy for ChronosLOB artefacts.",
                safe_rewording=_safe_forbidden_rewording(claim),
                category="forbidden or high-risk",
            )
        )
    return entries


def _needs_real_evidence(
    claim_id: str,
    claim_text: str,
    required: Sequence[str],
    *,
    reason: str = "Required real non-smoke artefacts are missing.",
) -> ClaimAuditEntry:
    return ClaimAuditEntry(
        claim_id=claim_id,
        claim_text=claim_text,
        status="needs_real_evidence",
        supporting_artefacts=[],
        required_artefacts=list(required),
        reason=reason,
        safe_rewording=(
            "Built infrastructure exists, but this result claim needs real aggregate artefacts."
        ),
        category="empirical result",
    )


def _smoke_only_claim(
    claim_id: str,
    claim_text: str,
    required: Sequence[str],
    supporting: Sequence[str],
) -> ClaimAuditEntry:
    return ClaimAuditEntry(
        claim_id=claim_id,
        claim_text=claim_text,
        status="smoke_only",
        supporting_artefacts=list(supporting),
        required_artefacts=list(required),
        reason="Only smoke-test artefacts support this code path; they are not empirical evidence.",
        safe_rewording=(
            "Smoke diagnostics exercise the code path only; no empirical result is claimed."
        ),
        category="empirical result",
    )


def _safe_forbidden_rewording(claim: str) -> str:
    lowered = claim.lower()
    if "ofi" in lowered or "queue" in lowered:
        return (
            "Use FI-2010 snapshot-derived proxy language only and state that event-level "
            "order flow and queue position are not observed."
        )
    if "pnl" in lowered or "alpha" in lowered or "profitable" in lowered:
        return "Describe offline predictive or proxy diagnostics; do not imply returns."
    if "live" in lowered or "production" in lowered:
        return "Describe offline research software and stored-artefact diagnostics only."
    if "state" in lowered or "foundation" in lowered:
        return "Describe the implemented research platform and audited artefacts conservatively."
    return "Use a narrower artefact-backed statement with explicit limitations."


def _comparison_rows(record: ArtefactInventoryRow) -> list[dict[str, str]]:
    rows = record.csv_rows.get("ssl_comparison")
    if rows is not None:
        return rows
    return record.csv_rows.get("comparison_summary", [])


def _best_macro_f1(record: ArtefactInventoryRow) -> dict[str, str] | None:
    candidate_tables = (
        "results_summary",
        "aggregate_summary",
        "feature_delta_summary",
    )
    best: dict[str, str] | None = None
    best_value = -math.inf
    for table in candidate_tables:
        for row in record.csv_rows.get(table, []):
            split = str(row.get("split", "test")).lower()
            if split and split != "test":
                continue
            value = (
                _row_float(row, "macro_f1_mean")
                or _row_float(row, "mean_macro_f1")
                or _row_float(row, "macro_f1")
            )
            if value is None:
                continue
            if value > best_value:
                best_value = value
                best = dict(row)
    return best


def _confidence_filtering_improvement(rows: Sequence[Mapping[str, str]]) -> bool | None:
    if not rows:
        return None
    grouped: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    for row in rows:
        if row.get("status", "ok") not in {"", "ok"}:
            continue
        threshold = _row_float(row, "threshold")
        value = _row_float(row, "mean_net_cost_adjusted_proxy")
        if threshold is None or value is None:
            continue
        key = (
            row.get("model_family", ""),
            row.get("pretraining_objective", ""),
            row.get("horizon", ""),
        )
        grouped.setdefault(key, []).append((threshold, value))
    if not grouped:
        return None
    saw_comparison = False
    for values in grouped.values():
        ordered = sorted(values)
        if len(ordered) < 2:
            continue
        saw_comparison = True
        reference = ordered[0][1]
        if any(value > reference for _, value in ordered[1:]):
            return True
    return False if saw_comparison else None


def _attach_claim_support(
    records: Sequence[ArtefactInventoryRow],
    claims: Sequence[ClaimAuditEntry],
) -> None:
    by_name = {record.artefact_name: record for record in records}
    for claim in claims:
        if claim.status not in {"supported", "partially_supported"}:
            continue
        for name in claim.supporting_artefacts:
            record = by_name.get(name)
            if record is not None:
                record.supports_claims.append(claim.claim_id)


def _write_pack_outputs(
    *,
    out_dir: Path,
    config: EvidencePackConfig,
    inventory: Sequence[ArtefactInventoryRow],
    claim_audit: Sequence[ClaimAuditEntry],
    warnings: Sequence[str],
    strict_violations: Sequence[str],
) -> list[Path]:
    files: list[Path] = []
    claim_counts = Counter(entry.status for entry in claim_audit)
    status_counts = Counter(record.status for record in inventory)
    manifest = {
        "builder_version": EVIDENCE_PACK_VERSION,
        "package_version": __version__,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _current_git_commit(),
        "strict": config.strict,
        "allow_smoke_test": config.allow_smoke_test,
        "input_paths": _config_paths(config),
        "artefact_status_counts": dict(sorted(status_counts.items())),
        "claim_status_counts": dict(sorted(claim_counts.items())),
        "artefacts": [record.to_manifest_payload() for record in inventory],
        "warnings": list(warnings),
        "strict_violations": list(strict_violations),
        "outputs": {filename: filename for filename in _OUTPUT_FILENAMES},
        "claim_boundary": (
            "Smoke diagnostics are not empirical evidence. Execution metrics are "
            "offline proxies. FI-2010 snapshot features do not expose event-level "
            "order flow, cancellations, trades or queue position."
        ),
    }
    files.append(_write_text(out_dir / "evidence_pack_manifest.json", stable_json_dumps(manifest)))
    files.append(_write_text(out_dir / "evidence_pack_summary.md", _render_summary(inventory)))
    files.append(_write_inventory_csv(out_dir / "artefact_inventory.csv", inventory))
    files.append(
        _write_text(
            out_dir / "claim_audit.json",
            stable_json_dumps(
                {
                    "builder_version": EVIDENCE_PACK_VERSION,
                    "generated_at": datetime.now(UTC).isoformat(),
                    "claim_status_counts": dict(sorted(claim_counts.items())),
                    "claims": [entry.to_payload() for entry in claim_audit],
                }
            ),
        )
    )
    files.append(_write_text(out_dir / "claim_audit.md", _render_claim_audit(claim_audit)))
    files.append(
        _write_text(out_dir / "supported_claims.md", _render_supported_claims(claim_audit))
    )
    files.append(
        _write_text(out_dir / "unsupported_claims.md", _render_unsupported_claims(claim_audit))
    )
    files.append(
        _write_text(
            out_dir / _PUBLIC_BULLET_CONSERVATIVE,
            _render_conservative_bullets(claim_audit),
        )
    )
    files.append(
        _write_text(
            out_dir / _PUBLIC_BULLET_STRONG,
            _render_strong_bullets(inventory, claim_audit),
        )
    )
    files.append(
        _write_text(out_dir / "reproduction_commands.md", _render_reproduction_commands(config))
    )
    files.append(_write_text(out_dir / "release_checklist.md", _render_release_checklist()))
    files.append(
        _write_text(
            out_dir / "readme_result_snapshot.md",
            _render_readme_snapshot(inventory, claim_audit),
        )
    )
    return files


def _render_summary(inventory: Sequence[ArtefactInventoryRow]) -> str:
    rows = [
        (
            record.artefact_name,
            record.status,
            "yes" if record.smoke_test else "no",
            _optional_int_text(record.completed_runs),
            _optional_int_text(record.failed_runs),
            record.notes,
        )
        for record in inventory
    ]
    lines = [
        "# ChronosLOB Evidence Pack Summary",
        "",
        (
            "This pack inventories stored artefacts and separates smoke diagnostics "
            "from real evidence."
        ),
        "",
        "`complete_real` means required non-smoke artefacts are present and pass "
        "completion checks.",
        "`partial_real` means real artefacts exist but the scope is incomplete, "
        "mixed or has explicit skipped diagnostics.",
        "Smoke rows remain code-path checks only.",
        "",
        *_markdown_table(
            ("artefact", "status", "smoke", "completed", "failed", "notes"),
            rows,
        ),
        "",
        "Smoke-test artefacts are code-path diagnostics only. They are not empirical evidence.",
        "Generated evidence-pack files should be regenerated from artefacts "
        "rather than hand-edited.",
        "",
    ]
    return "\n".join(lines)


def _render_claim_audit(claims: Sequence[ClaimAuditEntry]) -> str:
    rows = [
        (
            entry.claim_id,
            entry.status,
            entry.claim_text,
            ", ".join(entry.supporting_artefacts) or "none",
            entry.reason,
            entry.safe_rewording,
        )
        for entry in claims
    ]
    return "\n".join(
        [
            "# Claim Audit",
            "",
            (
                "Every claim below is checked against stored artefacts. Broad "
                "improvement claims require consistent aggregate support; isolated "
                "positive rows are not enough."
            ),
            "",
            *_markdown_table(
                ("id", "status", "claim", "supporting artefacts", "reason", "safe wording"),
                rows,
            ),
            "",
        ]
    )


def _render_supported_claims(claims: Sequence[ClaimAuditEntry]) -> str:
    supported = [entry for entry in claims if entry.status == "supported"]
    lines = ["# Supported Claims", ""]
    if not supported:
        lines.append("No fully supported public claims were found.")
        lines.append("")
        return "\n".join(lines)
    for entry in supported:
        lines.extend(
            [
                f"- {entry.claim_text}",
                f"  - support: {', '.join(entry.supporting_artefacts)}",
                f"  - safe wording: {entry.safe_rewording}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _render_unsupported_claims(claims: Sequence[ClaimAuditEntry]) -> str:
    unsupported = [entry for entry in claims if entry.status != "supported"]
    lines = ["# Unsupported Or Limited Claims", ""]
    for entry in unsupported:
        lines.extend(
            [
                f"- {entry.claim_text}",
                f"  - status: {entry.status}",
                f"  - reason: {entry.reason}",
                f"  - safe wording: {entry.safe_rewording}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _render_conservative_bullets(claims: Sequence[ClaimAuditEntry]) -> str:
    supported = {entry.claim_id for entry in claims if entry.status == "supported"}
    lines = ["# Conservative Public Summary Bullets", ""]
    if "general.reproducible_platform" in supported:
        lines.append(
            "- Built a reproducible limit-order-book research platform with stored "
            "artefact inventories, local audit checks and release-readiness gates."
        )
    if "general.leakage_safe_fi2010" in supported:
        lines.append(
            "- Implemented leakage-aware FI-2010 evaluation workflows with traceable "
            "stored benchmark outputs."
        )
    if "general.train_only_ssl" in supported:
        lines.append(
            "- Added train-only self-supervised pretraining paths for transformer "
            "experiments with artefact-backed comparison hooks."
        )
    if "general.execution_proxy_diagnostics" in supported:
        lines.append(
            "- Added offline execution-aware proxy diagnostics with explicit scope "
            "and release-claim boundaries."
        )
    if "general.feature_ablations" in supported:
        lines.append(
            "- Added FI-2010 microstructure feature-ablation diagnostics that separate "
            "snapshot-supported features from unsupported event-level concepts."
        )
    if not any(line.startswith("- ") for line in lines):
        lines.append(
            "- Built release infrastructure for auditing experiment artefacts and "
            "blocking unsupported empirical claims."
        )
    lines.append("")
    return "\n".join(lines)


def _render_strong_bullets(
    inventory: Sequence[ArtefactInventoryRow],
    claims: Sequence[ClaimAuditEntry],
) -> str:
    by_name = {record.artefact_name: record for record in inventory}
    supported = {entry.claim_id for entry in claims if entry.status == "supported"}
    grid = by_name.get("fi2010_neural_full_grid")
    grid_status = grid.status if grid is not None else "missing"
    lines = ["# Stronger Public Bullets With Support Conditions", ""]
    lines.extend(
        [
            "Bullet:",
            (
                '"Compared supervised and self-supervised transformer variants across '
                'FI-2010 folds, horizons and seeds under leakage-safe evaluation."'
            ),
            "",
            "Status:",
            "supported only if:",
            "- neural full-grid status = complete_real",
            "- supervised, masked SSL and next-field SSL objectives completed",
            "- aggregate_summary.csv exists",
            "- ssl_comparison.csv exists",
            "- no broad SSL improvement is claimed unless aggregate deltas support it",
            "",
            "Current status:",
            f"- neural full-grid status = {grid_status}",
            "",
            "Safe fallback:",
            (
                '"Built infrastructure to compare supervised and self-supervised '
                'transformer variants under leakage-safe FI-2010 evaluation."'
            ),
            "",
        ]
    )
    lines.extend(
        [
            "Bullet:",
            (
                '"Generated artefact-backed execution-aware proxy diagnostics for '
                'stored FI-2010 predictions."'
            ),
            "",
            "Status:",
            "supported only if:",
            "- execution-v3 status = complete_real",
            "- summary.json and execution_v3_manifest.json exist",
            "- inputs are non-smoke and not stale",
            "",
            "Safe fallback:",
            (
                '"Built offline execution-aware proxy diagnostics with explicit '
                'release limitations."'
            ),
            "",
        ]
    )
    if "empirical.gradient_boosting_best" in supported:
        lines.extend(
            [
                "Bullet:",
                '"Identified the strongest stored classical FI-2010 baseline by macro-F1."',
                "",
                "Status:",
                "supported by classical result artefacts.",
                "",
            ]
        )
    return "\n".join(lines)


def _render_readme_snapshot(
    inventory: Sequence[ArtefactInventoryRow],
    claims: Sequence[ClaimAuditEntry],
) -> str:
    by_name = {record.artefact_name: record for record in inventory}
    claim_by_id = {entry.claim_id: entry for entry in claims}
    classical = by_name.get("fi2010_classical_benchmarks")
    grid = by_name.get("fi2010_neural_full_grid")
    execution = by_name.get("execution_v3_outputs")
    feature = by_name.get("feature_ablation_outputs")
    figures = by_name.get("fi2010_figures")
    lines = [
        "# README Result Snapshot",
        "",
        "## Current Evidence Status",
        "",
        *_markdown_table(
            ("component", "status", "smoke"),
            [
                (
                    record.artefact_name,
                    record.status,
                    "yes" if record.smoke_test else "no",
                )
                for record in inventory
            ],
        ),
        "",
        "## Results",
        "",
        _result_line("Best classical result", classical),
        _result_line("Best neural full-grid result", grid),
        _ssl_snapshot_line(claim_by_id),
        _component_status_line("Execution-v3", execution),
        _component_status_line("Feature ablations", feature),
        _component_status_line("Figures", figures),
        "",
        "## Limitations",
        "",
        "- Smoke diagnostics are labelled as smoke diagnostics and are not empirical evidence.",
        _full_grid_limitation_line(grid),
        "- SSL improvement language is blocked unless real aggregate deltas support it.",
        "- Execution-v3 metrics are offline proxy diagnostics, not deployed execution results.",
        "- FI-2010 snapshot features do not expose event-level order flow or queue position.",
        "",
    ]
    return "\n".join(lines)


def _full_grid_limitation_line(record: ArtefactInventoryRow | None) -> str:
    if record is not None and record.status == "complete_real":
        return (
            "- Full-grid neural artefacts are present, but public wording must quote "
            "stored scope and metrics exactly."
        )
    return "- Missing real full-grid artefacts mean no full-grid empirical result is claimed."


def _result_line(label: str, record: ArtefactInventoryRow | None) -> str:
    if record is None:
        return f"- {label}: not available."
    if record.status == "smoke_test_only":
        return f"- {label}: smoke diagnostics only; no empirical result claimed."
    if record.status != "complete_real":
        return f"- {label}: not cleanly supported ({record.status})."
    best = _best_macro_f1(record)
    if best is None:
        return f"- {label}: no usable macro-F1 row found."
    model = best.get("model_name") or best.get("model") or best.get("model_family") or "model"
    metric = (
        _row_float(best, "macro_f1_mean")
        or _row_float(best, "mean_macro_f1")
        or _row_float(best, "macro_f1")
    )
    return f"- {label}: {model} macro-F1 {_format_float(metric)} from stored artefacts."


def _ssl_snapshot_line(claims: Mapping[str, ClaimAuditEntry]) -> str:
    macro = claims.get("empirical.ssl_improved_macro_f1")
    calibration = claims.get("empirical.ssl_improved_calibration")
    if macro is None or calibration is None:
        return "- SSL comparison: not audited."
    if macro.status == "supported" and calibration.status == "supported":
        return (
            "- SSL comparison: stored aggregate deltas support macro-F1 and "
            "calibration improvements."
        )
    if macro.status == "smoke_only" or calibration.status == "smoke_only":
        return "- SSL comparison: smoke diagnostics only; no SSL improvement claimed."
    return (
        "- SSL comparison: no broad SSL improvement claim; see claim_audit.md for "
        f"macro-F1={macro.status}, calibration={calibration.status}."
    )


def _component_status_line(label: str, record: ArtefactInventoryRow | None) -> str:
    if record is None:
        return f"- {label}: not available."
    if record.status == "smoke_test_only":
        return f"- {label}: smoke diagnostics only."
    if record.status == "complete_real":
        return f"- {label}: complete real artefacts present."
    return f"- {label}: {record.status}."


def _render_reproduction_commands(config: EvidencePackConfig) -> str:
    lines = [
        "# Reproduction Commands",
        "",
        "Commands are ordered from feature checks through release audit. Real runs require "
        "local FI-2010 data prepared under `data/processed/fi2010`; no runtime "
        "duration is promised.",
        "",
    ]
    sections = [
        (
            "Feature Audit",
            "python -m chronoslob.cli audit-fi2010-features "
            "--path tests/fixtures/fi2010/tiny_fi2010_like.csv --feature-groups all --no-strict",
            "python -m chronoslob.cli audit-fi2010-features "
            "--path data/processed/fi2010/fold1_combined.csv --feature-groups all --strict",
            "read-only CLI output",
            "Requires a local FI-2010-style CSV with label and split columns.",
        ),
        (
            "Feature Ablations",
            "python -m chronoslob.cli run-fi2010-feature-ablations "
            "--data-path tests/fixtures/fi2010/tiny_fi2010_like.csv --out "
            f"{config.feature_ablations_dir.as_posix()} --smoke-test --no-strict",
            "python -m chronoslob.cli run-fi2010-feature-ablations "
            "--config configs/experiments/fi2010_multifold.yaml "
            "--processed-root data/processed/fi2010 --folds all --horizons 10,20,50 "
            f"--out {config.feature_ablations_dir.as_posix()} --strict",
            config.feature_ablations_dir.as_posix(),
            "Real run depends on prepared folds and selected model/grid scope.",
        ),
        (
            "Neural Full Grid",
            "python -m chronoslob.cli run-fi2010-neural-full-grid "
            "--processed-root data/processed/fi2010 "
            f"--out {config.neural_full_grid_dir.as_posix()} --smoke-test",
            "python -m chronoslob.cli run-fi2010-neural-full-grid "
            "--processed-root data/processed/fi2010 --folds 1,2,3,4,5 "
            "--horizons 10,20,50 --seeds 0,1,2 "
            f"--out {config.neural_full_grid_dir.as_posix()}",
            config.neural_full_grid_dir.as_posix(),
            "Real run requires local compute suitable for neural training.",
        ),
        (
            "Proper-Training Neural Subset",
            "python -m chronoslob.cli run-fi2010-neural-proper-training-subset "
            "--config configs/experiments/fi2010_neural_proper_training_smoke.yaml "
            "--processed-root data/processed/fi2010 "
            f"--out {config.proper_training_dir.as_posix()} --folds 1 --horizons 10 "
            "--seeds 0 --lookbacks 10 --objectives supervised,masked_reconstruction,next_field "
            "--pretrain-epochs 1 --max-epochs 2 --patience 1 --batch-size 16 --smoke-test",
            "python -m chronoslob.cli run-fi2010-neural-proper-training-subset "
            "--config configs/experiments/fi2010_neural_proper_training.yaml "
            "--processed-root data/processed/fi2010 "
            f"--out {config.proper_training_dir.as_posix()} --folds 1,2,3 "
            "--horizons 10,50 --seeds 0 --lookbacks 50 "
            "--objectives supervised,masked_reconstruction,next_field "
            "--pretrain-epochs 5 --max-epochs 25 --patience 5 --batch-size 1024 --device cpu",
            config.proper_training_dir.as_posix(),
            "Fallback real evidence is partial_real; complete_real requires folds "
            "1-5 at horizons 10 and 50 with the same longer-training protocol.",
        ),
        (
            "SSL Benchmark",
            "python -m chronoslob.cli run-fi2010-ssl-neural-benchmark "
            "--config configs/experiments/fi2010_ssl_smoke.yaml "
            "--processed-root data/processed/fi2010 "
            f"--out {config.ssl_dir.as_posix()} --folds fold_1 --seeds 0 --lookbacks 10",
            "python -m chronoslob.cli run-fi2010-ssl-neural-benchmark "
            "--config configs/experiments/fi2010_ssl_smoke.yaml "
            "--processed-root data/processed/fi2010 "
            f"--out {config.ssl_dir.as_posix()} --folds all --seeds 0,1,2 "
            "--objective both",
            config.ssl_dir.as_posix(),
            "Real SSL evidence requires train-only pretraining and verified checkpoints.",
        ),
        (
            "Execution-V3",
            "python -m chronoslob.cli build-fi2010-execution-v3 "
            f"--neural-full-grid {config.neural_full_grid_dir.as_posix()} "
            f"--out {config.execution_v3_dir.as_posix()} --allow-smoke-test --no-strict",
            "python -m chronoslob.cli build-fi2010-execution-v3 "
            f"--neural-full-grid {config.neural_full_grid_dir.as_posix()} "
            f"--out {config.execution_v3_dir.as_posix()} --strict",
            config.execution_v3_dir.as_posix(),
            "Consumes stored predictions; outputs are offline proxy diagnostics.",
        ),
        (
            "Figures",
            "python -m chronoslob.cli build-fi2010-figures "
            f"--neural-full-grid {config.neural_full_grid_dir.as_posix()} "
            f"--out {config.figures_dir.as_posix()} --allow-smoke-test --no-strict",
            "python -m chronoslob.cli build-fi2010-figures "
            f"--neural-full-grid {config.neural_full_grid_dir.as_posix()} "
            f"--execution-v3 {config.execution_v3_dir.as_posix()} "
            f"--out {config.figures_dir.as_posix()} --strict",
            config.figures_dir.as_posix(),
            "Requires aggregate tables and optional execution-v3 inputs.",
        ),
        (
            "Ablation Figures",
            "python -m chronoslob.cli build-fi2010-ablation-figures "
            f"--feature-ablations {config.feature_ablations_dir.as_posix()} "
            f"--out {config.ablation_figures_dir.as_posix()} --allow-smoke-test",
            "python -m chronoslob.cli build-fi2010-ablation-figures "
            f"--feature-ablations {config.feature_ablations_dir.as_posix()} "
            f"--out {config.ablation_figures_dir.as_posix()}",
            config.ablation_figures_dir.as_posix(),
            "Requires stored feature-ablation tables.",
        ),
        (
            "Feature Ablation Analysis",
            "python -m chronoslob.cli analyse-fi2010-feature-ablations "
            f"--feature-ablations {config.feature_ablations_dir.as_posix()} "
            f"--out {config.feature_ablation_analysis_dir.as_posix()} "
            "--allow-smoke-test --overwrite",
            "python -m chronoslob.cli analyse-fi2010-feature-ablations "
            f"--feature-ablations {config.feature_ablations_dir.as_posix()} "
            f"--out {config.feature_ablation_analysis_dir.as_posix()} --overwrite",
            config.feature_ablation_analysis_dir.as_posix(),
            "Consumes retained lightweight feature-ablation tables; prediction-level "
            "execution-aware diagnostics require a targeted prediction-retaining rerun.",
        ),
        (
            "Final Empirical Report",
            "python -m chronoslob.cli build-final-empirical-report "
            "--classical experiments/fi2010_multifold_classical "
            "--neural experiments/fi2010_multifold_neural "
            "--uncertainty experiments/fi2010_uncertainty "
            f"--neural-full-grid {config.neural_full_grid_dir.as_posix()} "
            f"--proper-training {config.proper_training_dir.as_posix()} "
            f"--feature-ablations {config.feature_ablations_dir.as_posix()} "
            f"--feature-ablation-analysis {config.feature_ablation_analysis_dir.as_posix()} "
            f"--execution-v3 {config.execution_v3_dir.as_posix()} "
            f"--out {config.final_report_path.as_posix()} --overwrite",
            "python -m chronoslob.cli build-final-empirical-report "
            "--classical experiments/fi2010_multifold_classical "
            "--neural experiments/fi2010_multifold_neural "
            "--uncertainty experiments/fi2010_uncertainty "
            "--ablations experiments/fi2010_brutal_ablations "
            "--external experiments/fi2010_external_context "
            f"--neural-full-grid {config.neural_full_grid_dir.as_posix()} "
            f"--proper-training {config.proper_training_dir.as_posix()} "
            f"--feature-ablations {config.feature_ablations_dir.as_posix()} "
            f"--feature-ablation-analysis {config.feature_ablation_analysis_dir.as_posix()} "
            f"--execution-v3 {config.execution_v3_dir.as_posix()} "
            f"--evidence-pack {config.out_dir.as_posix()} "
            f"--out {config.final_report_path.as_posix()} --overwrite",
            config.final_report_path.as_posix(),
            "Consumes stored artefacts only.",
        ),
        (
            "Evidence Pack",
            "python -m chronoslob.cli build-evidence-pack "
            f"--out {config.out_dir.as_posix()} "
            f"--neural-full-grid {config.neural_full_grid_dir.as_posix()} "
            f"--figures {config.figures_dir.as_posix()} "
            f"--execution-v3 {config.execution_v3_dir.as_posix()} "
            f"--feature-ablations {config.feature_ablations_dir.as_posix()} "
            f"--feature-ablation-analysis {config.feature_ablation_analysis_dir.as_posix()} "
            f"--ablation-figures {config.ablation_figures_dir.as_posix()} "
            f"--final-report {config.final_report_path.as_posix()} "
            "--allow-smoke-test --no-strict --overwrite",
            "python -m chronoslob.cli build-evidence-pack "
            f"--out {config.out_dir.as_posix()} "
            f"--neural-full-grid {config.neural_full_grid_dir.as_posix()} "
            f"--figures {config.figures_dir.as_posix()} "
            f"--execution-v3 {config.execution_v3_dir.as_posix()} "
            f"--feature-ablations {config.feature_ablations_dir.as_posix()} "
            f"--feature-ablation-analysis {config.feature_ablation_analysis_dir.as_posix()} "
            f"--ablation-figures {config.ablation_figures_dir.as_posix()} "
            f"--final-report {config.final_report_path.as_posix()} "
            "--strict --overwrite",
            config.out_dir.as_posix(),
            "Builds summaries and claim audits; it does not train models.",
        ),
        (
            "Project Audit",
            "python -m chronoslob.cli run-project-audit",
            "python -m chronoslob.cli run-project-audit --strict",
            "read-only CLI output",
            "Review unrelated or untracked worktree files before release.",
        ),
    ]
    for title, smoke, real, output, caveat in sections:
        lines.extend(
            [
                f"## {title}",
                "",
                "Smoke-test version:",
                "",
                "```bash",
                smoke,
                "```",
                "",
                "Real-run version:",
                "",
                "```bash",
                real,
                "```",
                "",
                f"Expected output: `{output}`",
                "",
                f"Compute and dependency caveat: {caveat}",
                "",
            ]
        )
    return "\n".join(lines)


def _render_release_checklist() -> str:
    sections = {
        "Git Hygiene": [
            "Review `git status --short` before release.",
            (
                "If the worktree contains unrelated or untracked files, they must be "
                "reviewed before release. Do not automatically delete or stage them."
            ),
        ],
        "Tests": ["Run `python -m pytest`."],
        "Lint And Types": ["Run `python -m ruff check .`.", "Run `python -m mypy chronoslob`."],
        "Project Audit": [
            "Run `python -m chronoslob.cli doctor`.",
            "Run `python -m chronoslob.cli inspect-release-readiness`.",
            "Run `python -m chronoslob.cli run-project-audit --strict`.",
        ],
        "Artefact Hashes": [
            "Confirm input hashes are present where artefacts record source files.",
            "Investigate stale and unknown-staleness rows in artefact_inventory.csv.",
        ],
        "Smoke And Real Separation": [
            "Confirm smoke_test_only rows are not cited as empirical evidence.",
            "Use `--allow-smoke-test` only for diagnostics packs.",
        ],
        "Public Snapshot": ["Review `readme_result_snapshot.md` before copying any result text."],
        "Docs And Limits": [
            "Docs updated.",
            "Limitations updated.",
            "Unsupported claims removed.",
        ],
        "Public Bullets": ["Check conservative and stronger public bullet files."],
        "Paper Boundary": ["Manual paper not yet written."],
        "Blocked Claims": ["No live-trading or profitability claims."],
    }
    lines = ["# Release Checklist", ""]
    for title, items in sections.items():
        lines.extend([f"## {title}", ""])
        lines.extend(f"- [ ] {item}" for item in items)
        lines.append("")
    return "\n".join(lines)


def _write_inventory_csv(
    path: Path,
    inventory: Sequence[ArtefactInventoryRow],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ARTEFACT_INVENTORY_COLUMNS))
        writer.writeheader()
        for record in inventory:
            writer.writerow(record.to_csv_row())
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"evidence-pack output path is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    existing = [path / filename for filename in _OUTPUT_FILENAMES if (path / filename).exists()]
    if existing and not overwrite:
        rendered = ", ".join(_display_path(item) for item in existing[:5])
        raise FileExistsError(
            "refusing to overwrite existing evidence-pack outputs; "
            f"pass overwrite=True to replace them: {rendered}"
        )


def _strict_violations(
    inventory: Sequence[ArtefactInventoryRow],
    *,
    public_claim_violations: Sequence[str],
    allow_smoke_test: bool,
) -> list[str]:
    violations: list[str] = []
    for record in inventory:
        if record.status == "invalid":
            violations.append(f"invalid artefact: {record.artefact_name}")
        if (
            record.status == "smoke_test_only"
            and not allow_smoke_test
            and record.artefact_name
            in {
                "fi2010_neural_full_grid",
                "fi2010_neural_proper_training_subset",
                "fi2010_figures",
                "execution_v3_outputs",
                "feature_ablation_outputs",
                "feature_ablation_analysis_report",
                "ablation_figures",
            }
        ):
            violations.append(
                f"smoke-test artefact requires --allow-smoke-test: {record.artefact_name}"
            )
    violations.extend(public_claim_violations)
    return violations


def _build_warnings(
    inventory: Sequence[ArtefactInventoryRow],
    *,
    public_claim_violations: Sequence[str],
    allow_smoke_test: bool,
) -> list[str]:
    warnings: list[str] = []
    for record in inventory:
        if record.status in {"stale", "unknown_staleness", "missing", "unsupported"}:
            warnings.append(f"{record.artefact_name}: {record.status}")
        if record.status == "smoke_test_only" and not allow_smoke_test:
            warnings.append(f"{record.artefact_name}: smoke-test-only and not allowed for claims")
    warnings.extend(public_claim_violations)
    return warnings


def _scan_public_claim_violations(final_report_path: Path) -> list[str]:
    path = Path(final_report_path)
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    violations: list[str] = []
    lowered_claims = tuple((claim, claim.lower()) for claim in _FORBIDDEN_CLAIMS)
    for index, line in enumerate(lines):
        lowered = line.lower()
        for claim, lowered_claim in lowered_claims:
            if lowered_claim not in lowered:
                continue
            if _allowed_public_claim_context(lines, index):
                continue
            violations.append(
                f"forbidden public claim in {_display_path(path)}:{index + 1}: {claim}"
            )
    return violations


def _allowed_public_claim_context(lines: Sequence[str], index: int) -> bool:
    start = max(0, index - 8)
    end = min(len(lines), index + 2)
    context = " ".join(lines[start:end]).lower()
    return any(marker in context for marker in _PUBLIC_CLAIM_ALLOW_MARKERS)


def _config_paths(config: EvidencePackConfig) -> dict[str, str | None]:
    return {
        "out_dir": _display_path(config.out_dir),
        "classical_dir": _display_path(config.classical_dir),
        "ssl_dir": _display_path(config.ssl_dir),
        "neural_full_grid_dir": _display_path(config.neural_full_grid_dir),
        "proper_training_dir": _display_path(config.proper_training_dir),
        "figures_dir": _display_path(config.figures_dir),
        "execution_v3_dir": _display_path(config.execution_v3_dir),
        "feature_audit_dir": (
            None if config.feature_audit_dir is None else _display_path(config.feature_audit_dir)
        ),
        "feature_ablations_dir": _display_path(config.feature_ablations_dir),
        "feature_ablation_analysis_dir": _display_path(config.feature_ablation_analysis_dir),
        "ablation_figures_dir": _display_path(config.ablation_figures_dir),
        "final_report_path": _display_path(config.final_report_path),
        "synthetic_lob_dir": _display_path(config.synthetic_lob_dir),
        "project_audit_dir": (
            None if config.project_audit_dir is None else _display_path(config.project_audit_dir)
        ),
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("JSON artefact must contain an object")
    return dict(payload)


def _read_csv(path: Path) -> list[dict[str, str]]:
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


def _row_ok(row: Mapping[str, str]) -> bool:
    return str(row.get("status", "ok")).strip().lower() in {
        "",
        "ok",
        "completed",
        "matched",
    }


def _row_float(row: Mapping[str, str], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        numeric = float(text)
    except ValueError:
        return None
    return numeric if math.isfinite(numeric) else None


def _format_float(value: float | None) -> str:
    if value is None:
        return "not available"
    return f"{value:.4f}"


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(float(value.strip()))
        except ValueError:
            return None
    return None


def _optional_int_text(value: int | None) -> str:
    return "" if value is None else str(value)


def _path_like(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    text = value.strip()
    return (
        "/" in text
        or "\\" in text
        or Path(text).suffix != ""
        or text.startswith(("data", "configs", "experiments", "reports", "runs"))
    )


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip().lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _resolve_recorded_path(raw_path: str, *, base_path: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    root_candidate = project_root() / candidate
    if root_candidate.exists():
        return root_candidate
    if base_path.is_file():
        return base_path.parent / candidate
    return base_path / candidate


def _summary_filename(path: Path) -> str:
    return f"{Path(path).stem}_summary.json"


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


def _join_notes(*parts: str) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip())


def _markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_escape_table_cell(cell) for cell in row) + " |")
    return lines


def _escape_table_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()

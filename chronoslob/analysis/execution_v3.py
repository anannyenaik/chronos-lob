"""FI-2010 execution-aware proxy diagnostic v3.

This module consumes stored FI-2010 neural full-grid prediction artefacts and
builds an offline execution-aware proxy diagnostic.  It is intentionally not a
broker integration, live trading component, profitability study or realistic
execution simulator.  Every output is an artefact-backed diagnostic describing
how directional signal quality changes under confidence filtering, costs,
latency and proxy fill assumptions.
"""

from __future__ import annotations

import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from chronoslob import __version__
from chronoslob.analysis.fi2010_label_mapping import (
    FI2010_CANONICAL_CLASS_ORDER,
    FI2010_RAW_LABEL_TO_CLASS,
    canonical_class_name,
    probability_columns_for_order,
    validate_class_order,
    validate_probability_columns,
)
from chronoslob.experiments.manifests import sha256_file, stable_json_dumps
from chronoslob.training.experiment import get_git_commit
from chronoslob.utils.paths import project_root

__all__ = [
    "DEFAULT_CONFIDENCE_THRESHOLDS",
    "DEFAULT_EXECUTION_V3_OUTPUT_DIR",
    "DEFAULT_FEE_BPS",
    "DEFAULT_FILL_ASSUMPTIONS",
    "DEFAULT_LATENCY_STEPS",
    "DEFAULT_SPREAD_MULTIPLIERS",
    "EXECUTION_V3_VERSION",
    "ExecutionV3Summary",
    "build_fi2010_execution_v3",
    "load_feature_ablation_predictions",
]

EXECUTION_V3_VERSION = "fi2010-execution-aware-proxy-diagnostic/v3"
DEFAULT_EXECUTION_V3_OUTPUT_DIR = Path("experiments/fi2010_execution_v3")
DEFAULT_CONFIDENCE_THRESHOLDS: tuple[float, ...] = (
    0.33,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
)
DEFAULT_FEE_BPS: tuple[float, ...] = (0.0, 1.0, 2.0, 5.0, 10.0)
DEFAULT_SPREAD_MULTIPLIERS: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
DEFAULT_LATENCY_STEPS: tuple[int, ...] = (0, 1, 2, 5, 10)
DEFAULT_FILL_ASSUMPTIONS: tuple[str, ...] = (
    "aggressive_crossing",
    "passive_optimistic",
    "passive_conservative",
    "abstain_only",
)

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)
_SUCCESS_STATUSES = {"completed", "skipped_existing", "ok"}
_GROUP_COLUMNS: tuple[str, ...] = (
    "model_family",
    "pretraining_objective",
    "horizon",
    "fold",
    "seed",
    "lookback",
)
_REGIME_COLUMNS: dict[str, tuple[str, ...]] = {
    "volatility_regime": ("volatility_regime", "volatility_label"),
    "spread_regime": ("spread_regime", "spread_label"),
    "imbalance_regime": ("imbalance_regime", "imbalance_label"),
    "liquidity_regime": ("liquidity_regime", "liquidity_label"),
    "regime": ("regime",),
}
_SIGNED_RETURN_COLUMNS: tuple[str, ...] = (
    "future_return",
    "future_mid_return",
    "future_midprice_return",
    "realised_return",
    "realized_return",
    "post_fill_return",
    "future_move",
    "future_price_move",
    "mid_price_change",
    "future_mid_price_change",
)
_MAGNITUDE_COLUMNS: tuple[str, ...] = (
    "future_move_magnitude",
    "future_return_magnitude",
    "move_magnitude",
)
_SPREAD_COLUMNS: tuple[str, ...] = (
    "relative_spread",
    "spread_bps",
    "spread",
)
_LIQUIDITY_COLUMNS: tuple[str, ...] = (
    "liquidity",
    "depth",
    "top_of_book_depth",
    "bid_ask_depth",
    "bid_depth_1",
    "ask_depth_1",
)
_IMBALANCE_COLUMNS: tuple[str, ...] = ("imbalance", "order_imbalance")
_CONFIDENCE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0.33-0.50", 0.33, 0.50),
    ("0.50-0.70", 0.50, 0.70),
    ("0.70-0.85", 0.70, 0.85),
    ("0.85-1.00", 0.85, 1.0000001),
)
_UNIT_SPREAD_COST = 0.01


class ExecutionV3Summary(BaseModel):
    """Summary returned after building execution-aware proxy diagnostic v3."""

    model_config = _MODEL_CONFIG

    output_dir: str
    neural_full_grid_dir: str
    feature_ablation_dir: str | None = None
    manifest_path: str
    summary_path: str
    prediction_row_count: int
    run_group_count: int
    payoff_mode: str
    cost_mode: str
    smoke_test: bool
    strict: bool
    diagnostics_produced: list[str] = Field(default_factory=list)
    diagnostics_skipped: list[str] = Field(default_factory=list)
    skipped_diagnostics: list[dict[str, str]] = Field(default_factory=list)
    output_files: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime
    builder_version: str

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @field_validator(
        "output_dir",
        "neural_full_grid_dir",
        "manifest_path",
        "summary_path",
        "payoff_mode",
        "cost_mode",
        "builder_version",
    )
    @classmethod
    def _validate_non_empty_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("execution v3 summary text fields must be non-empty")
        return value.strip()


@dataclass
class _LoadedInputs:
    grid_dir: Path
    output_dir: Path
    input_kind: str
    summary: dict[str, Any]
    results: pd.DataFrame
    predictions_raw: pd.DataFrame
    predictions: pd.DataFrame
    input_files: dict[str, Path]
    input_hashes: dict[str, str]
    label_mapping_audit: dict[str, Any]
    smoke_test: bool
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _RuntimeConfig:
    models: tuple[str, ...] | None
    horizons: tuple[int, ...] | None
    folds: tuple[str, ...] | None
    seeds: tuple[int, ...] | None
    confidence_thresholds: tuple[float, ...]
    fee_bps: tuple[float, ...]
    spread_multipliers: tuple[float, ...]
    latency_steps: tuple[int, ...]
    fill_assumptions: tuple[str, ...]
    allow_smoke_test: bool
    strict: bool

    @property
    def reference_threshold(self) -> float:
        return self.confidence_thresholds[0]

    @property
    def reference_fee_bps(self) -> float:
        return self.fee_bps[0]

    @property
    def reference_spread_multiplier(self) -> float:
        return self.spread_multipliers[0]


def build_fi2010_execution_v3(
    *,
    neural_full_grid_dir: Path,
    out_dir: Path = DEFAULT_EXECUTION_V3_OUTPUT_DIR,
    feature_ablation_dir: Path | None = None,
    models: Sequence[str] | str | None = None,
    horizons: Sequence[int] | str | None = None,
    folds: Sequence[int | str] | str | None = None,
    seeds: Sequence[int] | str | None = None,
    confidence_thresholds: Sequence[float] | str | None = None,
    fee_bps: Sequence[float] | str | None = None,
    spread_multipliers: Sequence[float] | str | None = None,
    latency_steps: Sequence[int] | str | None = None,
    fill_assumptions: Sequence[str] | str | None = None,
    allow_smoke_test: bool = False,
    strict: bool = True,
    overwrite: bool = False,
) -> ExecutionV3Summary:
    """Build FI-2010 execution-aware proxy diagnostic v3 from stored artefacts."""
    config = _RuntimeConfig(
        models=_parse_model_selection(models),
        horizons=_parse_int_selection(horizons, positive=True),
        folds=_parse_fold_selection(folds),
        seeds=_parse_int_selection(seeds, positive=False),
        confidence_thresholds=_parse_float_selection(
            confidence_thresholds,
            default=DEFAULT_CONFIDENCE_THRESHOLDS,
            lower_bound=0.0,
        ),
        fee_bps=_parse_float_selection(
            fee_bps,
            default=DEFAULT_FEE_BPS,
            lower_bound=0.0,
        ),
        spread_multipliers=_parse_float_selection(
            spread_multipliers,
            default=DEFAULT_SPREAD_MULTIPLIERS,
            lower_bound=0.0,
        ),
        latency_steps=_parse_int_selection(
            latency_steps,
            default=DEFAULT_LATENCY_STEPS,
            positive=False,
        )
        or DEFAULT_LATENCY_STEPS,
        fill_assumptions=_parse_fill_assumptions(fill_assumptions),
        allow_smoke_test=bool(allow_smoke_test),
        strict=bool(strict),
    )
    loaded = _load_inputs(
        neural_full_grid_dir=Path(neural_full_grid_dir),
        feature_ablation_dir=Path(feature_ablation_dir)
        if feature_ablation_dir is not None
        else None,
        out_dir=Path(out_dir),
        config=config,
        overwrite=overwrite,
    )

    predictions = _prepare_payoff_columns(loaded.predictions)
    payoff_mode = _payoff_mode(predictions)
    cost_mode = _cost_mode(predictions)

    skipped: list[dict[str, str]] = []
    output_files: dict[str, Path] = {}

    confidence_rows = _confidence_threshold_rows(
        predictions,
        config=config,
        cost_mode=cost_mode,
    )
    confidence_path = loaded.output_dir / "confidence_threshold_summary.csv"
    _write_csv(confidence_rows, confidence_path)
    output_files["confidence_threshold_summary"] = confidence_path

    confidence_aggregate_rows = _confidence_threshold_aggregate_rows(confidence_rows)
    confidence_aggregate_path = loaded.output_dir / "confidence_threshold_aggregate.csv"
    _write_csv(confidence_aggregate_rows, confidence_aggregate_path)
    output_files["confidence_threshold_aggregate"] = confidence_aggregate_path

    cost_rows = _cost_sensitivity_rows(
        predictions,
        config=config,
        payoff_mode=payoff_mode,
        cost_mode=cost_mode,
    )
    cost_path = loaded.output_dir / "cost_sensitivity_summary.csv"
    _write_csv(cost_rows, cost_path)
    output_files["cost_sensitivity_summary"] = cost_path

    latency_rows, latency_skips = _latency_sensitivity_rows(
        predictions,
        config=config,
        cost_mode=cost_mode,
    )
    skipped.extend(latency_skips)
    latency_path = loaded.output_dir / "latency_sensitivity_summary.csv"
    _write_csv(latency_rows, latency_path)
    output_files["latency_sensitivity_summary"] = latency_path

    fill_rows, fill_skips = _fill_assumption_rows(
        predictions,
        config=config,
        cost_mode=cost_mode,
    )
    skipped.extend(fill_skips)
    fill_path = loaded.output_dir / "fill_assumption_summary.csv"
    _write_csv(fill_rows, fill_path)
    output_files["fill_assumption_summary"] = fill_path

    adverse_rows = _adverse_selection_rows(predictions, config=config)
    adverse_path = loaded.output_dir / "adverse_selection_summary.csv"
    _write_csv(adverse_rows, adverse_path)
    output_files["adverse_selection_summary"] = adverse_path

    regime_rows, regime_skips = _regime_execution_rows(
        predictions,
        config=config,
        cost_mode=cost_mode,
    )
    skipped.extend(regime_skips)
    regime_path = loaded.output_dir / "regime_execution_summary.csv"
    _write_csv(regime_rows, regime_path)
    output_files["regime_execution_summary"] = regime_path

    diagnostics = {
        "confidence_threshold": bool(confidence_rows),
        "cost_sensitivity": bool(cost_rows),
        "latency_sensitivity": any(row.get("status") == "ok" for row in latency_rows),
        "fill_assumptions": any(row.get("status") == "ok" for row in fill_rows),
        "adverse_selection": any(row.get("status") == "ok" for row in adverse_rows),
        "regime_execution": any(row.get("status") == "ok" for row in regime_rows),
    }
    diagnostics_produced = [name for name, ok in diagnostics.items() if ok]
    diagnostics_skipped = [name for name, ok in diagnostics.items() if not ok]
    for name in diagnostics_skipped:
        if not any(item.get("diagnostic") == name for item in skipped):
            skipped.append(
                {
                    "diagnostic": name,
                    "scope": "all",
                    "skip_reason": "no usable rows were available",
                }
            )

    skipped_path = loaded.output_dir / "skipped_diagnostics.json"
    skipped_path.write_text(
        stable_json_dumps({"skipped_count": len(skipped), "skipped": skipped}),
        encoding="utf-8",
    )
    output_files["skipped_diagnostics"] = skipped_path

    notes_path = loaded.output_dir / "execution_v3_notes.md"
    notes_path.write_text(
        _notes_markdown(
            payoff_mode=payoff_mode,
            cost_mode=cost_mode,
            smoke_test=loaded.smoke_test,
        ),
        encoding="utf-8",
    )
    output_files["execution_v3_notes"] = notes_path

    summary_payload = {
        "builder_version": EXECUTION_V3_VERSION,
        "package_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
        "offline_execution_aware_proxy_diagnostic": True,
        "claim_boundary": (
            "This is an offline execution-aware proxy diagnostic. It is not a "
            "live trading system, broker integration, profitability claim or "
            "realistic execution simulator."
        ),
        "neural_full_grid_dir": _display_path(loaded.grid_dir),
        "feature_ablation_dir": (
            _display_path(loaded.grid_dir) if loaded.input_kind == "feature_ablation" else None
        ),
        "input_kind": loaded.input_kind,
        "output_dir": _display_path(loaded.output_dir),
        "prediction_row_count": len(predictions),
        "run_group_count": _group_count(predictions),
        "payoff_mode": payoff_mode,
        "cost_mode": cost_mode,
        "fill_modes_used": list(config.fill_assumptions),
        "latency_modes_used": list(config.latency_steps),
        "smoke_test": loaded.smoke_test,
        "smoke_test_status": (
            "smoke-test-only; not empirical evidence" if loaded.smoke_test else "not smoke-test"
        ),
        "strict": config.strict,
        "diagnostics_produced": diagnostics_produced,
        "diagnostics_skipped": diagnostics_skipped,
        "skipped_diagnostics": skipped,
        "label_mapping_audit_status": loaded.label_mapping_audit.get("status"),
        "outputs": {key: _display_path(path) for key, path in output_files.items()},
        "warnings": loaded.warnings,
    }
    summary_path = loaded.output_dir / "summary.json"
    summary_path.write_text(stable_json_dumps(summary_payload), encoding="utf-8")
    output_files["summary"] = summary_path

    manifest_path = loaded.output_dir / "execution_v3_manifest.json"
    manifest = _execution_manifest(
        loaded=loaded,
        output_files=output_files,
        payoff_mode=payoff_mode,
        cost_mode=cost_mode,
        diagnostics_produced=diagnostics_produced,
        diagnostics_skipped=diagnostics_skipped,
        skipped=skipped,
        config=config,
    )
    manifest_path.write_text(stable_json_dumps(manifest), encoding="utf-8")
    output_files["execution_v3_manifest"] = manifest_path

    return ExecutionV3Summary(
        output_dir=str(loaded.output_dir),
        neural_full_grid_dir=str(loaded.grid_dir),
        feature_ablation_dir=(
            str(loaded.grid_dir) if loaded.input_kind == "feature_ablation" else None
        ),
        manifest_path=str(manifest_path),
        summary_path=str(summary_path),
        prediction_row_count=len(predictions),
        run_group_count=_group_count(predictions),
        payoff_mode=payoff_mode,
        cost_mode=cost_mode,
        smoke_test=loaded.smoke_test,
        strict=config.strict,
        diagnostics_produced=diagnostics_produced,
        diagnostics_skipped=diagnostics_skipped,
        skipped_diagnostics=skipped,
        output_files={key: _display_path(path) for key, path in output_files.items()},
        warnings=loaded.warnings,
        created_at=datetime.now(UTC),
        builder_version=EXECUTION_V3_VERSION,
    )


def _load_inputs(
    *,
    neural_full_grid_dir: Path,
    feature_ablation_dir: Path | None,
    out_dir: Path,
    config: _RuntimeConfig,
    overwrite: bool,
) -> _LoadedInputs:
    if feature_ablation_dir is not None:
        return _load_feature_ablation_inputs(
            ablation_dir=feature_ablation_dir,
            out_dir=out_dir,
            config=config,
            overwrite=overwrite,
        )
    grid_dir = Path(neural_full_grid_dir)
    if not grid_dir.exists():
        raise FileNotFoundError(f"neural full-grid directory does not exist: {grid_dir}")
    if not grid_dir.is_dir():
        raise NotADirectoryError(f"neural full-grid path is not a directory: {grid_dir}")
    summary_path = grid_dir / "summary.json"
    results_path = grid_dir / "results_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"full-grid summary.json is missing: {summary_path}")
    if not results_path.is_file():
        raise FileNotFoundError(f"full-grid results_summary.csv is missing: {results_path}")

    summary = _read_json(summary_path)
    smoke_test = _is_smoke_summary(summary)
    if smoke_test and not config.allow_smoke_test:
        raise ValueError("execution-v3 refuses smoke-test artefacts unless allow_smoke_test=True")

    _ensure_output_dir(out_dir, input_dir=grid_dir, overwrite=overwrite)

    input_files: dict[str, Path] = {
        "full_grid_summary": summary_path,
        "full_grid_results_summary": results_path,
    }
    input_hashes: dict[str, str] = {}
    warnings: list[str] = []
    for key, filename in (
        ("full_grid_aggregate_summary", "aggregate_summary.csv"),
        ("full_grid_aggregate_summary_json", "aggregate_summary.json"),
        ("full_grid_ssl_comparison", "ssl_comparison.csv"),
        ("full_grid_label_mapping_audit", "label_mapping_audit.json"),
    ):
        candidate = grid_dir / filename
        if candidate.is_file():
            input_files[key] = candidate
        else:
            warnings.append(f"optional input artefact missing: {candidate}")
    for key, path in input_files.items():
        input_hashes[key] = sha256_file(path)

    results = pd.read_csv(results_path)
    results = _filter_result_rows(results, config=config)
    predictions_raw, prediction_paths, prediction_warnings = _load_prediction_frame(
        results,
        grid_dir,
    )
    warnings.extend(prediction_warnings)
    for index, path in enumerate(prediction_paths):
        key = f"prediction_file_{index:04d}"
        input_files[key] = path
        input_hashes[key] = sha256_file(path)

    label_mapping_audit = _build_label_mapping_audit(
        predictions_raw,
        results,
        existing_audit_path=grid_dir / "label_mapping_audit.json",
        warnings=warnings,
    )
    if config.strict and label_mapping_audit["status"] != "pass":
        audit_path = out_dir / "label_mapping_audit.json"
        audit_path.write_text(stable_json_dumps(label_mapping_audit), encoding="utf-8")
        raise ValueError(f"FI-2010 label mapping audit failed in strict mode; see {audit_path}")
    (out_dir / "label_mapping_audit.json").write_text(
        stable_json_dumps(label_mapping_audit),
        encoding="utf-8",
    )

    predictions = _canonicalise_predictions(
        predictions_raw,
        strict=config.strict,
        warnings=warnings,
    )
    if predictions.empty:
        raise ValueError("no usable prediction rows were available for execution-v3")
    return _LoadedInputs(
        grid_dir=grid_dir,
        output_dir=out_dir,
        input_kind="neural_full_grid",
        summary=summary,
        results=results,
        predictions_raw=predictions_raw,
        predictions=predictions,
        input_files=input_files,
        input_hashes=input_hashes,
        label_mapping_audit=label_mapping_audit,
        smoke_test=smoke_test,
        warnings=warnings,
    )


def load_feature_ablation_predictions(
    ablation_dir: Path,
) -> tuple[pd.DataFrame, list[Path], list[str]]:
    """Load execution-v3-compatible prediction rows from feature ablations."""
    root = Path(ablation_dir)
    warnings: list[str] = []
    paths = sorted(root.glob("runs/*/predictions.csv"))
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        if frame.empty:
            warnings.append(f"feature-ablation prediction file is empty: {path}")
            continue
        if "partition" not in frame.columns:
            frame["partition"] = frame["split"] if "split" in frame.columns else "test"
        if "model_family" not in frame.columns and "model" in frame.columns:
            frame["model_family"] = frame["model"]
        if "pretraining_objective" not in frame.columns:
            frame["pretraining_objective"] = "feature_ablation"
        if {"ablation_mode", "feature_group"} <= set(frame.columns):
            frame["pretraining_objective"] = frame.apply(
                lambda row: (
                    f"feature_ablation:{row.get('ablation_mode')}:{row.get('feature_group')}"
                ),
                axis=1,
            )
        if "lookback" not in frame.columns:
            frame["lookback"] = 0
        else:
            frame["lookback"] = frame["lookback"].fillna(0)
        frame["_source_prediction_file"] = _display_path(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), paths, ["no feature-ablation prediction files were available"]
    return pd.concat(frames, ignore_index=True, sort=False), paths, warnings


def _load_feature_ablation_inputs(
    *,
    ablation_dir: Path,
    out_dir: Path,
    config: _RuntimeConfig,
    overwrite: bool,
) -> _LoadedInputs:
    root = Path(ablation_dir)
    if not root.exists():
        raise FileNotFoundError(f"feature-ablation directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"feature-ablation path is not a directory: {root}")
    summary_path = root / "summary.json"
    results_path = root / "results_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"feature-ablation summary.json is missing: {summary_path}")
    if not results_path.is_file():
        raise FileNotFoundError(f"feature-ablation results_summary.csv is missing: {results_path}")
    summary = _read_json(summary_path)
    smoke_test = _is_smoke_summary(summary)
    if smoke_test and not config.allow_smoke_test:
        raise ValueError(
            "execution-v3 refuses smoke-test feature-ablation artefacts unless "
            "allow_smoke_test=True"
        )
    _ensure_output_dir(out_dir, input_dir=root, overwrite=overwrite)
    input_files: dict[str, Path] = {
        "feature_ablation_summary": summary_path,
        "feature_ablation_results_summary": results_path,
    }
    for optional_name in (
        "aggregate_summary.csv",
        "feature_delta_summary.csv",
        "ablation_manifest.json",
    ):
        candidate = root / optional_name
        if candidate.is_file():
            input_files[f"feature_ablation_{Path(optional_name).stem}"] = candidate
    input_hashes = {key: sha256_file(path) for key, path in input_files.items()}
    warnings: list[str] = []
    results = _filter_result_rows(pd.read_csv(results_path), config=config)
    predictions_raw, prediction_paths, prediction_warnings = load_feature_ablation_predictions(root)
    warnings.extend(prediction_warnings)
    for index, path in enumerate(prediction_paths):
        key = f"feature_ablation_prediction_file_{index:04d}"
        input_files[key] = path
        input_hashes[key] = sha256_file(path)
    if config.models or config.horizons or config.folds or config.seeds:
        predictions_raw = _filter_prediction_rows(predictions_raw, config=config)
    label_mapping_audit = _build_label_mapping_audit(
        predictions_raw,
        results,
        existing_audit_path=root / "label_mapping_audit.json",
        warnings=warnings,
    )
    if config.strict and label_mapping_audit["status"] != "pass":
        audit_path = out_dir / "label_mapping_audit.json"
        audit_path.write_text(stable_json_dumps(label_mapping_audit), encoding="utf-8")
        raise ValueError(f"FI-2010 label mapping audit failed in strict mode; see {audit_path}")
    (out_dir / "label_mapping_audit.json").write_text(
        stable_json_dumps(label_mapping_audit),
        encoding="utf-8",
    )
    predictions = _canonicalise_predictions(
        predictions_raw,
        strict=config.strict,
        warnings=warnings,
    )
    if predictions.empty:
        raise ValueError("no usable feature-ablation prediction rows were available")
    return _LoadedInputs(
        grid_dir=root,
        output_dir=out_dir,
        input_kind="feature_ablation",
        summary=summary,
        results=results,
        predictions_raw=predictions_raw,
        predictions=predictions,
        input_files=input_files,
        input_hashes=input_hashes,
        label_mapping_audit=label_mapping_audit,
        smoke_test=smoke_test,
        warnings=warnings,
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = cast(dict[str, Any], {})
    raw = path.read_text(encoding="utf-8")
    import json

    loaded = json.loads(raw)
    if not isinstance(loaded, dict):
        raise ValueError(f"expected JSON object: {path}")
    payload = cast(dict[str, Any], loaded)
    return payload


def _ensure_output_dir(out_dir: Path, *, input_dir: Path, overwrite: bool) -> None:
    resolved_out = out_dir.resolve(strict=False)
    resolved_input = input_dir.resolve(strict=False)
    if resolved_out == resolved_input or resolved_input.is_relative_to(resolved_out):
        raise ValueError("execution-v3 output directory must not contain the input grid")
    if out_dir.exists():
        if not out_dir.is_dir():
            raise FileExistsError(f"output path exists and is not a directory: {out_dir}")
        if any(out_dir.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    "refusing to overwrite existing execution-v3 directory; "
                    f"pass overwrite=True: {out_dir}"
                )
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _is_smoke_summary(summary: Mapping[str, Any]) -> bool:
    return (
        bool(summary.get("smoke_test")) or str(summary.get("execution_mode", "")).lower() == "smoke"
    )


def _filter_result_rows(results: pd.DataFrame, *, config: _RuntimeConfig) -> pd.DataFrame:
    if results.empty:
        return results
    frame = results.copy()
    if "status" in frame.columns:
        frame = frame[frame["status"].astype(str).isin(_SUCCESS_STATUSES)].copy()
    frame = _result_frame_with_keys(frame)
    mask = pd.Series(True, index=frame.index)
    if config.models:
        selected = {item.lower() for item in config.models}
        mask &= frame.apply(lambda row: _matches_model_selection(row, selected), axis=1)
    if config.horizons:
        selected_horizons = set(config.horizons)
        mask &= frame["horizon"].apply(lambda value: _optional_int(value) in selected_horizons)
    if config.folds:
        selected_folds = set(config.folds)
        mask &= frame["fold"].apply(lambda value: _normalise_fold(value) in selected_folds)
    if config.seeds:
        selected_seeds = set(config.seeds)
        mask &= frame["seed"].apply(lambda value: _optional_int(value) in selected_seeds)
    return frame.loc[mask].reset_index(drop=True)


def _result_frame_with_keys(results: pd.DataFrame) -> pd.DataFrame:
    frame = results.copy()
    if "fold" not in frame.columns and "fold_id" in frame.columns:
        frame["fold"] = frame["fold_id"]
    if "fold" not in frame.columns:
        frame["fold"] = None
    for column in ("horizon", "seed", "lookback"):
        if column not in frame.columns:
            frame[column] = None
    if "model_family" not in frame.columns:
        if "model_name" in frame.columns:
            frame["model_family"] = frame["model_name"].astype(str)
        elif "model" in frame.columns:
            frame["model_family"] = frame["model"].astype(str)
        else:
            frame["model_family"] = "unknown"
    if "pretraining_objective" not in frame.columns:
        frame["pretraining_objective"] = "none"
    return frame


def _matches_model_selection(row: Mapping[str, Any], selected: set[str]) -> bool:
    family = str(row.get("model_family", "")).strip().lower()
    objective = _objective_label(row.get("pretraining_objective")).lower()
    return bool({family, objective, f"{family}/{objective}"} & selected)


def _filter_prediction_rows(
    predictions: pd.DataFrame,
    *,
    config: _RuntimeConfig,
) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    frame = _result_frame_with_keys(predictions)
    mask = pd.Series(True, index=frame.index)
    if config.models:
        selected = {item.lower() for item in config.models}
        mask &= frame.apply(lambda row: _matches_model_selection(row, selected), axis=1)
    if config.horizons:
        selected_horizons = set(config.horizons)
        mask &= frame["horizon"].apply(lambda value: _optional_int(value) in selected_horizons)
    if config.folds:
        selected_folds = set(config.folds)
        mask &= frame["fold"].apply(lambda value: _normalise_fold(value) in selected_folds)
    if config.seeds:
        selected_seeds = set(config.seeds)
        mask &= frame["seed"].apply(lambda value: _optional_int(value) in selected_seeds)
    return frame.loc[mask].reset_index(drop=True)


def _load_prediction_frame(
    results: pd.DataFrame,
    grid_dir: Path,
) -> tuple[pd.DataFrame, list[Path], list[str]]:
    rows: list[pd.DataFrame] = []
    paths: list[Path] = []
    warnings: list[str] = []
    if results.empty:
        return pd.DataFrame(), [], ["no completed full-grid result rows were available"]
    for _, row in results.iterrows():
        path = _prediction_path(row, grid_dir)
        if path is None or not path.is_file():
            warnings.append(
                "prediction file missing for "
                f"{row.get('model_family')} objective={row.get('pretraining_objective')} "
                f"horizon={row.get('horizon')} fold={row.get('fold')} seed={row.get('seed')}"
            )
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            warnings.append(f"prediction file is empty: {path}")
            continue
        for column in _GROUP_COLUMNS:
            if column not in frame.columns:
                frame[column] = row.get(column)
        if "partition" not in frame.columns:
            if "split" in frame.columns:
                frame["partition"] = frame["split"]
            else:
                frame["partition"] = "test"
        frame["_source_prediction_file"] = _display_path(path)
        rows.append(frame)
        paths.append(path)
    if not rows:
        return pd.DataFrame(), paths, warnings
    return pd.concat(rows, ignore_index=True, sort=False), paths, warnings


def _prediction_path(row: Mapping[str, Any], grid_dir: Path) -> Path | None:
    raw = row.get("prediction_file") or row.get("prediction_path")
    if raw is not None and str(raw).strip():
        path = Path(str(raw))
        if path.is_absolute():
            return path
        return grid_dir / path
    run_dir = row.get("run_dir")
    if run_dir is not None and str(run_dir).strip():
        return grid_dir / str(run_dir) / "predictions.csv"
    return None


def _build_label_mapping_audit(
    predictions: pd.DataFrame,
    results: pd.DataFrame,
    *,
    existing_audit_path: Path,
    warnings: Sequence[str],
) -> dict[str, Any]:
    errors: list[str] = []
    audit_warnings = list(warnings)
    existing_status = "not_available"
    if existing_audit_path.is_file():
        try:
            existing = _read_json(existing_audit_path)
            existing_status = str(existing.get("status", "unknown"))
            if existing_status != "pass":
                errors.append(f"existing label_mapping_audit.json status is {existing_status!r}")
        except (OSError, ValueError) as exc:
            existing_status = "unreadable"
            errors.append(f"existing label_mapping_audit.json could not be read: {exc}")

    probability_validation = validate_probability_columns(tuple(str(col) for col in predictions))
    class_order_validation = validate_class_order(FI2010_CANONICAL_CLASS_ORDER)
    for validation in (probability_validation, class_order_validation):
        errors.extend(validation.errors)
        audit_warnings.extend(validation.warnings)
    if predictions.empty:
        errors.append("no prediction rows available for label-mapping audit")
    for label in _detected_prediction_labels(predictions):
        try:
            canonical_class_name(label)
        except ValueError as exc:
            errors.append(str(exc))

    probability_details = probability_validation.details or {}
    return {
        "status": "pass" if not errors else "fail",
        "mode": "existing_audit_checked_and_canonical_validation_rerun",
        "existing_audit_status": existing_status,
        "expected_labels": {str(key): value for key, value in FI2010_RAW_LABEL_TO_CLASS.items()},
        "detected_labels": _detected_prediction_labels(predictions),
        "probability_columns_found": probability_details.get(
            "probability_columns_found",
            [],
        ),
        "probability_column_order_used": list(
            probability_details.get("probability_column_order_used", ())
        ),
        "class_order_used_for_metrics": list(FI2010_CANONICAL_CLASS_ORDER),
        "result_rows_checked": len(results),
        "prediction_rows_checked": len(predictions),
        "errors": errors,
        "warnings": audit_warnings,
    }


def _detected_prediction_labels(predictions: pd.DataFrame) -> list[Any]:
    if predictions.empty:
        return []
    labels: list[Any] = []
    for column in ("y_true", "y_pred", "label", "prediction"):
        if column in predictions.columns:
            labels.extend(
                value for value in predictions[column].dropna().tolist() if str(value) != ""
            )
    return sorted({_json_scalar(value) for value in labels}, key=lambda item: str(item))


def _canonicalise_predictions(
    predictions: pd.DataFrame,
    *,
    strict: bool,
    warnings: list[str],
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    try:
        probability_columns = probability_columns_for_order(
            tuple(str(col) for col in predictions.columns)
        )
    except ValueError as exc:
        if strict:
            raise
        warnings.append(f"prediction probabilities unavailable: {exc}")
        return pd.DataFrame()

    frame = predictions.copy()
    y_true_column = "y_true" if "y_true" in frame.columns else "label"
    y_pred_column = "y_pred" if "y_pred" in frame.columns else "prediction"
    if y_true_column not in frame.columns or y_pred_column not in frame.columns:
        raise ValueError("prediction rows must contain y_true/y_pred or label/prediction")
    for group_column in _GROUP_COLUMNS:
        if group_column not in frame.columns:
            raise ValueError(f"prediction rows must contain or inherit {group_column}")
    if "confidence" not in frame.columns:
        message = "prediction rows do not include confidence"
        if strict:
            raise ValueError(message)
        warnings.append(f"{message}; derived from named probabilities")
    frame["y_true_class"] = frame[y_true_column].apply(canonical_class_name)
    frame["y_pred_class"] = frame[y_pred_column].apply(canonical_class_name)
    for class_name, column in zip(
        FI2010_CANONICAL_CLASS_ORDER,
        probability_columns,
        strict=True,
    ):
        frame[f"prob_{class_name}"] = pd.to_numeric(frame[column], errors="coerce")
    probability_matrix = frame[[f"prob_{label}" for label in FI2010_CANONICAL_CLASS_ORDER]]
    if "confidence" in frame.columns:
        frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce")
    else:
        frame["confidence"] = probability_matrix.max(axis=1)
    frame["confidence"] = frame["confidence"].fillna(probability_matrix.max(axis=1))
    frame["fold"] = frame["fold"].apply(_normalise_fold)
    for column in ("horizon", "seed", "lookback"):
        frame[column] = frame[column].apply(_optional_int)
    frame["pretraining_objective"] = frame["pretraining_objective"].apply(_objective_label)
    frame["model_key"] = frame.apply(
        lambda row: f"{row.get('model_family')}/{row.get('pretraining_objective')}",
        axis=1,
    )
    frame["signal_direction"] = frame["y_pred_class"].map({"up": 1, "stationary": 0, "down": -1})
    frame["is_active_signal"] = frame["signal_direction"] != 0
    frame["directional_correct"] = (
        (frame["signal_direction"] == 1) & (frame["y_true_class"] == "up")
    ) | ((frame["signal_direction"] == -1) & (frame["y_true_class"] == "down"))
    if "partition" not in frame.columns:
        frame["partition"] = "test"
    frame["_row_order"] = _row_order(frame)
    return frame.dropna(
        subset=[
            "confidence",
            "y_true_class",
            "y_pred_class",
            "signal_direction",
            "horizon",
            "seed",
            "lookback",
        ]
    ).reset_index(drop=True)


def _row_order(frame: pd.DataFrame) -> pd.Series:
    for column in ("row_id", "sample_id", "timestamp", "event_index"):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().any():
                return values.fillna(pd.RangeIndex(len(frame)).to_series(index=frame.index))
    return pd.Series(range(len(frame)), index=frame.index)


def _prepare_payoff_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    signed_return = _first_present_column(frame, _SIGNED_RETURN_COLUMNS)
    magnitude = _first_present_column(frame, _MAGNITUDE_COLUMNS)
    if signed_return is not None:
        move = pd.to_numeric(frame[signed_return], errors="coerce")
        frame["_payoff_mode"] = "realised_return"
        frame["_payoff_source_column"] = signed_return
    elif magnitude is not None:
        raw_magnitude = pd.to_numeric(frame[magnitude], errors="coerce").abs()
        sign = frame["y_true_class"].map({"up": 1.0, "stationary": 0.0, "down": -1.0})
        move = raw_magnitude * sign
        frame["_payoff_mode"] = "realised_return"
        frame["_payoff_source_column"] = magnitude
    else:
        move = pd.Series([math.nan] * len(frame), index=frame.index)
        frame["_payoff_mode"] = "unit_payoff"
        frame["_payoff_source_column"] = ""
    frame["_future_move_proxy"] = move
    direction = pd.to_numeric(frame["signal_direction"], errors="coerce").fillna(0.0)
    if str(frame["_payoff_mode"].iloc[0]) == "realised_return":
        frame["gross_directional_payoff"] = direction * move.fillna(0.0)
        frame.loc[~frame["is_active_signal"].astype(bool), "gross_directional_payoff"] = 0.0
    else:
        frame["gross_directional_payoff"] = 0.0
        active = frame["is_active_signal"].astype(bool)
        correct = frame["directional_correct"].astype(bool)
        frame.loc[active & correct, "gross_directional_payoff"] = 1.0
        frame.loc[active & ~correct, "gross_directional_payoff"] = -1.0
    return frame


def _payoff_mode(predictions: pd.DataFrame) -> str:
    values = predictions.get("_payoff_mode")
    if values is None or values.empty:
        return "unit_payoff"
    return str(values.iloc[0])


def _cost_mode(predictions: pd.DataFrame) -> str:
    return "spread_proxy" if _spread_column(predictions) is not None else "unit_proxy"


def _confidence_threshold_rows(
    predictions: pd.DataFrame,
    *,
    config: _RuntimeConfig,
    cost_mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(list(_GROUP_COLUMNS), dropna=False, sort=True):
        base = _group_key_dict(keys)
        total = len(group)
        for threshold in config.confidence_thresholds:
            retained = group[group["confidence"] >= threshold]
            active = retained[retained["is_active_signal"].astype(bool)]
            status = "ok"
            reason = ""
            if len(retained) < len(FI2010_CANONICAL_CLASS_ORDER):
                status = "skipped"
                reason = "too few samples remain after confidence filtering"
            costs = _cost_series(
                retained,
                fee_bps=config.reference_fee_bps,
                spread_multiplier=config.reference_spread_multiplier,
                cost_mode=cost_mode,
            )
            net_payoff = retained["gross_directional_payoff"] - costs
            row: dict[str, Any] = {
                **base,
                "threshold": threshold,
                "status": status,
                "skipped_reason": reason,
                "sample_count": total,
                "retained_count": len(retained),
                "retained_sample_fraction": _safe_ratio(len(retained), total),
                "active_trade_count": len(active),
                "active_trade_fraction": _safe_ratio(len(active), total),
                "classification_accuracy": _classification_accuracy(retained)
                if status == "ok"
                else None,
                "macro_f1": _macro_f1(
                    retained["y_true_class"].tolist(),
                    retained["y_pred_class"].tolist(),
                )
                if status == "ok"
                else None,
                "directional_trade_hit_rate": _directional_hit_rate(active),
                "gross_directional_proxy": _sum(retained["gross_directional_payoff"]),
                "net_cost_adjusted_proxy": _sum(net_payoff),
                "turnover_proxy": _safe_ratio(len(active), max(total - 1, 1)),
                "payoff_mode": (
                    _payoff_mode(retained) if not retained.empty else _payoff_mode(group)
                ),
                "cost_mode": cost_mode,
                "reference_fee_bps": config.reference_fee_bps,
                "reference_spread_multiplier": config.reference_spread_multiplier,
                **_class_mix(retained, prefix="true"),
                **_predicted_class_mix(retained),
            }
            rows.append(row)
    return rows


def _confidence_threshold_aggregate_rows(
    confidence_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not confidence_rows:
        return []
    frame = pd.DataFrame(confidence_rows)
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        ["model_family", "pretraining_objective", "horizon", "threshold"],
        dropna=False,
        sort=True,
    ):
        model_family, objective, horizon, threshold = keys
        ok = group[group["status"] == "ok"]
        source = ok if not ok.empty else group
        rows.append(
            {
                "model_family": str(model_family),
                "pretraining_objective": str(objective),
                "horizon": _optional_int(horizon),
                "threshold": float(threshold),
                "run_group_count": len(group),
                "ok_group_count": len(ok),
                "status": "ok" if not ok.empty else "skipped",
                "mean_retained_sample_fraction": _mean(source["retained_sample_fraction"]),
                "mean_active_trade_fraction": _mean(source["active_trade_fraction"]),
                "mean_classification_accuracy": _mean(source["classification_accuracy"]),
                "mean_macro_f1": _mean(source["macro_f1"]),
                "mean_directional_trade_hit_rate": _mean(source["directional_trade_hit_rate"]),
                "mean_gross_directional_proxy": _mean(source["gross_directional_proxy"]),
                "mean_net_cost_adjusted_proxy": _mean(source["net_cost_adjusted_proxy"]),
                "payoff_mode": _first_non_empty(source.get("payoff_mode")),
                "cost_mode": _first_non_empty(source.get("cost_mode")),
            }
        )
    return rows


def _cost_sensitivity_rows(
    predictions: pd.DataFrame,
    *,
    config: _RuntimeConfig,
    payoff_mode: str,
    cost_mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(list(_GROUP_COLUMNS), dropna=False, sort=True):
        base = _group_key_dict(keys)
        retained = group[group["confidence"] >= config.reference_threshold]
        active_count = int(retained["is_active_signal"].astype(bool).sum())
        gross = _sum(retained["gross_directional_payoff"])
        for fee_bps in config.fee_bps:
            for spread_multiplier in config.spread_multipliers:
                costs = _cost_series(
                    retained,
                    fee_bps=fee_bps,
                    spread_multiplier=spread_multiplier,
                    cost_mode=cost_mode,
                )
                total_cost = _sum(costs)
                net = gross - total_cost
                rows.append(
                    {
                        **base,
                        "confidence_threshold": config.reference_threshold,
                        "fee_bps": fee_bps,
                        "spread_multiplier": spread_multiplier,
                        "gross_proxy": gross,
                        "net_proxy": net,
                        "degradation_percentage": _degradation_percentage(gross, net),
                        "active_trade_count": active_count,
                        "active_trade_fraction": _safe_ratio(active_count, len(group)),
                        "average_cost_per_active_trade": _safe_ratio(
                            total_cost,
                            active_count,
                        ),
                        "cost_mode": cost_mode,
                        "payoff_mode": payoff_mode,
                        "status": "ok",
                        "skipped_reason": "",
                    }
                )
    return rows


def _latency_sensitivity_rows(
    predictions: pd.DataFrame,
    *,
    config: _RuntimeConfig,
    cost_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    group_columns = [*_GROUP_COLUMNS, "partition"]
    for keys, group in predictions.groupby(group_columns, dropna=False, sort=True):
        group = group.sort_values("_row_order").reset_index(drop=True)
        base_key = _group_key_dict(keys[: len(_GROUP_COLUMNS)])
        base_key["partition"] = str(keys[-1])
        base_gross: float | None = None
        base_net: float | None = None
        for latency in config.latency_steps:
            shifted = _latency_shifted_group(group, latency)
            status = "ok"
            reason = ""
            if shifted.empty:
                status = "skipped"
                reason = "latency step drops all samples within fold/partition boundary"
                skipped.append(
                    {
                        "diagnostic": "latency_sensitivity",
                        "scope": _scope_from_key(base_key),
                        "skip_reason": reason,
                    }
                )
            costs = _cost_series(
                shifted,
                fee_bps=config.reference_fee_bps,
                spread_multiplier=config.reference_spread_multiplier,
                cost_mode=cost_mode,
            )
            gross = _sum(shifted.get("latency_gross_directional_payoff"))
            net = gross - _sum(costs)
            if latency == 0 and status == "ok":
                base_gross = gross
                base_net = net
            rows.append(
                {
                    **base_key,
                    "latency_step": latency,
                    "retained_samples": len(shifted),
                    "dropped_samples": max(len(group) - len(shifted), 0),
                    "active_trades": int(shifted["is_active_signal"].astype(bool).sum())
                    if not shifted.empty
                    else 0,
                    "directional_hit_rate": _directional_hit_rate(
                        shifted[shifted["is_active_signal"].astype(bool)]
                    ),
                    "gross_proxy": gross,
                    "net_proxy": net,
                    "degradation_vs_latency_0": _degradation_absolute(base_gross, gross),
                    "net_degradation_vs_latency_0": _degradation_absolute(base_net, net),
                    "payoff_mode": _payoff_mode(group),
                    "cost_mode": cost_mode,
                    "status": status,
                    "skipped_reason": reason,
                }
            )
    if not rows:
        skipped.append(
            {
                "diagnostic": "latency_sensitivity",
                "scope": "all",
                "skip_reason": "no row-level prediction groups were available",
            }
        )
    return rows, skipped


def _latency_shifted_group(group: pd.DataFrame, latency: int) -> pd.DataFrame:
    if latency < 0:
        raise ValueError("latency steps must be non-negative")
    if latency == 0:
        shifted = group.copy()
        shifted["latency_y_true_class"] = shifted["y_true_class"]
        shifted["latency_future_move_proxy"] = shifted["_future_move_proxy"]
    elif latency >= len(group):
        return pd.DataFrame(columns=[*list(group.columns), "latency_gross_directional_payoff"])
    else:
        shifted = group.iloc[:-latency].copy()
        outcomes = group.iloc[latency:].reset_index(drop=True)
        shifted = shifted.reset_index(drop=True)
        shifted["latency_y_true_class"] = outcomes["y_true_class"]
        shifted["latency_future_move_proxy"] = outcomes["_future_move_proxy"]
    active = shifted["is_active_signal"].astype(bool)
    direction = pd.to_numeric(shifted["signal_direction"], errors="coerce").fillna(0.0)
    if _payoff_mode(group) == "realised_return":
        shifted_move = pd.to_numeric(
            shifted["latency_future_move_proxy"],
            errors="coerce",
        ).fillna(0.0)
        shifted["latency_gross_directional_payoff"] = direction * shifted_move
        shifted.loc[~active, "latency_gross_directional_payoff"] = 0.0
    else:
        correct = (
            (shifted["signal_direction"] == 1) & (shifted["latency_y_true_class"] == "up")
        ) | ((shifted["signal_direction"] == -1) & (shifted["latency_y_true_class"] == "down"))
        shifted["directional_correct"] = correct
        shifted["latency_gross_directional_payoff"] = 0.0
        shifted.loc[active & correct, "latency_gross_directional_payoff"] = 1.0
        shifted.loc[active & ~correct, "latency_gross_directional_payoff"] = -1.0
    shifted["gross_directional_payoff"] = shifted["latency_gross_directional_payoff"]
    return shifted


def _fill_assumption_rows(
    predictions: pd.DataFrame,
    *,
    config: _RuntimeConfig,
    cost_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for keys, group in predictions.groupby(list(_GROUP_COLUMNS), dropna=False, sort=True):
        base = _group_key_dict(keys)
        active_count = int(group["is_active_signal"].astype(bool).sum())
        for fill_mode in config.fill_assumptions:
            filled_mask, reason, fallback = _fill_mask(
                group,
                fill_mode=fill_mode,
                threshold=config.reference_threshold,
            )
            filled = group[filled_mask]
            costs = _cost_series(
                filled,
                fee_bps=max(config.fee_bps),
                spread_multiplier=max(config.spread_multipliers),
                cost_mode=cost_mode,
            ) * _fill_cost_multiplier(fill_mode)
            status = "ok"
            if reason:
                status = "skipped" if fill_mode not in DEFAULT_FILL_ASSUMPTIONS else "ok"
                skipped.append(
                    {
                        "diagnostic": "fill_assumptions",
                        "scope": f"{_scope_from_key(base)}:{fill_mode}",
                        "skip_reason": reason,
                    }
                )
            gross = _sum(filled["gross_directional_payoff"])
            total_cost = _sum(costs)
            rows.append(
                {
                    **base,
                    "confidence_threshold": config.reference_threshold,
                    "fill_mode": fill_mode,
                    "active_signal_count": active_count,
                    "filled_count": len(filled),
                    "fill_fraction": _safe_ratio(len(filled), active_count),
                    "directional_hit_rate_on_filled_trades": _directional_hit_rate(filled),
                    "gross_proxy": gross,
                    "net_proxy": gross - total_cost,
                    "average_cost": _safe_ratio(total_cost, len(filled)),
                    "cost_mode": cost_mode,
                    "fill_fallback": fallback,
                    "status": status,
                    "skipped_reason": reason if status == "skipped" else "",
                }
            )
    return rows, skipped


def _fill_mask(
    group: pd.DataFrame,
    *,
    fill_mode: str,
    threshold: float,
) -> tuple[pd.Series, str, str]:
    active = group["is_active_signal"].astype(bool)
    if fill_mode == "aggressive_crossing":
        return active, "", "none"
    if fill_mode == "abstain_only":
        return pd.Series([False] * len(group), index=group.index), "", "baseline_no_trades"
    if fill_mode not in {"passive_optimistic", "passive_conservative"}:
        return (
            pd.Series([False] * len(group), index=group.index),
            f"unknown fill assumption {fill_mode!r}",
            "unavailable",
        )

    required_confidence = threshold
    if fill_mode == "passive_conservative":
        required_confidence = max(threshold + 0.10, 0.75)
    mask = active & (group["confidence"] >= required_confidence)
    fallback_notes: list[str] = []

    spread_col = _spread_column(group)
    if spread_col is not None:
        spread = pd.to_numeric(group[spread_col], errors="coerce")
        quantile = 0.75 if fill_mode == "passive_optimistic" else 0.40
        threshold_value = _series_quantile(spread, quantile)
        if threshold_value is not None:
            mask &= spread <= threshold_value
    else:
        fallback_notes.append("spread_proxy_missing")

    liquidity = _liquidity_series(group)
    if liquidity is not None:
        quantile = 0.40 if fill_mode == "passive_optimistic" else 0.60
        threshold_value = _series_quantile(liquidity, quantile)
        if threshold_value is not None:
            mask &= liquidity >= threshold_value
    else:
        fallback_notes.append("liquidity_proxy_missing")

    fallback = "none" if not fallback_notes else "confidence_only:" + ",".join(fallback_notes)
    return mask, "", fallback


def _adverse_selection_rows(
    predictions: pd.DataFrame,
    *,
    config: _RuntimeConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mode = "future_move_proxy" if _payoff_mode(predictions) == "realised_return" else "label_proxy"
    for fill_mode in config.fill_assumptions:
        filled_mask, _reason, _fallback = _fill_mask(
            predictions,
            fill_mode=fill_mode,
            threshold=config.reference_threshold,
        )
        scoped = predictions[filled_mask].copy()
        if scoped.empty:
            rows.append(
                {
                    "model_family": "all",
                    "pretraining_objective": "all",
                    "horizon": None,
                    "confidence_bucket": "all",
                    "fill_assumption": fill_mode,
                    "adverse_selection_mode": mode,
                    "filled_count": 0,
                    "adverse_count": 0,
                    "adverse_fraction": None,
                    "status": "skipped",
                    "skipped_reason": "fill assumption produced no filled trades",
                }
            )
            continue
        scoped["confidence_bucket"] = scoped["confidence"].apply(_confidence_bucket)
        scoped["adverse_selected"] = _adverse_selected(scoped, mode=mode)
        for keys, group in scoped.groupby(
            ["model_family", "pretraining_objective", "horizon", "confidence_bucket"],
            dropna=False,
            sort=True,
        ):
            model_family, objective, horizon, bucket = keys
            adverse_count = int(group["adverse_selected"].sum())
            rows.append(
                {
                    "model_family": str(model_family),
                    "pretraining_objective": str(objective),
                    "horizon": _optional_int(horizon),
                    "confidence_bucket": str(bucket),
                    "fill_assumption": fill_mode,
                    "adverse_selection_mode": mode,
                    "filled_count": len(group),
                    "adverse_count": adverse_count,
                    "adverse_fraction": _safe_ratio(adverse_count, len(group)),
                    "status": "ok",
                    "skipped_reason": "",
                }
            )
    return rows


def _adverse_selected(frame: pd.DataFrame, *, mode: str) -> pd.Series:
    long_signal = frame["signal_direction"] == 1
    short_signal = frame["signal_direction"] == -1
    if mode == "future_move_proxy":
        move = pd.to_numeric(frame["_future_move_proxy"], errors="coerce").fillna(0.0)
        return (long_signal & (move < 0.0)) | (short_signal & (move > 0.0))
    return (long_signal & (frame["y_true_class"] == "down")) | (
        short_signal & (frame["y_true_class"] == "up")
    )


def _regime_execution_rows(
    predictions: pd.DataFrame,
    *,
    config: _RuntimeConfig,
    cost_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for regime_type, candidates in _REGIME_COLUMNS.items():
        column = _first_present_column(predictions, candidates)
        if column is None:
            reason = f"{regime_type} labels unavailable in prediction artefacts"
            skipped.append(
                {
                    "diagnostic": "regime_execution",
                    "scope": regime_type,
                    "skip_reason": reason,
                }
            )
            rows.append(
                {
                    "model_family": "all",
                    "pretraining_objective": "all",
                    "horizon": None,
                    "regime_type": regime_type,
                    "regime_label": "unavailable",
                    "sample_count": 0,
                    "active_trade_count": 0,
                    "directional_hit_rate": None,
                    "gross_proxy": None,
                    "net_proxy": None,
                    "payoff_mode": _payoff_mode(predictions),
                    "cost_mode": cost_mode,
                    "status": "skipped",
                    "skipped_reason": reason,
                }
            )
            continue
        retained = predictions[predictions["confidence"] >= config.reference_threshold].copy()
        for keys, group in retained.groupby(
            ["model_family", "pretraining_objective", "horizon", column],
            dropna=False,
            sort=True,
        ):
            model_family, objective, horizon, label = keys
            costs = _cost_series(
                group,
                fee_bps=config.reference_fee_bps,
                spread_multiplier=config.reference_spread_multiplier,
                cost_mode=cost_mode,
            )
            rows.append(
                {
                    "model_family": str(model_family),
                    "pretraining_objective": str(objective),
                    "horizon": _optional_int(horizon),
                    "regime_type": regime_type,
                    "regime_label": str(label),
                    "sample_count": len(group),
                    "active_trade_count": int(group["is_active_signal"].astype(bool).sum()),
                    "directional_hit_rate": _directional_hit_rate(
                        group[group["is_active_signal"].astype(bool)]
                    ),
                    "gross_proxy": _sum(group["gross_directional_payoff"]),
                    "net_proxy": _sum(group["gross_directional_payoff"] - costs),
                    "payoff_mode": _payoff_mode(group),
                    "cost_mode": cost_mode,
                    "status": "ok",
                    "skipped_reason": "",
                }
            )
    return rows, skipped


def _cost_series(
    frame: pd.DataFrame,
    *,
    fee_bps: float,
    spread_multiplier: float,
    cost_mode: str,
) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    active = frame["is_active_signal"].astype(bool)
    fee_cost = float(fee_bps) / 10_000.0
    if cost_mode == "spread_proxy":
        spread_component = _spread_cost_component(frame)
        cost = fee_cost + float(spread_multiplier) * spread_component
    else:
        cost = fee_cost + float(spread_multiplier) * _UNIT_SPREAD_COST
    if not isinstance(cost, pd.Series):
        cost = pd.Series([cost] * len(frame), index=frame.index)
    return pd.to_numeric(cost, errors="coerce").fillna(0.0).where(active, 0.0)


def _spread_cost_component(frame: pd.DataFrame) -> pd.Series:
    column = _spread_column(frame)
    if column is None:
        return pd.Series([_UNIT_SPREAD_COST] * len(frame), index=frame.index)
    spread = pd.to_numeric(frame[column], errors="coerce").abs()
    if column == "spread_bps":
        return spread / 10_000.0
    if column == "spread" and "mid_price" in frame.columns:
        mid = pd.to_numeric(frame["mid_price"], errors="coerce").abs()
        return spread / mid.where(mid > 0.0)
    return spread


def _spread_column(frame: pd.DataFrame) -> str | None:
    return _first_present_column(frame, _SPREAD_COLUMNS)


def _liquidity_series(frame: pd.DataFrame) -> pd.Series | None:
    available = [column for column in _LIQUIDITY_COLUMNS if column in frame.columns]
    if not available:
        return None
    if {"bid_depth_1", "ask_depth_1"} <= set(available):
        return pd.to_numeric(frame["bid_depth_1"], errors="coerce") + pd.to_numeric(
            frame["ask_depth_1"],
            errors="coerce",
        )
    return pd.to_numeric(frame[available[0]], errors="coerce")


def _class_mix(frame: pd.DataFrame, *, prefix: str) -> dict[str, float | None]:
    if frame.empty:
        return {f"{prefix}_{label}_fraction": None for label in FI2010_CANONICAL_CLASS_ORDER}
    return {
        f"{prefix}_{label}_fraction": float((frame["y_true_class"] == label).mean())
        for label in FI2010_CANONICAL_CLASS_ORDER
    }


def _predicted_class_mix(frame: pd.DataFrame) -> dict[str, float | None]:
    if frame.empty:
        return {f"predicted_{label}_fraction": None for label in FI2010_CANONICAL_CLASS_ORDER}
    return {
        f"predicted_{label}_fraction": float((frame["y_pred_class"] == label).mean())
        for label in FI2010_CANONICAL_CLASS_ORDER
    }


def _classification_accuracy(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    return float((frame["y_true_class"] == frame["y_pred_class"]).mean())


def _macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("macro-F1 inputs must have matching lengths")
    if not y_true:
        return 0.0
    return float(
        sum(
            _binary_f1(y_true, y_pred, positive_label=label)
            for label in FI2010_CANONICAL_CLASS_ORDER
        )
        / len(FI2010_CANONICAL_CLASS_ORDER)
    )


def _binary_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    positive_label: str,
) -> float:
    tp = sum(
        1
        for actual, predicted in zip(y_true, y_pred, strict=True)
        if actual == positive_label and predicted == positive_label
    )
    fp = sum(
        1
        for actual, predicted in zip(y_true, y_pred, strict=True)
        if actual != positive_label and predicted == positive_label
    )
    fn = sum(
        1
        for actual, predicted in zip(y_true, y_pred, strict=True)
        if actual == positive_label and predicted != positive_label
    )
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else float(2 * tp / denom)


def _directional_hit_rate(active: pd.DataFrame) -> float | None:
    if active.empty:
        return None
    return float(active["directional_correct"].astype(bool).mean())


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([_clean_mapping(row) for row in rows]) if rows else pd.DataFrame()
    frame.to_csv(path, index=False)


def _execution_manifest(
    *,
    loaded: _LoadedInputs,
    output_files: Mapping[str, Path],
    payoff_mode: str,
    cost_mode: str,
    diagnostics_produced: Sequence[str],
    diagnostics_skipped: Sequence[str],
    skipped: Sequence[Mapping[str, str]],
    config: _RuntimeConfig,
) -> dict[str, Any]:
    output_hashes = {
        key: sha256_file(path) for key, path in sorted(output_files.items()) if path.is_file()
    }
    return {
        "builder_version": EXECUTION_V3_VERSION,
        "package_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
        "offline_execution_aware_proxy_diagnostic": True,
        "claim_boundary": (
            "This is an offline execution-aware proxy diagnostic. It is not a "
            "live trading system, broker integration, profitability claim or "
            "realistic execution simulator."
        ),
        "input_artefact_paths": {
            key: _display_path(path) for key, path in sorted(loaded.input_files.items())
        },
        "input_file_hashes": {
            _display_path(loaded.input_files[key]): value
            for key, value in sorted(loaded.input_hashes.items())
            if key in loaded.input_files
        },
        "output_files": {key: _display_path(path) for key, path in sorted(output_files.items())},
        "output_file_hashes": output_hashes,
        "label_mapping_audit_status": loaded.label_mapping_audit.get("status"),
        "label_mapping_audit": loaded.label_mapping_audit,
        "payoff_mode": payoff_mode,
        "cost_mode": cost_mode,
        "fill_modes_used": list(config.fill_assumptions),
        "latency_modes_used": list(config.latency_steps),
        "confidence_thresholds": list(config.confidence_thresholds),
        "fee_bps": list(config.fee_bps),
        "spread_multipliers": list(config.spread_multipliers),
        "skipped_diagnostics": list(skipped),
        "diagnostics_produced": list(diagnostics_produced),
        "diagnostics_skipped": list(diagnostics_skipped),
        "smoke_test": loaded.smoke_test,
        "smoke_test_status": (
            "smoke-test-only; not empirical evidence" if loaded.smoke_test else "not smoke-test"
        ),
        "strict": config.strict,
        "warnings": loaded.warnings,
    }


def _notes_markdown(*, payoff_mode: str, cost_mode: str, smoke_test: bool) -> str:
    smoke_line = (
        "- Smoke-test artefacts are marked smoke-test-only and are not empirical evidence."
        if smoke_test
        else "- Inputs were not marked as smoke-test artefacts."
    )
    return "\n".join(
        [
            "# FI-2010 Execution-Aware Proxy Diagnostic V3",
            "",
            "This is an offline execution-aware proxy diagnostic. It is not a live "
            "trading system, broker integration, profitability claim or realistic "
            "execution simulator.",
            "",
            f"- Payoff mode: `{payoff_mode}`.",
            f"- Cost mode: `{cost_mode}`.",
            "- Reported cost-adjusted values are proxy diagnostics, not PnL.",
            "- Stationary predictions are no-trade signals in trade-specific metrics.",
            "- Costs are subtracted from active or filled directional trades only.",
            smoke_line,
            "",
        ]
    )


def _parse_model_selection(value: Sequence[str] | str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "all":
            return None
        tokens: Sequence[str] = [token.strip() for token in text.split(",")]
    else:
        tokens = list(value)
    cleaned: list[str] = []
    for token in tokens:
        item = str(token).strip()
        if item and item not in cleaned:
            cleaned.append(item)
    return tuple(cleaned) if cleaned else None


def _parse_float_selection(
    value: Sequence[float] | str | None,
    *,
    default: tuple[float, ...],
    lower_bound: float,
) -> tuple[float, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "all":
            return default
        tokens: Sequence[Any] = [token.strip() for token in text.split(",")]
    else:
        tokens = list(value)
    cleaned: list[float] = []
    for token in tokens:
        if str(token).strip() == "":
            continue
        number = float(token)
        if number < lower_bound:
            raise ValueError(f"numeric option values must be >= {lower_bound:g}")
        if number not in cleaned:
            cleaned.append(number)
    return tuple(sorted(cleaned)) if cleaned else default


def _parse_int_selection(
    value: Sequence[int] | str | None,
    *,
    positive: bool,
    default: tuple[int, ...] | None = None,
) -> tuple[int, ...] | None:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "all":
            return default
        tokens: Sequence[Any] = [token.strip() for token in text.split(",")]
    else:
        tokens = list(value)
    cleaned: list[int] = []
    for token in tokens:
        if str(token).strip() == "":
            continue
        number = int(token)
        if positive and number <= 0:
            raise ValueError("integer option values must be positive")
        if not positive and number < 0:
            raise ValueError("integer option values must be non-negative")
        if number not in cleaned:
            cleaned.append(number)
    return tuple(sorted(cleaned)) if cleaned else default


def _parse_fold_selection(value: Sequence[int | str] | str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "all":
            return None
        tokens: Sequence[Any] = [token.strip() for token in text.split(",")]
    else:
        tokens = list(value)
    cleaned: list[str] = []
    for token in tokens:
        fold = _normalise_fold(token)
        if fold and fold not in cleaned:
            cleaned.append(fold)
    return tuple(cleaned) if cleaned else None


def _parse_fill_assumptions(value: Sequence[str] | str | None) -> tuple[str, ...]:
    parsed = _parse_model_selection(value)
    if parsed is None:
        return DEFAULT_FILL_ASSUMPTIONS
    return parsed


def _normalise_fold(value: Any) -> str:
    text = str(value).strip().lower()
    if not text or text == "none" or text == "nan":
        return "unknown"
    if text.startswith("fold_"):
        suffix = text.removeprefix("fold_")
        if suffix.isdigit():
            return f"fold_{int(suffix)}"
    if text.isdigit():
        return f"fold_{int(text)}"
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return f"fold_{int(number)}"
    return text


def _objective_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"", "none", "nan", "supervised", "random_init"}:
        return "supervised"
    if text == "masked_field":
        return "masked_reconstruction"
    return text


def _group_key_dict(keys: Any) -> dict[str, Any]:
    if not isinstance(keys, tuple):
        keys = (keys,)
    return {
        "model_family": str(keys[0]),
        "pretraining_objective": str(keys[1]),
        "horizon": _optional_int(keys[2]),
        "fold": str(keys[3]),
        "seed": _optional_int(keys[4]),
        "lookback": _optional_int(keys[5]),
    }


def _scope_from_key(row: Mapping[str, Any]) -> str:
    return (
        f"{row.get('model_family')}/{row.get('pretraining_objective')}/"
        f"h{row.get('horizon')}/{row.get('fold')}/s{row.get('seed')}/"
        f"lb{row.get('lookback')}"
    )


def _first_present_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _safe_ratio(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    try:
        den = float(denominator)
        num = float(numerator)
    except (TypeError, ValueError):
        return None
    if den == 0.0 or not math.isfinite(den) or not math.isfinite(num):
        return None
    return num / den


def _sum(values: Any) -> float:
    if values is None:
        return 0.0
    series = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return float(series.sum())


def _mean(values: Any) -> float | None:
    if values is None:
        return None
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.mean())


def _first_non_empty(values: Any) -> str:
    if values is None:
        return ""
    for value in values:
        text = str(value)
        if text and text.lower() != "nan":
            return text
    return ""


def _series_quantile(series: pd.Series, quantile: float) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.quantile(quantile))


def _degradation_percentage(gross: float, net: float) -> float | None:
    if gross == 0.0 or not math.isfinite(gross) or not math.isfinite(net):
        return None
    return float((gross - net) / abs(gross) * 100.0)


def _degradation_absolute(base: float | None, value: float) -> float | None:
    if base is None:
        return None
    return float(value - base)


def _fill_cost_multiplier(fill_mode: str) -> float:
    if fill_mode == "aggressive_crossing":
        return 1.5
    if fill_mode == "passive_conservative":
        return 0.5
    if fill_mode == "abstain_only":
        return 0.0
    return 1.0


def _confidence_bucket(value: Any) -> str:
    numeric = float(value)
    for label, lower, upper in _CONFIDENCE_BUCKETS:
        if lower <= numeric < upper:
            return label
    if numeric < _CONFIDENCE_BUCKETS[0][1]:
        return f"<{_CONFIDENCE_BUCKETS[0][1]:.2f}"
    return ">=1.00"


def _group_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    return int(frame.groupby(list(_GROUP_COLUMNS), dropna=False).ngroups)


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    item = value.item() if hasattr(value, "item") else value
    if isinstance(item, float) and item.is_integer():
        return int(item)
    return item


def _clean_mapping(row: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _clean_value(value) for key, value in row.items()}


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _display_path(path: Path) -> str:
    candidate = Path(path)
    try:
        resolved = candidate.resolve(strict=False)
        root = project_root().resolve(strict=False)
        return resolved.relative_to(root).as_posix()
    except (OSError, ValueError):
        return candidate.as_posix()

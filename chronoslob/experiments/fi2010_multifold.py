"""Local multi-fold FI-2010 preparation layer.

This module turns the official FI-2010 ``BenchmarkDatasets`` directory
into one combined CSV file per configured fold, plus a per-fold manifest
and a top-level summary. It only reads user-supplied local files and
never downloads anything. The combined CSV files produced under the
caller-supplied processed root are intentionally not committed to the
repository.

The layer does not train models, does not produce predictions and does
not produce paper benchmark results. It is preparation only and is
gated by the wider research protocol documented at
``docs/RESEARCH_PROTOCOL.md``.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from chronoslob.data.fi2010_official import (
    OFFICIAL_FI2010_LABEL_HORIZONS,
    convert_official_fi2010_to_csv,
)
from chronoslob.experiments.manifests import sha256_file, stable_json_dumps
from chronoslob.training.experiment import get_git_commit

__all__ = [
    "FoldFilePlan",
    "FoldPreparationManifest",
    "MultifoldPreparationOptions",
    "MultifoldPreparationResult",
    "MultifoldPreparationSummary",
    "MultifoldStudyConfig",
    "discover_fold_files",
    "inspect_multifold_files",
    "load_multifold_config",
    "prepare_multifold",
]

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)
_PREPARATION_VERSION = "phase-c/fi2010-multifold-preparation/v1"
_DEFAULT_LOCAL_NOTE = (
    "Raw FI-2010 .txt matrices under the extracted dataset root and "
    "processed combined CSV files under the processed output root are "
    "local-only and gitignored. The repository does not download FI-2010 "
    "and does not commit raw or processed FI-2010 data."
)


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


class MultifoldFoldOverride(BaseModel):
    """Optional per-fold explicit path overrides relative to the dataset root."""

    model_config = _MODEL_CONFIG

    train_path: str | None = None
    test_path: str | None = None

    @field_validator("train_path", "test_path")
    @classmethod
    def _validate_optional_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("override path must be non-empty when provided")
        return value.strip()


class MultifoldPreparationOptions(BaseModel):
    """Preparation-only options for the multi-fold runner."""

    model_config = _MODEL_CONFIG

    extracted_dataset_root_placeholder: str
    processed_output_root_placeholder: str
    combined_csv_filename_template: str = "fold{fold}_combined.csv"
    train_filename_template: str
    test_filename_template: str
    fold_overrides: dict[str, MultifoldFoldOverride] = Field(default_factory=dict)
    compute_source_hashes: bool = True
    local_only_note: str = _DEFAULT_LOCAL_NOTE

    @field_validator(
        "extracted_dataset_root_placeholder",
        "processed_output_root_placeholder",
        "combined_csv_filename_template",
        "train_filename_template",
        "test_filename_template",
        "local_only_note",
    )
    @classmethod
    def _validate_required_string(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("preparation field must be a non-empty string")
        return value.strip()

    @model_validator(mode="after")
    def _validate_templates(self) -> MultifoldPreparationOptions:
        if "{fold}" not in self.combined_csv_filename_template:
            raise ValueError(
                "combined_csv_filename_template must contain a '{fold}' "
                "placeholder",
            )
        if "{fold}" not in self.train_filename_template:
            raise ValueError(
                "train_filename_template must contain a '{fold}' placeholder",
            )
        if "{fold}" not in self.test_filename_template:
            raise ValueError(
                "test_filename_template must contain a '{fold}' placeholder",
            )
        return self


class MultifoldStudyConfig(BaseModel):
    """Validated multi-fold preparation configuration."""

    model_config = _MODEL_CONFIG

    study_name: str
    dataset_name: str
    folds: tuple[int, ...]
    label_columns: tuple[str, ...]
    target_horizon: int
    split_column: str
    train_value: str
    test_value: str
    preparation: MultifoldPreparationOptions

    @field_validator("study_name", "dataset_name", "split_column")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("config string field must be non-empty")
        return value.strip()

    @field_validator("train_value", "test_value")
    @classmethod
    def _validate_split_value(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("split values must be non-empty strings")
        cleaned = value.strip().lower()
        if cleaned not in {"train", "test"}:
            raise ValueError(
                "official split values must be one of {'train', 'test'} to "
                "match the official FI-2010 adapter convention",
            )
        return cleaned

    @field_validator("folds")
    @classmethod
    def _validate_folds(cls, value: Sequence[int]) -> tuple[int, ...]:
        cleaned: list[int] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError("folds must be integers")
            if item <= 0:
                raise ValueError("folds must be positive integers")
            cleaned.append(int(item))
        if not cleaned:
            raise ValueError("folds must contain at least one entry")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("folds must not contain duplicates")
        return tuple(cleaned)

    @field_validator("label_columns")
    @classmethod
    def _validate_label_columns(cls, value: Sequence[str]) -> tuple[str, ...]:
        cleaned: list[str] = []
        for column in value:
            if not isinstance(column, str) or not column.strip():
                raise ValueError("label_columns entries must be non-empty strings")
            cleaned.append(column.strip())
        if not cleaned:
            raise ValueError("label_columns must contain at least one entry")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("label_columns must not contain duplicates")
        allowed = {f"label_{h}" for h in OFFICIAL_FI2010_LABEL_HORIZONS}
        unexpected = [column for column in cleaned if column not in allowed]
        if unexpected:
            raise ValueError(
                "label_columns must only reference official FI-2010 label "
                f"columns {sorted(allowed)}; got {unexpected}",
            )
        return tuple(cleaned)

    @field_validator("target_horizon")
    @classmethod
    def _validate_target_horizon(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("target_horizon must be an integer")
        if value not in OFFICIAL_FI2010_LABEL_HORIZONS:
            raise ValueError(
                "target_horizon must be one of the official FI-2010 "
                f"horizons {list(OFFICIAL_FI2010_LABEL_HORIZONS)}; got {value}",
            )
        return int(value)

    @model_validator(mode="after")
    def _validate_target_horizon_matches_label_columns(self) -> MultifoldStudyConfig:
        if self.train_value.casefold() == self.test_value.casefold():
            raise ValueError("train_value and test_value must be distinct")
        expected_column = f"label_{self.target_horizon}"
        if expected_column not in self.label_columns:
            raise ValueError(
                f"label_columns must include {expected_column!r} so the "
                "target horizon is retained in the combined CSV",
            )
        return self


def load_multifold_config(path: str | Path) -> MultifoldStudyConfig:
    """Load and validate a multi-fold preparation YAML config."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"multi-fold config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"multi-fold config must be a YAML mapping: {config_path}")

    target = payload.get("target")
    if not isinstance(target, Mapping):
        raise ValueError("multi-fold config must contain a 'target' mapping")
    official_split = payload.get("official_split")
    if not isinstance(official_split, Mapping):
        raise ValueError("multi-fold config must contain an 'official_split' mapping")
    preparation = payload.get("preparation")
    if not isinstance(preparation, Mapping):
        raise ValueError(
            "multi-fold config must contain a 'preparation' mapping; this "
            "config is the executable preparation layer config",
        )

    label_columns = target.get("label_columns")
    if not isinstance(label_columns, Sequence) or isinstance(label_columns, (str, bytes)):
        raise ValueError("target.label_columns must be a sequence of column names")

    return MultifoldStudyConfig.model_validate(
        {
            "study_name": payload.get("study_name"),
            "dataset_name": payload.get("dataset_name"),
            "folds": tuple(payload.get("folds", ())),
            "label_columns": tuple(label_columns),
            "target_horizon": target.get("horizon"),
            "split_column": official_split.get("split_column"),
            "train_value": official_split.get("train_value"),
            "test_value": official_split.get("test_value"),
            "preparation": dict(preparation),
        }
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldFilePlan:
    """File plan for a single fold prior to conversion."""

    fold: int
    train_path: Path
    test_path: Path
    combined_output_path: Path
    train_present: bool
    test_present: bool

    @property
    def is_ready(self) -> bool:
        """Return ``True`` when both train and test source files exist."""
        return self.train_present and self.test_present


def _resolve_template(template: str, fold: int) -> str:
    return template.replace("{fold}", str(fold))


def _resolve_fold_paths(
    config: MultifoldStudyConfig,
    *,
    extracted_root: Path,
    processed_root: Path,
    fold: int,
) -> tuple[Path, Path, Path]:
    options = config.preparation
    override = options.fold_overrides.get(str(fold))
    train_relative = (
        override.train_path
        if override is not None and override.train_path is not None
        else _resolve_template(options.train_filename_template, fold)
    )
    test_relative = (
        override.test_path
        if override is not None and override.test_path is not None
        else _resolve_template(options.test_filename_template, fold)
    )
    combined_relative = _resolve_template(
        options.combined_csv_filename_template, fold
    )
    train_path = (extracted_root / train_relative).resolve(strict=False)
    test_path = (extracted_root / test_relative).resolve(strict=False)
    combined_path = (processed_root / combined_relative).resolve(strict=False)
    return train_path, test_path, combined_path


def _filter_folds(
    configured: Sequence[int],
    requested: Sequence[int] | None,
) -> tuple[int, ...]:
    if requested is None:
        return tuple(configured)
    configured_set = set(configured)
    cleaned: list[int] = []
    for item in requested:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError("requested folds must be integers")
        if item not in configured_set:
            raise ValueError(
                f"requested fold {item} is not configured; configured folds "
                f"are {sorted(configured_set)}",
            )
        if item not in cleaned:
            cleaned.append(int(item))
    if not cleaned:
        raise ValueError("requested fold list is empty after filtering")
    return tuple(cleaned)


def discover_fold_files(
    config: MultifoldStudyConfig,
    *,
    extracted_root: str | Path,
    processed_root: str | Path,
    folds: Sequence[int] | None = None,
) -> list[FoldFilePlan]:
    """Return a file plan for each requested (or all configured) folds."""
    extracted = Path(extracted_root)
    processed = Path(processed_root)
    selected = _filter_folds(config.folds, folds)
    plans: list[FoldFilePlan] = []
    for fold in selected:
        train_path, test_path, combined_path = _resolve_fold_paths(
            config,
            extracted_root=extracted,
            processed_root=processed,
            fold=fold,
        )
        plans.append(
            FoldFilePlan(
                fold=fold,
                train_path=train_path,
                test_path=test_path,
                combined_output_path=combined_path,
                train_present=train_path.is_file(),
                test_present=test_path.is_file(),
            )
        )
    return plans


def inspect_multifold_files(
    config: MultifoldStudyConfig,
    *,
    extracted_root: str | Path,
    processed_root: str | Path,
    folds: Sequence[int] | None = None,
) -> list[FoldFilePlan]:
    """Return a file plan without converting any data."""
    return discover_fold_files(
        config,
        extracted_root=extracted_root,
        processed_root=processed_root,
        folds=folds,
    )


# ---------------------------------------------------------------------------
# Per-fold manifest and summary models
# ---------------------------------------------------------------------------


class FoldSourceFileRecord(BaseModel):
    """Provenance record for one official FI-2010 source file."""

    model_config = _MODEL_CONFIG

    relative_path: str
    absolute_path: str
    byte_size: int
    sha256: str | None
    row_count: int
    column_count: int
    split_label: str

    @field_validator("relative_path", "absolute_path", "split_label")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("source-file string field must be non-empty")
        return value.strip()

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("sha256 must be non-empty when provided")
        return value.strip()

    @field_validator("byte_size", "row_count", "column_count")
    @classmethod
    def _validate_non_negative_int(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("source-file counts must be integers")
        if value < 0:
            raise ValueError("source-file counts must be non-negative")
        return value


class FoldPreparationManifest(BaseModel):
    """Manifest written for each prepared fold."""

    model_config = _MODEL_CONFIG

    fold: int
    study_name: str
    dataset_name: str
    target_horizon: int
    label_columns: list[str]
    split_column: str
    train_value: str
    test_value: str
    train_source: FoldSourceFileRecord
    test_source: FoldSourceFileRecord
    combined_csv_absolute_path: str
    combined_csv_byte_size: int
    combined_csv_sha256: str | None
    combined_row_count: int
    combined_column_count: int
    split_counts: dict[str, int]
    preparation_version: str
    created_at: datetime
    warnings: list[str] = Field(default_factory=list)

    @field_validator(
        "study_name",
        "dataset_name",
        "split_column",
        "train_value",
        "test_value",
        "combined_csv_absolute_path",
        "preparation_version",
    )
    @classmethod
    def _validate_required_string(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("manifest string field must be non-empty")
        return value.strip()

    @field_validator("fold", "target_horizon")
    @classmethod
    def _validate_positive_int(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("manifest integer field must be an int")
        if value <= 0:
            raise ValueError("manifest integer field must be positive")
        return int(value)

    @field_validator("combined_csv_byte_size", "combined_row_count", "combined_column_count")
    @classmethod
    def _validate_non_negative_int(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("manifest counts must be integers")
        if value < 0:
            raise ValueError("manifest counts must be non-negative")
        return value

    @field_validator("split_counts")
    @classmethod
    def _validate_split_counts(cls, value: dict[str, int]) -> dict[str, int]:
        cleaned: dict[str, int] = {}
        for key, count in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("split_counts keys must be non-empty strings")
            if isinstance(count, bool) or not isinstance(count, int):
                raise ValueError("split_counts values must be integers")
            if count < 0:
                raise ValueError("split_counts values must be non-negative")
            cleaned[key.strip()] = int(count)
        if not cleaned:
            raise ValueError("split_counts must contain at least one entry")
        return cleaned

    @field_validator("warnings")
    @classmethod
    def _validate_warnings(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for warning in value:
            if not isinstance(warning, str) or not warning.strip():
                raise ValueError("warnings must be non-empty strings")
            cleaned.append(warning.strip())
        return cleaned


class MultifoldPreparationSummary(BaseModel):
    """Top-level summary written under ``--out``."""

    model_config = _MODEL_CONFIG

    study_name: str
    dataset_name: str
    config_path: str
    config_sha256: str
    extracted_dataset_root: str
    processed_output_root: str
    output_dir: str
    folds_configured: list[int]
    folds_requested: list[int]
    folds_prepared: list[int]
    folds_skipped: list[int]
    fold_manifest_paths: dict[str, str]
    per_fold_row_counts: dict[str, int]
    per_fold_split_counts: dict[str, dict[str, int]]
    per_fold_source_hashes: dict[str, dict[str, str | None]]
    git_commit: str | None
    preparation_version: str
    created_at: datetime
    warnings: list[str] = Field(default_factory=list)

    @field_validator(
        "study_name",
        "dataset_name",
        "config_path",
        "config_sha256",
        "extracted_dataset_root",
        "processed_output_root",
        "output_dir",
        "preparation_version",
    )
    @classmethod
    def _validate_required_string(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("summary string field must be non-empty")
        return value.strip()

    @field_validator("git_commit")
    @classmethod
    def _validate_git_commit(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("git_commit must be non-empty when provided")
        return value.strip()

    @field_validator("warnings")
    @classmethod
    def _validate_warnings(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for warning in value:
            if not isinstance(warning, str) or not warning.strip():
                raise ValueError("warnings must be non-empty strings")
            cleaned.append(warning.strip())
        return cleaned


@dataclass(frozen=True)
class MultifoldPreparationResult:
    """Aggregate result returned by :func:`prepare_multifold`."""

    summary: MultifoldPreparationSummary
    fold_manifests: tuple[FoldPreparationManifest, ...]
    output_dir: Path
    written_files: tuple[Path, ...]


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


def _count_csv_rows_and_columns(path: Path) -> tuple[int, int, dict[str, int], int]:
    """Return (row_count, column_count, split_counts, header_column_count).

    The function does a single streaming pass over the file. ``row_count``
    excludes the header. ``split_counts`` is empty if no ``split`` column
    exists. ``header_column_count`` matches ``column_count``.
    """
    row_count = 0
    column_count = 0
    split_counts: dict[str, int] = {}
    split_index: int | None = None
    with path.open("r", encoding="utf-8") as handle:
        header_line = handle.readline()
        if not header_line:
            return 0, 0, {}, 0
        header = [token.strip() for token in header_line.rstrip("\n").split(",")]
        column_count = len(header)
        if "split" in header:
            split_index = header.index("split")
        for line in handle:
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            row_count += 1
            if split_index is None:
                continue
            tokens = stripped.split(",")
            if split_index >= len(tokens):
                continue
            value = tokens[split_index].strip()
            if not value:
                continue
            split_counts[value] = split_counts.get(value, 0) + 1
    return row_count, column_count, split_counts, column_count


def _resolve_existing_path(value: str | Path, *, label: str) -> Path:
    candidate = Path(value)
    if not candidate.exists():
        raise FileNotFoundError(f"{label} not found: {candidate}")
    return candidate


def _convert_pair_to_combined_csv(
    *,
    train_input: Path,
    test_input: Path,
    train_value: str,
    test_value: str,
    combined_output: Path,
    overwrite: bool,
) -> tuple[int, int, int]:
    """Convert one train/test pair into a single split-aware combined CSV.

    Returns ``(n_train_rows, n_test_rows, column_count)``.
    """
    combined_output.parent.mkdir(parents=True, exist_ok=True)
    if combined_output.exists() and not overwrite:
        raise FileExistsError(
            f"combined output already exists: {combined_output}; pass overwrite=True",
        )

    with tempfile.TemporaryDirectory(
        prefix="fi2010_multifold_", dir=combined_output.parent
    ) as tmp_root_name:
        tmp_root = Path(tmp_root_name)
        train_tmp = tmp_root / "train.csv"
        test_tmp = tmp_root / "test.csv"
        train_report = convert_official_fi2010_to_csv(
            train_input,
            train_tmp,
            split_label=train_value,
            overwrite=False,
        )
        test_report = convert_official_fi2010_to_csv(
            test_input,
            test_tmp,
            split_label=test_value,
            overwrite=False,
        )
        with combined_output.open("w", encoding="utf-8", newline="") as out_handle:
            with train_tmp.open("r", encoding="utf-8") as train_handle:
                header = train_handle.readline()
                if not header:
                    raise ValueError(
                        f"converted train CSV is empty: {train_tmp}",
                    )
                out_handle.write(header)
                shutil.copyfileobj(train_handle, out_handle)
            with test_tmp.open("r", encoding="utf-8") as test_handle:
                test_header = test_handle.readline()
                if test_header != header:
                    raise ValueError(
                        "train and test CSV headers disagree after conversion; "
                        "refusing to write a malformed combined file",
                    )
                shutil.copyfileobj(test_handle, out_handle)
        header_columns = [token.strip() for token in header.rstrip("\n").split(",")]
        column_count = len(header_columns)
        return train_report.n_samples, test_report.n_samples, column_count


def _source_record(
    *,
    extracted_root: Path,
    source_path: Path,
    split_label: str,
    compute_hash: bool,
    matrix_row_count: int,
    matrix_column_count: int,
) -> FoldSourceFileRecord:
    try:
        relative = source_path.resolve().relative_to(extracted_root.resolve())
        relative_text = str(relative)
    except ValueError:
        relative_text = str(source_path)
    sha256 = sha256_file(source_path) if compute_hash else None
    return FoldSourceFileRecord(
        relative_path=relative_text,
        absolute_path=str(source_path),
        byte_size=source_path.stat().st_size,
        sha256=sha256,
        row_count=matrix_row_count,
        column_count=matrix_column_count,
        split_label=split_label,
    )


def _write_json(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json_dumps(model), encoding="utf-8")


def _ensure_directory_writable(
    path: Path, *, overwrite: bool, label: str
) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{label} already exists and is non-empty: {path}; pass overwrite=True",
        )


def _validate_extracted_root(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"extracted FI-2010 dataset root not found: {path}",
        )
    if not path.is_dir():
        raise NotADirectoryError(
            f"extracted FI-2010 dataset root is not a directory: {path}",
        )
    return path


def _prepare_single_fold(
    *,
    config: MultifoldStudyConfig,
    plan: FoldFilePlan,
    extracted_root: Path,
    overwrite: bool,
    created_at: datetime,
) -> FoldPreparationManifest:
    if not plan.train_present:
        raise FileNotFoundError(
            f"fold {plan.fold} train file is missing: {plan.train_path}",
        )
    if not plan.test_present:
        raise FileNotFoundError(
            f"fold {plan.fold} test file is missing: {plan.test_path}",
        )

    train_input = _resolve_existing_path(plan.train_path, label="train file")
    test_input = _resolve_existing_path(plan.test_path, label="test file")

    n_train_rows, n_test_rows, column_count = _convert_pair_to_combined_csv(
        train_input=train_input,
        test_input=test_input,
        train_value=config.train_value,
        test_value=config.test_value,
        combined_output=plan.combined_output_path,
        overwrite=overwrite,
    )

    observed_rows, observed_columns, split_counts, _ = (
        _count_csv_rows_and_columns(plan.combined_output_path)
    )
    if observed_rows != n_train_rows + n_test_rows:
        raise ValueError(
            f"fold {plan.fold} combined CSV row count {observed_rows} does not "
            f"equal train+test rows {n_train_rows + n_test_rows}",
        )
    if observed_columns != column_count:
        raise ValueError(
            f"fold {plan.fold} combined CSV column count {observed_columns} "
            f"does not match converter column count {column_count}",
        )
    expected_split_counts = {
        config.train_value: n_train_rows,
        config.test_value: n_test_rows,
    }
    if split_counts != expected_split_counts:
        raise ValueError(
            f"fold {plan.fold} combined CSV split counts {split_counts} do not "
            f"match expected {expected_split_counts}",
        )

    compute_hash = config.preparation.compute_source_hashes
    train_record = _source_record(
        extracted_root=extracted_root,
        source_path=train_input,
        split_label=config.train_value,
        compute_hash=compute_hash,
        matrix_row_count=149,
        matrix_column_count=n_train_rows,
    )
    test_record = _source_record(
        extracted_root=extracted_root,
        source_path=test_input,
        split_label=config.test_value,
        compute_hash=compute_hash,
        matrix_row_count=149,
        matrix_column_count=n_test_rows,
    )

    combined_sha256 = (
        sha256_file(plan.combined_output_path) if compute_hash else None
    )
    return FoldPreparationManifest(
        fold=plan.fold,
        study_name=config.study_name,
        dataset_name=config.dataset_name,
        target_horizon=config.target_horizon,
        label_columns=list(config.label_columns),
        split_column=config.split_column,
        train_value=config.train_value,
        test_value=config.test_value,
        train_source=train_record,
        test_source=test_record,
        combined_csv_absolute_path=str(plan.combined_output_path),
        combined_csv_byte_size=plan.combined_output_path.stat().st_size,
        combined_csv_sha256=combined_sha256,
        combined_row_count=observed_rows,
        combined_column_count=observed_columns,
        split_counts=split_counts,
        preparation_version=_PREPARATION_VERSION,
        created_at=created_at,
        warnings=[],
    )


def prepare_multifold(
    *,
    config: MultifoldStudyConfig,
    config_source_path: str | Path,
    extracted_root: str | Path,
    processed_root: str | Path,
    output_dir: str | Path,
    folds: Sequence[int] | None = None,
    overwrite: bool = False,
    created_at: datetime | None = None,
) -> MultifoldPreparationResult:
    """Prepare all requested folds and write per-fold manifests and a summary.

    No model is trained and no prediction is produced. The combined CSV
    files are written under ``processed_root`` and the manifests are
    written under ``output_dir``. Both directories are expected to be
    ignored by git; the configuration's ``local_only_note`` is preserved
    in the per-fold manifests for traceability.
    """
    extracted = _validate_extracted_root(Path(extracted_root))
    processed = Path(processed_root)
    output_path = Path(output_dir)
    config_path = Path(config_source_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config source path is missing: {config_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    folds_dir = output_path / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "summary.json"
    if summary_path.exists() and not overwrite:
        raise FileExistsError(
            f"summary.json already exists: {summary_path}; pass overwrite=True",
        )

    plans = discover_fold_files(
        config,
        extracted_root=extracted,
        processed_root=processed,
        folds=folds,
    )

    timestamp = created_at or datetime.now(UTC)
    config_hash = sha256_file(config_path)
    warnings: list[str] = []
    written_files: list[Path] = []
    manifests: list[FoldPreparationManifest] = []
    folds_prepared: list[int] = []
    folds_skipped: list[int] = []

    for plan in plans:
        if not plan.is_ready:
            missing = []
            if not plan.train_present:
                missing.append(f"train={plan.train_path}")
            if not plan.test_present:
                missing.append(f"test={plan.test_path}")
            raise FileNotFoundError(
                f"fold {plan.fold} source file(s) missing: "
                + ", ".join(missing),
            )
        manifest = _prepare_single_fold(
            config=config,
            plan=plan,
            extracted_root=extracted,
            overwrite=overwrite,
            created_at=timestamp,
        )
        manifests.append(manifest)
        folds_prepared.append(plan.fold)
        manifest_path = folds_dir / f"fold_{plan.fold}_manifest.json"
        if manifest_path.exists() and not overwrite:
            raise FileExistsError(
                f"fold manifest already exists: {manifest_path}; pass overwrite=True",
            )
        _write_json(manifest_path, manifest)
        written_files.append(manifest_path)
        written_files.append(plan.combined_output_path)

    requested_list = [plan.fold for plan in plans]
    configured_list = list(config.folds)
    folds_skipped = [
        fold for fold in configured_list if fold not in folds_prepared
    ]

    fold_manifest_paths = {
        str(manifest.fold): str(folds_dir / f"fold_{manifest.fold}_manifest.json")
        for manifest in manifests
    }
    per_fold_row_counts = {
        str(manifest.fold): manifest.combined_row_count for manifest in manifests
    }
    per_fold_split_counts = {
        str(manifest.fold): dict(manifest.split_counts) for manifest in manifests
    }
    per_fold_source_hashes = {
        str(manifest.fold): {
            "train": manifest.train_source.sha256,
            "test": manifest.test_source.sha256,
            "combined": manifest.combined_csv_sha256,
        }
        for manifest in manifests
    }

    if folds_skipped:
        warnings.append(
            "configured folds not requested by this run: "
            + ", ".join(str(fold) for fold in folds_skipped),
        )

    summary = MultifoldPreparationSummary(
        study_name=config.study_name,
        dataset_name=config.dataset_name,
        config_path=str(config_path),
        config_sha256=config_hash,
        extracted_dataset_root=str(extracted),
        processed_output_root=str(processed),
        output_dir=str(output_path),
        folds_configured=configured_list,
        folds_requested=requested_list,
        folds_prepared=folds_prepared,
        folds_skipped=folds_skipped,
        fold_manifest_paths=fold_manifest_paths,
        per_fold_row_counts=per_fold_row_counts,
        per_fold_split_counts=per_fold_split_counts,
        per_fold_source_hashes=per_fold_source_hashes,
        git_commit=get_git_commit(),
        preparation_version=_PREPARATION_VERSION,
        created_at=timestamp,
        warnings=warnings,
    )
    _write_json(summary_path, summary)
    written_files.append(summary_path)

    config_snapshot_path = output_path / "config.yaml"
    if config_snapshot_path.exists() and not overwrite:
        raise FileExistsError(
            f"config snapshot already exists: {config_snapshot_path}; pass overwrite=True",
        )
    shutil.copyfile(config_path, config_snapshot_path)
    written_files.append(config_snapshot_path)

    return MultifoldPreparationResult(
        summary=summary,
        fold_manifests=tuple(manifests),
        output_dir=output_path,
        written_files=tuple(written_files),
    )

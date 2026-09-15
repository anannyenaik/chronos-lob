"""Planning helpers for serious FI-2010 neural benchmark runs.

This module is deliberately orchestration-only. It validates the neural
benchmark configuration, expands deterministic run plans and defines the
lightweight metadata contract for future runs. It does not train models and
does not write predictions or checkpoints by default.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

__all__ = [
    "NEURAL_BENCHMARK_ARTEFACTS",
    "SUPPORTED_NEURAL_BENCHMARK_MODELS",
    "CheckpointPolicy",
    "DeterministicSeedHandling",
    "DeviceResolution",
    "NeuralArtefactConfig",
    "NeuralBenchmarkConfig",
    "NeuralBenchmarkRunPlan",
    "NeuralDatasetAssumptions",
    "NeuralModelSpec",
    "NeuralOfficialSplitSemantics",
    "NeuralTargetConfig",
    "NeuralTrainingConfig",
    "TrainingRunMetadata",
    "build_training_metadata",
    "count_parameters",
    "expected_lightweight_artefacts",
    "generate_neural_run_plan",
    "load_neural_benchmark_config",
    "normalise_neural_fold_ids",
    "normalise_neural_model_names",
    "resolve_neural_device",
    "training_metadata_schema_fields",
    "validate_supported_neural_models",
]

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)

SUPPORTED_NEURAL_BENCHMARK_MODELS: tuple[str, ...] = (
    "deeplob_style",
    "matrix_transformer",
)

NEURAL_BENCHMARK_ARTEFACTS: tuple[str, ...] = (
    "summary.json",
    "run_plan.csv",
    "results_by_fold_seed.csv",
    "results_summary.csv",
    "training_summary.csv",
    "model_capacity_summary.csv",
    "model_failures.json",
)


def _validate_non_empty_text(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _validate_positive_int(value: int, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return int(value)


def _validate_non_negative_int(value: int, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return int(value)


def _validate_fraction(value: float, *, field_name: str, upper_open: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    numeric = float(value)
    upper_ok = numeric < 1.0 if upper_open else numeric <= 1.0
    if numeric < 0.0 or not upper_ok:
        relation = "< 1" if upper_open else "<= 1"
        raise ValueError(f"{field_name} must satisfy 0 <= value {relation}")
    return numeric


def _normalise_fold_id(value: str | int) -> str:
    if isinstance(value, bool):
        raise ValueError("fold identifiers must be positive integers or fold_N strings")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("fold identifiers must be positive")
        return f"fold_{value}"
    if not isinstance(value, str) or not value.strip():
        raise ValueError("fold identifiers must be non-empty")
    cleaned = value.strip().lower()
    if cleaned.isdigit():
        numeric = int(cleaned)
        if numeric <= 0:
            raise ValueError("fold identifiers must be positive")
        return f"fold_{numeric}"
    if cleaned.startswith("fold_") and cleaned.removeprefix("fold_").isdigit():
        numeric = int(cleaned.removeprefix("fold_"))
        if numeric <= 0:
            raise ValueError("fold identifiers must be positive")
        return f"fold_{numeric}"
    raise ValueError(
        f"fold identifier {value!r} is invalid; expected a positive integer "
        "or fold_N",
    )


def _fold_number(fold_id: str) -> int:
    return int(fold_id.removeprefix("fold_"))


class NeuralDatasetAssumptions(BaseModel):
    """Dataset assumptions recorded by the neural benchmark config."""

    model_config = _MODEL_CONFIG

    name: str
    source: str
    assumptions: tuple[str, ...]
    local_data_only: bool = True
    raw_data_committed: bool = False
    processed_csv_committed: bool = False

    @field_validator("name", "source")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        return _validate_non_empty_text(value, field_name="dataset field")

    @field_validator("assumptions")
    @classmethod
    def _validate_assumptions(cls, value: Sequence[str]) -> tuple[str, ...]:
        cleaned = tuple(
            _validate_non_empty_text(item, field_name="dataset assumption")
            for item in value
        )
        if not cleaned:
            raise ValueError("dataset assumptions must not be empty")
        return cleaned


class NeuralOfficialSplitSemantics(BaseModel):
    """Official train/test split handling for prepared FI-2010 folds."""

    model_config = _MODEL_CONFIG

    split_column: str = "split"
    train_value: str = "train"
    test_value: str = "test"
    validation_fraction_within_train: float = 0.15
    validation_source: str = "tail_of_official_train"
    test_usage: str = "final_evaluation_only"

    @field_validator("split_column", "train_value", "test_value", "validation_source", "test_usage")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        return _validate_non_empty_text(value, field_name="official split field")

    @field_validator("validation_fraction_within_train")
    @classmethod
    def _validate_validation_fraction(cls, value: float) -> float:
        return _validate_fraction(
            value,
            field_name="validation_fraction_within_train",
            upper_open=True,
        )

    @model_validator(mode="after")
    def _validate_distinct_split_values(self) -> NeuralOfficialSplitSemantics:
        if self.train_value.casefold() == self.test_value.casefold():
            raise ValueError("train_value and test_value must be distinct")
        return self


class NeuralTargetConfig(BaseModel):
    """Target label settings for FI-2010 neural benchmarks."""

    model_config = _MODEL_CONFIG

    horizon: int
    label_column: str

    @field_validator("horizon")
    @classmethod
    def _validate_horizon(cls, value: int) -> int:
        return _validate_positive_int(value, field_name="target horizon")

    @field_validator("label_column")
    @classmethod
    def _validate_label_column(cls, value: str) -> str:
        return _validate_non_empty_text(value, field_name="label_column")


class NeuralTrainingConfig(BaseModel):
    """Shared neural optimisation settings."""

    model_config = _MODEL_CONFIG

    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    early_stopping_metric: str = "validation_macro_f1"
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float | None = 1.0
    dropout: float

    @field_validator("batch_size", "max_epochs")
    @classmethod
    def _validate_positive_training_ints(cls, value: int) -> int:
        return _validate_positive_int(value, field_name="training integer")

    @field_validator("early_stopping_patience")
    @classmethod
    def _validate_patience(cls, value: int) -> int:
        return _validate_non_negative_int(value, field_name="early_stopping_patience")

    @field_validator("early_stopping_metric")
    @classmethod
    def _validate_early_stopping_metric(cls, value: str) -> str:
        metric = _validate_non_empty_text(value, field_name="early_stopping_metric")
        allowed = {"validation_macro_f1", "validation_loss"}
        if metric not in allowed:
            raise ValueError(
                f"early_stopping_metric must be one of {sorted(allowed)}; got {metric!r}",
            )
        return metric

    @field_validator("learning_rate")
    @classmethod
    def _validate_learning_rate(cls, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("learning_rate must be a finite number")
        numeric = float(value)
        if numeric <= 0.0:
            raise ValueError("learning_rate must be positive")
        return numeric

    @field_validator("weight_decay")
    @classmethod
    def _validate_weight_decay(cls, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("weight_decay must be a finite number")
        numeric = float(value)
        if numeric < 0.0:
            raise ValueError("weight_decay must be non-negative")
        return numeric

    @field_validator("gradient_clip_norm")
    @classmethod
    def _validate_gradient_clip_norm(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("gradient_clip_norm must be a finite number or null")
        numeric = float(value)
        if numeric <= 0.0:
            raise ValueError("gradient_clip_norm must be positive when provided")
        return numeric

    @field_validator("dropout")
    @classmethod
    def _validate_dropout(cls, value: float) -> float:
        return _validate_fraction(value, field_name="dropout", upper_open=True)


class NeuralModelSpec(BaseModel):
    """Model-specific architecture settings for the benchmark plan."""

    model_config = _MODEL_CONFIG

    enabled: bool = True
    architecture: str
    hidden_sizes: tuple[int, ...] = ()
    conv_channels: int | None = None
    lstm_hidden_size: int | None = None
    use_batch_norm: bool | None = None
    model_dim: int | None = None
    num_heads: int | None = None
    num_layers: int | None = None
    feedforward_dim: int | None = None
    dropout: float | None = None

    @field_validator("architecture")
    @classmethod
    def _validate_architecture(cls, value: str) -> str:
        return _validate_non_empty_text(value, field_name="architecture")

    @field_validator("hidden_sizes")
    @classmethod
    def _validate_hidden_sizes(cls, value: Sequence[int]) -> tuple[int, ...]:
        return tuple(
            _validate_positive_int(item, field_name="hidden_sizes entry")
            for item in value
        )

    @field_validator(
        "conv_channels",
        "lstm_hidden_size",
        "model_dim",
        "num_heads",
        "num_layers",
        "feedforward_dim",
    )
    @classmethod
    def _validate_optional_positive_int(cls, value: int | None) -> int | None:
        if value is None:
            return None
        return _validate_positive_int(value, field_name="model dimension")

    @field_validator("dropout")
    @classmethod
    def _validate_optional_dropout(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return _validate_fraction(value, field_name="model dropout", upper_open=True)

    @model_validator(mode="after")
    def _validate_transformer_heads(self) -> NeuralModelSpec:
        if (
            self.model_dim is not None
            and self.num_heads is not None
            and self.model_dim % self.num_heads != 0
        ):
            raise ValueError("model_dim must be divisible by num_heads")
        return self


class DeterministicSeedHandling(BaseModel):
    """Deterministic seed policy for neural benchmark runs."""

    model_config = _MODEL_CONFIG

    enabled: bool = True
    seed_source: str = "configured_seed_list"
    set_python_numpy_torch: bool = True
    deterministic_torch_algorithms: bool = True

    @field_validator("seed_source")
    @classmethod
    def _validate_seed_source(cls, value: str) -> str:
        return _validate_non_empty_text(value, field_name="seed_source")


class CheckpointPolicy(BaseModel):
    """Checkpoint writing policy for future neural runs."""

    model_config = _MODEL_CONFIG

    enabled: bool = False
    save_best_only: bool = True
    filename: str = "best_model.pt"
    write_by_default: bool = False

    @field_validator("filename")
    @classmethod
    def _validate_filename(cls, value: str) -> str:
        return _validate_non_empty_text(value, field_name="checkpoint filename")

    @model_validator(mode="after")
    def _validate_default_write_policy(self) -> CheckpointPolicy:
        if self.write_by_default and not self.enabled:
            raise ValueError("write_by_default requires checkpoint policy enabled")
        return self


class NeuralArtefactConfig(BaseModel):
    """Top-level lightweight output locations for neural benchmarks."""

    model_config = _MODEL_CONFIG

    output_root: str
    checkpoint_root: str
    summary: str = "summary.json"
    run_plan: str = "run_plan.csv"
    results_by_fold_seed: str = "results_by_fold_seed.csv"
    results_summary: str = "results_summary.csv"
    training_summary: str = "training_summary.csv"
    model_capacity_summary: str = "model_capacity_summary.csv"
    model_failures: str = "model_failures.json"
    write_full_predictions_by_default: bool = False
    write_checkpoints_by_default: bool = False

    @field_validator(
        "output_root",
        "checkpoint_root",
        "summary",
        "run_plan",
        "results_by_fold_seed",
        "results_summary",
        "training_summary",
        "model_capacity_summary",
        "model_failures",
    )
    @classmethod
    def _validate_text(cls, value: str) -> str:
        return _validate_non_empty_text(value, field_name="artefact field")


class NeuralBenchmarkConfig(BaseModel):
    """Validated FI-2010 neural benchmark planning config."""

    model_config = _MODEL_CONFIG

    study_name: str
    dataset: NeuralDatasetAssumptions
    official_split: NeuralOfficialSplitSemantics
    folds: tuple[str, ...]
    seeds: tuple[int, ...]
    target: NeuralTargetConfig
    neural_models: dict[str, NeuralModelSpec]
    lookbacks: tuple[int, ...]
    training: NeuralTrainingConfig
    device_selection: str
    deterministic_seed_handling: DeterministicSeedHandling
    validation_metric: str
    checkpoint_policy: CheckpointPolicy
    artefacts: NeuralArtefactConfig
    mode: str = "benchmark"
    benchmark_note: str

    @field_validator("study_name", "validation_metric", "mode", "benchmark_note")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        return _validate_non_empty_text(value, field_name="benchmark config field")

    @field_validator("folds", mode="before")
    @classmethod
    def _normalise_folds_before(cls, value: object) -> tuple[str, ...]:
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise ValueError("folds must be a sequence")
        return tuple(_normalise_fold_id(item) for item in value)

    @field_validator("folds")
    @classmethod
    def _validate_folds(cls, value: Sequence[str]) -> tuple[str, ...]:
        cleaned = tuple(value)
        if not cleaned:
            raise ValueError("folds must not be empty")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("folds must not contain duplicates")
        return cleaned

    @field_validator("seeds")
    @classmethod
    def _validate_seeds(cls, value: Sequence[int]) -> tuple[int, ...]:
        cleaned = tuple(
            _validate_non_negative_int(seed, field_name="seed") for seed in value
        )
        if not cleaned:
            raise ValueError("seeds must not be empty")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("seeds must not contain duplicates")
        return cleaned

    @field_validator("lookbacks")
    @classmethod
    def _validate_lookbacks(cls, value: Sequence[int]) -> tuple[int, ...]:
        cleaned = tuple(
            _validate_positive_int(lookback, field_name="lookback")
            for lookback in value
        )
        if not cleaned:
            raise ValueError("lookbacks must not be empty")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("lookbacks must not contain duplicates")
        return cleaned

    @field_validator("device_selection")
    @classmethod
    def _validate_device_selection(cls, value: str) -> str:
        cleaned = _validate_non_empty_text(value, field_name="device_selection").lower()
        if cleaned not in {"auto", "cpu", "cuda"} and not cleaned.startswith("cuda:"):
            raise ValueError("device_selection must be auto, cpu, cuda or cuda:<index>")
        return cleaned

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, value: str) -> str:
        mode = _validate_non_empty_text(value, field_name="mode").lower()
        if mode not in {"benchmark", "smoke"}:
            raise ValueError("mode must be either 'benchmark' or 'smoke'")
        return mode

    @field_validator("neural_models")
    @classmethod
    def _validate_neural_models(
        cls,
        value: Mapping[str, NeuralModelSpec],
    ) -> dict[str, NeuralModelSpec]:
        if not value:
            raise ValueError("neural_models must not be empty")
        cleaned: dict[str, NeuralModelSpec] = {}
        for name, spec in value.items():
            model_name = _validate_non_empty_text(
                name,
                field_name="neural model name",
            ).lower()
            if model_name in cleaned:
                raise ValueError("neural_models must not contain duplicates")
            cleaned[model_name] = spec
        validate_supported_neural_models(tuple(cleaned))
        return cleaned

    @model_validator(mode="after")
    def _validate_mode_and_models(self) -> NeuralBenchmarkConfig:
        if self.is_benchmark_mode and "not smoke" not in self.benchmark_note.lower():
            raise ValueError("benchmark_note must explicitly say this is not smoke")
        enabled = self.enabled_model_names
        if not enabled:
            raise ValueError("at least one neural model must be enabled")
        return self

    @property
    def enabled_model_names(self) -> tuple[str, ...]:
        """Return enabled model names in config order."""
        return tuple(
            name for name, spec in self.neural_models.items() if bool(spec.enabled)
        )

    @property
    def is_smoke_mode(self) -> bool:
        """Return true when the config is explicitly a smoke config."""
        return self.mode == "smoke"

    @property
    def is_benchmark_mode(self) -> bool:
        """Return true when the config is explicitly a benchmark config."""
        return self.mode == "benchmark"


class NeuralBenchmarkRunPlan(BaseModel):
    """One deterministic neural benchmark run specification."""

    model_config = _MODEL_CONFIG

    run_id: str
    fold_id: str
    seed: int
    model_name: str
    lookback: int
    target_horizon: int
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    early_stopping_metric: str
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float | None
    dropout: float
    validation_metric: str
    device_policy: str
    output_dir: str
    checkpoint_path: str | None
    mode: str

    def to_csv_row(self) -> dict[str, str | int | float | None]:
        """Return a stable row for future ``run_plan.csv`` writing."""
        return {
            "run_id": self.run_id,
            "fold_id": self.fold_id,
            "seed": self.seed,
            "model_name": self.model_name,
            "lookback": self.lookback,
            "target_horizon": self.target_horizon,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_metric": self.early_stopping_metric,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clip_norm": self.gradient_clip_norm,
            "dropout": self.dropout,
            "validation_metric": self.validation_metric,
            "device_policy": self.device_policy,
            "output_dir": self.output_dir,
            "checkpoint_path": self.checkpoint_path,
            "mode": self.mode,
        }


class TrainingRunMetadata(BaseModel):
    """Lightweight per-run metadata expected from future neural runs."""

    model_config = _MODEL_CONFIG

    fold_id: str
    seed: int
    model_name: str
    lookback: int
    device: str
    parameter_count: int
    max_epochs: int
    best_epoch: int | None
    early_stopped: bool
    training_seconds: float | None
    validation_metric: str
    validation_metric_value: float | None
    test_metrics: dict[str, float]
    status: str

    @field_validator("fold_id", "model_name", "device", "validation_metric", "status")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        return _validate_non_empty_text(value, field_name="metadata field")

    @field_validator("seed")
    @classmethod
    def _validate_seed(cls, value: int) -> int:
        return _validate_non_negative_int(value, field_name="seed")

    @field_validator("lookback", "max_epochs")
    @classmethod
    def _validate_positive_ints(cls, value: int) -> int:
        return _validate_positive_int(value, field_name="metadata positive integer")

    @field_validator("parameter_count")
    @classmethod
    def _validate_parameter_count(cls, value: int) -> int:
        return _validate_non_negative_int(value, field_name="parameter_count")

    @field_validator("best_epoch")
    @classmethod
    def _validate_best_epoch(cls, value: int | None) -> int | None:
        if value is None:
            return None
        return _validate_positive_int(value, field_name="best_epoch")

    @field_validator("training_seconds")
    @classmethod
    def _validate_training_seconds(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("training_seconds must be numeric")
        numeric = float(value)
        if numeric < 0.0:
            raise ValueError("training_seconds must be non-negative")
        return numeric


@dataclass(frozen=True)
class DeviceResolution:
    """Resolved device metadata for a requested device policy."""

    requested: str
    resolved: str
    torch_available: bool
    cuda_available: bool


def validate_supported_neural_models(models: Sequence[str]) -> tuple[str, ...]:
    """Validate that all requested neural models are supported."""
    cleaned: list[str] = []
    for model in models:
        model_name = _validate_non_empty_text(model, field_name="model name").lower()
        if model_name not in SUPPORTED_NEURAL_BENCHMARK_MODELS:
            raise ValueError(
                f"unsupported neural benchmark model {model!r}; supported: "
                f"{list(SUPPORTED_NEURAL_BENCHMARK_MODELS)}",
            )
        if model_name not in cleaned:
            cleaned.append(model_name)
    if not cleaned:
        raise ValueError("at least one neural benchmark model is required")
    return tuple(cleaned)


def normalise_neural_model_names(
    models: Sequence[str] | str | None,
    *,
    config: NeuralBenchmarkConfig,
) -> tuple[str, ...]:
    """Normalise optional user model selection against the config."""
    if models is None:
        return config.enabled_model_names
    if isinstance(models, str):
        tokens: Sequence[str] = [token.strip() for token in models.split(",")]
    else:
        tokens = models
    requested = validate_supported_neural_models(
        tuple(token for token in tokens if token),
    )
    enabled = set(config.enabled_model_names)
    disabled = [model for model in requested if model not in enabled]
    if disabled:
        raise ValueError(
            f"requested neural models are not enabled in the config: {disabled}",
        )
    return requested


def normalise_neural_fold_ids(
    folds: Sequence[str | int] | str | None,
    *,
    config: NeuralBenchmarkConfig,
) -> tuple[str, ...]:
    """Normalise optional fold selection against the config."""
    if folds is None:
        return config.folds
    if isinstance(folds, str):
        text = folds.strip()
        if not text or text.lower() == "all":
            return config.folds
        tokens: Sequence[str | int] = [token.strip() for token in text.split(",")]
    else:
        tokens = folds
    requested = tuple(_normalise_fold_id(item) for item in tokens if str(item).strip())
    if not requested:
        raise ValueError("fold selection must not be empty")
    configured = set(config.folds)
    unknown = [fold for fold in requested if fold not in configured]
    if unknown:
        raise ValueError(f"requested folds are not configured: {unknown}")
    return requested


def load_neural_benchmark_config(path: str | Path) -> NeuralBenchmarkConfig:
    """Load and validate an FI-2010 neural benchmark YAML config."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"neural benchmark config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"neural benchmark config must be a YAML mapping: {config_path}")
    return NeuralBenchmarkConfig.model_validate(dict(payload))


def _planned_output_dir(
    config: NeuralBenchmarkConfig,
    *,
    fold_id: str,
    seed: int,
    model_name: str,
    lookback: int,
) -> Path:
    return (
        Path(config.artefacts.output_root)
        / fold_id
        / f"seed_{seed}"
        / model_name
        / f"lookback_{lookback}"
    )


def _planned_checkpoint_path(
    config: NeuralBenchmarkConfig,
    *,
    fold_id: str,
    seed: int,
    model_name: str,
    lookback: int,
) -> Path | None:
    if (
        not config.checkpoint_policy.enabled
        or not config.checkpoint_policy.write_by_default
        or not config.artefacts.write_checkpoints_by_default
    ):
        return None
    return (
        Path(config.artefacts.checkpoint_root)
        / fold_id
        / f"seed_{seed}"
        / model_name
        / f"lookback_{lookback}"
        / config.checkpoint_policy.filename
    )


def generate_neural_run_plan(
    config: NeuralBenchmarkConfig,
    *,
    folds: Sequence[str | int] | str | None = None,
    models: Sequence[str] | str | None = None,
    lookbacks: Sequence[int] | None = None,
) -> tuple[NeuralBenchmarkRunPlan, ...]:
    """Expand ``folds x seeds x models x lookbacks`` into a deterministic plan."""
    selected_folds = normalise_neural_fold_ids(folds, config=config)
    selected_models = normalise_neural_model_names(models, config=config)
    selected_lookbacks = (
        config.lookbacks
        if lookbacks is None
        else tuple(
            _validate_positive_int(lookback, field_name="lookback")
            for lookback in lookbacks
        )
    )
    if not selected_lookbacks:
        raise ValueError("lookbacks must not be empty")

    plans: list[NeuralBenchmarkRunPlan] = []
    for fold_id in selected_folds:
        for seed in config.seeds:
            for model_name in selected_models:
                model_spec = config.neural_models[model_name]
                model_dropout = (
                    config.training.dropout
                    if model_spec.dropout is None
                    else model_spec.dropout
                )
                for lookback in selected_lookbacks:
                    run_id = (
                        f"{fold_id}__seed_{seed}__{model_name}"
                        f"__lookback_{lookback}"
                    )
                    output_dir = _planned_output_dir(
                        config,
                        fold_id=fold_id,
                        seed=seed,
                        model_name=model_name,
                        lookback=lookback,
                    )
                    checkpoint_path = _planned_checkpoint_path(
                        config,
                        fold_id=fold_id,
                        seed=seed,
                        model_name=model_name,
                        lookback=lookback,
                    )
                    plans.append(
                        NeuralBenchmarkRunPlan(
                            run_id=run_id,
                            fold_id=fold_id,
                            seed=seed,
                            model_name=model_name,
                            lookback=lookback,
                            target_horizon=config.target.horizon,
                            batch_size=config.training.batch_size,
                            max_epochs=config.training.max_epochs,
                            early_stopping_patience=(
                                config.training.early_stopping_patience
                            ),
                            early_stopping_metric=config.training.early_stopping_metric,
                            learning_rate=config.training.learning_rate,
                            weight_decay=config.training.weight_decay,
                            gradient_clip_norm=config.training.gradient_clip_norm,
                            dropout=model_dropout,
                            validation_metric=config.validation_metric,
                            device_policy=config.device_selection,
                            output_dir=output_dir.as_posix(),
                            checkpoint_path=(
                                None
                                if checkpoint_path is None
                                else checkpoint_path.as_posix()
                            ),
                            mode=config.mode,
                        )
                    )
    return tuple(plans)


def resolve_neural_device(device_policy: str) -> DeviceResolution:
    """Resolve ``auto``, ``cpu`` or ``cuda`` without training a model."""
    requested = _validate_non_empty_text(device_policy, field_name="device_policy").lower()
    try:
        import torch

        torch_available = True
        cuda_available = bool(torch.cuda.is_available())
    except ImportError:
        torch_available = False
        cuda_available = False

    if requested == "auto":
        resolved = "cuda" if cuda_available else "cpu"
        return DeviceResolution(
            requested=requested,
            resolved=resolved,
            torch_available=torch_available,
            cuda_available=cuda_available,
        )
    if requested == "cpu":
        return DeviceResolution(
            requested=requested,
            resolved="cpu",
            torch_available=torch_available,
            cuda_available=cuda_available,
        )
    if requested == "cuda" or requested.startswith("cuda:"):
        if not torch_available:
            raise RuntimeError("requested CUDA device but PyTorch is unavailable")
        if not cuda_available:
            raise RuntimeError("requested CUDA device but CUDA is not available")
        return DeviceResolution(
            requested=requested,
            resolved=requested,
            torch_available=True,
            cuda_available=True,
        )
    raise ValueError("device_policy must be auto, cpu, cuda or cuda:<index>")


def count_parameters(model: Any, *, trainable_only: bool = True) -> int:
    """Count model parameters for any object exposing ``parameters()``."""
    parameters = getattr(model, "parameters", None)
    if parameters is None or not callable(parameters):
        raise TypeError("model must expose a callable parameters() method")
    total = 0
    for parameter in parameters():
        requires_grad = bool(getattr(parameter, "requires_grad", True))
        if trainable_only and not requires_grad:
            continue
        numel = getattr(parameter, "numel", None)
        if numel is None or not callable(numel):
            raise TypeError("model parameters must expose numel()")
        total += int(numel())
    return total


def expected_lightweight_artefacts(config: NeuralBenchmarkConfig) -> dict[str, str]:
    """Return the lightweight top-level artefacts expected from future runs."""
    root = Path(config.artefacts.output_root)
    return {
        "summary": (root / config.artefacts.summary).as_posix(),
        "run_plan": (root / config.artefacts.run_plan).as_posix(),
        "results_by_fold_seed": (
            root / config.artefacts.results_by_fold_seed
        ).as_posix(),
        "results_summary": (root / config.artefacts.results_summary).as_posix(),
        "training_summary": (root / config.artefacts.training_summary).as_posix(),
        "model_capacity_summary": (
            root / config.artefacts.model_capacity_summary
        ).as_posix(),
        "model_failures": (root / config.artefacts.model_failures).as_posix(),
    }


def build_training_metadata(
    *,
    plan: NeuralBenchmarkRunPlan,
    device: str,
    parameter_count: int,
    best_epoch: int | None,
    early_stopped: bool,
    training_seconds: float | None,
    validation_metric_value: float | None,
    test_metrics: Mapping[str, float],
    status: str,
) -> TrainingRunMetadata:
    """Build a validated per-run metadata record."""
    return TrainingRunMetadata(
        fold_id=plan.fold_id,
        seed=plan.seed,
        model_name=plan.model_name,
        lookback=plan.lookback,
        device=device,
        parameter_count=parameter_count,
        max_epochs=plan.max_epochs,
        best_epoch=best_epoch,
        early_stopped=early_stopped,
        training_seconds=training_seconds,
        validation_metric=plan.validation_metric,
        validation_metric_value=validation_metric_value,
        test_metrics=dict(test_metrics),
        status=status,
    )


def training_metadata_schema_fields() -> tuple[str, ...]:
    """Return stable field names for per-run neural metadata."""
    return tuple(TrainingRunMetadata.model_fields)

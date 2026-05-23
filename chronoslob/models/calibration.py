"""Calibration and uncertainty utilities for classifier logits.

This module provides Phase 15 calibration primitives for supervised
classification outputs, including the Phase 14 multi-task transformer heads.
It works on already-produced logits or probabilities and does not train market
forecasting models, simulate execution, run backtests or make alpha claims.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

try:  # pragma: no cover - exercised when torch is unavailable
    import torch
    from torch.nn import functional as torch_functional

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when torch is unavailable
    torch = None  # type: ignore[assignment]
    torch_functional = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

__all__ = [
    "CalibrationErrorConfig",
    "CalibrationSummary",
    "MultiTaskTemperatureScaler",
    "ReliabilityBin",
    "TemperatureScaler",
    "brier_score",
    "classification_confidence",
    "expected_calibration_error",
    "negative_log_likelihood",
    "reliability_bins",
    "softmax_probabilities",
]


def _require_torch() -> Any:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for calibration utilities. Install the "
            "'torch' optional dependency: pip install -e '.[torch]'"
        )
    return torch


def _validate_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_positive_float(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if numeric <= 0.0:
        raise ValueError(f"{name} must be positive")
    return numeric


def _validate_probability(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must satisfy 0 <= {name} <= 1")
    return numeric


def _state_float(
    state: Mapping[str, object],
    key: str,
    default: float,
) -> float:
    value = state.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be numeric")
    return float(value)


def _state_int(
    state: Mapping[str, object],
    key: str,
    default: int,
) -> int:
    value = state.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return int(value)


@dataclass(frozen=True)
class CalibrationErrorConfig:
    """Configuration for confidence-bin calibration summaries."""

    n_bins: int = 10
    min_confidence: float = 0.0
    max_confidence: float = 1.0

    def __post_init__(self) -> None:
        _validate_positive_int(self.n_bins, name="n_bins")
        minimum = _validate_probability(
            self.min_confidence,
            name="min_confidence",
        )
        maximum = _validate_probability(
            self.max_confidence,
            name="max_confidence",
        )
        if minimum >= maximum:
            raise ValueError("min_confidence must be smaller than max_confidence")

    @property
    def bin_edges(self) -> tuple[float, ...]:
        """Return evenly spaced bin edges for the configured confidence range."""
        width = (self.max_confidence - self.min_confidence) / self.n_bins
        return tuple(
            self.min_confidence + width * index
            for index in range(self.n_bins + 1)
        )


@dataclass(frozen=True)
class ReliabilityBin:
    """One confidence bin used to compute reliability and ECE."""

    bin_index: int
    lower_edge: float
    upper_edge: float
    count: int
    accuracy: float
    average_confidence: float
    calibration_gap: float
    contribution_to_ece: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a serialisable representation."""
        return {
            "bin_index": self.bin_index,
            "lower_edge": self.lower_edge,
            "upper_edge": self.upper_edge,
            "count": self.count,
            "accuracy": self.accuracy,
            "average_confidence": self.average_confidence,
            "calibration_gap": self.calibration_gap,
            "contribution_to_ece": self.contribution_to_ece,
        }


@dataclass(frozen=True)
class CalibrationSummary:
    """Summary of probabilistic classifier calibration."""

    n_examples: int
    n_bins: int
    ece: float
    average_confidence: float
    accuracy: float
    brier_score: float | None = None
    nll: float | None = None
    bins: list[ReliabilityBin] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable representation."""
        return {
            "n_examples": self.n_examples,
            "n_bins": self.n_bins,
            "ece": self.ece,
            "average_confidence": self.average_confidence,
            "accuracy": self.accuracy,
            "brier_score": self.brier_score,
            "nll": self.nll,
            "bins": [item.to_dict() for item in self.bins],
        }


def _validate_logits(logits: torch.Tensor, *, name: str = "logits") -> None:
    torch_module = _require_torch()
    if not torch_module.is_tensor(logits):
        raise TypeError(f"{name} must be a torch.Tensor")
    if logits.ndim != 2:
        raise ValueError(f"{name} must be 2D [n_examples, n_classes]")
    if int(logits.shape[0]) == 0:
        raise ValueError(f"{name} must contain at least one example")
    if int(logits.shape[1]) < 2:
        raise ValueError(f"{name} must contain at least two classes")
    if not torch_module.isfinite(logits).all().item():
        raise ValueError(f"{name} must contain only finite values")


def _validate_targets(
    targets: torch.Tensor,
    *,
    n_examples: int,
    n_classes: int,
    ignore_index: int,
) -> torch.Tensor:
    torch_module = _require_torch()
    if not torch_module.is_tensor(targets):
        raise TypeError("targets must be a torch.Tensor")
    if targets.ndim != 1:
        raise ValueError("targets must be 1D [n_examples]")
    if int(targets.shape[0]) != n_examples:
        raise ValueError(
            "targets length must match the number of examples: "
            f"{int(targets.shape[0])} != {n_examples}"
        )
    if targets.dtype != torch_module.long:
        raise TypeError("targets must have dtype torch.long")
    valid_mask = targets != int(ignore_index)
    if not bool(valid_mask.any().item()):
        raise ValueError("no valid targets after applying ignore_index")
    valid_targets = targets[valid_mask]
    below = valid_targets < 0
    above = valid_targets >= int(n_classes)
    if bool((below | above).any().item()):
        raise ValueError(
            "valid targets must be class indices in "
            f"[0, {int(n_classes) - 1}]"
        )
    return valid_mask


def _validate_probabilities(
    probabilities: torch.Tensor,
    *,
    name: str = "probabilities",
) -> None:
    torch_module = _require_torch()
    _validate_logits(probabilities, name=name)
    if bool((probabilities < 0.0).any().item()):
        raise ValueError(f"{name} must be non-negative")
    row_sums = probabilities.sum(dim=-1)
    expected = torch_module.ones_like(row_sums)
    if not torch_module.allclose(row_sums, expected, atol=1e-5, rtol=1e-5):
        raise ValueError(f"{name} rows must sum to 1")


def _probabilities_from_input(
    logits_or_probs: torch.Tensor,
    *,
    from_logits: bool,
) -> torch.Tensor:
    if from_logits:
        return softmax_probabilities(logits_or_probs)
    _validate_probabilities(logits_or_probs)
    return logits_or_probs


def _classification_arrays(
    logits_or_probs: torch.Tensor,
    targets: torch.Tensor,
    *,
    from_logits: bool,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = _probabilities_from_input(logits_or_probs, from_logits=from_logits)
    n_examples = int(probabilities.shape[0])
    n_classes = int(probabilities.shape[1])
    valid_mask = _validate_targets(
        targets,
        n_examples=n_examples,
        n_classes=n_classes,
        ignore_index=ignore_index,
    )
    valid_probabilities = probabilities[valid_mask]
    valid_targets = targets[valid_mask]
    confidence = classification_confidence(valid_probabilities)
    predictions = valid_probabilities.argmax(dim=-1)
    correct = predictions == valid_targets
    return valid_probabilities, valid_targets, confidence, predictions, correct


def softmax_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert classifier logits to row-wise probabilities."""
    _validate_logits(logits)
    return torch.softmax(logits, dim=-1)


def classification_confidence(probabilities: torch.Tensor) -> torch.Tensor:
    """Return the maximum class probability for each example."""
    _validate_probabilities(probabilities)
    return probabilities.max(dim=-1).values


def negative_log_likelihood(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Return mean cross-entropy over non-ignored classification targets."""
    _validate_logits(logits)
    valid_mask = _validate_targets(
        targets,
        n_examples=int(logits.shape[0]),
        n_classes=int(logits.shape[1]),
        ignore_index=ignore_index,
    )
    return torch_functional.cross_entropy(
        logits[valid_mask],
        targets[valid_mask],
        reduction="mean",
    )


def _probability_negative_log_likelihood(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int,
) -> torch.Tensor:
    torch_module = _require_torch()
    _validate_probabilities(probabilities)
    valid_mask = _validate_targets(
        targets,
        n_examples=int(probabilities.shape[0]),
        n_classes=int(probabilities.shape[1]),
        ignore_index=ignore_index,
    )
    valid_probabilities = probabilities[valid_mask]
    valid_targets = targets[valid_mask]
    selected = valid_probabilities[
        torch_module.arange(valid_targets.shape[0], device=valid_targets.device),
        valid_targets,
    ]
    return -torch_module.log(selected.clamp_min(1e-12)).mean()


def brier_score(
    logits_or_probs: torch.Tensor,
    targets: torch.Tensor,
    from_logits: bool = True,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Return the multiclass Brier score over non-ignored targets."""
    probabilities, valid_targets, _, _, _ = _classification_arrays(
        logits_or_probs,
        targets,
        from_logits=from_logits,
        ignore_index=ignore_index,
    )
    n_classes = int(probabilities.shape[1])
    one_hot = torch_functional.one_hot(valid_targets, num_classes=n_classes).to(
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    return ((probabilities - one_hot) ** 2).sum(dim=-1).mean()


def reliability_bins(
    logits_or_probs: torch.Tensor,
    targets: torch.Tensor,
    config: CalibrationErrorConfig,
    from_logits: bool = True,
    ignore_index: int = -100,
) -> list[ReliabilityBin]:
    """Build reliability-bin data for plotting or tabular reporting."""
    if not isinstance(config, CalibrationErrorConfig):
        raise TypeError("config must be a CalibrationErrorConfig instance")
    valid_probabilities, _, confidence, _, correct = _classification_arrays(
        logits_or_probs,
        targets,
        from_logits=from_logits,
        ignore_index=ignore_index,
    )
    n_valid = int(valid_probabilities.shape[0])
    width = (config.max_confidence - config.min_confidence) / config.n_bins
    scaled = (confidence - config.min_confidence) / width
    bin_ids = scaled.floor().clamp(min=0, max=config.n_bins - 1).to(torch.long)

    bins: list[ReliabilityBin] = []
    for index in range(config.n_bins):
        lower = config.min_confidence + width * index
        upper = config.min_confidence + width * (index + 1)
        in_bin = bin_ids == index
        count = int(in_bin.sum().item())
        if count == 0:
            accuracy = 0.0
            average_confidence = 0.0
            gap = 0.0
            contribution = 0.0
        else:
            accuracy = float(correct[in_bin].to(torch.float32).mean().item())
            average_confidence = float(confidence[in_bin].mean().item())
            gap = abs(accuracy - average_confidence)
            contribution = (count / n_valid) * gap
        bins.append(
            ReliabilityBin(
                bin_index=index,
                lower_edge=float(lower),
                upper_edge=float(upper),
                count=count,
                accuracy=accuracy,
                average_confidence=average_confidence,
                calibration_gap=gap,
                contribution_to_ece=contribution,
            )
        )
    return bins


def expected_calibration_error(
    logits_or_probs: torch.Tensor,
    targets: torch.Tensor,
    config: CalibrationErrorConfig,
    from_logits: bool = True,
    ignore_index: int = -100,
) -> CalibrationSummary:
    """Compute ECE and associated classifier calibration diagnostics."""
    probabilities, valid_targets, confidence, _, correct = _classification_arrays(
        logits_or_probs,
        targets,
        from_logits=from_logits,
        ignore_index=ignore_index,
    )
    bins = reliability_bins(
        logits_or_probs,
        targets,
        config,
        from_logits=from_logits,
        ignore_index=ignore_index,
    )
    ece = float(sum(item.contribution_to_ece for item in bins))
    if from_logits:
        nll = float(
            negative_log_likelihood(
                logits_or_probs,
                targets,
                ignore_index=ignore_index,
            )
            .detach()
            .cpu()
            .item()
        )
    else:
        nll = float(
            _probability_negative_log_likelihood(
                probabilities,
                targets,
                ignore_index=ignore_index,
            )
            .detach()
            .cpu()
            .item()
        )
    return CalibrationSummary(
        n_examples=int(valid_targets.shape[0]),
        n_bins=int(config.n_bins),
        ece=ece,
        average_confidence=float(confidence.mean().item()),
        accuracy=float(correct.to(torch.float32).mean().item()),
        brier_score=float(
            brier_score(
                logits_or_probs,
                targets,
                from_logits=from_logits,
                ignore_index=ignore_index,
            )
            .detach()
            .cpu()
            .item()
        ),
        nll=nll,
        bins=bins,
    )


class TemperatureScaler:
    """One-parameter post-hoc temperature scaling for classifier logits."""

    def __init__(
        self,
        *,
        initial_temperature: float = 1.0,
        max_iterations: int = 50,
        learning_rate: float = 0.1,
    ) -> None:
        _require_torch()
        self.initial_temperature = _validate_positive_float(
            initial_temperature,
            name="initial_temperature",
        )
        self.max_iterations = _validate_positive_int(
            max_iterations,
            name="max_iterations",
        )
        self.learning_rate = _validate_positive_float(
            learning_rate,
            name="learning_rate",
        )
        self._temperature = float(self.initial_temperature)

    @property
    def temperature(self) -> float:
        """Return the fitted positive scalar temperature."""
        return self._temperature

    def fit(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100,
    ) -> TemperatureScaler:
        """Fit the temperature on calibration logits and targets only."""
        torch_module = _require_torch()
        _validate_logits(logits)
        valid_mask = _validate_targets(
            targets,
            n_examples=int(logits.shape[0]),
            n_classes=int(logits.shape[1]),
            ignore_index=ignore_index,
        )
        logits_cpu = logits.detach().clone().to(
            device="cpu",
            dtype=torch_module.float64,
        )
        targets_cpu = targets.detach().clone().to(device="cpu")
        valid_logits = logits_cpu[valid_mask.detach().cpu()]
        valid_targets = targets_cpu[valid_mask.detach().cpu()]

        initial_temperature = torch_module.tensor(
            self._temperature,
            dtype=valid_logits.dtype,
            device=valid_logits.device,
        )
        initial_loss = torch_functional.cross_entropy(
            valid_logits / initial_temperature,
            valid_targets,
            reduction="mean",
        )
        log_temperature = torch_module.nn.Parameter(
            torch_module.tensor(
                math.log(self._temperature),
                dtype=valid_logits.dtype,
                device=valid_logits.device,
            )
        )
        optimiser = torch_module.optim.LBFGS(
            [log_temperature],
            lr=float(self.learning_rate),
            max_iter=int(self.max_iterations),
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimiser.zero_grad()
            temperature = log_temperature.exp()
            loss = torch_functional.cross_entropy(
                valid_logits / temperature,
                valid_targets,
                reduction="mean",
            )
            loss.backward()
            return loss

        optimiser.step(closure)
        with torch_module.no_grad():
            candidate_temperature = float(log_temperature.exp().item())
            candidate_tensor = torch_module.tensor(
                candidate_temperature,
                dtype=valid_logits.dtype,
                device=valid_logits.device,
            )
            candidate_loss = torch_functional.cross_entropy(
                valid_logits / candidate_tensor,
                valid_targets,
                reduction="mean",
            )
            if (
                not math.isfinite(candidate_temperature)
                or not torch_module.isfinite(candidate_loss).item()
                or float(candidate_loss.item()) > float(initial_loss.item()) + 1e-10
            ):
                candidate_temperature = float(initial_temperature.item())
            self._temperature = candidate_temperature
        return self

    def transform_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Return temperature-scaled logits without mutating the input tensor."""
        _validate_logits(logits)
        return logits / float(self._temperature)

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """Return probabilities after applying the fitted temperature."""
        return softmax_probabilities(self.transform_logits(logits))

    def to_dict(self) -> dict[str, float | int]:
        """Return a serialisable scaler state."""
        return {
            "temperature": float(self._temperature),
            "initial_temperature": float(self.initial_temperature),
            "max_iterations": int(self.max_iterations),
            "learning_rate": float(self.learning_rate),
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, object]) -> TemperatureScaler:
        """Construct a scaler from :meth:`to_dict` output."""
        if "temperature" not in state:
            raise ValueError("temperature scaler state is missing 'temperature'")
        scaler = cls(
            initial_temperature=_state_float(
                state,
                "initial_temperature",
                _state_float(state, "temperature", 1.0),
            ),
            max_iterations=_state_int(state, "max_iterations", 50),
            learning_rate=_state_float(state, "learning_rate", 0.1),
        )
        scaler._temperature = _validate_positive_float(
            _state_float(state, "temperature", 1.0),
            name="temperature",
        )
        return scaler


class MultiTaskTemperatureScaler:
    """Per-task temperature scaling for multi-task classifier logits."""

    def __init__(
        self,
        *,
        task_names: Sequence[str] | None = None,
        allow_missing_tasks: bool = False,
        allow_empty_tasks: bool = False,
        initial_temperature: float = 1.0,
        max_iterations: int = 50,
        learning_rate: float = 0.1,
    ) -> None:
        _require_torch()
        if task_names is not None:
            cleaned = tuple(str(name) for name in task_names)
            if not cleaned:
                raise ValueError("task_names must not be empty when provided")
            if len(set(cleaned)) != len(cleaned):
                raise ValueError("task_names must be unique")
            if any(not name.strip() or name != name.strip() for name in cleaned):
                raise ValueError("task_names must contain clean non-empty names")
            self.task_names: tuple[str, ...] | None = cleaned
        else:
            self.task_names = None
        if not isinstance(allow_missing_tasks, bool):
            raise TypeError("allow_missing_tasks must be a bool")
        if not isinstance(allow_empty_tasks, bool):
            raise TypeError("allow_empty_tasks must be a bool")
        self.allow_missing_tasks = allow_missing_tasks
        self.allow_empty_tasks = allow_empty_tasks
        self.initial_temperature = _validate_positive_float(
            initial_temperature,
            name="initial_temperature",
        )
        self.max_iterations = _validate_positive_int(
            max_iterations,
            name="max_iterations",
        )
        self.learning_rate = _validate_positive_float(
            learning_rate,
            name="learning_rate",
        )
        self._scalers: dict[str, TemperatureScaler] = {}

    @property
    def temperatures(self) -> dict[str, float]:
        """Return fitted temperatures by task name."""
        return {
            task_name: scaler.temperature
            for task_name, scaler in self._scalers.items()
        }

    def _tasks_to_fit(
        self,
        logits: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
    ) -> tuple[str, ...]:
        if self.task_names is not None:
            tasks = self.task_names
        else:
            tasks = tuple(logits.keys())
            if not tasks:
                raise ValueError("logits mapping must contain at least one task")
        missing_logits = [name for name in tasks if name not in logits]
        missing_targets = [name for name in tasks if name not in targets]
        missing = sorted({*missing_logits, *missing_targets})
        if missing and not self.allow_missing_tasks:
            raise ValueError(f"missing logits or targets for task(s): {missing}")
        if self.allow_missing_tasks:
            tasks = tuple(
                name for name in tasks if name in logits and name in targets
            )
        if not tasks:
            raise ValueError("no tasks available for temperature scaling")
        return tasks

    def _new_scaler(self) -> TemperatureScaler:
        return TemperatureScaler(
            initial_temperature=self.initial_temperature,
            max_iterations=self.max_iterations,
            learning_rate=self.learning_rate,
        )

    def fit(
        self,
        logits: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        ignore_index: int | Mapping[str, int] = -100,
    ) -> MultiTaskTemperatureScaler:
        """Fit one temperature per task on calibration logits and targets."""
        if not isinstance(logits, Mapping):
            raise TypeError("logits must be a mapping from task name to tensor")
        if not isinstance(targets, Mapping):
            raise TypeError("targets must be a mapping from task name to tensor")
        tasks = self._tasks_to_fit(logits, targets)
        fitted: dict[str, TemperatureScaler] = {}
        for task_name in tasks:
            task_ignore = (
                int(ignore_index[task_name])
                if isinstance(ignore_index, Mapping)
                else int(ignore_index)
            )
            scaler = self._new_scaler()
            try:
                scaler.fit(
                    logits[task_name],
                    targets[task_name],
                    ignore_index=task_ignore,
                )
            except ValueError as exc:
                if self.allow_empty_tasks and "no valid targets" in str(exc):
                    continue
                raise
            fitted[task_name] = scaler
        if not fitted:
            raise ValueError("no task had valid targets for temperature scaling")
        self._scalers = fitted
        return self

    def transform_logits(
        self,
        logits: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Apply fitted per-task temperatures to a logits mapping."""
        if not isinstance(logits, Mapping):
            raise TypeError("logits must be a mapping from task name to tensor")
        transformed: dict[str, torch.Tensor] = {}
        for task_name, task_logits in logits.items():
            if task_name not in self._scalers:
                raise KeyError(f"no fitted temperature for task {task_name!r}")
            transformed[task_name] = self._scalers[task_name].transform_logits(
                task_logits
            )
        return transformed

    def predict_proba(
        self,
        logits: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return per-task probabilities after temperature scaling."""
        return {
            task_name: softmax_probabilities(task_logits)
            for task_name, task_logits in self.transform_logits(logits).items()
        }

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable multi-task scaler state."""
        return {
            "task_names": list(self.task_names) if self.task_names is not None else None,
            "allow_missing_tasks": self.allow_missing_tasks,
            "allow_empty_tasks": self.allow_empty_tasks,
            "initial_temperature": self.initial_temperature,
            "max_iterations": self.max_iterations,
            "learning_rate": self.learning_rate,
            "scalers": {
                task_name: scaler.to_dict()
                for task_name, scaler in self._scalers.items()
            },
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, object]) -> MultiTaskTemperatureScaler:
        """Construct a multi-task scaler from :meth:`to_dict` output."""
        raw_task_names = state.get("task_names")
        if raw_task_names is None:
            task_names = None
        else:
            if not isinstance(raw_task_names, Sequence) or isinstance(
                raw_task_names,
                (str, bytes),
            ):
                raise TypeError("task_names in state must be a sequence or None")
            task_names = [str(name) for name in raw_task_names]
        scaler = cls(
            task_names=task_names,
            allow_missing_tasks=bool(state.get("allow_missing_tasks", False)),
            allow_empty_tasks=bool(state.get("allow_empty_tasks", False)),
            initial_temperature=_state_float(state, "initial_temperature", 1.0),
            max_iterations=_state_int(state, "max_iterations", 50),
            learning_rate=_state_float(state, "learning_rate", 0.1),
        )
        raw_scalers = state.get("scalers", {})
        if not isinstance(raw_scalers, Mapping):
            raise TypeError("scalers in state must be a mapping")
        scaler._scalers = {
            str(task_name): TemperatureScaler.from_dict(task_state)
            for task_name, task_state in raw_scalers.items()
            if isinstance(task_state, Mapping)
        }
        if len(scaler._scalers) != len(raw_scalers):
            raise TypeError("each scaler state must be a mapping")
        return scaler

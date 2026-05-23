"""Calibration smoke experiments and confidence filtering utilities.

Phase 15 keeps calibration separate from execution-aware validation. The
helpers in this module evaluate classifier probabilities, fitted temperature
scalers and abstention-style coverage diagnostics, but they do not implement
execution simulation, transaction costs, backtesting or trading claims.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from chronoslob.models.calibration import (
    CalibrationErrorConfig,
    CalibrationSummary,
    TemperatureScaler,
    expected_calibration_error,
    softmax_probabilities,
)

try:  # pragma: no cover - exercised when torch is unavailable
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when torch is unavailable
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

__all__ = [
    "AbstentionCurvePoint",
    "ConfidenceBucket",
    "ConfidenceFilterConfig",
    "ConfidenceFilteringResult",
    "abstention_curve",
    "build_confidence_filter",
    "evaluate_confidence_filter",
    "run_calibration_smoke",
    "summarise_multitask_calibration",
]

DEFAULT_CONFIDENCE_THRESHOLDS: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)
DEFAULT_ABSTENTION_COVERAGE_LEVELS: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4, 0.2)
_SYNTHETIC_CALIBRATION_WARNING = (
    "Synthetic calibration plumbing only; metrics are not benchmark evidence, "
    "alpha evidence, tradability evidence or execution-aware validation."
)


def _require_torch() -> Any:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for calibration smoke utilities. Install the "
            "'torch' optional dependency: pip install -e '.[torch]'"
        )
    return torch


def _validate_probability(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must satisfy 0 <= {name} <= 1")
    return numeric


def _validate_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True)
class ConfidenceFilterConfig:
    """Thresholds and missing-label policy for confidence filtering."""

    thresholds: tuple[float, ...] = DEFAULT_CONFIDENCE_THRESHOLDS
    ignore_index: int = -100

    def __post_init__(self) -> None:
        if not isinstance(self.thresholds, Sequence):
            raise TypeError("thresholds must be a sequence")
        cleaned = tuple(
            _validate_probability(value, name="threshold")
            for value in self.thresholds
        )
        if not cleaned:
            raise ValueError("thresholds must not be empty")
        object.__setattr__(self, "thresholds", cleaned)
        if isinstance(self.ignore_index, bool) or not isinstance(
            self.ignore_index,
            int,
        ):
            raise TypeError("ignore_index must be an integer")


@dataclass(frozen=True)
class ConfidenceBucket:
    """Coverage and accuracy for one confidence threshold."""

    threshold: float
    coverage: float
    abstention_rate: float
    accuracy_on_covered: float | None
    n_covered: int
    n_total: int

    def to_dict(self) -> dict[str, float | int | None]:
        """Return a serialisable representation."""
        return {
            "threshold": self.threshold,
            "coverage": self.coverage,
            "abstention_rate": self.abstention_rate,
            "accuracy_on_covered": self.accuracy_on_covered,
            "n_covered": self.n_covered,
            "n_total": self.n_total,
        }


@dataclass(frozen=True)
class ConfidenceFilteringResult:
    """Confidence-threshold evaluation over valid labelled examples."""

    n_total: int
    buckets: list[ConfidenceBucket] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable representation."""
        return {
            "n_total": self.n_total,
            "buckets": [bucket.to_dict() for bucket in self.buckets],
        }


@dataclass(frozen=True)
class AbstentionCurvePoint:
    """Accuracy retained at a requested coverage level."""

    coverage_level: float
    realised_coverage: float
    accuracy: float
    n_retained: int
    n_total: int
    min_confidence: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a serialisable representation."""
        return {
            "coverage_level": self.coverage_level,
            "realised_coverage": self.realised_coverage,
            "accuracy": self.accuracy,
            "n_retained": self.n_retained,
            "n_total": self.n_total,
            "min_confidence": self.min_confidence,
        }


def build_confidence_filter(
    thresholds: Sequence[float] | None = None,
    *,
    ignore_index: int = -100,
) -> ConfidenceFilterConfig:
    """Build a validated confidence-filtering configuration."""
    return ConfidenceFilterConfig(
        thresholds=(
            DEFAULT_CONFIDENCE_THRESHOLDS
            if thresholds is None
            else tuple(float(value) for value in thresholds)
        ),
        ignore_index=ignore_index,
    )


def _probabilities_from_input(
    logits_or_probs: torch.Tensor,
    *,
    from_logits: bool,
) -> torch.Tensor:
    torch_module = _require_torch()
    if from_logits:
        return softmax_probabilities(logits_or_probs)
    if not torch_module.is_tensor(logits_or_probs):
        raise TypeError("probabilities must be a torch.Tensor")
    if logits_or_probs.ndim != 2:
        raise ValueError("probabilities must be 2D [n_examples, n_classes]")
    if int(logits_or_probs.shape[1]) < 2:
        raise ValueError("probabilities must contain at least two classes")
    if bool((logits_or_probs < 0.0).any().item()):
        raise ValueError("probabilities must be non-negative")
    row_sums = logits_or_probs.sum(dim=-1)
    if not torch_module.allclose(
        row_sums,
        torch_module.ones_like(row_sums),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("probability rows must sum to 1")
    return logits_or_probs


def _valid_arrays(
    logits_or_probs: torch.Tensor,
    targets: torch.Tensor,
    *,
    from_logits: bool,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch_module = _require_torch()
    probabilities = _probabilities_from_input(
        logits_or_probs,
        from_logits=from_logits,
    )
    if not torch_module.is_tensor(targets):
        raise TypeError("targets must be a torch.Tensor")
    if targets.ndim != 1:
        raise ValueError("targets must be 1D [n_examples]")
    if int(targets.shape[0]) != int(probabilities.shape[0]):
        raise ValueError("targets length must match probabilities")
    if targets.dtype != torch_module.long:
        raise TypeError("targets must have dtype torch.long")
    valid_mask = targets != int(ignore_index)
    if not bool(valid_mask.any().item()):
        raise ValueError("no valid targets after applying ignore_index")
    valid_targets = targets[valid_mask]
    n_classes = int(probabilities.shape[1])
    invalid = (valid_targets < 0) | (valid_targets >= n_classes)
    if bool(invalid.any().item()):
        raise ValueError(
            f"valid targets must be class indices in [0, {n_classes - 1}]"
        )
    valid_probabilities = probabilities[valid_mask]
    confidence = valid_probabilities.max(dim=-1).values
    predictions = valid_probabilities.argmax(dim=-1)
    correct = predictions == valid_targets
    return valid_probabilities, confidence, predictions, correct


def evaluate_confidence_filter(
    logits_or_probs: torch.Tensor,
    targets: torch.Tensor,
    config: ConfidenceFilterConfig | None = None,
    *,
    from_logits: bool = True,
    ignore_index: int | None = None,
) -> ConfidenceFilteringResult:
    """Evaluate threshold-based coverage and accuracy diagnostics."""
    resolved_config = config if config is not None else ConfidenceFilterConfig()
    if not isinstance(resolved_config, ConfidenceFilterConfig):
        raise TypeError("config must be a ConfidenceFilterConfig instance")
    resolved_ignore = (
        int(resolved_config.ignore_index)
        if ignore_index is None
        else int(ignore_index)
    )
    _, confidence, _, correct = _valid_arrays(
        logits_or_probs,
        targets,
        from_logits=from_logits,
        ignore_index=resolved_ignore,
    )
    n_total = int(confidence.shape[0])
    buckets: list[ConfidenceBucket] = []
    for threshold in resolved_config.thresholds:
        covered = confidence >= float(threshold)
        n_covered = int(covered.sum().item())
        coverage = n_covered / n_total
        accuracy = (
            float(correct[covered].to(torch.float32).mean().item())
            if n_covered > 0
            else None
        )
        buckets.append(
            ConfidenceBucket(
                threshold=float(threshold),
                coverage=coverage,
                abstention_rate=1.0 - coverage,
                accuracy_on_covered=accuracy,
                n_covered=n_covered,
                n_total=n_total,
            )
        )
    return ConfidenceFilteringResult(n_total=n_total, buckets=buckets)


def _validate_coverage_levels(
    coverage_levels: Sequence[float],
) -> tuple[float, ...]:
    if not isinstance(coverage_levels, Sequence):
        raise TypeError("coverage_levels must be a sequence")
    cleaned = tuple(
        _validate_probability(value, name="coverage_level")
        for value in coverage_levels
    )
    if not cleaned:
        raise ValueError("coverage_levels must not be empty")
    if any(value <= 0.0 for value in cleaned):
        raise ValueError("coverage_levels must be greater than zero")
    return cleaned


def abstention_curve(
    logits_or_probs: torch.Tensor,
    targets: torch.Tensor,
    coverage_levels: Sequence[float] = DEFAULT_ABSTENTION_COVERAGE_LEVELS,
    *,
    from_logits: bool = True,
    ignore_index: int = -100,
) -> list[AbstentionCurvePoint]:
    """Evaluate accuracy when retaining the most confident examples."""
    _, confidence, _, correct = _valid_arrays(
        logits_or_probs,
        targets,
        from_logits=from_logits,
        ignore_index=ignore_index,
    )
    resolved_levels = _validate_coverage_levels(coverage_levels)
    order = torch.argsort(confidence, descending=True, stable=True)
    sorted_confidence = confidence[order]
    sorted_correct = correct[order]
    n_total = int(confidence.shape[0])
    points: list[AbstentionCurvePoint] = []
    for coverage_level in resolved_levels:
        n_retained = min(n_total, max(1, math.ceil(n_total * coverage_level)))
        retained_correct = sorted_correct[:n_retained]
        retained_confidence = sorted_confidence[:n_retained]
        points.append(
            AbstentionCurvePoint(
                coverage_level=float(coverage_level),
                realised_coverage=n_retained / n_total,
                accuracy=float(retained_correct.to(torch.float32).mean().item()),
                n_retained=n_retained,
                n_total=n_total,
                min_confidence=float(retained_confidence[-1].item()),
            )
        )
    return points


def _compact_summary(summary: CalibrationSummary) -> dict[str, object]:
    return {
        "n_examples": summary.n_examples,
        "n_bins": summary.n_bins,
        "ece": summary.ece,
        "average_confidence": summary.average_confidence,
        "accuracy": summary.accuracy,
        "brier_score": summary.brier_score,
        "nll": summary.nll,
        "bins": [item.to_dict() for item in summary.bins],
    }


def _task_ignore_index(
    ignore_index: int | Mapping[str, int],
    task_name: str,
) -> int:
    if isinstance(ignore_index, Mapping):
        if task_name not in ignore_index:
            raise KeyError(f"missing ignore_index for task {task_name!r}")
        return int(ignore_index[task_name])
    return int(ignore_index)


def summarise_multitask_calibration(
    logits: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    *,
    error_config: CalibrationErrorConfig | None = None,
    filter_config: ConfidenceFilterConfig | None = None,
    from_logits: bool = True,
    ignore_index: int | Mapping[str, int] = -100,
    allow_empty_tasks: bool = False,
) -> dict[str, object]:
    """Return per-task calibration summaries for multi-task classifiers."""
    if not isinstance(logits, Mapping):
        raise TypeError("logits must be a mapping from task name to tensor")
    if not isinstance(targets, Mapping):
        raise TypeError("targets must be a mapping from task name to tensor")
    missing_targets = sorted(set(logits) - set(targets))
    if missing_targets:
        raise ValueError(f"missing targets for task(s): {missing_targets}")
    config = error_config if error_config is not None else CalibrationErrorConfig()
    confidence_config = (
        filter_config if filter_config is not None else ConfidenceFilterConfig()
    )
    task_summaries: dict[str, object] = {}
    for task_name, task_logits in logits.items():
        task_ignore = _task_ignore_index(ignore_index, task_name)
        try:
            summary = expected_calibration_error(
                task_logits,
                targets[task_name],
                config,
                from_logits=from_logits,
                ignore_index=task_ignore,
            )
            filtering = evaluate_confidence_filter(
                task_logits,
                targets[task_name],
                confidence_config,
                from_logits=from_logits,
                ignore_index=task_ignore,
            )
        except ValueError as exc:
            if allow_empty_tasks and "no valid targets" in str(exc):
                continue
            raise
        task_summaries[task_name] = {
            "summary": _compact_summary(summary),
            "confidence_filtering": filtering.to_dict(),
        }
    if not task_summaries:
        raise ValueError("no task had valid targets for calibration summary")
    return {
        "task_count": len(task_summaries),
        "task_summaries": task_summaries,
        "averaging": "none; per-task summaries are reported separately",
    }


def _synthetic_logits_and_targets(
    *,
    n_examples: int,
    num_classes: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch_module = _require_torch()
    generator = torch_module.Generator(device="cpu")
    generator.manual_seed(int(seed))
    targets = torch_module.arange(n_examples, dtype=torch_module.long) % int(
        num_classes
    )
    logits = torch_module.randn(
        (n_examples, num_classes),
        generator=generator,
        dtype=torch_module.float32,
    ) * 0.35
    rows = torch_module.arange(n_examples)
    logits[rows, targets] += 1.4

    mistake_rows = torch_module.arange(0, n_examples, 7)
    if int(mistake_rows.numel()) > 0:
        wrong_classes = (targets[mistake_rows] + 1) % int(num_classes)
        logits[mistake_rows, targets[mistake_rows]] -= 1.1
        logits[mistake_rows, wrong_classes] += 1.3

    return logits * 2.25, targets


def run_calibration_smoke(
    *,
    n_examples: int = 60,
    num_classes: int = 3,
    seed: int = 42,
    ece_bins: int = 10,
    thresholds: Sequence[float] = DEFAULT_CONFIDENCE_THRESHOLDS,
    max_temperature_iterations: int = 50,
    temperature_learning_rate: float = 0.1,
) -> dict[str, object]:
    """Run a deterministic synthetic calibration plumbing check.

    The first half of the synthetic sequence is the calibration split used to
    fit temperature scaling. The second half is the evaluation split used for
    pre/post calibration diagnostics. No files are written.
    """
    _require_torch()
    _validate_positive_int(n_examples, name="n_examples")
    _validate_positive_int(num_classes, name="num_classes")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if n_examples < 4:
        raise ValueError("n_examples must be at least 4")
    _validate_non_negative_int(seed, name="seed")

    logits, targets = _synthetic_logits_and_targets(
        n_examples=n_examples,
        num_classes=num_classes,
        seed=seed,
    )
    split_index = n_examples // 2
    calibration_logits = logits[:split_index]
    calibration_targets = targets[:split_index]
    evaluation_logits = logits[split_index:]
    evaluation_targets = targets[split_index:]

    error_config = CalibrationErrorConfig(n_bins=ece_bins)
    filter_config = build_confidence_filter(thresholds)
    scaler = TemperatureScaler(
        max_iterations=max_temperature_iterations,
        learning_rate=temperature_learning_rate,
    )
    scaler.fit(calibration_logits, calibration_targets)
    calibrated_evaluation_logits = scaler.transform_logits(evaluation_logits)

    pre_summary = expected_calibration_error(
        evaluation_logits,
        evaluation_targets,
        error_config,
    )
    post_summary = expected_calibration_error(
        calibrated_evaluation_logits,
        evaluation_targets,
        error_config,
    )
    filtering = evaluate_confidence_filter(
        calibrated_evaluation_logits,
        evaluation_targets,
        filter_config,
    )
    curve = abstention_curve(
        calibrated_evaluation_logits,
        evaluation_targets,
        DEFAULT_ABSTENTION_COVERAGE_LEVELS,
    )

    return {
        "synthetic_plumbing_only": True,
        "notes": _SYNTHETIC_CALIBRATION_WARNING,
        "n_examples": int(n_examples),
        "num_classes": int(num_classes),
        "seed": int(seed),
        "calibration_examples": int(calibration_logits.shape[0]),
        "evaluation_examples": int(evaluation_logits.shape[0]),
        "split_discipline": (
            "temperature fitted on the first synthetic calibration subset; "
            "evaluation subset is transformed only with the fitted parameter"
        ),
        "fitted_temperature": float(scaler.temperature),
        "temperature_state": scaler.to_dict(),
        "pre_calibration": _compact_summary(pre_summary),
        "post_calibration": _compact_summary(post_summary),
        "confidence_filtering": filtering.to_dict(),
        "abstention_curve": [point.to_dict() for point in curve],
        "write_outputs": False,
        "checkpoints_written": False,
        "network_calls": "none",
    }

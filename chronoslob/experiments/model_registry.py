"""Model registry for the paper experiment runner.

This module is the single source of truth for the short model names
exposed via ``--models`` on ``run-paper-experiment``. Each entry maps a
short, lower-case name to:

* the runner family (classical or neural),
* the underlying model type,
* whether the model requires train-only feature standardisation,
* whether the model is expected to emit class probabilities, and
* a short human-readable description.

The registry is intentionally kept small and conservative. ``ssl_transformer``
is not registered until a genuine train-only pretraining plus supervised
fine-tuning path is available in the paper runner.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from chronoslob.models.baselines import BaselineModelConfig

__all__ = [
    "DEFAULT_PAPER_MODELS",
    "REQUIRED_PAPER_MODELS",
    "SUPPORTED_PAPER_MODELS",
    "PaperModelSpec",
    "build_paper_baseline_config",
    "get_paper_model_spec",
    "list_supported_paper_models",
    "normalise_paper_model_names",
]


@dataclass(frozen=True)
class PaperModelSpec:
    """Registry entry for one paper-runner model."""

    name: str
    model_type: str
    requires_standardisation: bool
    emits_probabilities: bool
    description: str
    model_family: str = "classical"


_REGISTRY: tuple[PaperModelSpec, ...] = (
    PaperModelSpec(
        name="majority",
        model_type="majority_class",
        requires_standardisation=False,
        emits_probabilities=True,
        description=(
            "Deterministic majority-class baseline; predicts the most "
            "frequent training class for every test row."
        ),
    ),
    PaperModelSpec(
        name="logistic",
        model_type="logistic_regression",
        requires_standardisation=True,
        emits_probabilities=True,
        description=(
            "L2-regularised logistic regression on train-only "
            "standardised features."
        ),
    ),
    PaperModelSpec(
        name="ridge",
        model_type="ridge_classifier",
        requires_standardisation=True,
        emits_probabilities=False,
        description=(
            "Ridge classifier on train-only standardised features. "
            "Does not emit calibrated class probabilities."
        ),
    ),
    PaperModelSpec(
        name="elastic_net",
        model_type="elastic_net_logistic",
        requires_standardisation=True,
        emits_probabilities=True,
        description=(
            "Elastic-net logistic regression on train-only standardised "
            "features."
        ),
    ),
    PaperModelSpec(
        name="random_forest",
        model_type="random_forest",
        requires_standardisation=False,
        emits_probabilities=True,
        description=(
            "Random forest classifier on raw features; "
            "scaling is not required."
        ),
    ),
    PaperModelSpec(
        name="gradient_boosting",
        model_type="gradient_boosting",
        requires_standardisation=False,
        emits_probabilities=True,
        description=(
            "Gradient-boosting classifier on raw features; "
            "scaling is not required."
        ),
    ),
    PaperModelSpec(
        name="deeplob_style",
        model_type="deeplob_style",
        requires_standardisation=True,
        emits_probabilities=True,
        description=(
            "Compact DeepLOB-style CNN-LSTM baseline over train-only "
            "standardised FI-2010 windows. This is not an exact "
            "reproduction of the original architecture."
        ),
        model_family="neural",
    ),
    PaperModelSpec(
        name="transformer",
        model_type="normalised_matrix_transformer",
        requires_standardisation=True,
        emits_probabilities=True,
        description=(
            "Supervised transformer baseline over normalised FI-2010 "
            "matrix windows. The paper-runner path does not construct raw "
            "order-book snapshots from z-score values."
        ),
        model_family="neural",
    ),
    PaperModelSpec(
        name="matrix_transformer",
        model_type="normalised_matrix_transformer",
        requires_standardisation=True,
        emits_probabilities=True,
        description=(
            "Explicit alias for the supervised normalised FI-2010 matrix "
            "transformer baseline."
        ),
        model_family="neural",
    ),
)


_REGISTRY_BY_NAME: dict[str, PaperModelSpec] = {spec.name: spec for spec in _REGISTRY}


SUPPORTED_PAPER_MODELS: tuple[str, ...] = tuple(spec.name for spec in _REGISTRY)
DEFAULT_PAPER_MODELS: tuple[str, ...] = ("majority",)
REQUIRED_PAPER_MODELS: tuple[str, ...] = ("majority",)


def list_supported_paper_models() -> tuple[str, ...]:
    """Return the supported paper-runner model short names."""
    return SUPPORTED_PAPER_MODELS


def get_paper_model_spec(name: str) -> PaperModelSpec:
    """Return the registry spec for ``name`` or raise a clear error."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("paper model name must be a non-empty string")
    cleaned = name.strip().lower()
    spec = _REGISTRY_BY_NAME.get(cleaned)
    if spec is None:
        raise ValueError(
            f"unsupported model name: {name!r}; supported: "
            f"{list(SUPPORTED_PAPER_MODELS)}",
        )
    return spec


def normalise_paper_model_names(
    models: Sequence[str] | None,
    *,
    require: Sequence[str] = REQUIRED_PAPER_MODELS,
) -> tuple[str, ...]:
    """Normalise, deduplicate and validate user-supplied model names.

    Names are case-folded to lower-case. Duplicates are removed while
    preserving first-seen order. The function returns the default model
    list when ``models`` is ``None``. The required-model rule keeps the
    majority baseline as a deterministic interpretable floor.
    """
    if models is None:
        return DEFAULT_PAPER_MODELS

    if isinstance(models, str) or not isinstance(models, Sequence):
        raise TypeError("models must be a sequence of strings")

    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in models:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("model names must be non-empty strings")
        spec = get_paper_model_spec(raw)
        if spec.name in seen:
            continue
        cleaned.append(spec.name)
        seen.add(spec.name)

    if not cleaned:
        raise ValueError("models must contain at least one supported entry")

    for required in require:
        if required not in cleaned:
            raise ValueError(
                "the majority baseline must be included; "
                f"add {required!r} to --models",
            )

    return tuple(cleaned)


def build_paper_baseline_config(name: str, *, seed: int) -> BaselineModelConfig:
    """Build a :class:`BaselineModelConfig` for the registry ``name``."""
    spec = get_paper_model_spec(name)
    if spec.model_family != "classical":
        raise ValueError(
            f"paper model {spec.name!r} is not a classical baseline and "
            "does not have a BaselineModelConfig"
        )
    return BaselineModelConfig(
        name=spec.name,
        model_type=spec.model_type,
        random_state=seed,
    )

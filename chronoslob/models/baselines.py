"""Classical baseline model interfaces for leakage-safe experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier

__all__ = [
    "SUPPORTED_BASELINE_MODEL_TYPES",
    "BaseBaselineModel",
    "BaselineModelConfig",
    "MajorityClassBaseline",
    "SklearnBaselineModel",
    "create_baseline_model",
]

SUPPORTED_BASELINE_MODEL_TYPES = (
    "majority_class",
    "logistic_regression",
    "ridge_classifier",
    "elastic_net_logistic",
    "random_forest",
    "gradient_boosting",
)

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)
_HyperparameterValue = str | int | float | bool


class BaselineModelConfig(BaseModel):
    """Configuration for one classical baseline model."""

    model_config = _MODEL_CONFIG

    name: str
    model_type: str
    task_type: str = "classification"
    random_state: int = 42
    class_weight: str | None = None
    hyperparameters: dict[str, _HyperparameterValue] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("name must be a non-empty string")
        return value

    @field_validator("model_type")
    @classmethod
    def _validate_model_type(cls, value: str) -> str:
        if value not in SUPPORTED_BASELINE_MODEL_TYPES:
            raise ValueError(
                "model_type must be one of "
                f"{list(SUPPORTED_BASELINE_MODEL_TYPES)}"
            )
        return value

    @field_validator("task_type")
    @classmethod
    def _validate_task_type(cls, value: str) -> str:
        if value != "classification":
            raise ValueError("task_type currently only supports 'classification'")
        return value

    @field_validator("random_state")
    @classmethod
    def _validate_random_state(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("random_state must be an integer")
        if value < 0:
            raise ValueError("random_state must be non-negative")
        return value

    @field_validator("class_weight")
    @classmethod
    def _validate_class_weight(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("class_weight must be a non-empty string when provided")
        return value

    @field_validator("hyperparameters")
    @classmethod
    def _validate_hyperparameters(
        cls,
        value: dict[str, _HyperparameterValue],
        info: ValidationInfo,
    ) -> dict[str, _HyperparameterValue]:
        for key in value:
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{info.field_name} keys must be non-empty strings")
        return dict(value)


class BaseBaselineModel(ABC):
    """Common interface for ChronosLOB classical baseline models."""

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> BaseBaselineModel:
        """Fit the model on training features and labels."""

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels for ``x``."""

    @abstractmethod
    def predict_proba(self, x: np.ndarray) -> np.ndarray | None:
        """Predict class probabilities for ``x`` when supported."""

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Return whether the model has been fitted."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the configured model name."""


def _validate_x(x: np.ndarray, *, name: str = "x") -> np.ndarray:
    if not isinstance(x, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D")
    numeric_x = x.astype(float, copy=False)
    if bool(np.isnan(numeric_x).any()):
        raise ValueError(f"{name} must not contain NaN values")
    if bool(np.isinf(numeric_x).any()):
        raise ValueError(f"{name} must not contain infinite values")
    return numeric_x


def _validate_y(y: np.ndarray, *, n_rows: int) -> np.ndarray:
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy.ndarray")
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    if len(y) != n_rows:
        raise ValueError("x and y must contain the same number of rows")
    if bool(np.asarray([value is None for value in y]).any()):
        raise ValueError("y must not contain None values")
    return y


def _label_sort_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def _sorted_labels(values: np.ndarray) -> list[Any]:
    unique = list(set(values.tolist()))
    try:
        return sorted(unique)
    except TypeError:
        return sorted(unique, key=_label_sort_key)


class MajorityClassBaseline(BaseBaselineModel):
    """Deterministic baseline that always predicts the most frequent class."""

    def __init__(self, name: str = "majority_class") -> None:
        if not name.strip():
            raise ValueError("name must be non-empty")
        self._name = name
        self._classes: list[Any] | None = None
        self._probabilities: np.ndarray | None = None
        self._majority_class: Any | None = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been fitted."""
        return self._is_fitted

    @property
    def model_name(self) -> str:
        """Return the configured model name."""
        return self._name

    @property
    def classes_(self) -> list[Any]:
        """Return fitted classes in deterministic order."""
        self._raise_if_unfitted()
        return list(self._classes or [])

    def _raise_if_unfitted(self) -> None:
        if not self._is_fitted:
            raise ValueError("MajorityClassBaseline must be fitted before prediction")

    def fit(self, x: np.ndarray, y: np.ndarray) -> MajorityClassBaseline:
        """Learn the most frequent training class and class frequencies."""
        numeric_x = _validate_x(x)
        labels = _validate_y(y, n_rows=numeric_x.shape[0])
        if len(labels) == 0:
            raise ValueError("y must contain at least one label")

        classes = _sorted_labels(labels)
        counts: Mapping[Any, int] = Counter(labels.tolist())
        majority_position = max(
            range(len(classes)),
            key=lambda position: (counts[classes[position]], -position),
        )
        total = float(len(labels))
        self._classes = classes
        self._majority_class = classes[majority_position]
        self._probabilities = np.asarray(
            [counts[class_label] / total for class_label in classes],
            dtype=float,
        )
        self._is_fitted = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the fitted majority class for every row in ``x``."""
        self._raise_if_unfitted()
        numeric_x = _validate_x(x)
        return np.asarray([self._majority_class] * numeric_x.shape[0])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return fitted class frequencies for every row in ``x``."""
        self._raise_if_unfitted()
        numeric_x = _validate_x(x)
        probabilities = np.asarray(self._probabilities, dtype=float)
        return np.tile(probabilities, (numeric_x.shape[0], 1))


class SklearnBaselineModel(BaseBaselineModel):
    """Wrapper for sklearn classifiers behind the baseline interface."""

    def __init__(self, name: str, estimator: Any) -> None:
        if not name.strip():
            raise ValueError("name must be non-empty")
        self._name = name
        self._estimator = estimator
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been fitted."""
        return self._is_fitted

    @property
    def model_name(self) -> str:
        """Return the configured model name."""
        return self._name

    @property
    def estimator(self) -> Any:
        """Return the wrapped sklearn estimator."""
        return self._estimator

    def _raise_if_unfitted(self) -> None:
        if not self._is_fitted:
            raise ValueError("SklearnBaselineModel must be fitted before prediction")

    def fit(self, x: np.ndarray, y: np.ndarray) -> SklearnBaselineModel:
        """Fit the wrapped sklearn estimator."""
        numeric_x = _validate_x(x)
        labels = _validate_y(y, n_rows=numeric_x.shape[0])
        self._estimator.fit(numeric_x, labels)
        self._is_fitted = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels using the wrapped estimator."""
        self._raise_if_unfitted()
        numeric_x = _validate_x(x)
        return np.asarray(self._estimator.predict(numeric_x))

    def predict_proba(self, x: np.ndarray) -> np.ndarray | None:
        """Return class probabilities when the wrapped estimator supports them."""
        self._raise_if_unfitted()
        if not hasattr(self._estimator, "predict_proba"):
            return None
        numeric_x = _validate_x(x)
        return np.asarray(self._estimator.predict_proba(numeric_x), dtype=float)


def _params_with_hyperparameters(
    defaults: dict[str, Any],
    hyperparameters: dict[str, _HyperparameterValue],
) -> dict[str, Any]:
    params = dict(defaults)
    params.update(hyperparameters)
    return params


def create_baseline_model(config: BaselineModelConfig) -> BaseBaselineModel:
    """Create a baseline model from a validated configuration."""
    if config.model_type == "majority_class":
        return MajorityClassBaseline(name=config.name)

    if config.model_type == "logistic_regression":
        params = _params_with_hyperparameters(
            {
                "max_iter": 1000,
                "random_state": config.random_state,
                "class_weight": config.class_weight,
            },
            config.hyperparameters,
        )
        return SklearnBaselineModel(config.name, LogisticRegression(**params))

    if config.model_type == "ridge_classifier":
        params = _params_with_hyperparameters(
            {"class_weight": config.class_weight},
            config.hyperparameters,
        )
        return SklearnBaselineModel(config.name, RidgeClassifier(**params))

    if config.model_type == "elastic_net_logistic":
        params = _params_with_hyperparameters(
            {
                "max_iter": 1000,
                "random_state": config.random_state,
                "class_weight": config.class_weight,
                "l1_ratio": 0.5,
            },
            config.hyperparameters,
        )
        params["penalty"] = "elasticnet"
        params["solver"] = "saga"
        return SklearnBaselineModel(config.name, LogisticRegression(**params))

    if config.model_type == "random_forest":
        params = _params_with_hyperparameters(
            {
                "n_estimators": 50,
                "random_state": config.random_state,
                "class_weight": config.class_weight,
            },
            config.hyperparameters,
        )
        return SklearnBaselineModel(config.name, RandomForestClassifier(**params))

    if config.model_type == "gradient_boosting":
        if config.class_weight is not None:
            raise ValueError("gradient_boosting does not support class_weight")
        params = _params_with_hyperparameters(
            {"random_state": config.random_state},
            config.hyperparameters,
        )
        return SklearnBaselineModel(config.name, GradientBoostingClassifier(**params))

    raise ValueError(f"unsupported baseline model_type: {config.model_type!r}")

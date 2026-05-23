"""Tests for classical baseline model interfaces."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from chronoslob.models.baselines import (
    SUPPORTED_BASELINE_MODEL_TYPES,
    BaselineModelConfig,
    MajorityClassBaseline,
    create_baseline_model,
)


def _tiny_classification_data() -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [3.0, 0.0],
            [3.0, 1.0],
        ]
    )
    y = np.asarray([0, 0, 0, 1, 1, 1, 1, 0])
    return x, y


def test_baseline_model_config_validation() -> None:
    config = BaselineModelConfig(name="majority", model_type="majority_class")
    assert config.task_type == "classification"

    with pytest.raises(ValidationError):
        BaselineModelConfig(name="", model_type="majority_class")
    with pytest.raises(ValidationError):
        BaselineModelConfig(name="bad", model_type="unsupported")
    with pytest.raises(ValidationError):
        BaselineModelConfig(
            name="bad",
            model_type="majority_class",
            random_state=-1,
        )


def test_majority_class_baseline_fit_and_predict() -> None:
    x, _ = _tiny_classification_data()
    y = np.asarray(["up", "up", "down", "up", "down", "up", "up", "down"])
    model = MajorityClassBaseline().fit(x, y)

    predictions = model.predict(x[:3])

    assert model.is_fitted
    assert predictions.tolist() == ["up", "up", "up"]


def test_majority_class_baseline_tie_breaking_is_deterministic() -> None:
    x = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    y = np.asarray(["b", "a", "b", "a"])

    model = MajorityClassBaseline().fit(x, y)

    assert model.predict(np.asarray([[10.0]])).tolist() == ["a"]


def test_majority_class_baseline_predict_proba_sums_to_one() -> None:
    x, y = _tiny_classification_data()
    model = MajorityClassBaseline().fit(x, y)

    probabilities = model.predict_proba(x[:2])

    assert probabilities.shape == (2, 2)
    assert probabilities.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])


def test_create_baseline_model_supports_all_model_types() -> None:
    for model_type in SUPPORTED_BASELINE_MODEL_TYPES:
        model = create_baseline_model(
            BaselineModelConfig(name=model_type, model_type=model_type)
        )
        assert model.model_name == model_type


def test_sklearn_baselines_fit_and_predict_on_tiny_data() -> None:
    x, y = _tiny_classification_data()
    model_types = [
        "logistic_regression",
        "ridge_classifier",
        "elastic_net_logistic",
        "random_forest",
        "gradient_boosting",
    ]

    for model_type in model_types:
        model = create_baseline_model(
            BaselineModelConfig(
                name=model_type,
                model_type=model_type,
                hyperparameters={"max_iter": 2000}
                if model_type in {"logistic_regression", "elastic_net_logistic"}
                else {},
            )
        )
        model.fit(x, y)
        predictions = model.predict(x)

        assert model.is_fitted
        assert predictions.shape == y.shape


def test_predict_before_fit_raises_clear_error() -> None:
    x, _ = _tiny_classification_data()

    with pytest.raises(ValueError, match="fitted"):
        MajorityClassBaseline().predict(x)

    model = create_baseline_model(
        BaselineModelConfig(name="logistic", model_type="logistic_regression")
    )
    with pytest.raises(ValueError, match="fitted"):
        model.predict(x)


def test_unsupported_model_type_raises() -> None:
    config = BaselineModelConfig.model_construct(
        name="unsupported",
        model_type="unsupported",
        task_type="classification",
        random_state=42,
        class_weight=None,
        hyperparameters={},
    )

    with pytest.raises(ValueError, match="unsupported"):
        create_baseline_model(config)

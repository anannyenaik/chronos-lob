"""Tests for the synthetic-label transformer smoke experiment runner."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from chronoslob.training.transformer_experiment import (  # noqa: E402
    run_transformer_smoke_from_event_log,
)
from chronoslob.utils.paths import project_root  # noqa: E402

_FIXTURE = (
    project_root() / "tests" / "fixtures" / "event_logs" / "synthetic_snapshots.jsonl"
)
_MIXED_FIXTURE = (
    project_root() / "tests" / "fixtures" / "event_logs" / "synthetic_mixed_log.jsonl"
)


def test_smoke_experiment_from_event_log_runs_and_returns_payload() -> None:
    result = run_transformer_smoke_from_event_log(
        path=_FIXTURE,
        symbol="TESTUSDT",
        window_length=4,
        batch_size=2,
        epochs=1,
        seed=42,
        num_classes=3,
        max_levels_per_side=2,
    )
    assert result["path"] == str(_FIXTURE)
    assert result["symbol_filter"] == "TESTUSDT"
    assert result["input_record_count"] > 0
    assert result["tokenised_record_count"] > 0
    assert result["window_count"] > 0
    assert result["num_classes"] == 3
    assert result["model_parameter_count"] > 0
    assert result["training_history"]
    assert isinstance(result["training_history"][-1]["train_loss"], float)
    assert torch.tensor(result["training_history"][-1]["train_loss"]).isfinite().item()
    assert "vocab_sizes" in result
    assert set(result["vocab_sizes"]).issuperset(
        {
            "event_type",
            "side",
            "price_bucket",
            "quantity_bucket",
            "time_delta_bucket",
            "context_bucket",
            "source",
        }
    )


def test_smoke_payload_marks_labels_synthetic_and_omits_market_claims() -> None:
    result = run_transformer_smoke_from_event_log(
        path=_FIXTURE,
        symbol="TESTUSDT",
        window_length=4,
        batch_size=2,
        epochs=1,
    )
    assert "synthetic" in result["notes"].lower()
    assert "synthetic" in result["label_source"].lower()
    forbidden_keys = {
        "alpha",
        "sharpe",
        "pnl",
        "profit",
        "trade_count",
        "backtest",
        "live_metrics",
    }
    assert forbidden_keys.isdisjoint(result.keys())
    smoke_metrics = result["synthetic_smoke_metrics"]
    assert set(smoke_metrics.keys()) == {"loss", "accuracy", "n_samples"}


def test_smoke_experiment_supports_mixed_event_log() -> None:
    result = run_transformer_smoke_from_event_log(
        path=_MIXED_FIXTURE,
        window_length=3,
        batch_size=2,
        epochs=1,
        seed=7,
    )
    assert result["window_count"] > 0
    assert result["model_parameter_count"] > 0


def test_smoke_experiment_invalid_path_raises_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.jsonl"
    with pytest.raises(FileNotFoundError):
        run_transformer_smoke_from_event_log(
            path=missing,
            window_length=4,
        )


def test_smoke_experiment_zero_window_case_raises_clear_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="zero windows"):
        run_transformer_smoke_from_event_log(
            path=empty,
            window_length=4,
        )


def test_smoke_experiment_does_not_write_files(tmp_path: Path) -> None:
    initial = sorted(tmp_path.iterdir())
    run_transformer_smoke_from_event_log(
        path=_FIXTURE,
        symbol="TESTUSDT",
        window_length=4,
        batch_size=2,
        epochs=1,
    )
    final = sorted(tmp_path.iterdir())
    assert initial == final


def test_smoke_payload_is_deterministic_for_fixed_seed() -> None:
    first = run_transformer_smoke_from_event_log(
        path=_FIXTURE,
        symbol="TESTUSDT",
        window_length=4,
        batch_size=2,
        epochs=1,
        seed=42,
    )
    second = run_transformer_smoke_from_event_log(
        path=_FIXTURE,
        symbol="TESTUSDT",
        window_length=4,
        batch_size=2,
        epochs=1,
        seed=42,
    )
    assert first["training_history"] == second["training_history"]
    assert first["synthetic_smoke_metrics"] == second["synthetic_smoke_metrics"]

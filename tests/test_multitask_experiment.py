"""Tests for the synthetic multi-task fine-tuning smoke runner."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from chronoslob.training.multitask_experiment import (  # noqa: E402
    run_multitask_smoke_from_event_log,
)
from chronoslob.utils.paths import project_root  # noqa: E402

_FIXTURE = (
    project_root() / "tests" / "fixtures" / "event_logs" / "synthetic_snapshots.jsonl"
)


def test_multitask_smoke_experiment_runs_and_returns_payload() -> None:
    result = run_multitask_smoke_from_event_log(
        path=_FIXTURE,
        symbol="TESTUSDT",
        window_length=4,
        batch_size=2,
        epochs=1,
        seed=42,
        max_levels_per_side=2,
    )

    assert result["path"] == str(_FIXTURE)
    assert result["symbol_filter"] == "TESTUSDT"
    assert result["input_record_count"] > 0
    assert result["tokenised_record_count"] > 0
    assert result["window_count"] > 0
    assert result["supervised_window_count"] > 0
    assert result["model_parameter_count"] > 0
    assert result["enabled_tasks"] == [
        "direction",
        "return_quantile",
        "volatility_regime",
        "spread_widening",
        "fill_probability",
        "adverse_selection",
    ]
    assert result["training_history"]
    assert result["final_train_loss"] is not None
    assert torch.tensor(result["final_train_loss"]).isfinite().item()


def test_multitask_smoke_payload_marks_run_as_synthetic_plumbing() -> None:
    result = run_multitask_smoke_from_event_log(
        path=_FIXTURE,
        symbol="TESTUSDT",
        window_length=4,
        batch_size=2,
        epochs=1,
    )

    assert "synthetic" in result["notes"].lower()
    assert "plumbing" in result["notes"].lower()
    forbidden_keys = {
        "sharpe",
        "pnl",
        "profit",
        "trade_count",
        "backtest",
        "live_metrics",
        "execution_simulation",
        "calibration",
    }
    assert forbidden_keys.isdisjoint(result.keys())


def test_valid_labels_and_loss_components_are_finite() -> None:
    result = run_multitask_smoke_from_event_log(
        path=_FIXTURE,
        symbol="TESTUSDT",
        window_length=4,
        batch_size=2,
        epochs=1,
    )

    valid_counts = result["valid_labels_per_task"]
    assert all(count > 0 for count in valid_counts.values())
    for name, value in result["final_train_loss_components"].items():
        assert torch.tensor(value).isfinite().item(), (
            f"component {name!r} is non-finite"
        )
    smoke_metrics = result["synthetic_smoke_metrics"]
    assert torch.tensor(smoke_metrics["loss"]).isfinite().item()
    for value in smoke_metrics["loss_components"].values():
        assert torch.tensor(value).isfinite().item()


def test_multitask_smoke_invalid_path_raises_file_not_found(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "does_not_exist.jsonl"

    with pytest.raises(FileNotFoundError):
        run_multitask_smoke_from_event_log(path=missing, window_length=4)


def test_multitask_smoke_zero_window_case_fails_clearly(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match=r"zero windows|empty"):
        run_multitask_smoke_from_event_log(path=empty, window_length=4)


def test_multitask_smoke_cpu_training_works() -> None:
    result = run_multitask_smoke_from_event_log(
        path=_FIXTURE,
        symbol="TESTUSDT",
        window_length=4,
        batch_size=2,
        epochs=1,
    )

    assert result["final_train_loss"] is not None
    assert torch.tensor(result["final_train_loss"]).isfinite().item()


def test_multitask_smoke_does_not_write_checkpoint_files(tmp_path: Path) -> None:
    before = sorted(tmp_path.iterdir())
    result = run_multitask_smoke_from_event_log(
        path=_FIXTURE,
        symbol="TESTUSDT",
        window_length=4,
        batch_size=2,
        epochs=1,
    )
    after = sorted(tmp_path.iterdir())

    assert before == after
    assert result["write_outputs"] is False

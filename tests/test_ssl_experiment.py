"""Tests for the synthetic SSL smoke experiment runner."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from chronoslob.training.ssl_experiment import (  # noqa: E402
    run_ssl_smoke_from_event_log,
)
from chronoslob.utils.paths import project_root  # noqa: E402

_FIXTURE = (
    project_root() / "tests" / "fixtures" / "event_logs" / "synthetic_snapshots.jsonl"
)


def test_ssl_smoke_experiment_runs_and_returns_payload() -> None:
    result = run_ssl_smoke_from_event_log(
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
    assert result["model_parameter_count"] > 0
    assert result["enabled_objectives"], "expected enabled SSL objectives"
    assert result["masked_fields"]
    assert result["next_fields"]
    assert result["training_history"], "expected at least one epoch"
    final = result["training_history"][-1]
    assert torch.tensor(final["train_loss"]).isfinite().item()
    assert result["final_train_loss"] is not None
    assert torch.tensor(result["final_train_loss"]).isfinite().item()
    assert isinstance(result["final_train_loss_components"], dict)
    assert "masked_field" in result["final_train_loss_components"]
    assert "next_field" in result["final_train_loss_components"]


def test_ssl_smoke_payload_marks_run_as_synthetic_plumbing() -> None:
    result = run_ssl_smoke_from_event_log(
        path=_FIXTURE,
        symbol="TESTUSDT",
        window_length=4,
        batch_size=2,
        epochs=1,
    )
    assert "synthetic" in result["notes"].lower()
    forbidden_keys = {
        "alpha",
        "sharpe",
        "pnl",
        "profit",
        "trade_count",
        "backtest",
        "live_metrics",
        "label_source",
    }
    assert forbidden_keys.isdisjoint(result.keys())


def test_ssl_smoke_loss_components_are_finite() -> None:
    result = run_ssl_smoke_from_event_log(
        path=_FIXTURE,
        symbol="TESTUSDT",
        window_length=4,
        batch_size=2,
        epochs=1,
    )
    smoke_metrics = result["synthetic_smoke_metrics"]
    assert "loss" in smoke_metrics
    assert "loss_components" in smoke_metrics
    assert torch.tensor(smoke_metrics["loss"]).isfinite().item()
    for name, value in smoke_metrics["loss_components"].items():
        assert torch.tensor(value).isfinite().item(), (
            f"component {name!r} is non-finite"
        )


def test_ssl_smoke_experiment_invalid_path_raises_file_not_found(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "does_not_exist.jsonl"
    with pytest.raises(FileNotFoundError):
        run_ssl_smoke_from_event_log(
            path=missing,
            window_length=4,
        )


def test_ssl_smoke_experiment_zero_window_case_raises_clear_error(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="zero windows"):
        run_ssl_smoke_from_event_log(
            path=empty,
            window_length=4,
        )


def test_ssl_smoke_does_not_write_files(tmp_path: Path) -> None:
    initial = sorted(tmp_path.iterdir())
    run_ssl_smoke_from_event_log(
        path=_FIXTURE,
        symbol="TESTUSDT",
        window_length=4,
        batch_size=2,
        epochs=1,
    )
    final = sorted(tmp_path.iterdir())
    assert initial == final


def test_ssl_smoke_is_deterministic_for_fixed_seed() -> None:
    first = run_ssl_smoke_from_event_log(
        path=_FIXTURE,
        symbol="TESTUSDT",
        window_length=4,
        batch_size=2,
        epochs=1,
        seed=42,
    )
    second = run_ssl_smoke_from_event_log(
        path=_FIXTURE,
        symbol="TESTUSDT",
        window_length=4,
        batch_size=2,
        epochs=1,
        seed=42,
    )
    assert first["training_history"] == second["training_history"]
    assert first["synthetic_smoke_metrics"] == second["synthetic_smoke_metrics"]


def test_ssl_smoke_runs_on_cpu_default() -> None:
    result = run_ssl_smoke_from_event_log(
        path=_FIXTURE,
        symbol="TESTUSDT",
        window_length=4,
        batch_size=2,
        epochs=1,
    )
    # Just make sure we got a finite loss out of CPU training.
    assert result["final_train_loss"] is not None
    assert torch.tensor(result["final_train_loss"]).isfinite().item()

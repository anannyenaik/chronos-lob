"""Tests for the FI-2010 SSL-v2 objective, data path and artefacts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from chronoslob.analysis.ssl_v2_analysis import analyse_ssl_v2_results
from chronoslob.experiments import fi2010_ssl_v2_benchmark as runner
from chronoslob.experiments.manifests import stable_json_dumps
from chronoslob.training.matrix_ssl_v2_datasets import (
    MatrixSSLV2WindowDataset,
    build_ssl_v2_auxiliary_label_matrix,
    build_ssl_v2_windows,
    fit_ssl_v2_auxiliary_label_spec,
    infer_ssl_v2_feature_groups,
)
from tests.test_fi2010_neural_runner import _write_tiny_neural_config


def _tiny_lob_frame(n_rows: int = 18) -> pd.DataFrame:
    rows = np.arange(n_rows, dtype=float)
    return pd.DataFrame(
        {
            "split": ["train"] * n_rows,
            "bid_price_1": 100.0 + rows,
            "bid_quantity_1": 10.0 + (rows % 3),
            "ask_price_1": 101.0 + rows + (rows % 2) * 0.2,
            "ask_quantity_1": 8.0 + (rows % 4),
            "depth_feature": rows / 10.0,
            "imbalance_feature": (rows % 5) / 5.0,
            "spread_feature": 1.0 + (rows % 2) * 0.1,
            "context_feature": np.sin(rows),
            "label_10": [1, 2, 3] * (n_rows // 3),
            "label_50": [1, 2, 3] * (n_rows // 3),
        }
    )


def test_ssl_v2_windows_do_not_cross_future_partition_boundary() -> None:
    windows = build_ssl_v2_windows(
        n_rows=12,
        window_length=3,
        allowed_indices=list(range(8)),
        future_offset=2,
    )
    assert windows
    for sample in windows:
        assert sample.future_index <= 7
        assert sample.future_index > sample.window_end
        assert set(range(sample.window_start, sample.window_end + 1)).issubset(set(range(8)))


def test_ssl_v2_auxiliary_bins_fit_on_train_pairs_only() -> None:
    frame = _tiny_lob_frame()
    train = list(range(10))
    spec_a = fit_ssl_v2_auxiliary_label_spec(
        frame,
        train_indices=train,
        future_offset=2,
        bucket_count=3,
    )
    corrupted = frame.copy()
    corrupted.loc[10:, "ask_price_1"] = 10000.0
    corrupted.loc[10:, "bid_price_1"] = -10000.0
    spec_b = fit_ssl_v2_auxiliary_label_spec(
        corrupted,
        train_indices=train,
        future_offset=2,
        bucket_count=3,
    )
    assert spec_a.volatility_edges == spec_b.volatility_edges
    assert spec_a.return_edges == spec_b.return_edges
    assert spec_a.imbalance_edges == spec_b.imbalance_edges
    assert max(spec_a.train_future_indices) <= 9


def test_ssl_v2_future_labels_are_strictly_future() -> None:
    frame = _tiny_lob_frame()
    spec = fit_ssl_v2_auxiliary_label_spec(
        frame,
        train_indices=list(range(12)),
        future_offset=2,
        bucket_count=3,
    )
    labels = build_ssl_v2_auxiliary_label_matrix(frame, spec)
    # Row 0 compares spread at row 2 against row 0, not against row 1 or itself.
    expected = int(
        (frame.loc[2, "ask_price_1"] - frame.loc[2, "bid_price_1"])
        > (frame.loc[0, "ask_price_1"] - frame.loc[0, "bid_price_1"])
    )
    assert labels["future_spread_widening"][0] == expected
    assert labels["future_spread_widening"][len(frame) - 1] == -100


def test_ssl_v2_structured_masking_is_deterministic() -> None:
    torch = pytest.importorskip("torch")
    frame = _tiny_lob_frame()
    feature_columns = [
        "bid_price_1",
        "bid_quantity_1",
        "ask_price_1",
        "ask_quantity_1",
        "imbalance_feature",
        "spread_feature",
        "context_feature",
    ]
    matrix = frame[feature_columns].to_numpy(dtype=float)
    train = list(range(12))
    windows = build_ssl_v2_windows(
        n_rows=len(frame),
        window_length=4,
        allowed_indices=train,
        future_offset=2,
    )
    spec = fit_ssl_v2_auxiliary_label_spec(
        frame,
        train_indices=train,
        future_offset=2,
        bucket_count=3,
    )
    labels = build_ssl_v2_auxiliary_label_matrix(frame, spec)
    dataset_a = MatrixSSLV2WindowDataset(
        matrix,
        windows,
        feature_groups=infer_ssl_v2_feature_groups(feature_columns),
        auxiliary_labels=labels,
        mask_probability=0.3,
        mask_value=0.0,
        ignore_index=-100,
        enable_contrastive=False,
        base_seed=17,
    )
    dataset_b = MatrixSSLV2WindowDataset(
        matrix,
        windows,
        feature_groups=infer_ssl_v2_feature_groups(feature_columns),
        auxiliary_labels=labels,
        mask_probability=0.3,
        mask_value=0.0,
        ignore_index=-100,
        enable_contrastive=False,
        base_seed=17,
    )
    first = dataset_a[0]
    second = dataset_b[0]
    assert torch.equal(first["mask"], second["mask"])
    assert torch.equal(first["x"], second["x"])


def test_ssl_v2_model_emits_loss_components_and_transfers_encoder() -> None:
    torch = pytest.importorskip("torch")
    from chronoslob.models.matrix_ssl import load_encoder_state_into_classifier
    from chronoslob.models.matrix_ssl_v2 import MatrixSSLV2Config, create_matrix_ssl_v2_model
    from chronoslob.models.matrix_transformer import (
        MatrixTransformerConfig,
        create_matrix_transformer,
    )

    model = create_matrix_ssl_v2_model(
        MatrixSSLV2Config(
            input_features=5,
            model_dim=8,
            num_heads=2,
            num_layers=1,
            feedforward_dim=16,
            max_sequence_length=4,
            enable_contrastive=True,
        )
    )
    x = torch.randn(3, 4, 5)
    mask = torch.zeros(3, 4, 5, dtype=torch.bool)
    mask[:, :1, :2] = True
    future_labels = {
        "future_spread_widening": torch.tensor([0, 1, 0], dtype=torch.long),
        "future_volatility": torch.tensor([0, 1, 2], dtype=torch.long),
        "future_return": torch.tensor([2, 1, 0], dtype=torch.long),
        "future_imbalance": torch.tensor([1, 1, 2], dtype=torch.long),
    }
    output = model(
        x,
        mask=mask,
        masked_target=x,
        future_labels=future_labels,
        contrastive_labels=torch.tensor([0, 0, 1], dtype=torch.long),
    )
    assert output.loss is not None
    for component in (
        "reconstruction",
        "future_spread_widening",
        "future_volatility",
        "future_return",
        "future_imbalance",
        "contrastive",
        "total",
    ):
        assert component in output.loss_components

    classifier = create_matrix_transformer(
        MatrixTransformerConfig(
            input_features=5,
            n_classes=3,
            model_dim=8,
            num_heads=2,
            num_layers=1,
            feedforward_dim=16,
            max_sequence_length=4,
        )
    )
    loaded = load_encoder_state_into_classifier(classifier, model.encoder_state_dict())
    assert loaded


def _fake_result_row(spec: runner.FI2010SSLV2RunSpec) -> dict[str, Any]:
    macro = {
        "supervised": 0.40,
        "masked_reconstruction": 0.42,
        "market_state_multitask": 0.45,
    }[spec.objective]
    mcc = {
        "supervised": 0.10,
        "masked_reconstruction": 0.11,
        "market_state_multitask": 0.14,
    }[spec.objective]
    ece = {
        "supervised": 0.08,
        "masked_reconstruction": 0.10,
        "market_state_multitask": 0.07,
    }[spec.objective]
    return {
        "fold": spec.fold,
        "horizon": spec.horizon,
        "seed": spec.seed,
        "lookback": spec.lookback,
        "model_family": spec.model_family,
        "pretraining_objective": spec.pretraining_objective,
        "accuracy": 0.55,
        "macro_f1": macro,
        "mcc": mcc,
        "ece": ece,
        "brier_score": 0.30 if spec.objective != "market_state_multitask" else 0.28,
        "nll": 1.1,
        "class_f1_down": 0.3,
        "class_f1_stationary": 0.4,
        "class_f1_up": 0.5,
        "checkpoint_hash": "hash",
        "prediction_file": "",
        "status": "completed",
        "run_id": spec.run_id,
        "run_dir": "",
        "architecture_hash": "arch",
        "preprocessing_hash": "prep",
    }


def _fake_training_row(spec: runner.FI2010SSLV2RunSpec) -> dict[str, Any]:
    return {
        "run_id": spec.run_id,
        "fold": spec.fold,
        "horizon": spec.horizon,
        "seed": spec.seed,
        "lookback": spec.lookback,
        "objective": spec.objective,
        "pretraining_objective": spec.pretraining_objective,
        "max_epochs": 2,
        "epochs_ran": 2,
        "best_epoch": 1,
        "monitored_metric": "validation_macro_f1",
        "best_validation_score": 0.4,
        "early_stopping_patience": 1,
        "early_stopped": True,
        "training_seconds": 0.01,
        "test_macro_f1": _fake_result_row(spec)["macro_f1"],
        "test_accuracy": 0.55,
        "test_mcc": _fake_result_row(spec)["mcc"],
        "test_ece": _fake_result_row(spec)["ece"],
        "status": "completed",
    }


def test_ssl_v2_runner_writes_required_artefacts(tmp_path: Path, monkeypatch: Any) -> None:
    def fake_v1_or_supervised(
        spec: runner.FI2010SSLV2RunSpec,
        **_: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return _fake_result_row(spec), _fake_training_row(spec)

    def fake_v2(
        spec: runner.FI2010SSLV2RunSpec,
        **_: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        return (
            _fake_result_row(spec),
            _fake_training_row(spec),
            [
                {
                    "epoch": 1,
                    "train_loss": 1.0,
                    "validation_loss": 0.9,
                    "train_reconstruction": 0.2,
                    "train_future_spread_widening": 0.3,
                    "train_future_volatility": 0.4,
                    "train_future_return": 0.5,
                    "train_future_imbalance": 0.6,
                    "train_contrastive": None,
                    "train_total": 1.0,
                    "validation_reconstruction": 0.2,
                    "validation_future_spread_widening": 0.3,
                    "validation_future_volatility": 0.4,
                    "validation_future_return": 0.5,
                    "validation_future_imbalance": 0.6,
                    "validation_contrastive": None,
                    "validation_total": 0.9,
                }
            ],
        )

    monkeypatch.setattr(runner, "_execute_v1_or_supervised_run_spec", fake_v1_or_supervised)
    monkeypatch.setattr(runner, "_execute_ssl_v2_run_spec", fake_v2)
    out = tmp_path / "ssl_v2"
    summary = runner.run_fi2010_ssl_v2_benchmark(
        config_path=_write_tiny_neural_config(tmp_path),
        processed_root=tmp_path / "processed",
        out_dir=out,
        baseline_source_dir=None,
        folds=[1],
        horizons=[10],
        seeds=[0],
        lookbacks=[2],
        objectives=["supervised", "masked_reconstruction", "market_state_multitask"],
        pretrain_epochs=1,
        max_epochs=2,
        patience=1,
        batch_size=4,
        import_existing_baselines=False,
    )
    assert summary.completed_run_count == 3
    expected = {
        "README.md",
        "summary.json",
        "results_summary.csv",
        "aggregate_summary.csv",
        "aggregate_summary.json",
        "ssl_v2_comparison.csv",
        "failures.csv",
        "missing_pairs.csv",
        "training_curves_summary.csv",
        "ssl_v2_loss_components.csv",
        "config_snapshot.json",
        "sha256_manifest.json",
    }
    assert expected.issubset({path.name for path in out.iterdir()})
    comparison = pd.read_csv(out / "ssl_v2_comparison.csv")
    assert set(comparison["ssl_objective"]) == {
        "masked_reconstruction",
        "market_state_multitask",
    }
    assert set(comparison["status"]) == {"matched"}


def test_ssl_v2_analysis_computes_deltas_and_claims(tmp_path: Path) -> None:
    source = tmp_path / "ssl_v2_source"
    source.mkdir()
    (source / "summary.json").write_text(
        stable_json_dumps(
            {
                "evidence_level": "partial_real",
                "scope_label": "tiny",
                "folds": [1],
                "horizons": [10],
                "seeds": [0],
                "lookbacks": [2],
                "objectives": ["supervised", "market_state_multitask"],
                "completed_run_count": 2,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame([_fake_result_row(runner.FI2010SSLV2RunSpec(1, 10, 0, 2, "supervised"))]).to_csv(
        source / "results_summary.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "horizon": 10,
                "lookback": 2,
                "model_family": "matrix_transformer",
                "pretraining_objective": "market_state_multitask",
                "completed_run_count": 1,
                "failed_run_count": 0,
                "mean_accuracy": 0.55,
                "std_accuracy": 0.0,
                "mean_macro_f1": 0.45,
                "std_macro_f1": 0.0,
                "mean_mcc": 0.14,
                "std_mcc": 0.0,
                "mean_ece": 0.07,
                "std_ece": 0.0,
                "mean_brier_score": 0.28,
                "mean_nll": 1.1,
            }
        ]
    ).to_csv(source / "aggregate_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "fold": 1,
                "horizon": 10,
                "seed": 0,
                "lookback": 2,
                "model_family": "matrix_transformer",
                "ssl_objective": "market_state_multitask",
                "comparison": "supervised_vs_market_state_multitask",
                "supervised_run_id": "base",
                "ssl_run_id": "v2",
                "supervised_macro_f1": 0.40,
                "ssl_macro_f1": 0.45,
                "delta_macro_f1": 0.05,
                "supervised_accuracy": 0.55,
                "ssl_accuracy": 0.56,
                "delta_accuracy": 0.01,
                "supervised_mcc": 0.10,
                "ssl_mcc": 0.14,
                "delta_mcc": 0.04,
                "supervised_ece": 0.08,
                "ssl_ece": 0.07,
                "delta_ece": -0.01,
                "supervised_brier_score": 0.30,
                "ssl_brier_score": 0.28,
                "delta_brier_score": -0.02,
                "supervised_nll": 1.2,
                "ssl_nll": 1.1,
                "delta_nll": -0.1,
                "macro_f1_outcome": "win",
                "ece_outcome": "win",
                "mcc_outcome": "win",
                "status": "matched",
                "reason": "",
            }
        ]
    ).to_csv(source / "ssl_v2_comparison.csv", index=False)
    pd.DataFrame(columns=["status"]).to_csv(source / "failures.csv", index=False)
    pd.DataFrame([{"epoch": 1, "train_total": 1.0}]).to_csv(
        source / "ssl_v2_loss_components.csv",
        index=False,
    )
    out = tmp_path / "analysis"
    summary = analyse_ssl_v2_results(source, out_dir=out)
    assert summary.ssl_v2_matched_rows == 1
    assert summary.claim_statuses["ssl_v2_predictive_improvement"] == "supported"
    assert summary.claim_statuses["ssl_v2_calibration_improvement"] == "supported"
    assert (out / "ssl_v2_analysis.md").is_file()
    assert (out / "ssl_v2_delta_by_horizon.csv").is_file()
    retained_summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert retained_summary["git_commit"]
    assert retained_summary["ssl_v2_dir"] == source.as_posix()
    assert retained_summary["out_dir"] == out.as_posix()


def _write_ssl_v2_analysis_source(
    source: Path,
    *,
    evidence_level: str = "complete_real",
    folds: Sequence[int] = (1, 2, 3, 4, 5),
    delta_macro_f1: float = 0.05,
    delta_mcc: float = 0.04,
    delta_ece: float = -0.01,
    delta_brier: float = -0.02,
) -> None:
    """Write a minimal but complete SSL-v2 benchmark source directory."""
    source.mkdir(parents=True, exist_ok=True)
    (source / "summary.json").write_text(
        stable_json_dumps(
            {
                "evidence_level": evidence_level,
                "scope_label": "test",
                "folds": list(folds),
                "horizons": [10, 50],
                "seeds": [0],
                "lookbacks": [50],
                "objectives": ["supervised", "market_state_multitask"],
                "completed_run_count": 2 * len(folds) * 2,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [_fake_result_row(runner.FI2010SSLV2RunSpec(1, 10, 0, 50, "supervised"))]
    ).to_csv(source / "results_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "horizon": 10,
                "lookback": 50,
                "model_family": "matrix_transformer",
                "pretraining_objective": "market_state_multitask",
                "completed_run_count": len(folds),
                "failed_run_count": 0,
                "mean_macro_f1": 0.45,
                "mean_mcc": 0.14,
                "mean_ece": 0.07,
                "mean_brier_score": 0.28,
                "mean_nll": 1.1,
            }
        ]
    ).to_csv(source / "aggregate_summary.csv", index=False)
    rows = []
    for fold in folds:
        for horizon in (10, 50):
            rows.append(
                {
                    "fold": fold,
                    "horizon": horizon,
                    "seed": 0,
                    "lookback": 50,
                    "model_family": "matrix_transformer",
                    "ssl_objective": "market_state_multitask",
                    "comparison": "supervised_vs_market_state_multitask",
                    "supervised_run_id": "base",
                    "ssl_run_id": "v2",
                    "supervised_macro_f1": 0.40,
                    "ssl_macro_f1": 0.40 + delta_macro_f1,
                    "delta_macro_f1": delta_macro_f1,
                    "supervised_accuracy": 0.55,
                    "ssl_accuracy": 0.55,
                    "delta_accuracy": 0.0,
                    "supervised_mcc": 0.10,
                    "ssl_mcc": 0.10 + delta_mcc,
                    "delta_mcc": delta_mcc,
                    "supervised_ece": 0.08,
                    "ssl_ece": 0.08 + delta_ece,
                    "delta_ece": delta_ece,
                    "supervised_brier_score": 0.30,
                    "ssl_brier_score": 0.30 + delta_brier,
                    "delta_brier_score": delta_brier,
                    "supervised_nll": 1.2,
                    "ssl_nll": 1.1,
                    "delta_nll": -0.1,
                    "macro_f1_outcome": "win" if delta_macro_f1 > 0 else "loss",
                    "ece_outcome": "win" if delta_ece < 0 else "loss",
                    "mcc_outcome": "win" if delta_mcc > 0 else "loss",
                    "status": "matched",
                    "reason": "",
                }
            )
    pd.DataFrame(rows).to_csv(source / "ssl_v2_comparison.csv", index=False)
    pd.DataFrame(columns=["status"]).to_csv(source / "failures.csv", index=False)


def test_ssl_v2_calibration_claim_requires_both_ece_and_brier(tmp_path: Path) -> None:
    # ECE improves but Brier worsens: calibration improvement must NOT be claimed.
    source = tmp_path / "src"
    _write_ssl_v2_analysis_source(source, delta_ece=-0.01, delta_brier=0.03)
    summary = analyse_ssl_v2_results(source, out_dir=tmp_path / "out")
    assert summary.claim_statuses["ssl_v2_calibration_improvement"] == "unsupported"


def test_ssl_v2_broad_improvement_never_supported(tmp_path: Path) -> None:
    # Even with strong positive predictive and calibration deltas across folds,
    # the broad-SSL-improvement claim stays unsupported.
    source = tmp_path / "src"
    _write_ssl_v2_analysis_source(
        source,
        delta_macro_f1=0.20,
        delta_mcc=0.20,
        delta_ece=-0.05,
        delta_brier=-0.05,
    )
    summary = analyse_ssl_v2_results(source, out_dir=tmp_path / "out")
    assert summary.claim_statuses["ssl_v2_predictive_improvement"] == "supported"
    assert summary.claim_statuses["ssl_v2_calibration_improvement"] == "supported"
    assert summary.claim_statuses["broad_ssl_improvement"] == "unsupported"


def test_ssl_v2_predictive_claim_not_supported_for_smoke(tmp_path: Path) -> None:
    source = tmp_path / "src"
    _write_ssl_v2_analysis_source(
        source,
        evidence_level="smoke_test_only",
        delta_macro_f1=0.20,
        delta_mcc=0.20,
    )
    summary = analyse_ssl_v2_results(source, out_dir=tmp_path / "out")
    assert summary.claim_statuses["ssl_v2_predictive_improvement"] == "unsupported"
    assert summary.claim_statuses["ssl_v2_evaluated"] == "needs_real_evidence"


def test_ssl_v2_complete_real_scope_label_reflects_actual_folds() -> None:
    evidence_level, scope_label = runner._scope_labels(
        folds=(1, 2, 3, 4, 5),
        horizons=(10, 50),
        seeds=(0,),
        lookbacks=(50,),
        objectives=("supervised", "market_state_multitask"),
        max_epochs=25,
        patience=5,
        smoke_test=False,
        completed=20,
        planned=20,
    )
    assert evidence_level == "complete_real"
    assert scope_label == "folds_1_2_3_4_5_h10_h50_complete_real"


def test_ssl_v2_partial_real_when_not_all_runs_complete() -> None:
    # A reduced or incomplete scope must never be silently promoted to complete.
    evidence_level, scope_label = runner._scope_labels(
        folds=(1, 2, 3, 4, 5),
        horizons=(10, 50),
        seeds=(0,),
        lookbacks=(50,),
        objectives=("supervised", "market_state_multitask"),
        max_epochs=25,
        patience=5,
        smoke_test=False,
        completed=18,
        planned=20,
    )
    assert evidence_level == "partial_real"
    assert scope_label == "limited_ssl_v2_partial_real_slice"

    fallback_level, _ = runner._scope_labels(
        folds=(1, 2, 3),
        horizons=(10, 50),
        seeds=(0,),
        lookbacks=(50,),
        objectives=("supervised", "market_state_multitask"),
        max_epochs=25,
        patience=5,
        smoke_test=False,
        completed=12,
        planned=12,
    )
    assert fallback_level == "complete_real"


def test_ssl_v2_multiseed_complete_real_scope_label() -> None:
    evidence_level, scope_label = runner._scope_labels(
        folds=(1, 2, 3, 4, 5),
        horizons=(10, 50),
        seeds=(0, 1, 2),
        lookbacks=(50,),
        objectives=("supervised", "market_state_multitask"),
        max_epochs=25,
        patience=5,
        smoke_test=False,
        completed=60,
        planned=60,
    )
    assert evidence_level == "complete_real"
    assert scope_label == "folds_1_2_3_4_5_h10_h50_seeds_0_1_2_complete_real"


def test_ssl_v2_multiseed_partial_when_not_all_runs_complete() -> None:
    evidence_level, scope_label = runner._scope_labels(
        folds=(1, 2, 3, 4, 5),
        horizons=(10, 50),
        seeds=(0, 1, 2),
        lookbacks=(50,),
        objectives=("supervised", "market_state_multitask"),
        max_epochs=25,
        patience=5,
        smoke_test=False,
        completed=59,
        planned=60,
    )
    assert evidence_level == "partial_real"
    assert scope_label == "limited_ssl_v2_partial_real_slice"


def test_ssl_v2_unrecognised_seed_set_never_complete_real() -> None:
    # A seed set that is neither {0} nor {0,1,2} must stay partial_real even when
    # every planned run completes, so an ad-hoc sweep is never promoted silently.
    evidence_level, scope_label = runner._scope_labels(
        folds=(1, 2, 3, 4, 5),
        horizons=(10, 50),
        seeds=(0, 1),
        lookbacks=(50,),
        objectives=("supervised", "market_state_multitask"),
        max_epochs=25,
        patience=5,
        smoke_test=False,
        completed=40,
        planned=40,
    )
    assert evidence_level == "partial_real"
    assert scope_label == "limited_ssl_v2_partial_real_slice"


def _write_run_predictions(
    source: Path,
    *,
    fold: int,
    horizon: int,
    seed: int,
    lookback: int,
    objective: str,
    rng: np.random.Generator,
    n_rows: int = 80,
) -> None:
    """Write a tiny benchmark run predictions.csv at the canonical run path."""
    run_dir = (
        source
        / "runs"
        / f"fold_{fold}"
        / f"horizon_{horizon}"
        / f"seed_{seed}"
        / f"lookback_{lookback}"
        / objective
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    y_true = rng.integers(1, 4, size=n_rows)
    y_pred = y_true.copy()
    # SSL-v2 is made slightly more accurate than the supervised baseline.
    error_rate = 0.45 if objective == "supervised" else 0.30
    flip = rng.random(n_rows) < error_rate
    y_pred[flip] = rng.integers(1, 4, size=int(flip.sum()))
    confidence = rng.uniform(0.34, 0.99, size=n_rows)
    pd.DataFrame(
        {
            "split": "test",
            "y_true": y_true,
            "y_pred": y_pred,
            "confidence": confidence,
        }
    ).to_csv(run_dir / "predictions.csv", index=False)


def test_ssl_v2_confidence_filtered_diagnostics_from_predictions(tmp_path: Path) -> None:
    source = tmp_path / "src"
    _write_ssl_v2_analysis_source(source, folds=(1, 2))
    rng = np.random.default_rng(7)
    for fold in (1, 2):
        for horizon in (10, 50):
            for objective in ("supervised", "market_state_multitask"):
                _write_run_predictions(
                    source,
                    fold=fold,
                    horizon=horizon,
                    seed=0,
                    lookback=50,
                    objective=objective,
                    rng=rng,
                )
    out = tmp_path / "out"
    summary = analyse_ssl_v2_results(source, out_dir=out)
    assert summary.execution_proxy_available is True
    assert summary.confidence_filtered_rows > 0

    confidence = pd.read_csv(out / "ssl_v2_confidence_filtered.csv")
    assert set(confidence["grouping"]) == {"by_horizon", "overall"}
    thresholds = sorted({round(float(value), 2) for value in confidence["threshold"]})
    assert thresholds == [0.33, 0.5, 0.7, 0.85, 0.95]
    overall = confidence[confidence["grouping"] == "overall"]
    # Active fraction must fall (and abstention rise) as the threshold tightens.
    by_threshold = overall.sort_values("threshold")
    active = by_threshold["mean_ssl_active_fraction"].tolist()
    assert active == sorted(active, reverse=True)
    assert (by_threshold["mean_ssl_active_fraction"].iloc[0]) == 1.0

    # The grouped delta artefacts the multi-seed scope requires must all exist.
    for name in (
        "ssl_v2_delta_overall.csv",
        "ssl_v2_delta_by_seed.csv",
        "ssl_v2_delta_by_fold_horizon.csv",
    ):
        assert (out / name).is_file()

    note = json.loads((out / "ssl_v2_execution_proxy.json").read_text(encoding="utf-8"))
    assert note["active_fraction_reported"] is True
    assert note["prediction_level_artefacts_required"] is True
    assert note["computable_from_retained_artefacts"] is False
    assert note["execution_hook"] == "execution_centrepiece"
    assert set(note["deferred_proxies"]) == {
        "turnover_proxy",
        "cost_adjusted_proxy",
        "latency_sensitivity",
        "adverse_selection_proxy",
    }


def test_ssl_v2_confidence_filtered_absent_without_predictions(tmp_path: Path) -> None:
    # Retained summary-light artefacts have no per-run predictions: diagnostics must
    # degrade gracefully and flag that prediction-level artefacts are required.
    source = tmp_path / "src"
    _write_ssl_v2_analysis_source(source, folds=(1,))
    out = tmp_path / "out"
    summary = analyse_ssl_v2_results(source, out_dir=out)
    assert summary.execution_proxy_available is False
    assert summary.confidence_filtered_rows == 0
    assert pd.read_csv(out / "ssl_v2_confidence_filtered.csv").empty
    note = json.loads((out / "ssl_v2_execution_proxy.json").read_text(encoding="utf-8"))
    assert note["active_fraction_reported"] is False
    assert note["prediction_level_artefacts_required"] is True

"""Tests for the FI-2010 matrix self-supervised model, datasets and loop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chronoslob.training.matrix_ssl_datasets import (
    MatrixSSLWindowSample,
    bucketise_matrix,
    build_contiguous_windows,
    fit_feature_bucket_edges,
)


def _synthetic_matrix(n_rows: int = 24, n_features: int = 6) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((n_rows, n_features))


def _windows_in(
    allowed: list[int],
    *,
    n_rows: int,
    window_length: int,
) -> list[MatrixSSLWindowSample]:
    return build_contiguous_windows(
        n_rows=n_rows,
        window_length=window_length,
        allowed_indices=allowed,
    )


# ---------------------------------------------------------------------------
# Data-path leakage and bucketisation (do not require torch)
# ---------------------------------------------------------------------------


def test_window_builder_excludes_out_of_partition_rows() -> None:
    train = list(range(0, 10))
    windows = _windows_in(train, n_rows=24, window_length=3)
    assert windows
    for window in windows:
        rows = set(range(window.window_start, window.window_end + 1))
        # Every row of every window must live inside the training partition.
        assert rows.issubset(set(train))
    # The first allowed window cannot start before row zero.
    assert min(window.window_start for window in windows) == 0
    # No window may reach into validation/test rows (>= 10).
    assert max(window.window_end for window in windows) <= 9


def test_window_builder_rejects_cross_boundary_windows() -> None:
    # A gap in the allowed set must prevent a window from spanning it.
    allowed = [0, 1, 2, 5, 6, 7]
    windows = _windows_in(allowed, n_rows=10, window_length=3)
    ends = {window.window_end for window in windows}
    # Window ending at row 5 would need rows {3,4,5}; rows 3,4 are excluded.
    assert 5 not in ends
    assert {2, 7}.issubset(ends)


def test_bucket_edges_fit_on_train_rows_only() -> None:
    matrix = _synthetic_matrix()
    train = list(range(0, 14))
    edges_a = fit_feature_bucket_edges(matrix, train_indices=train, bucket_count=3)

    # Corrupting validation/test rows must not change train-fitted edges.
    corrupted = matrix.copy()
    corrupted[14:, :] = 1e3
    edges_b = fit_feature_bucket_edges(corrupted, train_indices=train, bucket_count=3)
    assert edges_a == edges_b


def test_bucketise_matrix_values_in_range() -> None:
    matrix = _synthetic_matrix()
    train = list(range(0, 14))
    edges = fit_feature_bucket_edges(matrix, train_indices=train, bucket_count=4)
    buckets = bucketise_matrix(matrix, feature_edges=edges, bucket_count=4)
    assert buckets.shape == matrix.shape
    assert int(buckets.min()) >= 0
    assert int(buckets.max()) <= 3


# ---------------------------------------------------------------------------
# Model, dataset and pretraining loop (require torch)
# ---------------------------------------------------------------------------


def _build_dataset(matrix, windows, *, masked: bool, nxt: bool, buckets=None):
    from chronoslob.training.matrix_ssl_datasets import MatrixSSLWindowDataset

    return MatrixSSLWindowDataset(
        matrix,
        windows,
        bucket_matrix=buckets,
        mask_probability=0.3 if masked else 0.0,
        mask_value=0.0,
        bucket_count=3,
        ignore_index=-100,
        enable_masked_field=masked,
        enable_next_field=nxt,
        base_seed=7,
    )


def test_masked_field_ssl_smoke_runs_on_synthetic_fixture() -> None:
    pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from chronoslob.models.matrix_ssl import MatrixSSLConfig, create_matrix_ssl_model
    from chronoslob.training.matrix_ssl_datasets import collate_matrix_ssl_windows
    from chronoslob.training.matrix_ssl_experiment import (
        MatrixSSLTrainingConfig,
        fit_matrix_ssl,
    )

    matrix = _synthetic_matrix()
    train = list(range(0, 14))
    windows = _windows_in(train, n_rows=matrix.shape[0], window_length=3)
    dataset = _build_dataset(matrix, windows, masked=True, nxt=False)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_matrix_ssl_windows)

    config = MatrixSSLConfig(
        input_features=matrix.shape[1],
        model_dim=8,
        num_heads=2,
        num_layers=1,
        feedforward_dim=16,
        max_sequence_length=3,
        enable_masked_field=True,
        enable_next_field=False,
        mask_probability=0.3,
    )
    model = create_matrix_ssl_model(config)
    history = fit_matrix_ssl(
        model,
        loader,
        None,
        MatrixSSLTrainingConfig(epochs=2, seed=11),
    )
    assert len(history) == 2
    assert np.isfinite(history[-1].train_loss)
    assert "masked_field" in history[-1].train_loss_components
    assert "next_field" not in history[-1].train_loss_components


def test_next_field_ssl_smoke_runs_on_synthetic_fixture() -> None:
    pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from chronoslob.models.matrix_ssl import MatrixSSLConfig, create_matrix_ssl_model
    from chronoslob.training.matrix_ssl_datasets import collate_matrix_ssl_windows
    from chronoslob.training.matrix_ssl_experiment import (
        MatrixSSLTrainingConfig,
        evaluate_matrix_ssl,
        fit_matrix_ssl,
    )

    matrix = _synthetic_matrix()
    train = list(range(0, 14))
    validation = list(range(14, 19))
    edges = fit_feature_bucket_edges(matrix, train_indices=train, bucket_count=3)
    buckets = bucketise_matrix(matrix, feature_edges=edges, bucket_count=3)

    train_windows = _windows_in(train, n_rows=matrix.shape[0], window_length=3)
    val_windows = _windows_in(validation, n_rows=matrix.shape[0], window_length=3)
    train_loader = DataLoader(
        _build_dataset(matrix, train_windows, masked=False, nxt=True, buckets=buckets),
        batch_size=4,
        collate_fn=collate_matrix_ssl_windows,
    )
    val_loader = DataLoader(
        _build_dataset(matrix, val_windows, masked=False, nxt=True, buckets=buckets),
        batch_size=4,
        collate_fn=collate_matrix_ssl_windows,
    )

    config = MatrixSSLConfig(
        input_features=matrix.shape[1],
        model_dim=8,
        num_heads=2,
        num_layers=1,
        feedforward_dim=16,
        max_sequence_length=3,
        enable_masked_field=False,
        enable_next_field=True,
        mask_probability=0.0,
        next_field_bucket_count=3,
    )
    model = create_matrix_ssl_model(config)
    history = fit_matrix_ssl(
        model,
        train_loader,
        val_loader,
        MatrixSSLTrainingConfig(epochs=1, seed=5),
    )
    assert "next_field" in history[-1].train_loss_components
    assert "masked_field" not in history[-1].train_loss_components
    assert history[-1].validation_loss is not None
    evaluation = evaluate_matrix_ssl(model, val_loader, device="cpu")
    assert np.isfinite(evaluation["loss"])


def test_pretrained_encoder_checkpoint_loads_into_finetuning_model(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    import torch
    from torch.utils.data import DataLoader

    from chronoslob.models.matrix_ssl import (
        MatrixSSLConfig,
        create_matrix_ssl_model,
        load_encoder_state_into_classifier,
    )
    from chronoslob.models.matrix_transformer import (
        MatrixTransformerConfig,
        create_matrix_transformer,
    )
    from chronoslob.training.matrix_ssl_datasets import collate_matrix_ssl_windows
    from chronoslob.training.matrix_ssl_experiment import (
        MatrixSSLTrainingConfig,
        fit_matrix_ssl,
        load_pretrained_encoder_state,
        save_pretrained_encoder,
    )

    matrix = _synthetic_matrix()
    train = list(range(0, 14))
    windows = _windows_in(train, n_rows=matrix.shape[0], window_length=3)
    loader = DataLoader(
        _build_dataset(matrix, windows, masked=True, nxt=False),
        batch_size=4,
        collate_fn=collate_matrix_ssl_windows,
    )
    config = MatrixSSLConfig(
        input_features=matrix.shape[1],
        model_dim=8,
        num_heads=2,
        num_layers=1,
        feedforward_dim=16,
        max_sequence_length=3,
        enable_masked_field=True,
        enable_next_field=False,
        mask_probability=0.3,
    )
    model = create_matrix_ssl_model(config)
    fit_matrix_ssl(model, loader, None, MatrixSSLTrainingConfig(epochs=1, seed=3))

    checkpoint = tmp_path / "encoder.pt"
    save_pretrained_encoder(model, checkpoint, metadata={"smoke": True})
    assert checkpoint.is_file()
    encoder_state = load_pretrained_encoder_state(checkpoint)

    classifier = create_matrix_transformer(
        MatrixTransformerConfig(
            input_features=matrix.shape[1],
            n_classes=3,
            model_dim=8,
            num_heads=2,
            num_layers=1,
            feedforward_dim=16,
            max_sequence_length=3,
        )
    )
    loaded_keys = load_encoder_state_into_classifier(classifier, encoder_state)
    assert loaded_keys
    # Every transferred weight must now equal the pretrained value.
    classifier_state = classifier.state_dict()
    for key in loaded_keys:
        assert torch.allclose(classifier_state[key], encoder_state[key])
    # The fine-tuning classifier must still run a forward pass.
    logits = classifier(torch.zeros(2, 3, matrix.shape[1]))
    assert tuple(logits.shape) == (2, 3)


def test_encoder_transfer_rejects_architecture_mismatch(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from chronoslob.models.matrix_ssl import (
        MatrixSSLConfig,
        create_matrix_ssl_model,
        load_encoder_state_into_classifier,
    )
    from chronoslob.models.matrix_transformer import (
        MatrixTransformerConfig,
        create_matrix_transformer,
    )

    model = create_matrix_ssl_model(
        MatrixSSLConfig(
            input_features=6,
            model_dim=8,
            num_heads=2,
            num_layers=1,
            feedforward_dim=16,
            max_sequence_length=3,
        )
    )
    encoder_state = model.encoder_state_dict()
    mismatched = create_matrix_transformer(
        MatrixTransformerConfig(
            input_features=6,
            n_classes=3,
            model_dim=16,  # different model_dim
            num_heads=2,
            num_layers=1,
            feedforward_dim=16,
            max_sequence_length=3,
        )
    )
    with pytest.raises(ValueError, match="shape"):
        load_encoder_state_into_classifier(mismatched, encoder_state)


def test_matrix_ssl_config_rejects_zero_mask_probability_when_masked() -> None:
    from chronoslob.models.matrix_ssl import MatrixSSLConfig

    with pytest.raises(ValueError, match="mask_probability must be positive"):
        MatrixSSLConfig(
            input_features=6,
            enable_masked_field=True,
            enable_next_field=False,
            mask_probability=0.0,
        )

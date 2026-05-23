"""Training, validation and experiment registry utilities."""

from chronoslob.training.artifacts import create_run_directory, safe_run_name, write_json
from chronoslob.training.baseline_experiment import (
    BaselineExperimentConfig,
    BaselinePreprocessingConfig,
    BaselineSplitConfig,
    create_default_baseline_configs,
    run_baseline_experiment,
)
from chronoslob.training.batching import (
    collate_fixed_length_batch,
    collate_variable_length_batch,
    pad_variable_length_sequences,
)
from chronoslob.training.config import load_yaml_config, resolve_config_path
from chronoslob.training.dataloaders import (
    DataLoaderConfig,
    build_dataloaders_for_split,
    create_sequence_dataloader,
)
from chronoslob.training.datasets import (
    SequenceDataset,
    SequenceSampleIndex,
    SequenceWindowConfig,
    TorchSequenceStandardiser,
    build_sequence_indices,
    encode_target_values,
    infer_torch_feature_columns,
    torch_is_available,
)
from chronoslob.training.evaluate import evaluate_classifier
from chronoslob.training.experiment import (
    ExperimentMetadata,
    create_experiment_metadata,
    get_git_commit,
    initialise_experiment_run,
)
from chronoslob.training.metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
    confusion_matrix_as_dict,
)
from chronoslob.training.splitters import (
    PurgedEmbargoConfig,
    SplitIndices,
    TemporalSplitConfig,
    TrainOnlyQuantileBinner,
    WalkForwardSplitConfig,
    apply_purge_and_embargo,
    apply_row_embargo,
    intervals_overlap,
    label_horizon_end_indices_from_rows,
    make_label_horizon_end_indices_from_frame,
    purge_overlapping_train_indices,
    temporal_train_validation_test_split,
    walk_forward_splits,
)
from chronoslob.training.torch_experiment import (
    DeepLOBExperimentConfig,
    run_deeplob_experiment,
    run_deeplob_smoke_from_fi2010_fixture,
)
from chronoslob.training.torch_training import (
    TorchEpochResult,
    TorchTrainingConfig,
    evaluate_torch_classifier,
    fit_torch_classifier,
    set_torch_deterministic,
    train_one_epoch,
)

__all__ = [
    "BaselineExperimentConfig",
    "BaselinePreprocessingConfig",
    "BaselineSplitConfig",
    "ClassificationMetrics",
    "DataLoaderConfig",
    "DeepLOBExperimentConfig",
    "ExperimentMetadata",
    "PurgedEmbargoConfig",
    "SequenceDataset",
    "SequenceSampleIndex",
    "SequenceWindowConfig",
    "SplitIndices",
    "TemporalSplitConfig",
    "TorchEpochResult",
    "TorchSequenceStandardiser",
    "TorchTrainingConfig",
    "TrainOnlyQuantileBinner",
    "WalkForwardSplitConfig",
    "apply_purge_and_embargo",
    "apply_row_embargo",
    "build_dataloaders_for_split",
    "build_sequence_indices",
    "collate_fixed_length_batch",
    "collate_variable_length_batch",
    "compute_classification_metrics",
    "confusion_matrix_as_dict",
    "create_default_baseline_configs",
    "create_experiment_metadata",
    "create_run_directory",
    "create_sequence_dataloader",
    "encode_target_values",
    "evaluate_classifier",
    "evaluate_torch_classifier",
    "fit_torch_classifier",
    "get_git_commit",
    "infer_torch_feature_columns",
    "initialise_experiment_run",
    "intervals_overlap",
    "label_horizon_end_indices_from_rows",
    "load_yaml_config",
    "make_label_horizon_end_indices_from_frame",
    "pad_variable_length_sequences",
    "purge_overlapping_train_indices",
    "resolve_config_path",
    "run_baseline_experiment",
    "run_deeplob_experiment",
    "run_deeplob_smoke_from_fi2010_fixture",
    "safe_run_name",
    "set_torch_deterministic",
    "temporal_train_validation_test_split",
    "torch_is_available",
    "train_one_epoch",
    "walk_forward_splits",
    "write_json",
]

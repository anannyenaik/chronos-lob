"""Typer application and public command registration."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from .analysis import (
    _analyse_fi2010_execution_v3_impl,
    _analyse_fi2010_feature_ablations_impl,
    _analyse_fi2010_ssl_results_impl,
    _analyse_fi2010_ssl_v2_results_impl,
    _analyse_fi2010_uncertainty_impl,
    _build_execution_centrepiece_impl,
    _build_paper_plots_impl,
    _build_paper_report_impl,
    _inspect_paper_experiment_impl,
    _inspect_paper_report_impl,
    _inspect_system_benchmarks_impl,
    _run_paper_ablations_impl,
    _run_paper_experiment_impl,
    _run_system_benchmarks_impl,
)
from .common import (
    _doctor_impl,
    _parse_fold_selection,
    _parse_int_selection,
    _parse_model_selection,
    _parse_neural_fold_selection,
    _version_impl,
)
from .data import (
    _convert_fi2010_official_impl,
    _event_log_to_features_impl,
    _init_run_impl,
    _inspect_baselines_impl,
    _inspect_event_log_impl,
    _inspect_event_tokens_impl,
    _inspect_experiment_artifacts_impl,
    _inspect_features_fi2010_impl,
    _inspect_fi2010_impl,
    _inspect_labels_fi2010_impl,
    _inspect_split_impl,
    _inspect_torch_dataset_impl,
    _prepare_fi2010_benchmark_impl,
    _run_baseline_smoke_impl,
    _verify_fi2010_local_impl,
)
from .fi2010 import (
    _audit_fi2010_features_impl,
    _build_fi2010_ablation_figures_impl,
    _build_fi2010_execution_v3_impl,
    _inspect_fi2010_multifold_impl,
    _prepare_fi2010_multifold_impl,
    _run_fi2010_brutal_ablations_impl,
    _run_fi2010_execution_v2_impl,
    _run_fi2010_feature_ablations_impl,
    _run_fi2010_multifold_classical_impl,
)
from .neural import (
    _build_fi2010_figures_impl,
    _inspect_fi2010_neural_plan_impl,
    _run_fi2010_neural_benchmark_impl,
    _run_fi2010_neural_full_grid_impl,
    _run_fi2010_neural_proper_training_subset_impl,
    _run_fi2010_ssl_neural_benchmark_impl,
    _run_fi2010_ssl_v2_benchmark_impl,
)
from .replay import (
    _inspect_binance_replay_impl,
    _replay_binance_l2_sample_impl,
    _run_synthetic_lob_benchmark_impl,
)
from .synthetic import (
    _inspect_analysis_impl,
    _inspect_calibration_impl,
    _inspect_deeplob_impl,
    _inspect_execution_validation_impl,
    _inspect_multitask_impl,
    _inspect_ssl_impl,
    _inspect_transformer_impl,
    _run_calibration_smoke_impl,
    _run_deeplob_smoke_impl,
    _run_execution_validation_smoke_impl,
    _run_multitask_smoke_impl,
    _run_robustness_analysis_smoke_impl,
    _run_ssl_smoke_impl,
    _run_transformer_smoke_impl,
)

_REUSE_COMPLETED_FLAG = "--resume"
_NO_REUSE_COMPLETED_FLAG = "--no-resume"

app = typer.Typer(
    add_completion=False,
    help="ChronosLOB limit order book research utilities.",
    no_args_is_help=True,
)


def version() -> None:
    """Print the installed ChronosLOB version."""
    _version_impl()


def doctor() -> None:
    """Print a lightweight environment and repository check."""
    _doctor_impl()


_INSPECT_PATH_OPTION = typer.Option(
    ...,
    "--path",
    help="Path to the local FI-2010-style file.",
)

_EVENT_LOG_PATH_OPTION = typer.Option(
    ...,
    "--path",
    help="Path to the local canonical event-log JSONL file.",
)

_EVENT_TOKENS_SYMBOL_OPTION = typer.Option(
    None,
    "--symbol",
    help="Optional symbol filter for event-token inspection.",
)

_EVENT_TOKENS_WINDOW_LENGTH_OPTION = typer.Option(
    8,
    "--window-length",
    help="Fixed token-window length used for inspection.",
)

_EVENT_TOKENS_MAX_LEVELS_OPTION = typer.Option(
    2,
    "--max-levels-per-side",
    help="Maximum snapshot levels per side to tokenise.",
)

_EVENT_TOKENS_INCLUDE_EOS_OPTION = typer.Option(
    False,
    "--include-eos/--no-include-eos",
    help="Append one [EOS] record to the inspected token sequence.",
)

_INSPECT_TIMESTAMP_OPTION = typer.Option(
    "timestamp",
    "--timestamp-column",
    help="Name of the timestamp column, if any.",
)

_INSPECT_SPLIT_OPTION = typer.Option(
    "split",
    "--split-column",
    help="Name of the split column, if any.",
)

_INSPECT_LEVEL_COUNT_OPTION = typer.Option(
    2,
    "--price-level-count",
    help="Expected number of LOB levels per side.",
)

_INSPECT_NO_TIMESTAMP_OPTION = typer.Option(
    False,
    "--no-timestamp-column",
    help="Treat the file as having no timestamp column.",
)

_INSPECT_NO_SPLIT_OPTION = typer.Option(
    False,
    "--no-split-column",
    help="Treat the file as having no split column.",
)

_EXPERIMENT_ARTIFACTS_EXPERIMENT_OPTION = typer.Option(
    ...,
    "--experiment",
    help="Path to the experiment directory to inspect.",
)

_PREPARE_FI2010_CONFIG_OPTION = typer.Option(
    ...,
    "--config",
    help="Path to the FI-2010 benchmark preparation YAML config.",
)

_PREPARE_FI2010_DATA_PATH_OPTION = typer.Option(
    ...,
    "--data-path",
    help="Local FI-2010-style file path supplied by the user.",
)

_PREPARE_FI2010_OUT_OPTION = typer.Option(
    ...,
    "--out",
    help="Output directory for FI-2010 preparation artefacts.",
)

_VERIFY_FI2010_DATA_PATH_OPTION = typer.Option(
    ...,
    "--data-path",
    help="Path to the local FI-2010 file to inspect.",
)

_CONVERT_FI2010_INPUT_OPTION = typer.Option(
    ...,
    "--input",
    help="Path to a single official FI-2010 .txt matrix file.",
)

_CONVERT_FI2010_OUTPUT_OPTION = typer.Option(
    ...,
    "--output",
    help="Destination CSV path for the converted FI-2010 file.",
)

_CONVERT_FI2010_SPLIT_OPTION = typer.Option(
    None,
    "--split",
    help=("Optional split label written to a 'split' column (train or test)."),
)

_CONVERT_FI2010_OVERWRITE_OPTION = typer.Option(
    False,
    "--overwrite",
    help="Replace the output file if it already exists.",
)

_MULTIFOLD_CONFIG_OPTION = typer.Option(
    ...,
    "--config",
    help="Path to the multi-fold preparation YAML config.",
)

_MULTIFOLD_EXTRACTED_ROOT_OPTION = typer.Option(
    ...,
    "--extracted-root",
    help=("Local extracted FI-2010 dataset root (e.g. the BenchmarkDatasets/ directory)."),
)

_MULTIFOLD_PROCESSED_ROOT_OPTION = typer.Option(
    None,
    "--processed-root",
    help=("Local processed CSV root. Defaults to the value in the config."),
)

_MULTIFOLD_OUT_OPTION = typer.Option(
    ...,
    "--out",
    help="Output directory for the multi-fold preparation artefacts.",
)

_MULTIFOLD_FOLDS_OPTION = typer.Option(
    "all",
    "--folds",
    help="'all' or a comma-separated list of fold integers.",
)

_MULTIFOLD_OVERWRITE_OPTION = typer.Option(
    False,
    "--overwrite",
    help=("Replace existing combined CSVs, manifests and summary.json."),
)

_RUN_MULTIFOLD_CLASSICAL_CONFIG_OPTION = typer.Option(
    ...,
    "--config",
    help="Path to the FI-2010 multi-fold YAML config.",
)

_RUN_MULTIFOLD_CLASSICAL_PROCESSED_ROOT_OPTION = typer.Option(
    None,
    "--processed-root",
    help=("Root containing prepared fold CSV files. Defaults to the value in the config."),
)

_RUN_MULTIFOLD_CLASSICAL_OUT_OPTION = typer.Option(
    ...,
    "--out",
    help="Output directory for multi-fold classical artefacts.",
)

_RUN_MULTIFOLD_CLASSICAL_MODELS_OPTION = typer.Option(
    None,
    "--models",
    help=("Comma-separated classical model list. Defaults to the classical list in the config."),
)

_RUN_MULTIFOLD_CLASSICAL_FOLDS_OPTION = typer.Option(
    "all",
    "--folds",
    help="'all' or a comma-separated list of fold integers.",
)

_RUN_MULTIFOLD_CLASSICAL_OVERWRITE_OPTION = typer.Option(
    False,
    "--overwrite",
    help="Replace the output directory if it already exists.",
)

_NEURAL_PLAN_CONFIG_OPTION = typer.Option(
    ...,
    "--config",
    help="Path to the FI-2010 neural benchmark YAML config.",
)

_NEURAL_PLAN_FOLDS_OPTION = typer.Option(
    "all",
    "--folds",
    help="'all' or a comma-separated list of fold integers.",
)

_NEURAL_PLAN_MODELS_OPTION = typer.Option(
    "all",
    "--models",
    help="'all' or a comma-separated neural model list.",
)

_RUN_NEURAL_CONFIG_OPTION = typer.Option(
    ...,
    "--config",
    help="Path to the FI-2010 neural benchmark YAML config.",
)

_RUN_NEURAL_PROCESSED_ROOT_OPTION = typer.Option(
    ...,
    "--processed-root",
    help="Root containing prepared fold CSV files.",
)

_RUN_NEURAL_OUT_OPTION = typer.Option(
    ...,
    "--out",
    help="Output directory for neural benchmark artefacts.",
)

_RUN_NEURAL_FOLDS_OPTION = typer.Option(
    "fold_1",
    "--folds",
    help="'all' or a comma-separated list such as fold_1,fold_2.",
)

_RUN_NEURAL_MODELS_OPTION = typer.Option(
    "deeplob_style",
    "--models",
    help="'all' or a comma-separated neural model list.",
)

_RUN_NEURAL_SEEDS_OPTION = typer.Option(
    "0",
    "--seeds",
    help="'all' or a comma-separated list of non-negative seeds.",
)

_RUN_NEURAL_LOOKBACKS_OPTION = typer.Option(
    "20",
    "--lookbacks",
    help="'all' or a comma-separated list of positive lookbacks.",
)

_RUN_NEURAL_MAX_EPOCHS_OPTION = typer.Option(
    1,
    "--max-epochs",
    help="Maximum training epochs. Default is smoke-level.",
)

_RUN_NEURAL_OVERWRITE_OPTION = typer.Option(
    False,
    "--overwrite",
    help="Replace the output directory if it already exists.",
)

_RUN_NEURAL_FAIL_FAST_OPTION = typer.Option(
    False,
    "--fail-fast",
    help="Stop at the first run failure.",
)

_RUN_NEURAL_WRITE_PREDICTIONS_OPTION = typer.Option(
    False,
    "--write-full-predictions",
    help="Write per-run row-level predictions.",
)

_RUN_NEURAL_WRITE_CHECKPOINTS_OPTION = typer.Option(
    False,
    "--write-checkpoints",
    help="Write best-model checkpoints.",
)

_RUN_NEURAL_ALLOW_FULL_OPTION = typer.Option(
    False,
    "--allow-full-benchmark",
    help="Allow the complete configured benchmark grid.",
)

_RUN_SSL_CONFIG_OPTION = typer.Option(
    ...,
    "--config",
    help="Path to the FI-2010 neural benchmark YAML config (matrix_transformer enabled).",
)

_RUN_SSL_PROCESSED_ROOT_OPTION = typer.Option(
    ...,
    "--processed-root",
    help="Root containing prepared fold CSV files.",
)

_RUN_SSL_OUT_OPTION = typer.Option(
    ...,
    "--out",
    help="Output directory for SSL pretraining and fine-tuning artefacts.",
)

_RUN_SSL_FOLDS_OPTION = typer.Option(
    "fold_1",
    "--folds",
    help="'all' or a comma-separated list such as fold_1,fold_2.",
)

_RUN_SSL_SEEDS_OPTION = typer.Option(
    "0",
    "--seeds",
    help="'all' or a comma-separated list of non-negative seeds.",
)

_RUN_SSL_LOOKBACKS_OPTION = typer.Option(
    "10",
    "--lookbacks",
    help="'all' or a comma-separated list of positive lookbacks.",
)

_RUN_SSL_OBJECTIVE_OPTION = typer.Option(
    "masked_field",
    "--objective",
    help="Self-supervised objective: masked_field, next_field or both.",
)

_RUN_SSL_MASK_PROBABILITY_OPTION = typer.Option(
    0.15,
    "--mask-probability",
    help="Per-entry mask probability for the masked-field objective.",
)

_RUN_SSL_BUCKET_COUNT_OPTION = typer.Option(
    3,
    "--next-field-bucket-count",
    help="Train-only quantile bucket count for the next-field objective.",
)

_RUN_SSL_PRETRAIN_EPOCHS_OPTION = typer.Option(
    1,
    "--pretrain-epochs",
    help="Self-supervised pretraining epochs. Default is smoke-level.",
)

_RUN_SSL_MAX_EPOCHS_OPTION = typer.Option(
    1,
    "--max-epochs",
    help="Fine-tuning and baseline epochs. Default is smoke-level.",
)

_RUN_SSL_BATCH_SIZE_OPTION = typer.Option(
    16,
    "--batch-size",
    help="Batch size for pretraining, fine-tuning and the baseline.",
)

_RUN_SSL_DEVICE_OPTION = typer.Option(
    "cpu",
    "--device",
    help="Device: cpu or a cuda-prefixed device.",
)

_RUN_SSL_OVERWRITE_OPTION = typer.Option(
    False,
    "--overwrite",
    help="Replace the output directory if it already exists.",
)

_RUN_SSL_FAIL_FAST_OPTION = typer.Option(
    False,
    "--fail-fast",
    help="Stop at the first run failure.",
)

_RUN_SSL_NO_PREDICTIONS_OPTION = typer.Option(
    False,
    "--no-write-full-predictions",
    help="Skip writing per-run row-level predictions.",
)

_RUN_SSL_V2_CONFIG_OPTION = typer.Option(
    Path("configs/experiments/fi2010_neural_proper_training.yaml"),
    "--config",
    help="Path to the FI-2010 validation-selected neural YAML config.",
)

_RUN_SSL_V2_PROCESSED_ROOT_OPTION = typer.Option(
    ...,
    "--processed-root",
    help="Root containing prepared fold CSV files.",
)

_RUN_SSL_V2_OUT_OPTION = typer.Option(
    Path("experiments/fi2010_ssl_v2_benchmark"),
    "--out",
    help="Output directory for SSL-v2 benchmark artefacts.",
)

_RUN_SSL_V2_BASELINE_SOURCE_OPTION = typer.Option(
    Path("experiments/fi2010_neural_proper_training_subset_v2"),
    "--baseline-source",
    help="Existing validation-selected artefact root for matched baselines.",
)

_RUN_SSL_V2_FOLDS_OPTION = typer.Option(
    "1",
    "--folds",
    help="'all' or comma-separated fold ids.",
)

_RUN_SSL_V2_HORIZONS_OPTION = typer.Option(
    "10,50",
    "--horizons",
    help="'all' or comma-separated target horizons.",
)

_RUN_SSL_V2_SEEDS_OPTION = typer.Option(
    "0",
    "--seeds",
    help="Comma-separated non-negative seeds.",
)

_RUN_SSL_V2_LOOKBACKS_OPTION = typer.Option(
    "50",
    "--lookbacks",
    help="Comma-separated positive lookbacks.",
)

_RUN_SSL_V2_OBJECTIVES_OPTION = typer.Option(
    "supervised,masked_reconstruction,market_state_multitask",
    "--objectives",
    help="supervised, masked_reconstruction and/or market_state_multitask.",
)

_RUN_SSL_V2_PRETRAIN_EPOCHS_OPTION = typer.Option(
    5,
    "--pretrain-epochs",
    help="SSL-v2 pretraining epochs.",
)

_RUN_SSL_V2_MAX_EPOCHS_OPTION = typer.Option(
    None,
    "--max-epochs",
    help="Maximum fine-tuning epochs; defaults to config.",
)

_RUN_SSL_V2_PATIENCE_OPTION = typer.Option(
    None,
    "--patience",
    help="Validation early-stopping patience; defaults to config.",
)

_RUN_SSL_V2_BATCH_SIZE_OPTION = typer.Option(
    None,
    "--batch-size",
    help="Training batch size; defaults to config.",
)

_RUN_SSL_V2_MASK_PROBABILITY_OPTION = typer.Option(
    0.30,
    "--mask-probability",
    help="Structured group-mask probability.",
)

_RUN_SSL_V2_BUCKET_COUNT_OPTION = typer.Option(
    3,
    "--future-bucket-count",
    help="Train-only auxiliary future-state bucket count.",
)

_RUN_SSL_V2_CONTRASTIVE_OPTION = typer.Option(
    False,
    "--contrastive",
    help="Enable the optional regime contrastive SSL-v2 term.",
)

_RUN_SSL_V2_DEVICE_OPTION = typer.Option(
    "cpu",
    "--device",
    help="Device: cpu or cuda-prefixed device.",
)

_RUN_SSL_V2_REUSE_OPTION = typer.Option(
    True,
    f"{_REUSE_COMPLETED_FLAG}/{_NO_REUSE_COMPLETED_FLAG}",
    help="Skip existing completed run directories when possible.",
)

_RUN_SSL_V2_IMPORT_BASELINES_OPTION = typer.Option(
    True,
    "--import-existing-baselines/--no-import-existing-baselines",
    help="Import matching supervised/SSL-v1 baselines from baseline-source when available.",
)

_RUN_SSL_V2_SMOKE_OPTION = typer.Option(
    False,
    "--smoke-test",
    help="Run a tiny CPU-safe subset and mark artefacts as smoke only.",
)

_RUN_FULL_GRID_CONFIG_OPTION = typer.Option(
    Path("configs/experiments/fi2010_neural_serious.yaml"),
    "--config",
    help="Path to the FI-2010 neural benchmark YAML config.",
)

_RUN_FULL_GRID_PROCESSED_ROOT_OPTION = typer.Option(
    ...,
    "--processed-root",
    help="Root containing prepared fold CSV files.",
)

_RUN_FULL_GRID_OUT_OPTION = typer.Option(
    Path("experiments/fi2010_neural_full_grid"),
    "--out",
    help="Output directory for full neural grid artefacts.",
)

_RUN_FULL_GRID_FOLDS_OPTION = typer.Option(
    "1,2,3,4,5",
    "--folds",
    help="'all' or comma-separated fold ids.",
)

_RUN_FULL_GRID_HORIZONS_OPTION = typer.Option(
    "10,20,50",
    "--horizons",
    help="'all' or comma-separated target horizons.",
)

_RUN_FULL_GRID_SEEDS_OPTION = typer.Option(
    "0,1,2",
    "--seeds",
    help="Comma-separated non-negative seeds.",
)

_RUN_FULL_GRID_LOOKBACKS_OPTION = typer.Option(
    "20",
    "--lookbacks",
    help="Comma-separated positive lookbacks.",
)

_RUN_FULL_GRID_OBJECTIVES_OPTION = typer.Option(
    "supervised,masked_reconstruction,next_field",
    "--objectives",
    help="supervised, masked_reconstruction and/or next_field.",
)

_RUN_FULL_GRID_PRETRAIN_EPOCHS_OPTION = typer.Option(
    1,
    "--pretrain-epochs",
    help="Self-supervised pretraining epochs.",
)

_RUN_FULL_GRID_MAX_EPOCHS_OPTION = typer.Option(
    1,
    "--max-epochs",
    help="Supervised fine-tuning epochs.",
)

_RUN_FULL_GRID_BATCH_SIZE_OPTION = typer.Option(
    16,
    "--batch-size",
    help="Batch size for lower-level neural runners.",
)

_RUN_FULL_GRID_DEVICE_OPTION = typer.Option(
    "cpu",
    "--device",
    help="Device: cpu or cuda-prefixed device.",
)

_RUN_FULL_GRID_REUSE_OPTION = typer.Option(
    True,
    f"{_REUSE_COMPLETED_FLAG}/{_NO_REUSE_COMPLETED_FLAG}",
    help="Skip existing completed run directories when possible.",
)

_RUN_FULL_GRID_SMOKE_OPTION = typer.Option(
    False,
    "--smoke-test",
    help="Run a tiny CPU-safe grid subset and mark artefacts as smoke only.",
)

_RUN_PT_CONFIG_OPTION = typer.Option(
    Path("configs/experiments/fi2010_neural_proper_training.yaml"),
    "--config",
    help="Path to the FI-2010 validation-selected neural YAML config.",
)

_RUN_PT_PROCESSED_ROOT_OPTION = typer.Option(
    ...,
    "--processed-root",
    help="Root containing prepared fold CSV files.",
)

_RUN_PT_OUT_OPTION = typer.Option(
    Path("experiments/fi2010_neural_proper_training_subset_v2"),
    "--out",
    help="Output directory for validation-selected benchmark artefacts.",
)

_RUN_PT_FOLDS_OPTION = typer.Option(
    "1,2,3,4,5",
    "--folds",
    help="'all' or comma-separated fold ids.",
)

_RUN_PT_HORIZONS_OPTION = typer.Option(
    "10,20,50",
    "--horizons",
    help="'all' or comma-separated target horizons.",
)

_RUN_PT_SEEDS_OPTION = typer.Option(
    "0,1,2",
    "--seeds",
    help="Comma-separated non-negative seeds.",
)

_RUN_PT_LOOKBACKS_OPTION = typer.Option(
    "50",
    "--lookbacks",
    help="Comma-separated positive lookbacks.",
)

_RUN_PT_MODELS_OPTION = typer.Option(
    "matrix_transformer",
    "--models",
    help="matrix_transformer and/or deeplob_style.",
)

_RUN_PT_OBJECTIVES_OPTION = typer.Option(
    "supervised,masked_reconstruction,next_field",
    "--objectives",
    help="supervised, masked_reconstruction and/or next_field.",
)

_RUN_PT_PRETRAIN_EPOCHS_OPTION = typer.Option(
    10,
    "--pretrain-epochs",
    help="Self-supervised pretraining epochs (SSL objectives only).",
)

_RUN_PT_MAX_EPOCHS_OPTION = typer.Option(
    25,
    "--max-epochs",
    help="Maximum training epochs before early stopping.",
)

_RUN_PT_PATIENCE_OPTION = typer.Option(
    5,
    "--patience",
    help="Validation early-stopping patience in epochs.",
)

_RUN_PT_BATCH_SIZE_OPTION = typer.Option(
    256,
    "--batch-size",
    help="Training batch size.",
)

_RUN_PT_DEVICE_OPTION = typer.Option(
    "cpu",
    "--device",
    help="Device: cpu or cuda-prefixed device.",
)

_RUN_PT_REUSE_OPTION = typer.Option(
    True,
    f"{_REUSE_COMPLETED_FLAG}/{_NO_REUSE_COMPLETED_FLAG}",
    help="Skip existing completed run directories when possible.",
)

_RUN_PT_SMOKE_OPTION = typer.Option(
    False,
    "--smoke-test",
    help="Run a tiny CPU-safe subset and mark artefacts as smoke only.",
)

_BUILD_FI2010_FIGURES_GRID_OPTION = typer.Option(
    ...,
    "--neural-full-grid",
    help="Path to the FI-2010 neural full-grid artefact directory.",
)

_BUILD_FI2010_FIGURES_OUT_OPTION = typer.Option(
    Path("reports/figures/fi2010_neural_full_grid"),
    "--out",
    help="Output directory for reproducible FI-2010 figures.",
)

_BUILD_FI2010_FIGURES_EXECUTION_V3_OPTION = typer.Option(
    None,
    "--execution-v3",
    help="Optional execution-v3 artefact directory for proxy diagnostic figures.",
)

_BUILD_FI2010_FIGURES_MODELS_OPTION = typer.Option(
    "all",
    "--models",
    help="'all' or comma-separated model/objective selectors.",
)

_BUILD_FI2010_FIGURES_HORIZONS_OPTION = typer.Option(
    "all",
    "--horizons",
    help="'all' or comma-separated horizons.",
)

_BUILD_FI2010_FIGURES_FOLDS_OPTION = typer.Option(
    "all",
    "--folds",
    help="'all' or comma-separated fold ids.",
)

_BUILD_FI2010_FIGURES_SEEDS_OPTION = typer.Option(
    "all",
    "--seeds",
    help="'all' or comma-separated seeds.",
)

_BUILD_FI2010_FIGURES_OVERWRITE_OPTION = typer.Option(
    False,
    "--overwrite/--no-overwrite",
    help="Replace the figure output directory if it already exists.",
)

_BUILD_FI2010_FIGURES_ALLOW_SMOKE_OPTION = typer.Option(
    False,
    "--allow-smoke-test",
    help="Permit smoke-test artefacts and label figures as diagnostics only.",
)

_BUILD_FI2010_FIGURES_STRICT_OPTION = typer.Option(
    True,
    "--strict/--no-strict",
    help="Fail when FI-2010 label mapping cannot be validated.",
)

_AUDIT_FI2010_FEATURES_PATH_OPTION = typer.Option(
    Path("tests/fixtures/fi2010/tiny_fi2010_like.csv"),
    "--path",
    help="Local FI-2010-style CSV to audit.",
)

_AUDIT_FI2010_FEATURES_GROUPS_OPTION = typer.Option(
    "all",
    "--feature-groups",
    help="'all' or comma-separated microstructure feature groups.",
)

_AUDIT_FI2010_FEATURES_LABELS_OPTION = typer.Option(
    None,
    "--label-columns",
    help="Optional comma-separated label columns to exclude.",
)

_AUDIT_FI2010_FEATURES_SPLIT_OPTION = typer.Option(
    "split",
    "--split-column",
    help="Optional split/partition column for boundary checks.",
)

_AUDIT_FI2010_FEATURES_STRICT_OPTION = typer.Option(
    True,
    "--strict/--no-strict",
    help="Fail on requested groups with no valid columns.",
)

_AUDIT_FI2010_FEATURES_VOL_WINDOW_OPTION = typer.Option(
    20,
    "--volatility-window",
    help="Past-looking rolling window for volatility proxy audit.",
)

_FEATURE_ABLATIONS_CONFIG_OPTION = typer.Option(
    Path("configs/experiments/fi2010_multifold.yaml"),
    "--config",
    help="Optional FI-2010 multi-fold config for prepared CSV discovery.",
)

_FEATURE_ABLATIONS_PROCESSED_ROOT_OPTION = typer.Option(
    None,
    "--processed-root",
    help="Optional root containing prepared fold CSVs.",
)

_FEATURE_ABLATIONS_DATA_PATH_OPTION = typer.Option(
    None,
    "--data-path",
    help="Optional single FI-2010-style CSV; useful for smoke/synthetic runs.",
)

_FEATURE_ABLATIONS_FOLDS_OPTION = typer.Option(
    "1",
    "--folds",
    help="'all' or comma-separated fold ids.",
)

_FEATURE_ABLATIONS_HORIZONS_OPTION = typer.Option(
    "10",
    "--horizons",
    help="'all' or comma-separated label horizons.",
)

_FEATURE_ABLATIONS_SEEDS_OPTION = typer.Option(
    "0",
    "--seeds",
    help="'all' or comma-separated non-negative seeds.",
)

_FEATURE_ABLATIONS_MODELS_OPTION = typer.Option(
    "logistic,ridge,elastic_net,gradient_boosting",
    "--models",
    help="'all' or comma-separated classical model names.",
)

_FEATURE_ABLATIONS_GROUPS_OPTION = typer.Option(
    "all",
    "--feature-groups",
    help="'all' or comma-separated feature group names.",
)

_FEATURE_ABLATIONS_MODES_OPTION = typer.Option(
    "all",
    "--ablation-modes",
    help="'all' or comma-separated ablation modes.",
)

_FEATURE_ABLATIONS_OUT_OPTION = typer.Option(
    Path("experiments/fi2010_feature_ablations"),
    "--out",
    help="Output directory for feature-ablation artefacts.",
)

_FEATURE_ABLATIONS_REUSE_OPTION = typer.Option(
    True,
    f"{_REUSE_COMPLETED_FLAG}/{_NO_REUSE_COMPLETED_FLAG}",
    help="Reuse completed run directories where possible.",
)

_FEATURE_ABLATIONS_STRICT_OPTION = typer.Option(
    True,
    "--strict/--no-strict",
    help="Fail on explicitly requested unsupported or empty feature groups.",
)

_FEATURE_ABLATIONS_SMOKE_OPTION = typer.Option(
    False,
    "--smoke-test",
    help="Use a tiny synthetic fixture when prepared FI-2010 inputs are absent.",
)

_FEATURE_ABLATIONS_SAVE_PREDICTIONS_OPTION = typer.Option(
    False,
    "--save-predictions/--no-save-predictions",
    help="Write per-run row-level prediction files.",
)

_FEATURE_ABLATIONS_SAVE_HEAVY_OPTION = typer.Option(
    False,
    "--save-heavy-artefacts/--no-save-heavy-artefacts",
    help="Write cached feature matrices and other regenerable intermediate artefacts.",
)

_FEATURE_ABLATIONS_SUMMARY_ONLY_OPTION = typer.Option(
    True,
    "--summary-only/--no-summary-only",
    help="Keep feature-ablation outputs compact by skipping large artefacts.",
)

_BUILD_ABLATION_FIGURES_INPUT_OPTION = typer.Option(
    Path("experiments/fi2010_feature_ablations"),
    "--feature-ablations",
    "--ablations",
    help="Feature-ablation artefact directory.",
)

_BUILD_ABLATION_FIGURES_OUT_OPTION = typer.Option(
    Path("reports/figures/fi2010_feature_ablations"),
    "--out",
    help="Output directory for feature-ablation figures.",
)

_BUILD_ABLATION_FIGURES_OVERWRITE_OPTION = typer.Option(
    False,
    "--overwrite/--no-overwrite",
    help="Replace the figure output directory if it already exists.",
)

_BUILD_ABLATION_FIGURES_ALLOW_SMOKE_OPTION = typer.Option(
    False,
    "--allow-smoke-test",
    help="Permit smoke-test ablation artefacts and label figures as diagnostics only.",
)

_ANALYSE_FEATURE_ABLATIONS_INPUT_OPTION = typer.Option(
    Path("experiments/fi2010_feature_ablations"),
    "--feature-ablations",
    "--ablations",
    help="Feature-ablation artefact directory.",
)

_ANALYSE_FEATURE_ABLATIONS_OUT_OPTION = typer.Option(
    Path("reports/feature_ablation_analysis"),
    "--out",
    help="Output directory for feature-ablation stability analysis.",
)

_ANALYSE_FEATURE_ABLATIONS_EXTRA_OPTION = typer.Option(
    None,
    "--extra-feature-ablations",
    help="Optional comma-separated extra feature-ablation directories to merge.",
)

_ANALYSE_FEATURE_ABLATIONS_FIGURES_OPTION = typer.Option(
    True,
    "--figures/--no-figures",
    help="Generate aggregate feature-ablation stability figures.",
)

_ANALYSE_FEATURE_ABLATIONS_OVERWRITE_OPTION = typer.Option(
    False,
    "--overwrite/--no-overwrite",
    help="Replace existing feature-ablation analysis outputs.",
)

_ANALYSE_FEATURE_ABLATIONS_ALLOW_SMOKE_OPTION = typer.Option(
    False,
    "--allow-smoke-test",
    help="Permit smoke-test ablation artefacts and label outputs as diagnostics only.",
)

_ANALYSE_UNCERTAINTY_CLASSICAL_OPTION = typer.Option(
    None,
    "--classical",
    help=("Path to the classical multi-fold artefact directory containing results_by_fold.csv."),
)

_ANALYSE_UNCERTAINTY_NEURAL_OPTION = typer.Option(
    None,
    "--neural",
    help=("Path to the neural multi-fold artefact directory containing results_by_fold_seed.csv."),
)

_ANALYSE_UNCERTAINTY_OUT_OPTION = typer.Option(
    ...,
    "--out",
    help="Output directory for the uncertainty artefacts.",
)

_ANALYSE_UNCERTAINTY_BASELINE_OPTION = typer.Option(
    "gradient_boosting",
    "--baseline",
    help=("Baseline model name for paired fold-level comparisons. Defaults to gradient_boosting."),
)

_ANALYSE_UNCERTAINTY_CI_OPTION = typer.Option(
    0.95,
    "--ci-level",
    help="Two-sided confidence level. Defaults to 0.95.",
)

_ANALYSE_UNCERTAINTY_BOOTSTRAP_ITER_OPTION = typer.Option(
    1000,
    "--bootstrap-iterations",
    help="Bootstrap iterations over folds. Defaults to 1000.",
)

_ANALYSE_UNCERTAINTY_BOOTSTRAP_SEED_OPTION = typer.Option(
    0,
    "--bootstrap-seed",
    help="Random seed for the bootstrap resampler. Defaults to 0.",
)

_ANALYSE_UNCERTAINTY_OVERWRITE_OPTION = typer.Option(
    False,
    "--overwrite",
    help="Replace the output directory if it already exists.",
)

_ANALYSE_SSL_FULL_GRID_OPTION = typer.Option(
    Path("experiments/fi2010_neural_full_grid"),
    "--full-grid",
    help="FI-2010 neural full-grid artefact directory.",
)

_ANALYSE_SSL_PROPER_TRAINING_OPTION = typer.Option(
    Path("experiments/fi2010_neural_proper_training_subset_v2"),
    "--proper-training",
    help="FI-2010 validation-selected neural benchmark artefact directory.",
)

_ANALYSE_SSL_OUT_OPTION = typer.Option(
    Path("reports/ssl_failure_analysis"),
    "--out",
    help="Output directory for the SSL failure-analysis artefacts.",
)

_ANALYSE_SSL_FIGURES_OPTION = typer.Option(
    True,
    "--figures/--no-figures",
    help="Generate lightweight delta figures. Use --no-figures to skip them.",
)

_ANALYSE_SSL_OVERWRITE_OPTION = typer.Option(
    False,
    "--overwrite",
    help="Replace the output directory if it already exists.",
)

_ANALYSE_SSL_V2_INPUT_OPTION = typer.Option(
    Path("experiments/fi2010_ssl_v2_benchmark"),
    "--ssl-v2",
    help="FI-2010 SSL-v2 benchmark artefact directory.",
)

_ANALYSE_SSL_V2_OUT_OPTION = typer.Option(
    Path("reports/ssl_v2_analysis"),
    "--out",
    help="Output directory for the SSL-v2 analysis artefacts.",
)

_ANALYSE_EXEC_V3_INPUT_OPTION = typer.Option(
    Path("experiments/fi2010_execution_v3"),
    "--execution-v3",
    help="Execution-v3 artefact directory produced by build-fi2010-execution-v3.",
)

_ANALYSE_EXEC_V3_OUT_OPTION = typer.Option(
    Path("reports/execution_v3_analysis"),
    "--out",
    help="Output directory for the execution-v3 analysis artefacts.",
)

_ANALYSE_EXEC_V3_FIGURES_OPTION = typer.Option(
    True,
    "--figures/--no-figures",
    help="Generate lightweight proxy figures. Use --no-figures to skip them.",
)

_ANALYSE_EXEC_V3_OVERWRITE_OPTION = typer.Option(
    False,
    "--overwrite",
    help="Replace the output directory if it already exists.",
)

_EXECUTION_CENTREPIECE_ANALYSIS_OPTION = typer.Option(
    Path("reports/execution_v3_analysis"),
    "--execution-analysis",
    help="Execution-v3 analysis directory produced by analyse-fi2010-execution-v3.",
)

_EXECUTION_CENTREPIECE_OUT_OPTION = typer.Option(
    Path("reports/execution_centrepiece"),
    "--out",
    help="Output directory for the execution centrepiece artefacts.",
)

_EXECUTION_CENTREPIECE_EXECUTION_V3_OPTION = typer.Option(
    Path("experiments/fi2010_execution_v3"),
    "--execution-v3",
    help="Optional execution-v3 artefact directory for manifest context.",
)

_EXECUTION_CENTREPIECE_FULL_GRID_OPTION = typer.Option(
    Path("experiments/fi2010_neural_full_grid"),
    "--neural-full-grid",
    help="Optional stored neural full-grid aggregate directory.",
)

_EXECUTION_CENTREPIECE_FIGURES_OPTION = typer.Option(
    True,
    "--figures/--no-figures",
    help="Generate the central centrepiece figure. Use --no-figures to skip it.",
)

_EXECUTION_CENTREPIECE_OVERWRITE_OPTION = typer.Option(
    False,
    "--overwrite",
    help="Replace the output directory if it already exists.",
)

_BRUTAL_ABLATIONS_CONFIG_OPTION = typer.Option(
    ...,
    "--config",
    help="Path to the FI-2010 multi-fold YAML config.",
)

_BRUTAL_ABLATIONS_NEURAL_CONFIG_OPTION = typer.Option(
    None,
    "--neural-config",
    help="Path to the neural benchmark YAML config (for the lookback sweep).",
)

_BRUTAL_ABLATIONS_PROCESSED_ROOT_OPTION = typer.Option(
    None,
    "--processed-root",
    help="Root containing prepared fold CSV files for the fit families.",
)

_BRUTAL_ABLATIONS_CLASSICAL_OPTION = typer.Option(
    None,
    "--classical",
    help="Stored classical multi-fold artefact directory.",
)

_BRUTAL_ABLATIONS_NEURAL_OPTION = typer.Option(
    None,
    "--neural",
    help="Stored neural multi-fold artefact directory.",
)

_BRUTAL_ABLATIONS_OUT_OPTION = typer.Option(
    ...,
    "--out",
    help="Output directory for brutal ablation artefacts.",
)

_BRUTAL_ABLATIONS_FAMILIES_OPTION = typer.Option(
    "all",
    "--families",
    help=(
        "'all' or a comma-separated subset of feature_groups,model_class,"
        "lookback,horizon,calibration,execution."
    ),
)

_BRUTAL_ABLATIONS_FOLDS_OPTION = typer.Option(
    "all",
    "--folds",
    help="'all' or a comma-separated list such as fold_1,fold_2.",
)

_BRUTAL_ABLATIONS_MODELS_OPTION = typer.Option(
    None,
    "--models",
    help=(
        "Optional comma-separated model filter. Classical names drive the "
        "fit families; neural names drive the lookback family."
    ),
)

_BRUTAL_ABLATIONS_LOOKBACKS_OPTION = typer.Option(
    None,
    "--neural-lookbacks",
    help=(
        "Optional comma-separated neural lookback subset. When given, the "
        "CPU-expensive lookback sweep is executed."
    ),
)

_BRUTAL_ABLATIONS_MAX_EPOCHS_OPTION = typer.Option(
    5,
    "--max-epochs",
    help="Maximum epochs for the optional neural lookback sweep.",
)

_BRUTAL_ABLATIONS_OVERWRITE_OPTION = typer.Option(
    False,
    "--overwrite",
    help="Replace the output directory if it already exists.",
)

_BRUTAL_ABLATIONS_DRY_RUN_OPTION = typer.Option(
    False,
    "--dry-run",
    help="Resolve the plan and write nothing.",
)

_EXECUTION_V2_CLASSICAL_OPTION = typer.Option(
    None,
    "--classical",
    help="Stored classical multi-fold artefact directory.",
)

_EXECUTION_V2_NEURAL_OPTION = typer.Option(
    None,
    "--neural",
    help="Stored neural multi-fold artefact directory.",
)

_EXECUTION_V2_ABLATIONS_OPTION = typer.Option(
    None,
    "--ablations",
    help="Stored brutal ablation artefact directory (cross-reference input).",
)

_EXECUTION_V2_OUT_OPTION = typer.Option(
    ...,
    "--out",
    help="Output directory for execution v2 proxy diagnostics.",
)

_EXECUTION_V2_MODELS_OPTION = typer.Option(
    None,
    "--models",
    help="Optional comma-separated model filter.",
)

_EXECUTION_V2_COST_BPS_OPTION = typer.Option(
    None,
    "--cost-bps",
    help="Optional comma-separated cost (bps) filter such as 0,1,5.",
)

_EXECUTION_V2_LATENCY_OPTION = typer.Option(
    None,
    "--latency-steps",
    help="Optional comma-separated latency-step filter such as 0,1.",
)

_EXECUTION_V2_THRESHOLDS_OPTION = typer.Option(
    None,
    "--confidence-thresholds",
    help="Optional comma-separated confidence-threshold filter such as 0,0.6.",
)

_EXECUTION_V2_OVERWRITE_OPTION = typer.Option(
    False,
    "--overwrite",
    help="Replace the output directory if it already exists.",
)

_EXECUTION_V3_GRID_OPTION = typer.Option(
    ...,
    "--neural-full-grid",
    help="Path to FI-2010 neural full-grid artefacts with predictions.",
)

_EXECUTION_V3_FEATURE_ABLATIONS_OPTION = typer.Option(
    None,
    "--feature-ablations",
    help=(
        "Optional FI-2010 feature-ablation artefact directory. When supplied, "
        "execution-v3 reads ablation prediction artefacts explicitly."
    ),
)

_EXECUTION_V3_OUT_OPTION = typer.Option(
    Path("experiments/fi2010_execution_v3"),
    "--out",
    help="Output directory for execution-aware proxy diagnostic v3.",
)

_EXECUTION_V3_MODELS_OPTION = typer.Option(
    "all",
    "--models",
    help="'all' or comma-separated model/objective selectors.",
)

_EXECUTION_V3_HORIZONS_OPTION = typer.Option(
    "all",
    "--horizons",
    help="'all' or comma-separated horizons.",
)

_EXECUTION_V3_FOLDS_OPTION = typer.Option(
    "all",
    "--folds",
    help="'all' or comma-separated fold ids.",
)

_EXECUTION_V3_SEEDS_OPTION = typer.Option(
    "all",
    "--seeds",
    help="'all' or comma-separated seeds.",
)

_EXECUTION_V3_THRESHOLDS_OPTION = typer.Option(
    None,
    "--confidence-thresholds",
    help="Optional comma-separated thresholds; defaults to 0.33 through 0.95.",
)

_EXECUTION_V3_FEE_BPS_OPTION = typer.Option(
    None,
    "--fee-bps",
    help="Optional comma-separated fee levels in bps; defaults to 0,1,2,5,10.",
)

_EXECUTION_V3_SPREAD_MULTIPLIERS_OPTION = typer.Option(
    None,
    "--spread-multipliers",
    help="Optional comma-separated spread multipliers; defaults to 0,0.5,1,2.",
)

_EXECUTION_V3_LATENCY_OPTION = typer.Option(
    None,
    "--latency-steps",
    help="Optional comma-separated row-step latencies; defaults to 0,1,2,5,10.",
)

_EXECUTION_V3_FILL_OPTION = typer.Option(
    None,
    "--fill-assumptions",
    help="Optional comma-separated fill proxy modes.",
)

_EXECUTION_V3_ALLOW_SMOKE_OPTION = typer.Option(
    False,
    "--allow-smoke-test",
    help="Permit smoke-test artefacts and mark v3 outputs as smoke only.",
)

_EXECUTION_V3_STRICT_OPTION = typer.Option(
    True,
    "--strict/--no-strict",
    help="Fail on ambiguous FI-2010 label/probability mapping.",
)

_EXECUTION_V3_OVERWRITE_OPTION = typer.Option(
    False,
    "--overwrite/--no-overwrite",
    help="Replace the execution-v3 output directory if it already exists.",
)

_RUN_PAPER_EXPERIMENT_CONFIG_OPTION = typer.Option(
    ...,
    "--config",
    help="Path to the paper experiment configuration YAML file.",
)

_RUN_PAPER_EXPERIMENT_DATA_PATH_OPTION = typer.Option(
    ...,
    "--data-path",
    help="Local FI-2010-style file path supplied by the user.",
)

_RUN_PAPER_EXPERIMENT_OUT_OPTION = typer.Option(
    ...,
    "--out",
    help="Output directory for the paper experiment artefacts.",
)

_RUN_PAPER_EXPERIMENT_MODELS_OPTION = typer.Option(
    "majority",
    "--models",
    help="Comma-separated model list. Defaults to 'majority'.",
)

_RUN_PAPER_EXPERIMENT_OVERWRITE_OPTION = typer.Option(
    False,
    "--overwrite",
    help="Replace the output directory if it already exists.",
)

_RUN_PAPER_EXPERIMENT_BUILD_PLOTS_OPTION = typer.Option(
    False,
    "--build-plots",
    help=("Generate paper experiment plots from stored artefacts after the run finishes."),
)

_RUN_PAPER_ABLATIONS_CONFIG_OPTION = typer.Option(
    ...,
    "--config",
    help="Path to the paper ablation base configuration YAML file.",
)

_RUN_PAPER_ABLATIONS_DATA_PATH_OPTION = typer.Option(
    ...,
    "--data-path",
    help="Local FI-2010-style file path supplied by the user.",
)

_RUN_PAPER_ABLATIONS_OUT_OPTION = typer.Option(
    ...,
    "--out",
    help="Output directory for aggregate ablation artefacts.",
)

_RUN_PAPER_ABLATIONS_MODELS_OPTION = typer.Option(
    "majority,logistic",
    "--models",
    help="Comma-separated model list. Defaults to 'majority,logistic'.",
)

_RUN_PAPER_ABLATIONS_SET_OPTION = typer.Option(
    "smoke",
    "--ablation-set",
    help="Named ablation set. Supported values: smoke, standard.",
)

_RUN_PAPER_ABLATIONS_OVERWRITE_OPTION = typer.Option(
    False,
    "--overwrite",
    help="Replace the output directory if it already exists.",
)

_RUN_PAPER_ABLATIONS_BUILD_PLOTS_OPTION = typer.Option(
    False,
    "--build-plots",
    help="Generate plots inside each child paper experiment directory.",
)

_RUN_SYSTEM_BENCHMARKS_CONFIG_OPTION = typer.Option(
    ...,
    "--config",
    help="Path to the systems benchmark configuration YAML file.",
)

_RUN_SYSTEM_BENCHMARKS_DATA_PATH_OPTION = typer.Option(
    ...,
    "--data-path",
    help="Local FI-2010-style file path supplied by the user.",
)

_RUN_SYSTEM_BENCHMARKS_OUT_OPTION = typer.Option(
    ...,
    "--out",
    help="Output directory for systems benchmark artefacts.",
)

_RUN_SYSTEM_BENCHMARKS_SET_OPTION = typer.Option(
    "smoke",
    "--benchmark-set",
    help="Named benchmark set. Supported values: smoke, standard.",
)

_RUN_SYSTEM_BENCHMARKS_MODELS_OPTION = typer.Option(
    "majority,logistic",
    "--models",
    help="Comma-separated paper-runner model list.",
)

_RUN_SYSTEM_BENCHMARKS_OVERWRITE_OPTION = typer.Option(
    False,
    "--overwrite",
    help="Replace the output directory if it already exists.",
)

_INSPECT_SYSTEM_BENCHMARKS_BENCHMARK_OPTION = typer.Option(
    ...,
    "--benchmark",
    help="Path to a completed systems benchmark directory.",
)

_BUILD_PAPER_PLOTS_EXPERIMENT_OPTION = typer.Option(
    ...,
    "--experiment",
    help="Path to a completed paper experiment artefact directory.",
)

_BUILD_PAPER_PLOTS_OVERWRITE_OPTION = typer.Option(
    False,
    "--overwrite",
    help="Replace existing plot files when generating.",
)

_INSPECT_PAPER_EXPERIMENT_EXPERIMENT_OPTION = typer.Option(
    ...,
    "--experiment",
    help="Path to a paper experiment artefact directory.",
)

_BUILD_PAPER_REPORT_EXPERIMENT_OPTION = typer.Option(
    ...,
    "--experiment",
    help="Path to a completed paper experiment artefact directory.",
)

_BUILD_PAPER_REPORT_ABLATIONS_OPTION = typer.Option(
    None,
    "--ablations",
    help="Optional path to a completed paper ablation directory.",
)

_BUILD_PAPER_REPORT_SYSTEMS_OPTION = typer.Option(
    None,
    "--systems",
    help="Optional path to a completed systems benchmark directory.",
)

_BUILD_PAPER_REPORT_OUT_OPTION = typer.Option(
    ...,
    "--out",
    help="Markdown report path to write.",
)

_BUILD_PAPER_REPORT_OVERWRITE_OPTION = typer.Option(
    False,
    "--overwrite",
    help="Replace the report and summary JSON if they already exist.",
)

_INSPECT_PAPER_REPORT_REPORT_OPTION = typer.Option(
    ...,
    "--report",
    help="Path to a generated empirical report Markdown file.",
)


def inspect_experiment_artifacts(
    experiment: Path = _EXPERIMENT_ARTIFACTS_EXPERIMENT_OPTION,
) -> None:
    """Inspect an experiment directory against the artefact contract."""
    exit_code = _inspect_experiment_artifacts_impl(experiment=experiment)
    if exit_code != 0:
        raise SystemExit(exit_code)


def prepare_fi2010_benchmark(
    config: Path = _PREPARE_FI2010_CONFIG_OPTION,
    data_path: Path = _PREPARE_FI2010_DATA_PATH_OPTION,
    out: Path = _PREPARE_FI2010_OUT_OPTION,
) -> None:
    """Prepare a local-only FI-2010 benchmark input."""
    exit_code = _prepare_fi2010_benchmark_impl(
        config_path=config,
        data_path=data_path,
        out=out,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def verify_fi2010_local(
    data_path: Path = _VERIFY_FI2010_DATA_PATH_OPTION,
) -> None:
    """Inspect a local FI-2010 file safely without loading it."""
    exit_code = _verify_fi2010_local_impl(data_path=data_path)
    if exit_code != 0:
        raise SystemExit(exit_code)


def convert_fi2010_official(
    input_path: Path = _CONVERT_FI2010_INPUT_OPTION,
    output_path: Path = _CONVERT_FI2010_OUTPUT_OPTION,
    split: str | None = _CONVERT_FI2010_SPLIT_OPTION,
    overwrite: bool = _CONVERT_FI2010_OVERWRITE_OPTION,
) -> None:
    """Convert a single official FI-2010 .txt matrix into a loader-ready CSV."""
    exit_code = _convert_fi2010_official_impl(
        input_path=input_path,
        output_path=output_path,
        split_label=split,
        overwrite=overwrite,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def inspect_fi2010_multifold(
    config: Path = _MULTIFOLD_CONFIG_OPTION,
    extracted_root: Path = _MULTIFOLD_EXTRACTED_ROOT_OPTION,
    processed_root: Path | None = _MULTIFOLD_PROCESSED_ROOT_OPTION,
    folds: str = _MULTIFOLD_FOLDS_OPTION,
) -> None:
    """Report configured FI-2010 folds and which expected files exist."""
    try:
        selection = _parse_fold_selection(folds)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    exit_code = _inspect_fi2010_multifold_impl(
        config_path=config,
        extracted_root=extracted_root,
        processed_root=processed_root,
        folds=selection,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def prepare_fi2010_multifold(
    config: Path = _MULTIFOLD_CONFIG_OPTION,
    extracted_root: Path = _MULTIFOLD_EXTRACTED_ROOT_OPTION,
    processed_root: Path | None = _MULTIFOLD_PROCESSED_ROOT_OPTION,
    out: Path = _MULTIFOLD_OUT_OPTION,
    folds: str = _MULTIFOLD_FOLDS_OPTION,
    overwrite: bool = _MULTIFOLD_OVERWRITE_OPTION,
) -> None:
    """Prepare multi-fold combined CSVs and manifests for the configured folds."""
    try:
        selection = _parse_fold_selection(folds)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    exit_code = _prepare_fi2010_multifold_impl(
        config_path=config,
        extracted_root=extracted_root,
        processed_root=processed_root,
        out=out,
        folds=selection,
        overwrite=overwrite,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def run_fi2010_multifold_classical(
    config: Path = _RUN_MULTIFOLD_CLASSICAL_CONFIG_OPTION,
    processed_root: Path | None = _RUN_MULTIFOLD_CLASSICAL_PROCESSED_ROOT_OPTION,
    out: Path = _RUN_MULTIFOLD_CLASSICAL_OUT_OPTION,
    models: str | None = _RUN_MULTIFOLD_CLASSICAL_MODELS_OPTION,
    folds: str = _RUN_MULTIFOLD_CLASSICAL_FOLDS_OPTION,
    overwrite: bool = _RUN_MULTIFOLD_CLASSICAL_OVERWRITE_OPTION,
) -> None:
    """Run classical models across prepared FI-2010 fold CSVs."""
    try:
        selection = _parse_fold_selection(folds)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    model_tokens = None
    if models is not None:
        model_tokens = [token.strip() for token in models.split(",") if token.strip()]
    exit_code = _run_fi2010_multifold_classical_impl(
        config_path=config,
        processed_root=processed_root,
        out=out,
        models=model_tokens,
        folds=selection,
        overwrite=overwrite,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def inspect_fi2010_neural_plan(
    config: Path = _NEURAL_PLAN_CONFIG_OPTION,
    folds: str = _NEURAL_PLAN_FOLDS_OPTION,
    models: str = _NEURAL_PLAN_MODELS_OPTION,
) -> None:
    """Inspect the FI-2010 neural benchmark run grid without training."""
    try:
        selection = _parse_fold_selection(folds)
        model_tokens = _parse_model_selection(models)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    exit_code = _inspect_fi2010_neural_plan_impl(
        config_path=config,
        folds=selection,
        models=model_tokens,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def run_fi2010_neural_benchmark(
    config: Path = _RUN_NEURAL_CONFIG_OPTION,
    processed_root: Path = _RUN_NEURAL_PROCESSED_ROOT_OPTION,
    out: Path = _RUN_NEURAL_OUT_OPTION,
    folds: str = _RUN_NEURAL_FOLDS_OPTION,
    models: str = _RUN_NEURAL_MODELS_OPTION,
    seeds: str = _RUN_NEURAL_SEEDS_OPTION,
    lookbacks: str = _RUN_NEURAL_LOOKBACKS_OPTION,
    max_epochs: int = _RUN_NEURAL_MAX_EPOCHS_OPTION,
    overwrite: bool = _RUN_NEURAL_OVERWRITE_OPTION,
    fail_fast: bool = _RUN_NEURAL_FAIL_FAST_OPTION,
    write_full_predictions: bool = _RUN_NEURAL_WRITE_PREDICTIONS_OPTION,
    write_checkpoints: bool = _RUN_NEURAL_WRITE_CHECKPOINTS_OPTION,
    allow_full_benchmark: bool = _RUN_NEURAL_ALLOW_FULL_OPTION,
) -> None:
    """Run selected FI-2010 supervised neural benchmark configurations."""
    try:
        fold_tokens = _parse_neural_fold_selection(folds)
        model_tokens = _parse_model_selection(models)
        seed_tokens = _parse_int_selection(
            seeds,
            option_name="--seeds",
            positive=False,
        )
        lookback_tokens = _parse_int_selection(
            lookbacks,
            option_name="--lookbacks",
            positive=True,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    exit_code = _run_fi2010_neural_benchmark_impl(
        config_path=config,
        processed_root=processed_root,
        out=out,
        folds=fold_tokens,
        models=model_tokens,
        seeds=seed_tokens,
        lookbacks=lookback_tokens,
        max_epochs=max_epochs,
        overwrite=overwrite,
        fail_fast=fail_fast,
        write_full_predictions=write_full_predictions,
        write_checkpoints=write_checkpoints,
        allow_full_benchmark=allow_full_benchmark,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def run_fi2010_ssl_neural_benchmark(
    config: Path = _RUN_SSL_CONFIG_OPTION,
    processed_root: Path = _RUN_SSL_PROCESSED_ROOT_OPTION,
    out: Path = _RUN_SSL_OUT_OPTION,
    folds: str = _RUN_SSL_FOLDS_OPTION,
    seeds: str = _RUN_SSL_SEEDS_OPTION,
    lookbacks: str = _RUN_SSL_LOOKBACKS_OPTION,
    objective: str = _RUN_SSL_OBJECTIVE_OPTION,
    mask_probability: float = _RUN_SSL_MASK_PROBABILITY_OPTION,
    next_field_bucket_count: int = _RUN_SSL_BUCKET_COUNT_OPTION,
    pretrain_epochs: int = _RUN_SSL_PRETRAIN_EPOCHS_OPTION,
    max_epochs: int = _RUN_SSL_MAX_EPOCHS_OPTION,
    batch_size: int = _RUN_SSL_BATCH_SIZE_OPTION,
    device: str = _RUN_SSL_DEVICE_OPTION,
    overwrite: bool = _RUN_SSL_OVERWRITE_OPTION,
    fail_fast: bool = _RUN_SSL_FAIL_FAST_OPTION,
    no_write_full_predictions: bool = _RUN_SSL_NO_PREDICTIONS_OPTION,
) -> None:
    """Pretrain, fine-tune and compare the FI-2010 ssl_transformer path."""
    try:
        fold_tokens = _parse_neural_fold_selection(folds)
        seed_tokens = _parse_int_selection(
            seeds,
            option_name="--seeds",
            positive=False,
        )
        lookback_tokens = _parse_int_selection(
            lookbacks,
            option_name="--lookbacks",
            positive=True,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    exit_code = _run_fi2010_ssl_neural_benchmark_impl(
        config_path=config,
        processed_root=processed_root,
        out=out,
        folds=fold_tokens,
        seeds=seed_tokens,
        lookbacks=lookback_tokens,
        objective=objective,
        mask_probability=mask_probability,
        next_field_bucket_count=next_field_bucket_count,
        pretrain_epochs=pretrain_epochs,
        max_epochs=max_epochs,
        batch_size=batch_size,
        device=device,
        overwrite=overwrite,
        fail_fast=fail_fast,
        write_full_predictions=not no_write_full_predictions,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def run_fi2010_ssl_v2_benchmark(
    config: Path = _RUN_SSL_V2_CONFIG_OPTION,
    processed_root: Path = _RUN_SSL_V2_PROCESSED_ROOT_OPTION,
    out: Path = _RUN_SSL_V2_OUT_OPTION,
    baseline_source: Path | None = _RUN_SSL_V2_BASELINE_SOURCE_OPTION,
    folds: str = _RUN_SSL_V2_FOLDS_OPTION,
    horizons: str = _RUN_SSL_V2_HORIZONS_OPTION,
    seeds: str = _RUN_SSL_V2_SEEDS_OPTION,
    lookbacks: str = _RUN_SSL_V2_LOOKBACKS_OPTION,
    objectives: str = _RUN_SSL_V2_OBJECTIVES_OPTION,
    pretrain_epochs: int = _RUN_SSL_V2_PRETRAIN_EPOCHS_OPTION,
    max_epochs: int | None = _RUN_SSL_V2_MAX_EPOCHS_OPTION,
    patience: int | None = _RUN_SSL_V2_PATIENCE_OPTION,
    batch_size: int | None = _RUN_SSL_V2_BATCH_SIZE_OPTION,
    mask_probability: float = _RUN_SSL_V2_MASK_PROBABILITY_OPTION,
    future_bucket_count: int = _RUN_SSL_V2_BUCKET_COUNT_OPTION,
    contrastive: bool = _RUN_SSL_V2_CONTRASTIVE_OPTION,
    device: str = _RUN_SSL_V2_DEVICE_OPTION,
    reuse_completed: bool = _RUN_SSL_V2_REUSE_OPTION,
    import_existing_baselines: bool = _RUN_SSL_V2_IMPORT_BASELINES_OPTION,
    smoke_test: bool = _RUN_SSL_V2_SMOKE_OPTION,
) -> None:
    """Run the FI-2010 second-generation SSL benchmark."""
    try:
        fold_tokens = _parse_neural_fold_selection(folds)
        horizon_tokens = _parse_int_selection(
            horizons,
            option_name="--horizons",
            positive=True,
        )
        seed_tokens = _parse_int_selection(
            seeds,
            option_name="--seeds",
            positive=False,
        )
        lookback_tokens = _parse_int_selection(
            lookbacks,
            option_name="--lookbacks",
            positive=True,
        )
        objective_tokens = _parse_model_selection(objectives)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    exit_code = _run_fi2010_ssl_v2_benchmark_impl(
        config_path=config,
        processed_root=processed_root,
        out=out,
        baseline_source=baseline_source,
        folds=fold_tokens,
        horizons=horizon_tokens,
        seeds=seed_tokens,
        lookbacks=lookback_tokens,
        objectives=objective_tokens,
        pretrain_epochs=pretrain_epochs,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        mask_probability=mask_probability,
        future_bucket_count=future_bucket_count,
        contrastive=contrastive,
        device=device,
        reuse_completed=reuse_completed,
        import_existing_baselines=import_existing_baselines,
        smoke_test=smoke_test,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def run_fi2010_neural_full_grid(
    config: Path = _RUN_FULL_GRID_CONFIG_OPTION,
    processed_root: Path = _RUN_FULL_GRID_PROCESSED_ROOT_OPTION,
    out: Path = _RUN_FULL_GRID_OUT_OPTION,
    folds: str = _RUN_FULL_GRID_FOLDS_OPTION,
    horizons: str = _RUN_FULL_GRID_HORIZONS_OPTION,
    seeds: str = _RUN_FULL_GRID_SEEDS_OPTION,
    lookbacks: str = _RUN_FULL_GRID_LOOKBACKS_OPTION,
    objectives: str = _RUN_FULL_GRID_OBJECTIVES_OPTION,
    pretrain_epochs: int = _RUN_FULL_GRID_PRETRAIN_EPOCHS_OPTION,
    max_epochs: int = _RUN_FULL_GRID_MAX_EPOCHS_OPTION,
    batch_size: int = _RUN_FULL_GRID_BATCH_SIZE_OPTION,
    device: str = _RUN_FULL_GRID_DEVICE_OPTION,
    reuse_completed: bool = _RUN_FULL_GRID_REUSE_OPTION,
    smoke_test: bool = _RUN_FULL_GRID_SMOKE_OPTION,
) -> None:
    """Run the FI-2010 supervised-versus-SSL neural comparison grid."""
    try:
        fold_tokens = _parse_neural_fold_selection(folds)
        horizon_tokens = _parse_int_selection(
            horizons,
            option_name="--horizons",
            positive=True,
        )
        seed_tokens = _parse_int_selection(
            seeds,
            option_name="--seeds",
            positive=False,
        )
        lookback_tokens = _parse_int_selection(
            lookbacks,
            option_name="--lookbacks",
            positive=True,
        )
        objective_tokens = _parse_model_selection(objectives)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    exit_code = _run_fi2010_neural_full_grid_impl(
        config_path=config,
        processed_root=processed_root,
        out=out,
        folds=fold_tokens,
        horizons=horizon_tokens,
        seeds=seed_tokens,
        lookbacks=lookback_tokens,
        objectives=objective_tokens,
        pretrain_epochs=pretrain_epochs,
        max_epochs=max_epochs,
        batch_size=batch_size,
        device=device,
        reuse_completed=reuse_completed,
        smoke_test=smoke_test,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def run_fi2010_neural_proper_training_subset(
    config: Path = _RUN_PT_CONFIG_OPTION,
    processed_root: Path = _RUN_PT_PROCESSED_ROOT_OPTION,
    out: Path = _RUN_PT_OUT_OPTION,
    folds: str = _RUN_PT_FOLDS_OPTION,
    horizons: str = _RUN_PT_HORIZONS_OPTION,
    seeds: str = _RUN_PT_SEEDS_OPTION,
    lookbacks: str = _RUN_PT_LOOKBACKS_OPTION,
    models: str = _RUN_PT_MODELS_OPTION,
    objectives: str = _RUN_PT_OBJECTIVES_OPTION,
    pretrain_epochs: int = _RUN_PT_PRETRAIN_EPOCHS_OPTION,
    max_epochs: int = _RUN_PT_MAX_EPOCHS_OPTION,
    patience: int = _RUN_PT_PATIENCE_OPTION,
    batch_size: int = _RUN_PT_BATCH_SIZE_OPTION,
    device: str = _RUN_PT_DEVICE_OPTION,
    reuse_completed: bool = _RUN_PT_REUSE_OPTION,
    smoke_test: bool = _RUN_PT_SMOKE_OPTION,
) -> None:
    """Run the FI-2010 validation-selected neural benchmark."""
    try:
        fold_tokens = _parse_neural_fold_selection(folds)
        horizon_tokens = _parse_int_selection(horizons, option_name="--horizons", positive=True)
        seed_tokens = _parse_int_selection(seeds, option_name="--seeds", positive=False)
        lookback_tokens = _parse_int_selection(lookbacks, option_name="--lookbacks", positive=True)
        model_tokens = _parse_model_selection(models)
        objective_tokens = _parse_model_selection(objectives)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    exit_code = _run_fi2010_neural_proper_training_subset_impl(
        config_path=config,
        processed_root=processed_root,
        out=out,
        folds=fold_tokens,
        horizons=horizon_tokens,
        seeds=seed_tokens,
        lookbacks=lookback_tokens,
        models=model_tokens,
        objectives=objective_tokens,
        pretrain_epochs=pretrain_epochs,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        device=device,
        reuse_completed=reuse_completed,
        smoke_test=smoke_test,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def build_fi2010_figures(
    neural_full_grid: Path = _BUILD_FI2010_FIGURES_GRID_OPTION,
    out: Path = _BUILD_FI2010_FIGURES_OUT_OPTION,
    execution_v3: Path | None = _BUILD_FI2010_FIGURES_EXECUTION_V3_OPTION,
    models: str = _BUILD_FI2010_FIGURES_MODELS_OPTION,
    horizons: str = _BUILD_FI2010_FIGURES_HORIZONS_OPTION,
    folds: str = _BUILD_FI2010_FIGURES_FOLDS_OPTION,
    seeds: str = _BUILD_FI2010_FIGURES_SEEDS_OPTION,
    overwrite: bool = _BUILD_FI2010_FIGURES_OVERWRITE_OPTION,
    allow_smoke_test: bool = _BUILD_FI2010_FIGURES_ALLOW_SMOKE_OPTION,
    strict: bool = _BUILD_FI2010_FIGURES_STRICT_OPTION,
) -> None:
    """Generate FI-2010 neural full-grid figures from stored artefacts."""
    try:
        figure_models = _parse_model_selection(models)
        figure_horizons = _parse_int_selection(
            horizons,
            option_name="--horizons",
            positive=True,
        )
        figure_folds = _parse_neural_fold_selection(folds)
        figure_seeds = _parse_int_selection(
            seeds,
            option_name="--seeds",
            positive=False,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    exit_code = _build_fi2010_figures_impl(
        neural_full_grid=neural_full_grid,
        out=out,
        execution_v3=execution_v3,
        models=figure_models,
        horizons=figure_horizons,
        folds=figure_folds,
        seeds=figure_seeds,
        overwrite=overwrite,
        allow_smoke_test=allow_smoke_test,
        strict=strict,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def audit_fi2010_features(
    path: Path = _AUDIT_FI2010_FEATURES_PATH_OPTION,
    feature_groups: str = _AUDIT_FI2010_FEATURES_GROUPS_OPTION,
    label_columns: str | None = _AUDIT_FI2010_FEATURES_LABELS_OPTION,
    split_column: str | None = _AUDIT_FI2010_FEATURES_SPLIT_OPTION,
    strict: bool = _AUDIT_FI2010_FEATURES_STRICT_OPTION,
    volatility_window: int = _AUDIT_FI2010_FEATURES_VOL_WINDOW_OPTION,
) -> None:
    """Audit leakage controls for FI-2010 microstructure features."""
    exit_code = _audit_fi2010_features_impl(
        path=path,
        feature_groups=feature_groups,
        label_columns=label_columns,
        split_column=split_column,
        strict=strict,
        volatility_window=volatility_window,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def run_fi2010_feature_ablations(
    config: Path | None = _FEATURE_ABLATIONS_CONFIG_OPTION,
    processed_root: Path | None = _FEATURE_ABLATIONS_PROCESSED_ROOT_OPTION,
    data_path: Path | None = _FEATURE_ABLATIONS_DATA_PATH_OPTION,
    folds: str = _FEATURE_ABLATIONS_FOLDS_OPTION,
    horizons: str = _FEATURE_ABLATIONS_HORIZONS_OPTION,
    seeds: str = _FEATURE_ABLATIONS_SEEDS_OPTION,
    models: str = _FEATURE_ABLATIONS_MODELS_OPTION,
    feature_groups: str = _FEATURE_ABLATIONS_GROUPS_OPTION,
    ablation_modes: str = _FEATURE_ABLATIONS_MODES_OPTION,
    out: Path = _FEATURE_ABLATIONS_OUT_OPTION,
    reuse_completed: bool = _FEATURE_ABLATIONS_REUSE_OPTION,
    strict: bool = _FEATURE_ABLATIONS_STRICT_OPTION,
    smoke_test: bool = _FEATURE_ABLATIONS_SMOKE_OPTION,
    save_predictions: bool = _FEATURE_ABLATIONS_SAVE_PREDICTIONS_OPTION,
    save_heavy_artefacts: bool = _FEATURE_ABLATIONS_SAVE_HEAVY_OPTION,
    summary_only: bool = _FEATURE_ABLATIONS_SUMMARY_ONLY_OPTION,
) -> None:
    """Run classical FI-2010 microstructure feature ablations."""
    exit_code = _run_fi2010_feature_ablations_impl(
        config_path=config,
        processed_root=processed_root,
        data_path=data_path,
        out=out,
        folds=folds,
        horizons=horizons,
        seeds=seeds,
        models=models,
        feature_groups=feature_groups,
        ablation_modes=ablation_modes,
        reuse_completed=reuse_completed,
        strict=strict,
        smoke_test=smoke_test,
        save_predictions=save_predictions,
        save_heavy_artefacts=save_heavy_artefacts,
        summary_only=summary_only,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def build_fi2010_ablation_figures(
    feature_ablations: Path = _BUILD_ABLATION_FIGURES_INPUT_OPTION,
    out: Path = _BUILD_ABLATION_FIGURES_OUT_OPTION,
    overwrite: bool = _BUILD_ABLATION_FIGURES_OVERWRITE_OPTION,
    allow_smoke_test: bool = _BUILD_ABLATION_FIGURES_ALLOW_SMOKE_OPTION,
) -> None:
    """Generate FI-2010 feature-ablation figures from stored artefacts."""
    exit_code = _build_fi2010_ablation_figures_impl(
        ablations=feature_ablations,
        out=out,
        overwrite=overwrite,
        allow_smoke_test=allow_smoke_test,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def analyse_fi2010_feature_ablations(
    feature_ablations: Path = _ANALYSE_FEATURE_ABLATIONS_INPUT_OPTION,
    extra_feature_ablations: str | None = _ANALYSE_FEATURE_ABLATIONS_EXTRA_OPTION,
    out: Path = _ANALYSE_FEATURE_ABLATIONS_OUT_OPTION,
    figures: bool = _ANALYSE_FEATURE_ABLATIONS_FIGURES_OPTION,
    overwrite: bool = _ANALYSE_FEATURE_ABLATIONS_OVERWRITE_OPTION,
    allow_smoke_test: bool = _ANALYSE_FEATURE_ABLATIONS_ALLOW_SMOKE_OPTION,
) -> None:
    """Build FI-2010 feature-ablation stability analysis from stored tables."""
    exit_code = _analyse_fi2010_feature_ablations_impl(
        feature_ablations=feature_ablations,
        extra_feature_ablations=extra_feature_ablations,
        out=out,
        figures=figures,
        overwrite=overwrite,
        allow_smoke_test=allow_smoke_test,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def analyse_fi2010_uncertainty(
    classical: Path | None = _ANALYSE_UNCERTAINTY_CLASSICAL_OPTION,
    neural: Path | None = _ANALYSE_UNCERTAINTY_NEURAL_OPTION,
    out: Path = _ANALYSE_UNCERTAINTY_OUT_OPTION,
    baseline: str = _ANALYSE_UNCERTAINTY_BASELINE_OPTION,
    ci_level: float = _ANALYSE_UNCERTAINTY_CI_OPTION,
    bootstrap_iterations: int = _ANALYSE_UNCERTAINTY_BOOTSTRAP_ITER_OPTION,
    bootstrap_seed: int = _ANALYSE_UNCERTAINTY_BOOTSTRAP_SEED_OPTION,
    overwrite: bool = _ANALYSE_UNCERTAINTY_OVERWRITE_OPTION,
) -> None:
    """Compute uncertainty artefacts from stored multi-fold tables."""
    exit_code = _analyse_fi2010_uncertainty_impl(
        classical_dir=classical,
        neural_dir=neural,
        out=out,
        baseline_model=baseline,
        ci_level=ci_level,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
        overwrite=overwrite,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def analyse_fi2010_ssl_v2_results(
    ssl_v2: Path = _ANALYSE_SSL_V2_INPUT_OPTION,
    out: Path = _ANALYSE_SSL_V2_OUT_OPTION,
) -> None:
    """Build the SSL-v2 analysis from stored benchmark artefacts."""
    exit_code = _analyse_fi2010_ssl_v2_results_impl(
        ssl_v2_dir=ssl_v2,
        out=out,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def analyse_fi2010_ssl_results(
    full_grid: Path = _ANALYSE_SSL_FULL_GRID_OPTION,
    proper_training: Path = _ANALYSE_SSL_PROPER_TRAINING_OPTION,
    out: Path = _ANALYSE_SSL_OUT_OPTION,
    figures: bool = _ANALYSE_SSL_FIGURES_OPTION,
    overwrite: bool = _ANALYSE_SSL_OVERWRITE_OPTION,
) -> None:
    """Build the SSL failure-analysis report from stored comparison tables."""
    exit_code = _analyse_fi2010_ssl_results_impl(
        full_grid_dir=full_grid,
        proper_training_dir=proper_training,
        out=out,
        make_figures=figures,
        overwrite=overwrite,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def analyse_fi2010_execution_v3(
    execution_v3: Path = _ANALYSE_EXEC_V3_INPUT_OPTION,
    out: Path = _ANALYSE_EXEC_V3_OUT_OPTION,
    figures: bool = _ANALYSE_EXEC_V3_FIGURES_OPTION,
    overwrite: bool = _ANALYSE_EXEC_V3_OVERWRITE_OPTION,
) -> None:
    """Build the execution-v3 proxy analysis from stored summary tables."""
    exit_code = _analyse_fi2010_execution_v3_impl(
        execution_v3_dir=execution_v3,
        out=out,
        make_figures=figures,
        overwrite=overwrite,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def build_execution_centrepiece(
    execution_analysis: Path = _EXECUTION_CENTREPIECE_ANALYSIS_OPTION,
    out: Path = _EXECUTION_CENTREPIECE_OUT_OPTION,
    execution_v3: Path | None = _EXECUTION_CENTREPIECE_EXECUTION_V3_OPTION,
    neural_full_grid: Path | None = _EXECUTION_CENTREPIECE_FULL_GRID_OPTION,
    figures: bool = _EXECUTION_CENTREPIECE_FIGURES_OPTION,
    overwrite: bool = _EXECUTION_CENTREPIECE_OVERWRITE_OPTION,
) -> None:
    """Build the forecasting-versus-signal-quality execution centrepiece."""
    exit_code = _build_execution_centrepiece_impl(
        execution_analysis=execution_analysis,
        out=out,
        execution_v3=execution_v3,
        neural_full_grid=neural_full_grid,
        make_figures=figures,
        overwrite=overwrite,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def run_fi2010_brutal_ablations(
    config: Path = _BRUTAL_ABLATIONS_CONFIG_OPTION,
    neural_config: Path | None = _BRUTAL_ABLATIONS_NEURAL_CONFIG_OPTION,
    processed_root: Path | None = _BRUTAL_ABLATIONS_PROCESSED_ROOT_OPTION,
    classical: Path | None = _BRUTAL_ABLATIONS_CLASSICAL_OPTION,
    neural: Path | None = _BRUTAL_ABLATIONS_NEURAL_OPTION,
    out: Path = _BRUTAL_ABLATIONS_OUT_OPTION,
    families: str = _BRUTAL_ABLATIONS_FAMILIES_OPTION,
    folds: str = _BRUTAL_ABLATIONS_FOLDS_OPTION,
    models: str | None = _BRUTAL_ABLATIONS_MODELS_OPTION,
    neural_lookbacks: str | None = _BRUTAL_ABLATIONS_LOOKBACKS_OPTION,
    max_epochs: int = _BRUTAL_ABLATIONS_MAX_EPOCHS_OPTION,
    overwrite: bool = _BRUTAL_ABLATIONS_OVERWRITE_OPTION,
    dry_run: bool = _BRUTAL_ABLATIONS_DRY_RUN_OPTION,
) -> None:
    """Run the FI-2010 brutal ablation families and write artefacts."""
    exit_code = _run_fi2010_brutal_ablations_impl(
        config_path=config,
        neural_config_path=neural_config,
        processed_root=processed_root,
        classical_dir=classical,
        neural_dir=neural,
        out=out,
        families=families,
        folds=folds,
        models=models,
        neural_lookbacks=neural_lookbacks,
        max_epochs=max_epochs,
        overwrite=overwrite,
        dry_run=dry_run,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def run_fi2010_execution_v2(
    classical: Path | None = _EXECUTION_V2_CLASSICAL_OPTION,
    neural: Path | None = _EXECUTION_V2_NEURAL_OPTION,
    ablations: Path | None = _EXECUTION_V2_ABLATIONS_OPTION,
    out: Path = _EXECUTION_V2_OUT_OPTION,
    models: str | None = _EXECUTION_V2_MODELS_OPTION,
    cost_bps: str | None = _EXECUTION_V2_COST_BPS_OPTION,
    latency_steps: str | None = _EXECUTION_V2_LATENCY_OPTION,
    confidence_thresholds: str | None = _EXECUTION_V2_THRESHOLDS_OPTION,
    overwrite: bool = _EXECUTION_V2_OVERWRITE_OPTION,
) -> None:
    """Build FI-2010 execution-aware v2 proxy diagnostics from stored artefacts."""
    exit_code = _run_fi2010_execution_v2_impl(
        classical_dir=classical,
        neural_dir=neural,
        ablations_dir=ablations,
        out=out,
        models=models,
        cost_bps=cost_bps,
        latency_steps=latency_steps,
        confidence_thresholds=confidence_thresholds,
        overwrite=overwrite,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def build_fi2010_execution_v3(
    neural_full_grid: Path = _EXECUTION_V3_GRID_OPTION,
    feature_ablations: Path | None = _EXECUTION_V3_FEATURE_ABLATIONS_OPTION,
    out: Path = _EXECUTION_V3_OUT_OPTION,
    models: str = _EXECUTION_V3_MODELS_OPTION,
    horizons: str = _EXECUTION_V3_HORIZONS_OPTION,
    folds: str = _EXECUTION_V3_FOLDS_OPTION,
    seeds: str = _EXECUTION_V3_SEEDS_OPTION,
    confidence_thresholds: str | None = _EXECUTION_V3_THRESHOLDS_OPTION,
    fee_bps: str | None = _EXECUTION_V3_FEE_BPS_OPTION,
    spread_multipliers: str | None = _EXECUTION_V3_SPREAD_MULTIPLIERS_OPTION,
    latency_steps: str | None = _EXECUTION_V3_LATENCY_OPTION,
    fill_assumptions: str | None = _EXECUTION_V3_FILL_OPTION,
    allow_smoke_test: bool = _EXECUTION_V3_ALLOW_SMOKE_OPTION,
    strict: bool = _EXECUTION_V3_STRICT_OPTION,
    overwrite: bool = _EXECUTION_V3_OVERWRITE_OPTION,
) -> None:
    """Build FI-2010 execution-aware proxy diagnostic v3 from predictions."""
    exit_code = _build_fi2010_execution_v3_impl(
        neural_full_grid=neural_full_grid,
        feature_ablations=feature_ablations,
        out=out,
        models=models,
        horizons=horizons,
        folds=folds,
        seeds=seeds,
        confidence_thresholds=confidence_thresholds,
        fee_bps=fee_bps,
        spread_multipliers=spread_multipliers,
        latency_steps=latency_steps,
        fill_assumptions=fill_assumptions,
        allow_smoke_test=allow_smoke_test,
        strict=strict,
        overwrite=overwrite,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def run_paper_experiment(
    config: Path = _RUN_PAPER_EXPERIMENT_CONFIG_OPTION,
    data_path: Path = _RUN_PAPER_EXPERIMENT_DATA_PATH_OPTION,
    out: Path = _RUN_PAPER_EXPERIMENT_OUT_OPTION,
    models: str = _RUN_PAPER_EXPERIMENT_MODELS_OPTION,
    overwrite: bool = _RUN_PAPER_EXPERIMENT_OVERWRITE_OPTION,
    build_plots: bool = _RUN_PAPER_EXPERIMENT_BUILD_PLOTS_OPTION,
) -> None:
    """Run the paper experiment runner and write artefacts."""
    model_tokens = [token.strip() for token in models.split(",") if token.strip()]
    exit_code = _run_paper_experiment_impl(
        config_path=config,
        data_path=data_path,
        out=out,
        models=model_tokens or None,
        overwrite=overwrite,
        build_plots=build_plots,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def run_paper_ablations(
    config: Path = _RUN_PAPER_ABLATIONS_CONFIG_OPTION,
    data_path: Path = _RUN_PAPER_ABLATIONS_DATA_PATH_OPTION,
    out: Path = _RUN_PAPER_ABLATIONS_OUT_OPTION,
    models: str = _RUN_PAPER_ABLATIONS_MODELS_OPTION,
    ablation_set: str = _RUN_PAPER_ABLATIONS_SET_OPTION,
    overwrite: bool = _RUN_PAPER_ABLATIONS_OVERWRITE_OPTION,
    build_plots: bool = _RUN_PAPER_ABLATIONS_BUILD_PLOTS_OPTION,
) -> None:
    """Run the paper-experiment ablation suite and write artefacts."""
    model_tokens = [token.strip() for token in models.split(",") if token.strip()]
    exit_code = _run_paper_ablations_impl(
        config_path=config,
        data_path=data_path,
        out=out,
        models=model_tokens or None,
        ablation_set=ablation_set,
        overwrite=overwrite,
        build_plots=build_plots,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def run_system_benchmarks(
    config: Path = _RUN_SYSTEM_BENCHMARKS_CONFIG_OPTION,
    data_path: Path = _RUN_SYSTEM_BENCHMARKS_DATA_PATH_OPTION,
    out: Path = _RUN_SYSTEM_BENCHMARKS_OUT_OPTION,
    benchmark_set: str = _RUN_SYSTEM_BENCHMARKS_SET_OPTION,
    models: str = _RUN_SYSTEM_BENCHMARKS_MODELS_OPTION,
    overwrite: bool = _RUN_SYSTEM_BENCHMARKS_OVERWRITE_OPTION,
) -> None:
    """Run local systems benchmarks and write artefacts."""
    model_tokens = [token.strip() for token in models.split(",") if token.strip()]
    exit_code = _run_system_benchmarks_impl(
        config_path=config,
        data_path=data_path,
        out=out,
        benchmark_set=benchmark_set,
        models=model_tokens or None,
        overwrite=overwrite,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def inspect_system_benchmarks(
    benchmark: Path = _INSPECT_SYSTEM_BENCHMARKS_BENCHMARK_OPTION,
) -> None:
    """Print a concise systems benchmark summary."""
    exit_code = _inspect_system_benchmarks_impl(benchmark=benchmark)
    if exit_code != 0:
        raise SystemExit(exit_code)


def build_paper_plots(
    experiment: Path = _BUILD_PAPER_PLOTS_EXPERIMENT_OPTION,
    overwrite: bool = _BUILD_PAPER_PLOTS_OVERWRITE_OPTION,
) -> None:
    """Generate paper experiment plots from stored artefacts."""
    exit_code = _build_paper_plots_impl(
        experiment=experiment,
        overwrite=overwrite,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def inspect_paper_experiment(
    experiment: Path = _INSPECT_PAPER_EXPERIMENT_EXPERIMENT_OPTION,
) -> None:
    """Print a concise paper experiment artefact summary."""
    exit_code = _inspect_paper_experiment_impl(experiment=experiment)
    if exit_code != 0:
        raise SystemExit(exit_code)


def build_paper_report(
    experiment: Path = _BUILD_PAPER_REPORT_EXPERIMENT_OPTION,
    ablations: Path | None = _BUILD_PAPER_REPORT_ABLATIONS_OPTION,
    systems: Path | None = _BUILD_PAPER_REPORT_SYSTEMS_OPTION,
    out: Path = _BUILD_PAPER_REPORT_OUT_OPTION,
    overwrite: bool = _BUILD_PAPER_REPORT_OVERWRITE_OPTION,
) -> None:
    """Build an empirical report from stored artefacts."""
    exit_code = _build_paper_report_impl(
        experiment=experiment,
        ablations=ablations,
        systems=systems,
        out=out,
        overwrite=overwrite,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def inspect_paper_report(
    report: Path = _INSPECT_PAPER_REPORT_REPORT_OPTION,
) -> None:
    """Inspect a generated empirical report summary."""
    exit_code = _inspect_paper_report_impl(report=report)
    if exit_code != 0:
        raise SystemExit(exit_code)


_SYNTHETIC_LOB_OUT_OPTION = typer.Option(
    Path("reports/synthetic_lob_extension"),
    "--out",
    help="Output directory for synthetic LOB extension artefacts.",
)

_SYNTHETIC_LOB_EVENTS_OPTION = typer.Option(
    3000,
    "--events-per-regime",
    help="Synthetic events generated per regime.",
)

_SYNTHETIC_LOB_SEED_OPTION = typer.Option(
    0,
    "--seed",
    help="Deterministic generation and benchmark seed.",
)

_SYNTHETIC_LOB_HORIZON_OPTION = typer.Option(
    20,
    "--horizon",
    help="Future label horizon in snapshot steps.",
)

_SYNTHETIC_LOB_SMOKE_OPTION = typer.Option(
    False,
    "--smoke",
    help="Run a tiny fast smoke configuration.",
)

_SYNTHETIC_LOB_FIGURES_OPTION = typer.Option(
    False,
    "--make-figures",
    help="Render compact figures when matplotlib is available.",
)

_SYNTHETIC_LOB_OVERWRITE_OPTION = typer.Option(
    False,
    "--overwrite",
    help="Overwrite an existing synthetic report directory.",
)


def run_synthetic_lob_benchmark(
    out: Path = _SYNTHETIC_LOB_OUT_OPTION,
    events_per_regime: int = _SYNTHETIC_LOB_EVENTS_OPTION,
    seed: int = _SYNTHETIC_LOB_SEED_OPTION,
    horizon: int = _SYNTHETIC_LOB_HORIZON_OPTION,
    smoke: bool = _SYNTHETIC_LOB_SMOKE_OPTION,
    make_figures: bool = _SYNTHETIC_LOB_FIGURES_OPTION,
    overwrite: bool = _SYNTHETIC_LOB_OVERWRITE_OPTION,
) -> None:
    """Run the synthetic event-level LOB pipeline and write artefacts."""
    exit_code = _run_synthetic_lob_benchmark_impl(
        out=out,
        events_per_regime=events_per_regime,
        seed=seed,
        horizon=horizon,
        smoke=smoke,
        make_figures=make_figures,
        overwrite=overwrite,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


_BINANCE_L2_OUT_OPTION = typer.Option(
    Path("reports/binance_l2_extension"),
    "--out",
    help="Output directory for Binance L2 extension artefacts.",
)

_BINANCE_L2_SNAPSHOT_OPTION = typer.Option(
    None,
    "--snapshot",
    help="Local depth snapshot JSON; defaults to the bundled fixture.",
)

_BINANCE_L2_UPDATES_OPTION = typer.Option(
    None,
    "--updates",
    help="Local diff-depth JSONL; defaults to the bundled fixture.",
)

_BINANCE_L2_SYMBOL_OPTION = typer.Option(
    None,
    "--symbol",
    help="Symbol override when the snapshot omits it.",
)

_BINANCE_L2_MAX_DEPTH_OPTION = typer.Option(
    None,
    "--max-depth",
    help="Trim the book to this many levels per side.",
)

_BINANCE_L2_WINDOW_OPTION = typer.Option(
    20,
    "--window-events",
    help="Trailing diff-event window for event-flow features.",
)

_BINANCE_L2_NO_STOP_ON_GAP_OPTION = typer.Option(
    False,
    "--no-stop-on-gap",
    help="Continue reconstruction after an update-id gap is detected.",
)

_BINANCE_L2_ALLOW_CROSSED_OPTION = typer.Option(
    False,
    "--allow-crossed",
    help="Permit crossed books instead of treating them as errors.",
)

_BINANCE_L2_FIGURES_OPTION = typer.Option(
    False,
    "--make-figures",
    help="Render compact replay figures when matplotlib is available.",
)

_BINANCE_L2_OVERWRITE_OPTION = typer.Option(
    False,
    "--overwrite",
    help="Overwrite an existing Binance L2 report directory.",
)


def replay_binance_l2_sample(
    out: Path = _BINANCE_L2_OUT_OPTION,
    snapshot: Path | None = _BINANCE_L2_SNAPSHOT_OPTION,
    updates: Path | None = _BINANCE_L2_UPDATES_OPTION,
    symbol: str | None = _BINANCE_L2_SYMBOL_OPTION,
    max_depth: int | None = _BINANCE_L2_MAX_DEPTH_OPTION,
    window_events: int = _BINANCE_L2_WINDOW_OPTION,
    no_stop_on_gap: bool = _BINANCE_L2_NO_STOP_ON_GAP_OPTION,
    allow_crossed: bool = _BINANCE_L2_ALLOW_CROSSED_OPTION,
    make_figures: bool = _BINANCE_L2_FIGURES_OPTION,
    overwrite: bool = _BINANCE_L2_OVERWRITE_OPTION,
) -> None:
    """Replay a local Binance L2 snapshot-plus-diff sample and write artefacts."""
    exit_code = _replay_binance_l2_sample_impl(
        out=out,
        snapshot=snapshot,
        updates=updates,
        symbol=symbol,
        max_depth=max_depth,
        window_events=window_events,
        stop_on_gap=not no_stop_on_gap,
        allow_crossed=allow_crossed,
        make_figures=make_figures,
        overwrite=overwrite,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def inspect_event_log(
    path: Path = _EVENT_LOG_PATH_OPTION,
) -> None:
    """Inspect a local canonical event-log JSONL file."""
    exit_code = _inspect_event_log_impl(path)
    if exit_code != 0:
        raise SystemExit(exit_code)


def inspect_event_tokens(
    path: Path = _EVENT_LOG_PATH_OPTION,
    symbol: str | None = _EVENT_TOKENS_SYMBOL_OPTION,
    window_length: int = _EVENT_TOKENS_WINDOW_LENGTH_OPTION,
    max_levels_per_side: int = _EVENT_TOKENS_MAX_LEVELS_OPTION,
    include_eos: bool = _EVENT_TOKENS_INCLUDE_EOS_OPTION,
) -> None:
    """Tokenise a local canonical event log and summarise IDs."""
    exit_code = _inspect_event_tokens_impl(
        path=path,
        symbol=symbol,
        window_length=window_length,
        max_levels_per_side=max_levels_per_side,
        include_eos=include_eos,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def event_log_to_features(
    path: Path = _EVENT_LOG_PATH_OPTION,
) -> None:
    """Replay a local event log into feature rows and summarise."""
    exit_code = _event_log_to_features_impl(path)
    if exit_code != 0:
        raise SystemExit(exit_code)


def inspect_fi2010(
    path: Path = _INSPECT_PATH_OPTION,
    timestamp_column: str = _INSPECT_TIMESTAMP_OPTION,
    split_column: str = _INSPECT_SPLIT_OPTION,
    price_level_count: int = _INSPECT_LEVEL_COUNT_OPTION,
    no_timestamp_column: bool = _INSPECT_NO_TIMESTAMP_OPTION,
    no_split_column: bool = _INSPECT_NO_SPLIT_OPTION,
) -> None:
    """Load an FI-2010 file and print a data-quality summary."""
    exit_code = _inspect_fi2010_impl(
        path=path,
        timestamp_column=(None if no_timestamp_column else timestamp_column),
        split_column=None if no_split_column else split_column,
        price_level_count=price_level_count,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


_INSPECT_FEAT_ALLOW_SYNTHETIC_OPTION = typer.Option(
    False,
    "--allow-synthetic-time",
    help=("Compute time-window features even when timestamps are synthetic. Off by default."),
)

_INSPECT_LABELS_GENERATE_OPTION = typer.Option(
    False,
    "--generate-labels",
    help=(
        "Generate ChronosLOB labels from snapshots instead of "
        "preferring configured FI-2010 benchmark labels."
    ),
)

_INSPECT_SPLIT_ROWS_OPTION = typer.Option(
    ...,
    "--rows",
    help="Number of ordered rows to split.",
)

_INIT_RUN_NAME_OPTION = typer.Option(
    ...,
    "--name",
    help="Readable run name.",
)

_INIT_RUN_PHASE_OPTION = typer.Option(
    ...,
    "--phase",
    help="Run grouping identifier.",
)

_INIT_RUN_SEED_OPTION = typer.Option(
    ...,
    "--seed",
    help="Deterministic run seed.",
)

_INIT_RUN_ROOT_OPTION = typer.Option(
    ...,
    "--root",
    help="Root directory for runs.",
)

_INIT_RUN_CONFIG_PATH_OPTION = typer.Option(
    None,
    "--config-path",
    help="Optional local config to copy into the run directory.",
)

_INIT_RUN_NOTES_OPTION = typer.Option(
    None,
    "--notes",
    help="Optional short run notes.",
)

_BASELINE_SMOKE_PATH_OPTION = typer.Option(
    ...,
    "--path",
    help="Path to the bundled synthetic FI-2010-style fixture.",
)

_BASELINE_SMOKE_WRITE_OUTPUTS_OPTION = typer.Option(
    False,
    "--write-outputs",
    help="Write run metadata and metrics under the output root.",
)

_BASELINE_SMOKE_OUTPUT_ROOT_OPTION = typer.Option(
    Path("runs"),
    "--output-root",
    help="Output root used only when --write-outputs is passed.",
)


def inspect_features_fi2010(
    path: Path = _INSPECT_PATH_OPTION,
    timestamp_column: str = _INSPECT_TIMESTAMP_OPTION,
    split_column: str = _INSPECT_SPLIT_OPTION,
    price_level_count: int = _INSPECT_LEVEL_COUNT_OPTION,
    no_timestamp_column: bool = _INSPECT_NO_TIMESTAMP_OPTION,
    no_split_column: bool = _INSPECT_NO_SPLIT_OPTION,
    allow_synthetic_time: bool = _INSPECT_FEAT_ALLOW_SYNTHETIC_OPTION,
) -> None:
    """Build microstructure features from an FI-2010 file and summarise."""
    exit_code = _inspect_features_fi2010_impl(
        path=path,
        timestamp_column=(None if no_timestamp_column else timestamp_column),
        split_column=None if no_split_column else split_column,
        price_level_count=price_level_count,
        allow_synthetic_timestamps_for_time_features=allow_synthetic_time,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def inspect_labels_fi2010(
    path: Path = _INSPECT_PATH_OPTION,
    timestamp_column: str = _INSPECT_TIMESTAMP_OPTION,
    split_column: str = _INSPECT_SPLIT_OPTION,
    price_level_count: int = _INSPECT_LEVEL_COUNT_OPTION,
    no_timestamp_column: bool = _INSPECT_NO_TIMESTAMP_OPTION,
    no_split_column: bool = _INSPECT_NO_SPLIT_OPTION,
    generate_labels: bool = _INSPECT_LABELS_GENERATE_OPTION,
) -> None:
    """Build or extract FI-2010 labels and summarise."""
    exit_code = _inspect_labels_fi2010_impl(
        path=path,
        timestamp_column=(None if no_timestamp_column else timestamp_column),
        split_column=None if no_split_column else split_column,
        price_level_count=price_level_count,
        prefer_existing_labels=not generate_labels,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def inspect_split(
    rows: int = _INSPECT_SPLIT_ROWS_OPTION,
) -> None:
    """Build a default temporal split and print partition counts."""
    exit_code = _inspect_split_impl(rows)
    if exit_code != 0:
        raise SystemExit(exit_code)


def init_run(
    name: str = _INIT_RUN_NAME_OPTION,
    phase: str = _INIT_RUN_PHASE_OPTION,
    seed: int = _INIT_RUN_SEED_OPTION,
    root: Path = _INIT_RUN_ROOT_OPTION,
    config_path: Path | None = _INIT_RUN_CONFIG_PATH_OPTION,
    notes: str | None = _INIT_RUN_NOTES_OPTION,
) -> None:
    """Create a metadata-only experiment run directory."""
    exit_code = _init_run_impl(
        name=name,
        phase=phase,
        seed=seed,
        root=root,
        config_path=config_path,
        notes=notes,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def inspect_baselines() -> None:
    """Print supported classical baseline model types."""
    exit_code = _inspect_baselines_impl()
    if exit_code != 0:
        raise SystemExit(exit_code)


def run_baseline_smoke(
    path: Path = _BASELINE_SMOKE_PATH_OPTION,
    write_outputs: bool = _BASELINE_SMOKE_WRITE_OUTPUTS_OPTION,
    output_root: Path = _BASELINE_SMOKE_OUTPUT_ROOT_OPTION,
) -> None:
    """Run a synthetic fixture baseline smoke test."""
    exit_code = _run_baseline_smoke_impl(
        path=path,
        write_outputs=write_outputs,
        output_root=output_root,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


_TORCH_DATASET_PATH_OPTION = typer.Option(
    ...,
    "--path",
    help="Path to the local FI-2010-style file.",
)

_TORCH_DATASET_LOOKBACK_OPTION = typer.Option(
    2,
    "--lookback",
    help="Number of past rows per sequence window.",
)

_TORCH_DATASET_BATCH_SIZE_OPTION = typer.Option(
    4,
    "--batch-size",
    help="Batch size for the smoke DataLoader.",
)

_TORCH_DATASET_TARGET_OPTION = typer.Option(
    "label_10",
    "--target-column",
    help="Label column to use as the supervised target.",
)

_TORCH_DATASET_TRAIN_FRACTION_OPTION = typer.Option(
    0.5,
    "--train-fraction",
    help="Train fraction for the temporal split.",
)

_TORCH_DATASET_VALIDATION_FRACTION_OPTION = typer.Option(
    0.34,
    "--validation-fraction",
    help="Validation fraction for the temporal split.",
)

_TORCH_DATASET_TEST_FRACTION_OPTION = typer.Option(
    0.16,
    "--test-fraction",
    help="Test fraction for the temporal split.",
)


def inspect_torch_dataset(
    path: Path = _TORCH_DATASET_PATH_OPTION,
    lookback: int = _TORCH_DATASET_LOOKBACK_OPTION,
    batch_size: int = _TORCH_DATASET_BATCH_SIZE_OPTION,
    target_column: str = _TORCH_DATASET_TARGET_OPTION,
    timestamp_column: str = _INSPECT_TIMESTAMP_OPTION,
    split_column: str = _INSPECT_SPLIT_OPTION,
    price_level_count: int = _INSPECT_LEVEL_COUNT_OPTION,
    no_timestamp_column: bool = _INSPECT_NO_TIMESTAMP_OPTION,
    no_split_column: bool = _INSPECT_NO_SPLIT_OPTION,
    train_fraction: float = _TORCH_DATASET_TRAIN_FRACTION_OPTION,
    validation_fraction: float = _TORCH_DATASET_VALIDATION_FRACTION_OPTION,
    test_fraction: float = _TORCH_DATASET_TEST_FRACTION_OPTION,
) -> None:
    """Build a tiny sequence DataLoader from an FI-2010 file and summarise."""
    exit_code = _inspect_torch_dataset_impl(
        path=path,
        lookback=lookback,
        batch_size=batch_size,
        target_column=target_column,
        timestamp_column=(None if no_timestamp_column else timestamp_column),
        split_column=None if no_split_column else split_column,
        price_level_count=price_level_count,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


_DEEPLOB_SMOKE_PATH_OPTION = typer.Option(
    ...,
    "--path",
    help="Path to the local FI-2010-style fixture file.",
)

_DEEPLOB_SMOKE_LOOKBACK_OPTION = typer.Option(
    2,
    "--lookback",
    help="Number of past rows per sequence window.",
)

_DEEPLOB_SMOKE_EPOCHS_OPTION = typer.Option(
    1,
    "--epochs",
    help="Number of training epochs for the smoke run.",
)

_DEEPLOB_SMOKE_BATCH_OPTION = typer.Option(
    4,
    "--batch-size",
    help="Batch size for the smoke DataLoader.",
)

_DEEPLOB_SMOKE_SEED_OPTION = typer.Option(
    42,
    "--seed",
    help="Deterministic seed for the smoke run.",
)

_DEEPLOB_SMOKE_WRITE_OUTPUTS_OPTION = typer.Option(
    False,
    "--write-outputs",
    help=(
        "The smoke command never writes outputs; this flag is accepted "
        "for symmetry with run-baseline-smoke but only prints a notice."
    ),
)

_DEEPLOB_SMOKE_OUTPUT_ROOT_OPTION = typer.Option(
    Path("runs"),
    "--output-root",
    help="Reserved for future use; ignored by the smoke command.",
)


def inspect_deeplob() -> None:
    """Print DeepLOB-style baseline defaults without training."""
    exit_code = _inspect_deeplob_impl()
    if exit_code != 0:
        raise SystemExit(exit_code)


def run_deeplob_smoke(
    path: Path = _DEEPLOB_SMOKE_PATH_OPTION,
    lookback: int = _DEEPLOB_SMOKE_LOOKBACK_OPTION,
    epochs: int = _DEEPLOB_SMOKE_EPOCHS_OPTION,
    batch_size: int = _DEEPLOB_SMOKE_BATCH_OPTION,
    seed: int = _DEEPLOB_SMOKE_SEED_OPTION,
    write_outputs: bool = _DEEPLOB_SMOKE_WRITE_OUTPUTS_OPTION,
    output_root: Path = _DEEPLOB_SMOKE_OUTPUT_ROOT_OPTION,
) -> None:
    """Run a synthetic fixture DeepLOB-style supervised smoke experiment."""
    exit_code = _run_deeplob_smoke_impl(
        path=path,
        lookback=lookback,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        write_outputs=write_outputs,
        output_root=output_root,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


_TRANSFORMER_SMOKE_PATH_OPTION = typer.Option(
    ...,
    "--path",
    help="Path to the local canonical event-log JSONL fixture.",
)

_TRANSFORMER_SMOKE_WINDOW_OPTION = typer.Option(
    4,
    "--window-length",
    help="Fixed token-window length used for the smoke run.",
)

_TRANSFORMER_SMOKE_BATCH_OPTION = typer.Option(
    4,
    "--batch-size",
    help="Batch size for the smoke DataLoader.",
)

_TRANSFORMER_SMOKE_EPOCHS_OPTION = typer.Option(
    1,
    "--epochs",
    help="Number of training epochs for the smoke run.",
)

_TRANSFORMER_SMOKE_SEED_OPTION = typer.Option(
    42,
    "--seed",
    help="Deterministic seed for the smoke run.",
)

_TRANSFORMER_SMOKE_NUM_CLASSES_OPTION = typer.Option(
    3,
    "--num-classes",
    help="Number of classes used by the synthetic smoke test.",
)

_TRANSFORMER_SMOKE_SYMBOL_OPTION = typer.Option(
    None,
    "--symbol",
    help="Optional symbol filter applied to the event log.",
)

_TRANSFORMER_SMOKE_LEVELS_OPTION = typer.Option(
    2,
    "--max-levels-per-side",
    help="Maximum snapshot levels per side to tokenise.",
)


def inspect_transformer() -> None:
    """Print market transformer encoder defaults without training."""
    exit_code = _inspect_transformer_impl()
    if exit_code != 0:
        raise SystemExit(exit_code)


def run_transformer_smoke(
    path: Path = _TRANSFORMER_SMOKE_PATH_OPTION,
    window_length: int = _TRANSFORMER_SMOKE_WINDOW_OPTION,
    batch_size: int = _TRANSFORMER_SMOKE_BATCH_OPTION,
    epochs: int = _TRANSFORMER_SMOKE_EPOCHS_OPTION,
    seed: int = _TRANSFORMER_SMOKE_SEED_OPTION,
    num_classes: int = _TRANSFORMER_SMOKE_NUM_CLASSES_OPTION,
    symbol: str | None = _TRANSFORMER_SMOKE_SYMBOL_OPTION,
    max_levels_per_side: int = _TRANSFORMER_SMOKE_LEVELS_OPTION,
) -> None:
    """Run a synthetic-label transformer smoke experiment."""
    exit_code = _run_transformer_smoke_impl(
        path=path,
        window_length=window_length,
        batch_size=batch_size,
        epochs=epochs,
        seed=seed,
        num_classes=num_classes,
        symbol=symbol,
        max_levels_per_side=max_levels_per_side,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


_SSL_SMOKE_PATH_OPTION = typer.Option(
    ...,
    "--path",
    help="Path to the local canonical event-log JSONL fixture.",
)

_SSL_SMOKE_WINDOW_OPTION = typer.Option(
    4,
    "--window-length",
    help="Fixed token-window length used for the SSL smoke run.",
)

_SSL_SMOKE_BATCH_OPTION = typer.Option(
    4,
    "--batch-size",
    help="Batch size for the SSL smoke DataLoader.",
)

_SSL_SMOKE_EPOCHS_OPTION = typer.Option(
    1,
    "--epochs",
    help="Number of pretraining epochs for the smoke run.",
)

_SSL_SMOKE_SEED_OPTION = typer.Option(
    42,
    "--seed",
    help="Deterministic seed for the SSL smoke run.",
)

_SSL_SMOKE_SYMBOL_OPTION = typer.Option(
    None,
    "--symbol",
    help="Optional symbol filter applied to the event log.",
)

_SSL_SMOKE_LEVELS_OPTION = typer.Option(
    2,
    "--max-levels-per-side",
    help="Maximum snapshot levels per side to tokenise.",
)

_SSL_SMOKE_MASK_PROBABILITY_OPTION = typer.Option(
    0.15,
    "--mask-probability",
    help="Probability of selecting a valid position for masking.",
)


def inspect_ssl() -> None:
    """Print SSL transformer wrapper defaults without training."""
    exit_code = _inspect_ssl_impl()
    if exit_code != 0:
        raise SystemExit(exit_code)


def run_ssl_smoke(
    path: Path = _SSL_SMOKE_PATH_OPTION,
    window_length: int = _SSL_SMOKE_WINDOW_OPTION,
    batch_size: int = _SSL_SMOKE_BATCH_OPTION,
    epochs: int = _SSL_SMOKE_EPOCHS_OPTION,
    seed: int = _SSL_SMOKE_SEED_OPTION,
    symbol: str | None = _SSL_SMOKE_SYMBOL_OPTION,
    max_levels_per_side: int = _SSL_SMOKE_LEVELS_OPTION,
    mask_probability: float = _SSL_SMOKE_MASK_PROBABILITY_OPTION,
) -> None:
    """Run a tiny synthetic SSL smoke experiment."""
    exit_code = _run_ssl_smoke_impl(
        path=path,
        window_length=window_length,
        batch_size=batch_size,
        epochs=epochs,
        seed=seed,
        symbol=symbol,
        max_levels_per_side=max_levels_per_side,
        mask_probability=mask_probability,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


_MULTITASK_SMOKE_PATH_OPTION = typer.Option(
    ...,
    "--path",
    help="Path to the local canonical event-log JSONL fixture.",
)

_MULTITASK_SMOKE_WINDOW_OPTION = typer.Option(
    4,
    "--window-length",
    help="Fixed token-window length used for the multi-task smoke run.",
)

_MULTITASK_SMOKE_BATCH_OPTION = typer.Option(
    4,
    "--batch-size",
    help="Batch size for the multi-task smoke DataLoader.",
)

_MULTITASK_SMOKE_EPOCHS_OPTION = typer.Option(
    1,
    "--epochs",
    help="Number of supervised fine-tuning epochs for the smoke run.",
)

_MULTITASK_SMOKE_SEED_OPTION = typer.Option(
    42,
    "--seed",
    help="Deterministic seed for the multi-task smoke run.",
)

_MULTITASK_SMOKE_SYMBOL_OPTION = typer.Option(
    None,
    "--symbol",
    help="Optional symbol filter applied to the event log.",
)

_MULTITASK_SMOKE_LEVELS_OPTION = typer.Option(
    2,
    "--max-levels-per-side",
    help="Maximum snapshot levels per side to tokenise.",
)


def inspect_multitask() -> None:
    """Print multi-task transformer defaults without training."""
    exit_code = _inspect_multitask_impl()
    if exit_code != 0:
        raise SystemExit(exit_code)


def run_multitask_smoke(
    path: Path = _MULTITASK_SMOKE_PATH_OPTION,
    window_length: int = _MULTITASK_SMOKE_WINDOW_OPTION,
    batch_size: int = _MULTITASK_SMOKE_BATCH_OPTION,
    epochs: int = _MULTITASK_SMOKE_EPOCHS_OPTION,
    seed: int = _MULTITASK_SMOKE_SEED_OPTION,
    symbol: str | None = _MULTITASK_SMOKE_SYMBOL_OPTION,
    max_levels_per_side: int = _MULTITASK_SMOKE_LEVELS_OPTION,
) -> None:
    """Run a tiny synthetic supervised multi-task smoke experiment."""
    exit_code = _run_multitask_smoke_impl(
        path=path,
        window_length=window_length,
        batch_size=batch_size,
        epochs=epochs,
        seed=seed,
        symbol=symbol,
        max_levels_per_side=max_levels_per_side,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


_CALIBRATION_SMOKE_N_EXAMPLES_OPTION = typer.Option(
    60,
    "--n-examples",
    help="Number of deterministic synthetic examples.",
)

_CALIBRATION_SMOKE_NUM_CLASSES_OPTION = typer.Option(
    3,
    "--num-classes",
    help="Number of synthetic classification classes.",
)

_CALIBRATION_SMOKE_SEED_OPTION = typer.Option(
    42,
    "--seed",
    help="Deterministic seed for synthetic logits.",
)

_CALIBRATION_SMOKE_ECE_BINS_OPTION = typer.Option(
    10,
    "--ece-bins",
    help="Number of bins for expected calibration error.",
)


def inspect_calibration() -> None:
    """Print calibration and uncertainty support without training."""
    exit_code = _inspect_calibration_impl()
    if exit_code != 0:
        raise SystemExit(exit_code)


def run_calibration_smoke(
    n_examples: int = _CALIBRATION_SMOKE_N_EXAMPLES_OPTION,
    num_classes: int = _CALIBRATION_SMOKE_NUM_CLASSES_OPTION,
    seed: int = _CALIBRATION_SMOKE_SEED_OPTION,
    ece_bins: int = _CALIBRATION_SMOKE_ECE_BINS_OPTION,
) -> None:
    """Run a deterministic synthetic calibration smoke check."""
    exit_code = _run_calibration_smoke_impl(
        n_examples=n_examples,
        num_classes=num_classes,
        seed=seed,
        ece_bins=ece_bins,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


_EXECUTION_SMOKE_N_SIGNALS_OPTION = typer.Option(
    24,
    "--n-signals",
    help="Number of deterministic synthetic prediction signals.",
)

_EXECUTION_SMOKE_SEED_OPTION = typer.Option(
    42,
    "--seed",
    help="Deterministic seed for synthetic market-state noise.",
)


def inspect_execution_validation() -> None:
    """Print execution-aware validation support without running a model."""
    exit_code = _inspect_execution_validation_impl()
    if exit_code != 0:
        raise SystemExit(exit_code)


def run_execution_validation_smoke(
    n_signals: int = _EXECUTION_SMOKE_N_SIGNALS_OPTION,
    seed: int = _EXECUTION_SMOKE_SEED_OPTION,
) -> None:
    """Run a deterministic synthetic execution-validation smoke check."""
    exit_code = _run_execution_validation_smoke_impl(
        n_signals=n_signals,
        seed=seed,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


_ROBUSTNESS_SMOKE_N_RECORDS_OPTION = typer.Option(
    36,
    "--n-records",
    help="Number of deterministic synthetic analysis records.",
)

_ROBUSTNESS_SMOKE_SEED_OPTION = typer.Option(
    42,
    "--seed",
    help="Deterministic seed for synthetic analysis records.",
)


def inspect_analysis() -> None:
    """Print supported analysis tools without running anything."""
    exit_code = _inspect_analysis_impl()
    if exit_code != 0:
        raise SystemExit(exit_code)


def run_robustness_analysis_smoke(
    n_records: int = _ROBUSTNESS_SMOKE_N_RECORDS_OPTION,
    seed: int = _ROBUSTNESS_SMOKE_SEED_OPTION,
) -> None:
    """Run a deterministic synthetic robustness-analysis smoke check."""
    exit_code = _run_robustness_analysis_smoke_impl(
        n_records=n_records,
        seed=seed,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


_BINANCE_SNAPSHOT_OPTION = typer.Option(
    ...,
    "--snapshot",
    help="Path to the local Binance-style snapshot JSON file.",
)

_BINANCE_UPDATES_OPTION = typer.Option(
    ...,
    "--updates",
    help="Path to the local Binance-style diff JSONL file.",
)

_BINANCE_SYMBOL_OPTION = typer.Option(
    None,
    "--symbol",
    help="Optional symbol override when the snapshot does not carry one.",
)

_BINANCE_MAX_DEPTH_OPTION = typer.Option(
    None,
    "--max-depth",
    help="Optional maximum depth per side to keep during replay.",
)

_BINANCE_STOP_ON_GAP_OPTION = typer.Option(
    True,
    "--stop-on-gap/--no-stop-on-gap",
    help="Stop reconstruction when an update-id gap is detected.",
)

_BINANCE_ALLOW_CROSSED_OPTION = typer.Option(
    False,
    "--allow-crossed",
    help="Permit crossed books instead of treating them as errors.",
)


def inspect_binance_replay(
    snapshot: Path = _BINANCE_SNAPSHOT_OPTION,
    updates: Path = _BINANCE_UPDATES_OPTION,
    symbol: str | None = _BINANCE_SYMBOL_OPTION,
    max_depth: int | None = _BINANCE_MAX_DEPTH_OPTION,
    stop_on_gap: bool = _BINANCE_STOP_ON_GAP_OPTION,
    allow_crossed: bool = _BINANCE_ALLOW_CROSSED_OPTION,
) -> None:
    """Reconstruct a local Binance-style book from a snapshot and diff JSONL."""
    exit_code = _inspect_binance_replay_impl(
        snapshot_path=snapshot,
        updates_path=updates,
        symbol=symbol,
        max_depth=max_depth,
        stop_on_gap=stop_on_gap,
        allow_crossed=allow_crossed,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


app.command()(version)
app.command()(doctor)
app.command("inspect-experiment-artifacts")(inspect_experiment_artifacts)
app.command("inspect-event-log")(inspect_event_log)
app.command("inspect-event-tokens")(inspect_event_tokens)
app.command("event-log-to-features")(event_log_to_features)
app.command("inspect-fi2010")(inspect_fi2010)
app.command("inspect-features-fi2010")(inspect_features_fi2010)
app.command("inspect-labels-fi2010")(inspect_labels_fi2010)
app.command("inspect-split")(inspect_split)
app.command("init-run")(init_run)
app.command("inspect-baselines")(inspect_baselines)
app.command("run-baseline-smoke")(run_baseline_smoke)
app.command("inspect-torch-dataset")(inspect_torch_dataset)
app.command("inspect-deeplob")(inspect_deeplob)
app.command("run-deeplob-smoke")(run_deeplob_smoke)
app.command("inspect-transformer")(inspect_transformer)
app.command("run-transformer-smoke")(run_transformer_smoke)
app.command("inspect-ssl")(inspect_ssl)
app.command("run-ssl-smoke")(run_ssl_smoke)
app.command("inspect-multitask")(inspect_multitask)
app.command("run-multitask-smoke")(run_multitask_smoke)
app.command("inspect-calibration")(inspect_calibration)
app.command("run-calibration-smoke")(run_calibration_smoke)
app.command("inspect-execution-validation")(inspect_execution_validation)
app.command("run-execution-validation-smoke")(run_execution_validation_smoke)
app.command("inspect-analysis")(inspect_analysis)
app.command("run-robustness-analysis-smoke")(run_robustness_analysis_smoke)
app.command("inspect-binance-replay")(inspect_binance_replay)
app.command("prepare-fi2010-benchmark")(prepare_fi2010_benchmark)
app.command("verify-fi2010-local")(verify_fi2010_local)
app.command("convert-fi2010-official")(convert_fi2010_official)
app.command("inspect-fi2010-multifold")(inspect_fi2010_multifold)
app.command("prepare-fi2010-multifold")(prepare_fi2010_multifold)
app.command("run-fi2010-multifold-classical")(run_fi2010_multifold_classical)
app.command("inspect-fi2010-neural-plan")(inspect_fi2010_neural_plan)
app.command("run-fi2010-neural-benchmark")(run_fi2010_neural_benchmark)
app.command("run-fi2010-ssl-neural-benchmark")(run_fi2010_ssl_neural_benchmark)
app.command("run-fi2010-ssl-v2-benchmark")(run_fi2010_ssl_v2_benchmark)
app.command("run-fi2010-neural-full-grid")(run_fi2010_neural_full_grid)
app.command("run-fi2010-neural-proper-training-subset")(run_fi2010_neural_proper_training_subset)
app.command("build-fi2010-figures")(build_fi2010_figures)
app.command("audit-fi2010-features")(audit_fi2010_features)
app.command("run-fi2010-feature-ablations")(run_fi2010_feature_ablations)
app.command("build-fi2010-ablation-figures")(build_fi2010_ablation_figures)
app.command("analyse-fi2010-feature-ablations")(analyse_fi2010_feature_ablations)
app.command("analyse-fi2010-uncertainty")(analyse_fi2010_uncertainty)
app.command("analyse-fi2010-ssl-results")(analyse_fi2010_ssl_results)
app.command("analyse-fi2010-ssl-v2-results")(analyse_fi2010_ssl_v2_results)
app.command("analyse-fi2010-execution-v3")(analyse_fi2010_execution_v3)
app.command("build-execution-centrepiece")(build_execution_centrepiece)
app.command("run-fi2010-brutal-ablations")(run_fi2010_brutal_ablations)
app.command("run-fi2010-execution-v2")(run_fi2010_execution_v2)
app.command("build-fi2010-execution-v3")(build_fi2010_execution_v3)
app.command("run-paper-experiment")(run_paper_experiment)
app.command("run-paper-ablations")(run_paper_ablations)
app.command("run-system-benchmarks")(run_system_benchmarks)
app.command("inspect-system-benchmarks")(inspect_system_benchmarks)
app.command("build-paper-plots")(build_paper_plots)
app.command("inspect-paper-experiment")(inspect_paper_experiment)
app.command("build-paper-report")(build_paper_report)
app.command("run-synthetic-lob-benchmark")(run_synthetic_lob_benchmark)
app.command("replay-binance-l2-sample")(replay_binance_l2_sample)
app.command("inspect-paper-report")(inspect_paper_report)

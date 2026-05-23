"""Import smoke tests for the initial package scaffold."""

from __future__ import annotations

import importlib


def test_chronoslob_imports() -> None:
    package = importlib.import_module("chronoslob")

    assert package.__version__


def test_key_subpackages_import() -> None:
    modules = [
        "chronoslob.data",
        "chronoslob.book",
        "chronoslob.features",
        "chronoslob.labels",
        "chronoslob.models",
        "chronoslob.training",
        "chronoslob.backtest",
        "chronoslob.analysis",
        "chronoslob.utils",
    ]

    for module in modules:
        assert importlib.import_module(module)


def test_cli_module_imports() -> None:
    assert importlib.import_module("chronoslob.cli")


def test_audit_utils_import() -> None:
    audit = importlib.import_module("chronoslob.utils.audit")

    for name in (
        "AuditStatus",
        "AuditIssue",
        "AuditResult",
        "PathInventory",
        "collect_cli_commands",
        "run_project_audit",
    ):
        assert hasattr(audit, name), f"chronoslob.utils.audit is missing {name}"


def test_report_archive_utils_import() -> None:
    report_archive = importlib.import_module("chronoslob.utils.report_archive")

    for name in (
        "ReportArchiveConfig",
        "ReportArchiveResult",
        "ReportArchiveSection",
        "CommandCapture",
        "build_report_archive",
        "collect_project_inventory",
        "collect_release_history",
        "collect_phase_timeline",
        "collect_cli_smoke_outputs",
        "write_report_archive",
    ):
        assert hasattr(report_archive, name), (
            f"chronoslob.utils.report_archive is missing {name}"
        )


def test_schema_modules_import() -> None:
    schemas = importlib.import_module("chronoslob.data.schemas")
    events = importlib.import_module("chronoslob.book.events")

    for name in (
        "OrderBookLevel",
        "OrderBookSnapshot",
        "BookEvent",
        "FeatureRow",
        "LabelRow",
        "DataQualityIssue",
        "Side",
        "EventType",
    ):
        assert hasattr(schemas, name), f"chronoslob.data.schemas is missing {name}"

    for name in (
        "sort_levels_for_side",
        "validate_book_side_order",
        "has_duplicate_prices",
        "top_of_book",
    ):
        assert hasattr(events, name), f"chronoslob.book.events is missing {name}"


def test_fi2010_modules_import() -> None:
    fi2010 = importlib.import_module("chronoslob.data.fi2010")
    validation = importlib.import_module("chronoslob.data.validation")

    for name in (
        "FI2010Config",
        "FI2010Dataset",
        "load_fi2010",
        "infer_fi2010_columns",
        "build_snapshot_from_row",
    ):
        assert hasattr(fi2010, name), f"chronoslob.data.fi2010 is missing {name}"

    for name in (
        "DataValidationResult",
        "DataValidationError",
        "validate_numeric_frame",
        "validate_fi2010_dataset",
    ):
        assert hasattr(validation, name), (
            f"chronoslob.data.validation is missing {name}"
        )


def test_binance_reconstruction_modules_import() -> None:
    binance = importlib.import_module("chronoslob.data.binance")
    event_store = importlib.import_module("chronoslob.data.event_store")
    manifests = importlib.import_module("chronoslob.data.manifests")
    event_replay = importlib.import_module("chronoslob.book.event_replay")
    local_order_book = importlib.import_module("chronoslob.book.local_order_book")
    reconstruction = importlib.import_module("chronoslob.book.reconstruction")
    replay = importlib.import_module("chronoslob.book.replay")

    for name in (
        "BinanceDepthLevel",
        "BinanceDepthSnapshot",
        "BinanceDiffDepthEvent",
        "parse_binance_snapshot",
        "parse_binance_diff_event",
        "load_binance_snapshot_json",
        "load_binance_diff_events_jsonl",
        "to_order_book_snapshot",
    ):
        assert hasattr(binance, name), f"chronoslob.data.binance is missing {name}"

    for name in ("LocalOrderBookConfig", "LocalOrderBook"):
        assert hasattr(local_order_book, name), (
            f"chronoslob.book.local_order_book is missing {name}"
        )

    for name in (
        "ReconstructionStatus",
        "ReconstructionIssue",
        "ReconstructionResult",
        "should_apply_first_diff",
        "is_stale_event",
        "has_update_gap",
        "reconstruct_order_book",
    ):
        assert hasattr(reconstruction, name), (
            f"chronoslob.book.reconstruction is missing {name}"
        )

    for name in ("ReplayConfig", "replay_binance_jsonl", "summarise_replay_result"):
        assert hasattr(replay, name), f"chronoslob.book.replay is missing {name}"

    for name in (
        "EventLogRecord",
        "EventLogRecordType",
        "serialise_book_event",
        "serialise_order_book_snapshot",
        "deserialise_event_log_record",
        "write_event_log_jsonl",
        "read_event_log_jsonl",
        "iter_event_log_jsonl",
        "filter_event_log_records",
        "sort_event_log_records",
    ):
        assert hasattr(event_store, name), (
            f"chronoslob.data.event_store is missing {name}"
        )

    for name in (
        "EventLogManifest",
        "sha256_file",
        "create_event_log_manifest",
        "write_manifest",
        "read_manifest",
    ):
        assert hasattr(manifests, name), f"chronoslob.data.manifests is missing {name}"

    for name in (
        "snapshots_from_event_log_records",
        "replay_event_log_to_feature_frame",
        "replay_event_log_to_label_frame",
        "replay_event_log_to_feature_label_frames",
        "write_binance_reconstruction_to_event_log",
    ):
        assert hasattr(event_replay, name), (
            f"chronoslob.book.event_replay is missing {name}"
        )


def test_feature_modules_import() -> None:
    microprice = importlib.import_module("chronoslob.features.microprice")
    imbalance = importlib.import_module("chronoslob.features.imbalance")
    order_flow = importlib.import_module("chronoslob.features.order_flow")
    volatility = importlib.import_module("chronoslob.features.volatility")
    regimes = importlib.import_module("chronoslob.features.regimes")
    pipeline = importlib.import_module("chronoslob.features.pipeline")
    features = importlib.import_module("chronoslob.features")

    for name in (
        "compute_mid_price",
        "compute_spread",
        "compute_relative_spread",
        "compute_microprice",
        "compute_snapshot_price_features",
    ):
        assert hasattr(microprice, name), (
            f"chronoslob.features.microprice is missing {name}"
        )

    for name in (
        "compute_depth",
        "compute_depth_imbalance",
        "compute_queue_imbalance",
        "compute_level_imbalances",
        "compute_depth_slope",
        "compute_liquidity_concentration",
    ):
        assert hasattr(imbalance, name), (
            f"chronoslob.features.imbalance is missing {name}"
        )

    for name in (
        "compute_order_flow_imbalance_from_snapshots",
        "compute_order_flow_imbalance_series",
        "compute_trade_imbalance_from_events",
    ):
        assert hasattr(order_flow, name), (
            f"chronoslob.features.order_flow is missing {name}"
        )

    for name in (
        "compute_log_returns",
        "compute_realised_volatility",
        "compute_rolling_realised_volatility",
        "compute_event_intensity",
        "compute_rolling_event_intensity",
    ):
        assert hasattr(volatility, name), (
            f"chronoslob.features.volatility is missing {name}"
        )

    for name in (
        "RegimeThresholds",
        "classify_spread_regime",
        "classify_volatility_regime",
        "classify_liquidity_regime",
        "classify_imbalance_regime",
        "compute_regime_thresholds_from_frame",
    ):
        assert hasattr(regimes, name), (
            f"chronoslob.features.regimes is missing {name}"
        )

    for name in (
        "FeaturePipelineConfig",
        "build_features_from_snapshot",
        "build_feature_frame_from_snapshots",
        "build_feature_frame_from_fi2010",
        "validate_feature_frame",
    ):
        assert hasattr(pipeline, name), (
            f"chronoslob.features.pipeline is missing {name}"
        )
        assert hasattr(features, name), (
            f"chronoslob.features is missing re-exported {name}"
        )


def test_label_modules_import() -> None:
    midprice = importlib.import_module("chronoslob.labels.midprice")
    volatility = importlib.import_module("chronoslob.labels.volatility")
    spread = importlib.import_module("chronoslob.labels.spread")
    fill_probability = importlib.import_module("chronoslob.labels.fill_probability")
    adverse_selection = importlib.import_module("chronoslob.labels.adverse_selection")
    leakage = importlib.import_module("chronoslob.labels.leakage")
    pipeline = importlib.import_module("chronoslob.labels.pipeline")
    labels = importlib.import_module("chronoslob.labels")

    for name in (
        "compute_future_return",
        "compute_future_returns",
        "classify_direction",
        "compute_direction_labels",
        "compute_return_quantile_labels",
    ):
        assert hasattr(midprice, name), f"chronoslob.labels.midprice is missing {name}"

    for name in (
        "compute_future_realised_volatility",
        "compute_future_volatility_series",
        "classify_volatility_labels",
    ):
        assert hasattr(volatility, name), (
            f"chronoslob.labels.volatility is missing {name}"
        )

    for name in (
        "compute_future_spread_change",
        "compute_spread_widening_label",
        "compute_spread_widening_labels",
    ):
        assert hasattr(spread, name), f"chronoslob.labels.spread is missing {name}"

    for name in ("compute_passive_fill_proxy", "compute_passive_fill_proxy_series"):
        assert hasattr(fill_probability, name), (
            f"chronoslob.labels.fill_probability is missing {name}"
        )

    for name in (
        "compute_adverse_selection_after_fill_proxy",
        "compute_adverse_selection_proxy_series",
    ):
        assert hasattr(adverse_selection, name), (
            f"chronoslob.labels.adverse_selection is missing {name}"
        )

    for name in (
        "LeakageCheckResult",
        "assert_feature_label_separation",
        "assert_temporal_label_alignment",
        "assert_no_future_feature_timestamps",
        "validate_no_lookahead",
    ):
        assert hasattr(leakage, name), f"chronoslob.labels.leakage is missing {name}"

    for name in (
        "LabelPipelineConfig",
        "build_label_rows_from_snapshots",
        "build_label_frame_from_snapshots",
        "build_label_frame_from_fi2010",
        "validate_label_frame",
    ):
        assert hasattr(pipeline, name), f"chronoslob.labels.pipeline is missing {name}"
        assert hasattr(labels, name), (
            f"chronoslob.labels is missing re-exported {name}"
        )


def test_training_modules_import() -> None:
    splitters = importlib.import_module("chronoslob.training.splitters")
    experiment = importlib.import_module("chronoslob.training.experiment")
    config = importlib.import_module("chronoslob.training.config")
    artifacts = importlib.import_module("chronoslob.training.artifacts")
    metrics = importlib.import_module("chronoslob.training.metrics")
    evaluate = importlib.import_module("chronoslob.training.evaluate")
    baseline_experiment = importlib.import_module(
        "chronoslob.training.baseline_experiment"
    )
    datasets = importlib.import_module("chronoslob.training.datasets")
    batching = importlib.import_module("chronoslob.training.batching")
    token_datasets = importlib.import_module("chronoslob.training.token_datasets")
    token_batching = importlib.import_module("chronoslob.training.token_batching")
    dataloaders = importlib.import_module("chronoslob.training.dataloaders")
    torch_training = importlib.import_module("chronoslob.training.torch_training")
    torch_experiment = importlib.import_module("chronoslob.training.torch_experiment")
    training = importlib.import_module("chronoslob.training")

    for name in (
        "SplitIndices",
        "TemporalSplitConfig",
        "WalkForwardSplitConfig",
        "PurgedEmbargoConfig",
        "TrainOnlyQuantileBinner",
        "temporal_train_validation_test_split",
        "walk_forward_splits",
        "apply_purge_and_embargo",
        "label_horizon_end_indices_from_rows",
        "make_label_horizon_end_indices_from_frame",
    ):
        assert hasattr(splitters, name), (
            f"chronoslob.training.splitters is missing {name}"
        )
        assert hasattr(training, name), (
            f"chronoslob.training is missing re-exported {name}"
        )

    for name in (
        "ExperimentMetadata",
        "create_experiment_metadata",
        "get_git_commit",
        "initialise_experiment_run",
    ):
        assert hasattr(experiment, name), (
            f"chronoslob.training.experiment is missing {name}"
        )

    for name in ("load_yaml_config", "resolve_config_path"):
        assert hasattr(config, name), f"chronoslob.training.config is missing {name}"

    for name in ("safe_run_name", "create_run_directory", "write_json"):
        assert hasattr(artifacts, name), (
            f"chronoslob.training.artifacts is missing {name}"
        )

    for name in (
        "ClassificationMetrics",
        "compute_classification_metrics",
        "confusion_matrix_as_dict",
    ):
        assert hasattr(metrics, name), f"chronoslob.training.metrics is missing {name}"

    assert hasattr(evaluate, "evaluate_classifier")

    for name in (
        "BaselineExperimentConfig",
        "BaselineSplitConfig",
        "BaselinePreprocessingConfig",
        "create_default_baseline_configs",
        "run_baseline_experiment",
    ):
        assert hasattr(baseline_experiment, name), (
            f"chronoslob.training.baseline_experiment is missing {name}"
        )
        assert hasattr(training, name), (
            f"chronoslob.training is missing re-exported {name}"
        )

    for name in (
        "SequenceDataset",
        "SequenceSampleIndex",
        "SequenceWindowConfig",
        "TorchSequenceStandardiser",
        "build_sequence_indices",
        "encode_target_values",
        "infer_torch_feature_columns",
        "torch_is_available",
    ):
        assert hasattr(datasets, name), (
            f"chronoslob.training.datasets is missing {name}"
        )
        assert hasattr(training, name), (
            f"chronoslob.training is missing re-exported {name}"
        )

    for name in (
        "collate_fixed_length_batch",
        "collate_variable_length_batch",
        "pad_variable_length_sequences",
    ):
        assert hasattr(batching, name), (
            f"chronoslob.training.batching is missing {name}"
        )
        assert hasattr(training, name), (
            f"chronoslob.training is missing re-exported {name}"
        )

    for name in (
        "TokenWindowConfig",
        "TokenWindowIndex",
        "TokenSequenceDataset",
        "build_token_window_indices",
    ):
        assert hasattr(token_datasets, name), (
            f"chronoslob.training.token_datasets is missing {name}"
        )
        assert hasattr(training, name), (
            f"chronoslob.training is missing re-exported {name}"
        )

    for name in ("collate_token_windows", "pad_variable_length_token_windows"):
        assert hasattr(token_batching, name), (
            f"chronoslob.training.token_batching is missing {name}"
        )
        assert hasattr(training, name), (
            f"chronoslob.training is missing re-exported {name}"
        )

    for name in (
        "DataLoaderConfig",
        "build_dataloaders_for_split",
        "create_sequence_dataloader",
    ):
        assert hasattr(dataloaders, name), (
            f"chronoslob.training.dataloaders is missing {name}"
        )
        assert hasattr(training, name), (
            f"chronoslob.training is missing re-exported {name}"
        )

    for name in (
        "TorchEpochResult",
        "TorchTrainingConfig",
        "evaluate_torch_classifier",
        "fit_torch_classifier",
        "set_torch_deterministic",
        "train_one_epoch",
    ):
        assert hasattr(torch_training, name), (
            f"chronoslob.training.torch_training is missing {name}"
        )
        assert hasattr(training, name), (
            f"chronoslob.training is missing re-exported {name}"
        )

    for name in (
        "DeepLOBExperimentConfig",
        "run_deeplob_experiment",
        "run_deeplob_smoke_from_fi2010_fixture",
    ):
        assert hasattr(torch_experiment, name), (
            f"chronoslob.training.torch_experiment is missing {name}"
        )
        assert hasattr(training, name), (
            f"chronoslob.training is missing re-exported {name}"
        )

    transformer_experiment = importlib.import_module(
        "chronoslob.training.transformer_experiment"
    )
    for name in (
        "TransformerEpochResult",
        "TransformerTrainingConfig",
        "evaluate_transformer_classifier",
        "fit_transformer_classifier",
        "run_transformer_smoke_from_event_log",
        "train_transformer_one_epoch",
    ):
        assert hasattr(transformer_experiment, name), (
            f"chronoslob.training.transformer_experiment is missing {name}"
        )
        assert hasattr(training, name), (
            f"chronoslob.training is missing re-exported {name}"
        )

    ssl_datasets = importlib.import_module("chronoslob.training.ssl_datasets")
    for name in (
        "DEFAULT_IGNORE_INDEX",
        "MaskedTokenBatch",
        "MaskingPolicy",
        "SSLTokenSequenceDataset",
        "apply_field_masking",
        "build_next_field_targets",
        "collate_ssl_token_windows",
    ):
        assert hasattr(ssl_datasets, name), (
            f"chronoslob.training.ssl_datasets is missing {name}"
        )
        assert hasattr(training, name), (
            f"chronoslob.training is missing re-exported {name}"
        )

    ssl_experiment = importlib.import_module("chronoslob.training.ssl_experiment")
    for name in (
        "SSLEpochResult",
        "SSLTrainingConfig",
        "evaluate_ssl",
        "fit_ssl_model",
        "run_ssl_smoke_from_event_log",
        "train_ssl_one_epoch",
    ):
        assert hasattr(ssl_experiment, name), (
            f"chronoslob.training.ssl_experiment is missing {name}"
        )
        assert hasattr(training, name), (
            f"chronoslob.training is missing re-exported {name}"
        )

    multitask_datasets = importlib.import_module(
        "chronoslob.training.multitask_datasets"
    )
    for name in (
        "DEFAULT_MULTITASK_IGNORE_INDEX",
        "MultiTaskLabelSpec",
        "MultiTaskSampleIndex",
        "MultiTaskTokenDataset",
        "MultiTaskWindowConfig",
        "build_multitask_sample_indices",
        "collate_multitask_token_windows",
    ):
        assert hasattr(multitask_datasets, name), (
            f"chronoslob.training.multitask_datasets is missing {name}"
        )
        assert hasattr(training, name), (
            f"chronoslob.training is missing re-exported {name}"
        )

    multitask_experiment = importlib.import_module(
        "chronoslob.training.multitask_experiment"
    )
    for name in (
        "MultiTaskEpochResult",
        "MultiTaskTrainingConfig",
        "evaluate_multitask_classifier",
        "fit_multitask_model",
        "run_multitask_smoke_from_event_log",
        "train_multitask_one_epoch",
    ):
        assert hasattr(multitask_experiment, name), (
            f"chronoslob.training.multitask_experiment is missing {name}"
        )
        assert hasattr(training, name), (
            f"chronoslob.training is missing re-exported {name}"
        )

    calibration_training = importlib.import_module(
        "chronoslob.training.calibration"
    )
    for name in (
        "ConfidenceFilterConfig",
        "ConfidenceBucket",
        "ConfidenceFilteringResult",
        "AbstentionCurvePoint",
        "build_confidence_filter",
        "evaluate_confidence_filter",
        "abstention_curve",
        "summarise_multitask_calibration",
        "run_calibration_smoke",
    ):
        assert hasattr(calibration_training, name), (
            f"chronoslob.training.calibration is missing {name}"
        )
        assert hasattr(training, name), (
            f"chronoslob.training is missing re-exported {name}"
        )


def test_model_baseline_modules_import() -> None:
    baselines = importlib.import_module("chronoslob.models.baselines")
    preprocessing = importlib.import_module("chronoslob.models.preprocessing")
    deeplob = importlib.import_module("chronoslob.models.deeplob")
    tokenisation = importlib.import_module("chronoslob.models.tokenisation")
    models = importlib.import_module("chronoslob.models")

    for name in (
        "BaselineModelConfig",
        "BaseBaselineModel",
        "MajorityClassBaseline",
        "SklearnBaselineModel",
        "create_baseline_model",
    ):
        assert hasattr(baselines, name), f"chronoslob.models.baselines is missing {name}"
        assert hasattr(models, name), f"chronoslob.models is missing re-exported {name}"

    for name in (
        "FeatureMatrix",
        "TargetVector",
        "TrainOnlyStandardScaler",
        "select_feature_columns",
        "build_feature_matrix",
        "build_target_vector",
        "align_feature_label_frames",
    ):
        assert hasattr(preprocessing, name), (
            f"chronoslob.models.preprocessing is missing {name}"
        )
        assert hasattr(models, name), f"chronoslob.models is missing re-exported {name}"

    for name in (
        "DeepLOBConfig",
        "DeepLOBModel",
        "create_deeplob_model",
    ):
        assert hasattr(deeplob, name), (
            f"chronoslob.models.deeplob is missing {name}"
        )
        assert hasattr(models, name), f"chronoslob.models is missing re-exported {name}"

    for name in (
        "TokenisationConfig",
        "TokenVocabulary",
        "TokenisedRecord",
        "TokenSequence",
        "build_static_token_vocabulary",
        "tokenise_records",
        "tokenise_event_log",
    ):
        assert hasattr(tokenisation, name), (
            f"chronoslob.models.tokenisation is missing {name}"
        )
        assert hasattr(models, name), f"chronoslob.models is missing re-exported {name}"

    transformer = importlib.import_module("chronoslob.models.transformer")
    for name in (
        "MarketTransformerConfig",
        "MarketTransformerEncoder",
        "MarketTransformerOutput",
        "TokenFieldEmbeddingConfig",
        "TransformerPooling",
        "create_market_transformer",
    ):
        assert hasattr(transformer, name), (
            f"chronoslob.models.transformer is missing {name}"
        )
        assert hasattr(models, name), f"chronoslob.models is missing re-exported {name}"

    ssl_module = importlib.import_module("chronoslob.models.ssl")
    for name in (
        "DEFAULT_MASKED_FIELDS",
        "DEFAULT_NEXT_FIELDS",
        "MarketSSLTransformer",
        "MaskingConfig",
        "SSLObjectiveName",
        "SSLTransformerConfig",
        "SSLTransformerOutput",
        "create_ssl_transformer",
    ):
        assert hasattr(ssl_module, name), (
            f"chronoslob.models.ssl is missing {name}"
        )
        assert hasattr(models, name), f"chronoslob.models is missing re-exported {name}"

    multitask = importlib.import_module("chronoslob.models.multitask")
    for name in (
        "DEFAULT_TASK_HEADS",
        "MultiTaskTransformer",
        "MultiTaskTransformerConfig",
        "MultiTaskTransformerOutput",
        "TaskHeadConfig",
        "TaskType",
        "copy_encoder_weights_from_ssl",
        "create_multitask_transformer",
    ):
        assert hasattr(multitask, name), (
            f"chronoslob.models.multitask is missing {name}"
        )
        assert hasattr(models, name), f"chronoslob.models is missing re-exported {name}"

    calibration = importlib.import_module("chronoslob.models.calibration")
    for name in (
        "CalibrationErrorConfig",
        "ReliabilityBin",
        "CalibrationSummary",
        "TemperatureScaler",
        "MultiTaskTemperatureScaler",
        "softmax_probabilities",
        "classification_confidence",
        "negative_log_likelihood",
        "brier_score",
        "expected_calibration_error",
        "reliability_bins",
    ):
        assert hasattr(calibration, name), (
            f"chronoslob.models.calibration is missing {name}"
        )
        assert hasattr(models, name), f"chronoslob.models is missing re-exported {name}"


def test_analysis_modules_import() -> None:
    regimes = importlib.import_module("chronoslob.analysis.regimes")
    transfer = importlib.import_module("chronoslob.analysis.transfer")
    ablations = importlib.import_module("chronoslob.analysis.ablations")
    sensitivity = importlib.import_module("chronoslob.analysis.sensitivity")
    summary = importlib.import_module("chronoslob.analysis.summary")
    analysis = importlib.import_module("chronoslob.analysis")

    for module, names in (
        (
            regimes,
            (
                "SUPPORTED_REGIME_KINDS",
                "UNKNOWN_REGIME_LABEL",
                "RegimeAssignment",
                "RegimeDefinition",
                "RegimeMetricSummary",
                "assign_confidence_bucket",
                "assign_latency_regime",
                "assign_liquidity_regime",
                "assign_spread_regime",
                "assign_volatility_regime",
                "fit_regime_boundaries",
                "summarise_by_regime",
            ),
        ),
        (
            transfer,
            (
                "TransferMatrix",
                "TransferResult",
                "TransferSplit",
                "build_transfer_matrix",
                "compare_in_domain_vs_out_of_domain",
                "summarise_transfer_results",
            ),
        ),
        (
            ablations,
            (
                "ABLATION_CATEGORIES",
                "AblationComparison",
                "AblationResult",
                "AblationSpec",
                "compare_against_baseline",
                "rank_ablations",
                "summarise_ablation_table",
            ),
        ),
        (
            sensitivity,
            (
                "SENSITIVITY_PARAMETERS",
                "SensitivityCurve",
                "SensitivityParameter",
                "SensitivityPoint",
                "build_sensitivity_curve",
                "compare_sensitivity_curves",
                "summarise_sensitivity_curve",
            ),
        ),
        (
            summary,
            (
                "ANALYSIS_TYPES",
                "EXECUTION_METRIC_NAMES",
                "FORBIDDEN_COMBINED_FIELDS",
                "METRIC_DIRECTIONS",
                "PREDICTIVE_METRIC_NAMES",
                "SUPPORTED_METRIC_NAMES",
                "SYNTHETIC_ANALYSIS_WARNING",
                "AnalysisMetric",
                "AnalysisRecord",
                "AnalysisSummary",
                "aggregate_metric",
                "aggregate_records",
                "format_summary_table",
                "run_robustness_analysis_smoke",
                "summarise_records",
            ),
        ),
    ):
        for name in names:
            assert hasattr(module, name), f"{module.__name__} is missing {name}"
            assert hasattr(analysis, name), (
                f"chronoslob.analysis is missing re-exported {name}"
            )


def test_execution_validation_modules_import() -> None:
    execution = importlib.import_module("chronoslob.backtest.execution")
    costs = importlib.import_module("chronoslob.backtest.costs")
    latency = importlib.import_module("chronoslob.backtest.latency")
    turnover = importlib.import_module("chronoslob.backtest.turnover")
    risk = importlib.import_module("chronoslob.backtest.risk")
    validation = importlib.import_module("chronoslob.backtest.validation")
    backtest = importlib.import_module("chronoslob.backtest")

    for module, names in (
        (
            execution,
            (
                "TradeSide",
                "ExecutionMode",
                "PredictionSignal",
                "MarketState",
                "ExecutionDecision",
                "ExecutionFill",
                "ExecutionResult",
            ),
        ),
        (
            costs,
            (
                "FeeModel",
                "SpreadCostModel",
                "ExecutionCostConfig",
                "estimate_aggressive_cost",
                "estimate_passive_cost",
                "estimate_total_cost",
            ),
        ),
        (
            latency,
            (
                "LatencyConfig",
                "apply_latency",
                "get_latency_state",
                "latency_sensitivity_grid",
            ),
        ),
        (
            turnover,
            (
                "TurnoverSummary",
                "compute_turnover",
                "compute_position_path",
                "compute_trade_count",
                "compute_average_holding_period",
            ),
        ),
        (
            risk,
            (
                "RiskConfig",
                "RiskState",
                "apply_inventory_limit",
                "apply_turnover_limit",
                "apply_drawdown_limit",
                "should_abstain_for_risk",
            ),
        ),
        (
            validation,
            (
                "ExecutionValidationConfig",
                "ExecutionValidationSummary",
                "ExecutionValidationResult",
                "run_execution_validation",
                "summarise_execution_results",
                "confidence_threshold_sweep",
                "latency_sensitivity_analysis",
                "run_execution_validation_smoke",
            ),
        ),
    ):
        for name in names:
            assert hasattr(module, name), f"{module.__name__} is missing {name}"
            assert hasattr(backtest, name), (
                f"chronoslob.backtest is missing re-exported {name}"
            )

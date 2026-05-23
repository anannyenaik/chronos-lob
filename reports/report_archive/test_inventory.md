# Test Inventory

Pytest files grouped by inferred area. Test source contents are not included here.

## analysis

- `tests/test_analysis_ablations.py`
- `tests/test_analysis_regimes.py`
- `tests/test_analysis_sensitivity.py`
- `tests/test_analysis_summary.py`
- `tests/test_analysis_transfer.py`

## backtest

- `tests/test_execution_costs.py`
- `tests/test_execution_validation.py`

## calibration

- `tests/test_calibration_experiment.py`
- `tests/test_calibration_metrics.py`

## configs

- `tests/test_config_inventory.py`

## data

- `tests/test_fi2010_loader.py`

## data/book

- `tests/test_binance_schemas.py`
- `tests/test_event_replay.py`
- `tests/test_event_store.py`
- `tests/test_events.py`

## features

- `tests/test_feature_pipeline.py`
- `tests/test_features_imbalance.py`
- `tests/test_features_microprice.py`
- `tests/test_features_order_flow.py`
- `tests/test_features_regimes.py`
- `tests/test_features_volatility.py`
- `tests/test_replay_to_features.py`

## general

- `tests/test_batching.py`
- `tests/test_cli_inventory.py`
- `tests/test_confidence_filtering.py`
- `tests/test_dataloaders.py`
- `tests/test_experiment_registry.py`
- `tests/test_imports.py`
- `tests/test_latency_model.py`
- `tests/test_local_order_book.py`
- `tests/test_manifests.py`
- `tests/test_metrics.py`
- `tests/test_no_lookahead.py`
- `tests/test_paths.py`
- `tests/test_preprocessing.py`
- `tests/test_public_release_readiness.py`
- `tests/test_reconstruction.py`
- `tests/test_replay.py`
- `tests/test_risk_constraints.py`
- `tests/test_schemas.py`
- `tests/test_seeding.py`
- `tests/test_sequence_indexing.py`
- `tests/test_temperature_scaling.py`
- `tests/test_token_batching.py`
- `tests/test_token_datasets.py`
- `tests/test_tokenisation.py`
- `tests/test_train_only_fitting.py`
- `tests/test_turnover.py`

## labels

- `tests/test_label_pipeline.py`
- `tests/test_labels_adverse_selection.py`
- `tests/test_labels_fill_probability.py`
- `tests/test_labels_midprice.py`
- `tests/test_labels_spread.py`
- `tests/test_labels_volatility.py`

## models

- `tests/test_deeplob_forward.py`

## models/training

- `tests/test_baseline_experiment.py`
- `tests/test_baselines.py`
- `tests/test_multitask_datasets.py`
- `tests/test_multitask_experiment.py`
- `tests/test_multitask_model.py`
- `tests/test_ssl_datasets.py`
- `tests/test_ssl_experiment.py`
- `tests/test_ssl_objectives.py`
- `tests/test_transformer_experiment.py`
- `tests/test_transformer_model.py`
- `tests/test_transformer_training.py`

## reports

- `tests/test_report_archive.py`
- `tests/test_report_inventory.py`

## training

- `tests/test_purged_embargoed_splitters.py`
- `tests/test_splitters.py`
- `tests/test_torch_datasets.py`
- `tests/test_torch_experiment.py`
- `tests/test_torch_training.py`

## utils

- `tests/test_audit_utils.py`

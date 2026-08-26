# Test Inventory

Pytest files grouped by inferred area. Test source contents are not included here.

## analysis

- `tests/test_analysis_ablations.py`
- `tests/test_analysis_regimes.py`
- `tests/test_analysis_sensitivity.py`
- `tests/test_analysis_summary.py`
- `tests/test_analysis_transfer.py`
- `tests/test_proper_neural_analysis.py`

## backtest

- `tests/test_execution_centrepiece.py`
- `tests/test_execution_costs.py`
- `tests/test_execution_v3_analysis.py`
- `tests/test_execution_validation.py`
- `tests/test_fi2010_execution_v2.py`
- `tests/test_fi2010_execution_v3.py`

## calibration

- `tests/test_calibration_experiment.py`
- `tests/test_calibration_metrics.py`

## configs

- `tests/test_config_inventory.py`

## data

- `tests/test_fi2010_benchmark_preparation.py`
- `tests/test_fi2010_brutal_ablations.py`
- `tests/test_fi2010_external_benchmarks.py`
- `tests/test_fi2010_figures.py`
- `tests/test_fi2010_loader.py`
- `tests/test_fi2010_multifold.py`
- `tests/test_fi2010_multifold_runner.py`
- `tests/test_fi2010_neural_grid.py`
- `tests/test_fi2010_neural_proper_training.py`
- `tests/test_fi2010_neural_runner.py`
- `tests/test_fi2010_official_adapter.py`
- `tests/test_fi2010_uncertainty.py`

## data/book

- `tests/test_binance_l2_extension.py`
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
- `tests/test_fi2010_feature_ablations_v2.py`
- `tests/test_fi2010_microstructure_features.py`
- `tests/test_replay_to_features.py`

## general

- `tests/test_batching.py`
- `tests/test_classical_paper_models.py`
- `tests/test_cli_inventory.py`
- `tests/test_confidence_filtering.py`
- `tests/test_dataloaders.py`
- `tests/test_evidence_pack.py`
- `tests/test_experiment_artifact_contract.py`
- `tests/test_experiment_registry.py`
- `tests/test_imports.py`
- `tests/test_latency_model.py`
- `tests/test_local_order_book.py`
- `tests/test_manifests.py`
- `tests/test_metrics.py`
- `tests/test_neural_benchmarking.py`
- `tests/test_neural_paper_models.py`
- `tests/test_no_lookahead.py`
- `tests/test_paper_ablations.py`
- `tests/test_paper_experiment_evidence.py`
- `tests/test_paper_experiment_inspection.py`
- `tests/test_paper_experiment_plots.py`
- `tests/test_paper_experiment_runner.py`
- `tests/test_paths.py`
- `tests/test_preprocessing.py`
- `tests/test_public_release_readiness.py`
- `tests/test_reconstruction.py`
- `tests/test_release_consistency.py`
- `tests/test_replay.py`
- `tests/test_research_protocol.py`
- `tests/test_risk_constraints.py`
- `tests/test_schemas.py`
- `tests/test_seeding.py`
- `tests/test_sequence_indexing.py`
- `tests/test_synthetic_lob.py`
- `tests/test_system_benchmarks.py`
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
- `tests/test_fi2010_ssl_runner.py`
- `tests/test_matrix_ssl.py`
- `tests/test_multitask_datasets.py`
- `tests/test_multitask_experiment.py`
- `tests/test_multitask_model.py`
- `tests/test_ssl_datasets.py`
- `tests/test_ssl_experiment.py`
- `tests/test_ssl_failure_analysis.py`
- `tests/test_ssl_objectives.py`
- `tests/test_ssl_v2.py`
- `tests/test_transformer_experiment.py`
- `tests/test_transformer_model.py`
- `tests/test_transformer_training.py`

## reports

- `tests/test_final_empirical_report.py`
- `tests/test_paper_report_builder.py`
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

# Experiment Evidence Index

This index maps research themes to the modules, configs, tests and CLI
commands that implement them. It is a navigation aid for anyone reading
or reproducing the platform.

## Data and Order Book

- Modules: `chronoslob/data/`, `chronoslob/book/`
- Configs: `configs/data/*.yaml`
- Tests: `tests/test_fi2010_loader.py`, `tests/test_binance_schemas.py`,
  `tests/test_event_store.py`, `tests/test_event_replay.py`
- CLI: `inspect-fi2010`, `inspect-binance-replay`, `inspect-event-log`
- Reports: [data_quality](../reports/data_quality.md),
  [order_book_reconstruction](../reports/order_book_reconstruction.md),
  [event_log_storage](../reports/event_log_storage.md)

## Features and Labels

- Modules: `chronoslob/features/`, `chronoslob/labels/`
- Configs: `configs/experiments/feature_audit_fi2010.yaml`,
  `configs/experiments/label_audit_fi2010.yaml`
- Tests: `tests/test_feature_pipeline.py`,
  `tests/test_label_pipeline.py`, `tests/test_no_lookahead.py`
- CLI: `inspect-features-fi2010`, `inspect-labels-fi2010`
- Reports: [feature_engine](../reports/feature_engine.md),
  [label_engine](../reports/label_engine.md),
  [leakage_controls](../reports/leakage_controls.md)

## Temporal Validation and Registry

- Modules: `chronoslob/training/splitters.py`,
  `chronoslob/training/experiment.py`
- Configs: `configs/experiments/fi2010_split_audit.yaml`
- Tests: `tests/test_splitters.py`,
  `tests/test_purged_embargoed_splitters.py`,
  `tests/test_experiment_registry.py`
- CLI: `inspect-split`, `init-run`
- Reports: [validation_protocol](../reports/validation_protocol.md),
  [experiment_registry](../reports/experiment_registry.md)

## Baselines and Sequence Models

- Modules: `chronoslob/models/`, `chronoslob/training/`
- Configs: `configs/models/*.yaml`,
  `configs/experiments/*smoke.yaml`
- Tests: `tests/test_baselines.py`,
  `tests/test_deeplob_forward.py`,
  `tests/test_transformer_model.py`
- CLI: `inspect-baselines`, `inspect-deeplob`, `inspect-transformer`
- Reports: [baselines](../reports/baselines.md),
  [deeplob_baseline](../reports/deeplob_baseline.md),
  [transformer_architecture](../reports/transformer_architecture.md)

## Self-Supervised and Multi-Task Learning

- Modules: `chronoslob/models/ssl.py`,
  `chronoslob/models/multitask.py`, `chronoslob/training/ssl_*`,
  `chronoslob/training/multitask_*`
- Configs: `configs/models/ssl_transformer.yaml`,
  `configs/models/multitask_transformer.yaml`,
  `configs/experiments/event_ssl_smoke.yaml`,
  `configs/experiments/event_multitask_smoke.yaml`
- Tests: `tests/test_ssl_datasets.py`,
  `tests/test_ssl_objectives.py`,
  `tests/test_multitask_model.py`,
  `tests/test_multitask_experiment.py`
- CLI: `inspect-ssl`, `inspect-multitask`
- Reports:
  [self_supervised_objectives](../reports/self_supervised_objectives.md),
  [multitask_finetuning](../reports/multitask_finetuning.md)

## Calibration and Uncertainty

- Modules: `chronoslob/models/calibration.py`,
  `chronoslob/training/calibration.py`,
  `chronoslob/experiments/evidence.py`
- Configs: `configs/experiments/calibration_smoke.yaml`,
  `configs/experiments/fi2010_midprice_h10.yaml`
- Tests: `tests/test_calibration_metrics.py`,
  `tests/test_confidence_filtering.py`,
  `tests/test_temperature_scaling.py`,
  `tests/test_paper_experiment_evidence.py`
- CLI: `inspect-calibration`, `run-paper-experiment`
- Reports:
  [calibration_uncertainty](../reports/calibration_uncertainty.md)
- Phase F artefact: `calibration_bins.csv` (reliability bins built
  from held-out test predictions).

## Execution-Aware Validation

- Modules: `chronoslob/backtest/`,
  `chronoslob/experiments/evidence.py`
- Configs: `configs/experiments/execution_validation_smoke.yaml`,
  `configs/experiments/fi2010_midprice_h10.yaml`
- Tests: `tests/test_execution_validation.py`,
  `tests/test_execution_costs.py`,
  `tests/test_latency_model.py`,
  `tests/test_turnover.py`,
  `tests/test_risk_constraints.py`,
  `tests/test_paper_experiment_evidence.py`
- CLI: `inspect-execution-validation`, `run-paper-experiment`
- Reports:
  [execution_aware_validation](../reports/execution_aware_validation.md)
- Phase F artefact: `execution_sensitivity.csv` (cost-aware signal
  quality rows under explicit cost assumptions; not a production
  backtest).

## Robustness Analysis

- Modules: `chronoslob/analysis/`
- Configs: `configs/experiments/robustness_analysis_smoke.yaml`
- Tests: `tests/test_analysis_transfer.py`,
  `tests/test_analysis_regimes.py`,
  `tests/test_analysis_ablations.py`,
  `tests/test_analysis_sensitivity.py`
- CLI: `inspect-analysis`
- Reports:
  [transfer_regime_ablation_analysis](../reports/transfer_regime_ablation_analysis.md)

## Experiment Artefact Contract

- Modules: `chronoslob/experiments/`
- Tests: `tests/test_experiment_artifact_contract.py`
- CLI: `inspect-experiment-artifacts`
- Docs: [EXPERIMENT_ARTIFACT_CONTRACT](EXPERIMENT_ARTIFACT_CONTRACT.md)

## FI-2010 Benchmark Preparation

- Modules: `chronoslob/experiments/fi2010_benchmark.py`
- Configs: `configs/experiments/fi2010_midprice_h10.yaml`
- Tests: `tests/test_fi2010_benchmark_preparation.py`
- CLI: `prepare-fi2010-benchmark`
- Docs: [FI2010_BENCHMARK](FI2010_BENCHMARK.md)

## Paper Experiment Runner

- Modules: `chronoslob/experiments/paper_runner.py`,
  `chronoslob/experiments/model_registry.py`,
  `chronoslob/experiments/neural_adapters.py`,
  `chronoslob/experiments/evidence.py`,
  `chronoslob/experiments/plots.py`
- Configs: `configs/experiments/fi2010_midprice_h10.yaml`
- Tests: `tests/test_paper_experiment_runner.py`,
  `tests/test_classical_paper_models.py`,
  `tests/test_neural_paper_models.py`,
  `tests/test_paper_experiment_evidence.py`,
  `tests/test_paper_experiment_plots.py`,
  `tests/test_paper_experiment_inspection.py`
- CLI: `run-paper-experiment`, `build-paper-plots`,
  `inspect-paper-experiment`
- Docs: [PAPER_EXPERIMENTS](PAPER_EXPERIMENTS.md)
- Phase G artefacts: `plots/reliability_curve.png`,
  `plots/cost_sensitivity.png`, `plots/confusion_matrix.png`,
  optionally `plots/regime_breakdown.png` (only when genuine regime
  data is available in stored artefacts) and `plot_summary.json`.

## Paper Ablation Suite

- Modules: `chronoslob/experiments/ablations.py`,
  `chronoslob/experiments/paper_runner.py`
- Configs: `configs/experiments/fi2010_midprice_h10.yaml`
- Tests: `tests/test_paper_ablations.py`
- CLI: `run-paper-ablations`
- Docs: [PAPER_ABLATIONS](PAPER_ABLATIONS.md),
  [PAPER_EXPERIMENTS](PAPER_EXPERIMENTS.md)
- Phase H artefacts: `ablation_summary.json`,
  `ablation_results.csv`, `ablation_manifest.json`, per-ablation
  Markdown reports and child paper experiment directories only for
  ablations that genuinely run. Skipped ablations, including SSL
  pretraining in this phase, are explicit status records rather than
  hidden omissions.

## Systems Benchmarks

- Modules: `chronoslob/experiments/system_benchmarks.py`
- Configs: `configs/experiments/fi2010_midprice_h10.yaml`
- Tests: `tests/test_system_benchmarks.py`
- CLI: `run-system-benchmarks`, `inspect-system-benchmarks`
- Docs: [SYSTEM_BENCHMARKS](SYSTEM_BENCHMARKS.md)
- Phase I artefacts: `system_benchmark_summary.json`,
  `system_benchmark_results.csv`, `environment.json`, per-category
  Markdown reports and a validated child paper experiment for runner
  timing. Smoke fixture timings are labelled as smoke measurements and
  are not benchmark evidence.

## Paper Report Builder

- Modules: `chronoslob/experiments/reporting.py`
- Configs: uses stored experiment config snapshots from completed paper
  experiment directories.
- Tests: `tests/test_paper_report_builder.py`
- CLI: `build-paper-report`, `inspect-paper-report`
- Docs: [PAPER_REPORTS](PAPER_REPORTS.md)
- Phase J artefacts: a Markdown empirical report plus
  `<report_stem>_summary.json`, both generated from stored paper experiment,
  ablation and systems benchmark artefacts. Fixture or smoke inputs remain
  labelled as smoke reports and are not benchmark evidence.

## Reproducibility and Audit

- Modules: `chronoslob/utils/audit.py`,
  `chronoslob/utils/report_archive.py`
- Configs: `configs/experiments/report_archive_smoke.yaml`,
  `configs/experiments/full_audit_smoke.yaml`
- Tests: `tests/test_audit_utils.py`,
  `tests/test_report_archive.py`,
  `tests/test_config_inventory.py`,
  `tests/test_report_inventory.py`
- CLI: `inspect-release-readiness`, `run-project-audit`,
  `build-report-archive`, `inspect-report-archive`
- Docs: [REPRODUCIBILITY](REPRODUCIBILITY.md),
  [SAFETY_AND_LIMITATIONS](SAFETY_AND_LIMITATIONS.md)

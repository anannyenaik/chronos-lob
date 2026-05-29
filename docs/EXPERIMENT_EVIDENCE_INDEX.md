# Experiment Evidence Index

This index maps research themes to the modules, configs, tests and CLI
commands that implement them. It is a navigation aid for anyone reading
or reproducing the platform.

## Research Protocol

- Public protocol: [RESEARCH_PROTOCOL](RESEARCH_PROTOCOL.md)
- Maintainer note: [reports/10_10_research_protocol.md](../reports/10_10_research_protocol.md)
- Multi-fold study skeleton:
  [`configs/experiments/fi2010_multifold.yaml`](../configs/experiments/fi2010_multifold.yaml)
- Multi-fold preparation runbook:
  [FI2010_MULTIFOLD_PROTOCOL](FI2010_MULTIFOLD_PROTOCOL.md)
- Multi-fold classical runner:
  [FI2010_MULTIFOLD_CLASSICAL](FI2010_MULTIFOLD_CLASSICAL.md)
- Neural benchmark protocol:
  [NEURAL_BENCHMARK_PROTOCOL](NEURAL_BENCHMARK_PROTOCOL.md)
- Neural benchmark runner:
  [FI2010_NEURAL_BENCHMARKS](FI2010_NEURAL_BENCHMARKS.md)
- External benchmark context:
  [FI2010_EXTERNAL_BENCHMARKS](FI2010_EXTERNAL_BENCHMARKS.md)

## Real FI-2010 Evidence Map

- Final generated empirical report:
  [`reports/chronoslob_final_empirical_report.md`](../reports/chronoslob_final_empirical_report.md)
  built by [FINAL_EMPIRICAL_REPORT](FINAL_EMPIRICAL_REPORT.md)
- Multi-fold classical evidence:
  [`experiments/fi2010_multifold_classical/`](../experiments/fi2010_multifold_classical/)
- Reduced-scope supervised neural evidence:
  [`experiments/fi2010_multifold_neural/`](../experiments/fi2010_multifold_neural/)
- Statistical uncertainty:
  [`experiments/fi2010_uncertainty/`](../experiments/fi2010_uncertainty/)
- Brutal ablations:
  [`experiments/fi2010_brutal_ablations/`](../experiments/fi2010_brutal_ablations/)
- Execution-aware proxy diagnostics:
  [`experiments/fi2010_execution_v2/`](../experiments/fi2010_execution_v2/)
- Execution-v3 offline execution-aware proxy diagnostic:
  [EXECUTION_VALIDATION_V3](EXECUTION_VALIDATION_V3.md)
- External benchmark context:
  [FI2010_EXTERNAL_BENCHMARKS](FI2010_EXTERNAL_BENCHMARKS.md) and
  [`experiments/fi2010_external_context/`](../experiments/fi2010_external_context/)
- FI-2010 neural figure pipeline:
  [FIGURE_INDEX](FIGURE_INDEX.md) and
  [`reports/figures/fi2010_neural_full_grid/`](../reports/figures/fi2010_neural_full_grid/)
- FI-2010 microstructure feature registry and ablations:
  [MICROSTRUCTURE_FEATURES](MICROSTRUCTURE_FEATURES.md),
  [FEATURE_ABLATIONS](FEATURE_ABLATIONS.md) and
  [`experiments/fi2010_feature_ablations/`](../experiments/fi2010_feature_ablations/)
- Acquisition and conversion:
  [FI2010_DATA_ACQUISITION](FI2010_DATA_ACQUISITION.md)
- Benchmark preparation:
  [FI2010_BENCHMARK](FI2010_BENCHMARK.md)
- Fold-1 paper experiment:
  [PAPER_EXPERIMENTS](PAPER_EXPERIMENTS.md) and
  [`experiments/fi2010_midprice_h10/`](../experiments/fi2010_midprice_h10/)
- Ablations:
  [PAPER_ABLATIONS](PAPER_ABLATIONS.md) and
  [`experiments/fi2010_midprice_h10_ablations/`](../experiments/fi2010_midprice_h10_ablations/)
- Systems benchmarks:
  [SYSTEM_BENCHMARKS](SYSTEM_BENCHMARKS.md) and
  [`experiments/fi2010_midprice_h10_systems/`](../experiments/fi2010_midprice_h10_systems/)
- Generated artefact report:
  [`reports/chronoslob_empirical_report.md`](../reports/chronoslob_empirical_report.md)
- Release evidence pack:
  [EVIDENCE_PACK](EVIDENCE_PACK.md) and
  [`reports/evidence_pack/`](../reports/evidence_pack/)
- Model card:
  [`experiments/fi2010_midprice_h10/model_card.md`](../experiments/fi2010_midprice_h10/model_card.md)

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
  `configs/experiments/*smoke.yaml`,
  `configs/experiments/fi2010_neural_serious.yaml`
- Tests: `tests/test_baselines.py`,
  `tests/test_deeplob_forward.py`,
  `tests/test_transformer_model.py`,
  `tests/test_neural_benchmarking.py`,
  `tests/test_fi2010_neural_runner.py`
- CLI: `inspect-baselines`, `inspect-deeplob`, `inspect-transformer`,
  `inspect-fi2010-neural-plan`, `run-fi2010-neural-benchmark`
- Reports: [baselines](../reports/baselines.md),
  [deeplob_baseline](../reports/deeplob_baseline.md),
  [transformer_architecture](../reports/transformer_architecture.md)
- Neural planning module:
  `chronoslob/experiments/neural_benchmarking.py`
- Neural execution module:
  `chronoslob/experiments/fi2010_neural_runner.py`
- Neural protocol:
  [NEURAL_BENCHMARK_PROTOCOL](NEURAL_BENCHMARK_PROTOCOL.md)

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

## FI-2010 Self-Supervised Pretraining And Fine-Tuning

- Scope: a leakage-safe `ssl_transformer` path that pretrains an encoder on
  training rows only, fine-tunes on mid-price direction and compares against a
  supervised baseline of identical architecture. No SSL effectiveness is
  claimed; the final report admits SSL rows only when artefacts are
  SHA256-verified.
- Modules: `chronoslob/models/matrix_ssl.py`,
  `chronoslob/training/matrix_ssl_datasets.py`,
  `chronoslob/training/matrix_ssl_experiment.py`,
  `chronoslob/experiments/fi2010_ssl_runner.py`
- Config: `configs/experiments/fi2010_ssl_smoke.yaml`
- Tests: `tests/test_matrix_ssl.py`,
  `tests/test_fi2010_ssl_runner.py`
- CLI: `run-fi2010-ssl-neural-benchmark`
- Docs: [FI2010_SSL_BENCHMARKS](FI2010_SSL_BENCHMARKS.md)

## Statistical Uncertainty

- Module: `chronoslob/experiments/statistics.py`
- Tests: `tests/test_fi2010_uncertainty.py`
- CLI: `analyse-fi2010-uncertainty`
- Docs: [STATISTICAL_UNCERTAINTY](STATISTICAL_UNCERTAINTY.md)
- Artefact directory:
  [`experiments/fi2010_uncertainty/`](../experiments/fi2010_uncertainty/)
- Outputs: `summary.json`, `metric_confidence_intervals.csv`,
  `paired_model_comparisons.csv`, `rank_stability.csv`,
  `model_ranking.csv`, `uncertainty_notes.md`. Diagnostic only.
  Neural numbers carry a reduced-scope, single-seed caveat in
  `uncertainty_notes.md`.

## Brutal Ablations

- Module: `chronoslob/experiments/fi2010_brutal_ablations.py`
- Tests: `tests/test_fi2010_brutal_ablations.py`
- CLI: `run-fi2010-brutal-ablations`
- Docs: [FI2010_BRUTAL_ABLATIONS](FI2010_BRUTAL_ABLATIONS.md)
- Artefact directory:
  [`experiments/fi2010_brutal_ablations/`](../experiments/fi2010_brutal_ablations/)
- Outputs: `summary.json`, `ablation_results.csv`, `ablation_summary.csv`,
  `skipped_ablations.json`, `feature_group_ablation.csv`,
  `model_class_ablation.csv`, `lookback_ablation.csv`,
  `horizon_ablation.csv`, `calibration_threshold_ablation.csv`,
  `execution_cost_latency_ablation.csv` and `ablation_notes.md`. The
  feature-group and horizon families refit a fast linear baseline on the
  real folds; the model-class, calibration and execution families reuse
  stored evidence; the neural lookback sweep is skipped by default and
  recorded with a reason. Execution numbers are proxy diagnostics only.

## Microstructure Feature Registry and Ablations

- Modules: `chronoslob/features/registry.py`,
  `chronoslob/features/microstructure_fi2010.py`,
  `chronoslob/experiments/fi2010_feature_ablations.py`,
  `chronoslob/analysis/fi2010_ablation_figures.py`
- Tests: `tests/test_fi2010_microstructure_features.py`,
  `tests/test_fi2010_feature_ablations_v2.py`
- CLI: `audit-fi2010-features`, `run-fi2010-feature-ablations`,
  `build-fi2010-ablation-figures`
- Docs: [MICROSTRUCTURE_FEATURES](MICROSTRUCTURE_FEATURES.md),
  [FEATURE_ABLATIONS](FEATURE_ABLATIONS.md),
  [FIGURE_INDEX](FIGURE_INDEX.md)
- Artefact directory:
  [`experiments/fi2010_feature_ablations/`](../experiments/fi2010_feature_ablations/)
- Outputs: `features.csv`, `feature_metadata.json`,
  `feature_group_manifest.json`, `results_summary.csv`,
  `aggregate_summary.csv`, `feature_delta_summary.csv`, per-run
  metrics, predictions where available, status records, SHA-256 manifests
  and ablation figures with source CSVs.
- Scope: separates raw LOB levels, top-of-book, spread, midprice,
  microprice, depth and concentration families from rolling volatility and
  clearly labelled snapshot-flow proxy features. Unsupported FI-2010 event
  families are recorded explicitly, so the evidence does not claim true
  order-flow, cancellation, trade or queue-position information unless a
  future data source directly supports those fields.

## Execution-Aware Evaluation v2

- Module: `chronoslob/experiments/execution_v2.py`
- Tests: `tests/test_fi2010_execution_v2.py`
- CLI: `run-fi2010-execution-v2`
- Docs: [FI2010_EXECUTION_V2](FI2010_EXECUTION_V2.md)
- Artefact directory:
  [`experiments/fi2010_execution_v2/`](../experiments/fi2010_execution_v2/)
- Outputs: `summary.json`, `execution_v2_results.csv`,
  `cost_latency_surface.csv`, `confidence_threshold_summary.csv`,
  `turnover_summary.csv`, `adverse_selection_summary.csv`,
  `fill_assumption_summary.csv`, `degradation_summary.csv`,
  `skipped_diagnostics.json`, `execution_assumptions.md` and
  `execution_notes.md`. The layer reuses the stored multi-fold and
  ablation artefacts to make the forecasting-versus-tradability gap
  explicit through cost, latency, confidence, turnover, adverse-selection,
  fill and statistical-to-execution degradation proxies. Neural runs ship
  no stored execution proxy rows, so their execution-aware diagnostics are
  recorded as explicit skips. Every metric is a proxy diagnostic; no
  profitability or live tradability claim is made.

## Execution-Aware Validation v3

- Module: `chronoslob/analysis/execution_v3.py`
- Tests: `tests/test_fi2010_execution_v3.py`
- CLI: `build-fi2010-execution-v3`
- Docs: [EXECUTION_VALIDATION_V3](EXECUTION_VALIDATION_V3.md)
- Default artefact directory:
  [`experiments/fi2010_execution_v3/`](../experiments/fi2010_execution_v3/)
- Outputs: `summary.json`, `execution_v3_manifest.json`,
  `confidence_threshold_summary.csv`, `confidence_threshold_aggregate.csv`,
  `cost_sensitivity_summary.csv`, `latency_sensitivity_summary.csv`,
  `fill_assumption_summary.csv`, `adverse_selection_summary.csv`,
  `regime_execution_summary.csv`, `skipped_diagnostics.json` and
  `execution_v3_notes.md`.
- Scope: consumes stored FI-2010 full-grid prediction artefacts and evaluates
  confidence filtering, costs, row-step latency, fill assumptions,
  adverse-selection proxies and explicit regime breakdowns when context exists.
  It is an offline execution-aware proxy diagnostic, not a live trading system,
  broker integration, profitability claim or realistic execution simulator.

## External Benchmark Context

- Docs: [FI2010_EXTERNAL_BENCHMARKS](FI2010_EXTERNAL_BENCHMARKS.md)
- Maintainer note:
  [`reports/external_benchmark_context.md`](../reports/external_benchmark_context.md)
- Artefact directory:
  [`experiments/fi2010_external_context/`](../experiments/fi2010_external_context/)
- Outputs: `benchmark_context.json`, `protocol_comparison.csv` and
  `comparison_notes.md`.
- Scope: protocol comparison only. The layer documents dataset variant,
  auction setting, normalisation, folds, horizon, label mapping, split
  protocol, metrics, preprocessing, model class and calibration/execution
  diagnostics. It includes the current ChronosLOB classical and
  reduced-scope neural result snapshot, carries the single-seed neural
  caveat and records no external numeric paper metrics. No SSL result is
  reported.

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
- FI-2010 artefact: `calibration_bins.csv` (reliability bins built
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
- FI-2010 artefact: `execution_sensitivity.csv` (cost-aware signal
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

- Modules: `chronoslob/experiments/fi2010_benchmark.py`,
  `chronoslob/data/fi2010_official.py`
- Configs: `configs/experiments/fi2010_midprice_h10.yaml`
- Tests: `tests/test_fi2010_benchmark_preparation.py`,
  `tests/test_fi2010_official_adapter.py`
- CLI: `prepare-fi2010-benchmark`, `verify-fi2010-local`,
  `convert-fi2010-official`, `inspect-fi2010-multifold`,
  `prepare-fi2010-multifold`, `run-fi2010-multifold-classical`,
  `inspect-fi2010-neural-plan`, `run-fi2010-neural-benchmark`
- Docs: [FI2010_BENCHMARK](FI2010_BENCHMARK.md),
  [FI2010_DATA_ACQUISITION](FI2010_DATA_ACQUISITION.md),
  [FI2010_MULTIFOLD_PROTOCOL](FI2010_MULTIFOLD_PROTOCOL.md),
  [FI2010_MULTIFOLD_CLASSICAL](FI2010_MULTIFOLD_CLASSICAL.md)
- Multi-fold modules:
  `chronoslob/experiments/fi2010_multifold.py`,
  `chronoslob/experiments/fi2010_multifold_runner.py`,
  `chronoslob/experiments/fi2010_neural_runner.py`
- Multi-fold tests: `tests/test_fi2010_multifold.py`,
  `tests/test_fi2010_multifold_runner.py`,
  `tests/test_fi2010_neural_runner.py`
- Split support: generic temporal split and official split-aware
  evaluation from the combined CSV `split` column. Multi-fold
  preparation produces the same split column per fold.
- Classical multi-fold artefacts: `summary.json`,
  `results_by_fold.csv`, `results_summary.csv`,
  `calibration_summary.csv`, `execution_summary.csv`,
  `model_failures.json` and per-fold lightweight evidence under
  `folds/fold_<N>/`.
- Real classical multi-fold evidence:
  [`experiments/fi2010_multifold_classical/`](../experiments/fi2010_multifold_classical/)
  contains a run across the five official NoAuction ZScore folds for the
  classical baseline set (`majority`, `logistic`, `ridge`, `elastic_net`,
  `random_forest`, `gradient_boosting`) at horizon `label_10`. Full
  predictions are not written.
- Serious neural plan: `configs/experiments/fi2010_neural_serious.yaml`
  and `inspect-fi2010-neural-plan` define the supervised neural run grid.
- Neural benchmark runner: `run-fi2010-neural-benchmark` executes selected
  supervised neural subsets and writes `summary.json`, `run_plan.csv`,
  `results_by_fold_seed.csv`, `results_summary.csv`,
  `training_summary.csv`, `model_capacity_summary.csv` and
  `model_failures.json`. Full prediction rows and checkpoints are not
  written by default.
- Real reduced-scope neural multi-fold evidence:
  [`experiments/fi2010_multifold_neural/`](../experiments/fi2010_multifold_neural/)
  contains a CPU run across the five official NoAuction ZScore folds for
  `deeplob_style` and `matrix_transformer` at horizon `label_10`, with a
  single seed (`0`), single lookback (`20`) and `max_epochs=25`. Scope is
  reduced from the configured grid because the full grid is impractical
  on CPU. All ten planned runs completed; zero failures. Full predictions
  and checkpoints are not written. The full configured grid is not yet
  reported here.
- Statistical uncertainty layer:
  [STATISTICAL_UNCERTAINTY](STATISTICAL_UNCERTAINTY.md) and
  [`experiments/fi2010_uncertainty/`](../experiments/fi2010_uncertainty/)
  contain fold-level confidence intervals, paired comparisons against the
  `gradient_boosting` baseline, rank stability and a combined ranking
  computed from the stored multi-fold tables. The neural numbers remain
  reduced-scope, single-seed evidence; the analysis does not promote any
  model beyond that evidence.

## Paper Experiment Runner

- Modules: `chronoslob/experiments/paper_runner.py`,
  `chronoslob/experiments/model_registry.py`,
  `chronoslob/experiments/neural_adapters.py`,
  `chronoslob/models/matrix_transformer.py`,
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
- Matrix path: `transformer` and `matrix_transformer` use the
  normalised FI-2010 matrix path; raw order-book schemas remain strict.
- FI-2010 plot artefacts: `plots/reliability_curve.png`,
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
- FI-2010 ablation artefacts: `ablation_summary.json`,
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
- FI-2010 systems artefacts: `system_benchmark_summary.json`,
  `system_benchmark_results.csv`, `environment.json`, per-category
  Markdown reports and a validated child paper experiment for runner
  timing. Smoke fixture timings are labelled as smoke measurements and
  are not benchmark evidence.
- Normalised FI-2010 support: feature throughput and inference latency
  can run in matrix mode without reconstructing raw order-book snapshots
  from z-score rows.

## Paper Report Builder

- Modules: `chronoslob/experiments/reporting.py`
- Configs: uses stored experiment config snapshots from completed paper
  experiment directories.
- Tests: `tests/test_paper_report_builder.py`
- CLI: `build-paper-report`, `inspect-paper-report`
- Docs: [PAPER_REPORTS](PAPER_REPORTS.md)
- Report artefacts: a Markdown empirical report plus
  `<report_stem>_summary.json`, both generated from stored paper experiment,
  ablation and systems benchmark artefacts. Fixture or smoke inputs remain
  labelled as smoke reports and are not benchmark evidence.
- Formatting: headings, tables and code fences are emitted as separate
  Markdown blocks, and repeated warnings are grouped into a summary plus
  detailed appendix.
- Real-data evidence: `reports/chronoslob_empirical_report.md` and
  `reports/chronoslob_empirical_report_summary.json` are built from the
  paper experiment under `experiments/fi2010_midprice_h10/` and the
  ablation suite under `experiments/fi2010_midprice_h10_ablations/`,
  with systems measurements under
  `experiments/fi2010_midprice_h10_systems/`,
  on the official FI-2010 NoAuction ZScore fold 1 file pair documented
  in [FI2010_DATA_ACQUISITION](FI2010_DATA_ACQUISITION.md).

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

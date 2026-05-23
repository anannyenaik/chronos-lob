# Report Writing Guide

This guide helps the user write the final ChronosLOB technical report manually.
It is not the report itself. It points to implemented artefacts, safe wording and
missing evidence so the final prose stays honest and reproducible.

## Abstract

- Discuss: ChronosLOB as a research-engineering platform for market
  microstructure modelling, representation learning and execution-aware
  validation.
- Reference: `README.md`, `docs/PROJECT_STATUS.md`,
  `reports/report_archive/project_inventory.md`.
- Safe claims: the repository implements infrastructure and validation tooling.
- Missing evidence: real benchmark outputs and report-ready result artefacts.

## Introduction

- Discuss: the forecasting-versus-tradability gap, leakage-safe labels, temporal
  splits and the need for execution-aware validation.
- Reference: `reports/limitations.md`, `docs/SAFETY_AND_LIMITATIONS.md`,
  `reports/report_archive/figures/architecture_overview.mmd`.
- Safe claims: prediction quality, calibration quality and simplified execution
  validation are separate questions.
- Missing evidence: real data experiment outputs tying those questions together.

## Related Work

- Discuss: FI-2010-style LOB forecasting, DeepLOB-style supervised baselines,
  transformer representations and self-supervised learning at a conceptual
  level.
- Reference: `reports/deeplob_baseline.md`,
  `reports/transformer_architecture.md`,
  `reports/self_supervised_objectives.md`.
- Safe claims: the repo includes compatible baseline and transformer plumbing.
- Missing evidence: literature comparison tables and reproduced benchmark
  results.

## Data

- Discuss: local FI-2010-style loading, offline Binance-style reconstruction,
  canonical event logs and fixture policy.
- Reference: `chronoslob/data/`, `chronoslob/book/`,
  `reports/data_quality.md`, `reports/order_book_reconstruction.md`,
  `reports/event_log_storage.md`,
  `reports/report_archive/figures/data_pipeline.mmd`.
- Safe claims: loaders and replay utilities are local-only and tested on small
  synthetic fixtures.
- Missing evidence: user-supplied real data provenance and preprocessing notes.

## Feature And Label Engineering

- Discuss: past-only microstructure features, future-window labels and
  no-look-ahead checks.
- Reference: `chronoslob/features/`, `chronoslob/labels/`,
  `reports/feature_engine.md`, `reports/label_engine.md`,
  `reports/leakage_controls.md`.
- Safe claims: feature and label logic is structured to keep future information
  out of model inputs.
- Missing evidence: dataset-specific feature audits on real experiment inputs.

## Modelling

- Discuss: classical baselines, DeepLOB-style supervised CNN-LSTM and the
  supervised transformer encoder.
- Reference: `chronoslob/models/`, `chronoslob/training/`,
  `reports/baselines.md`, `reports/deeplob_baseline.md`,
  `reports/transformer_architecture.md`,
  `reports/report_archive/figures/model_stack.mmd`.
- Safe claims: model and training code paths are implemented with deterministic
  smoke coverage.
- Missing evidence: benchmark comparison outputs from real temporal splits.

## Self-Supervised Pretraining

- Discuss: deterministic event tokenisation, masked-field objectives and
  next-field objectives.
- Reference: `chronoslob/models/tokenisation.py`, `chronoslob/models/ssl.py`,
  `chronoslob/training/ssl_*`, `reports/event_tokenisation.md`,
  `reports/self_supervised_objectives.md`.
- Safe claims: self-supervised objectives and datasets exist as infrastructure.
- Missing evidence: pretraining runs and downstream comparison artefacts.

## Calibration And Uncertainty

- Discuss: temperature scaling, calibration error, confidence filtering and
  abstention diagnostics.
- Reference: `chronoslob/models/calibration.py`,
  `chronoslob/training/calibration.py`,
  `reports/calibration_uncertainty.md`.
- Safe claims: calibration utilities separate confidence quality from predictive
  accuracy.
- Missing evidence: real reliability diagrams or calibration tables generated
  from documented runs.

## Execution-Aware Validation

- Discuss: costs, latency, turnover, simple risk constraints and passive-fill
  proxies as a simplified research simulation.
- Reference: `chronoslob/backtest/`,
  `reports/execution_aware_validation.md`,
  `reports/report_archive/figures/evaluation_stack.mmd`.
- Safe claims: execution-aware validation utilities expose assumptions instead
  of hiding them.
- Missing evidence: real signal outputs and documented cost assumptions for any
  future experiment.

## Transfer, Regime And Ablation Analysis

- Discuss: transfer matrices, regime summaries, ablation ranking and sensitivity
  curves for supplied experiment records.
- Reference: `chronoslob/analysis/`,
  `reports/transfer_regime_ablation_analysis.md`.
- Safe claims: analysis utilities organise reproducible result records.
- Missing evidence: upstream experiment records from real configs and data.

## Limitations

- Discuss: synthetic fixtures, public data caveats, crypto transfer limits,
  simplified execution, queue position, partial fills and market impact.
- Reference: `reports/limitations.md`,
  `docs/SAFETY_AND_LIMITATIONS.md`,
  `reports/report_archive/limitations_index.md`.
- Safe claims: limitations are explicit and part of the project design.
- Missing evidence: stronger claims remain out of scope until reproducible runs
  exist.

## Conclusion

- Discuss: what the repository demonstrates as a reproducible experiment
  artefact and what remains before any result claim.
- Reference: `docs/PROJECT_STATUS.md`,
  `reports/report_archive/report_claims_checklist.md`,
  `reports/report_archive/reproducibility_commands.md`.
- Safe claims: ChronosLOB is ready to support manual report writing and future
  reproducible experiments.
- Missing evidence: real results and report-ready findings should be handled
  only after reproducible experiments exist.

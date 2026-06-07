# README Result Snapshot

## Current Evidence Status

| component | status | smoke |
| --- | --- | --- |
| fi2010_classical_benchmarks | archived_valid | no |
| fi2010_ssl_runner_outputs | obsolete_superseded | no |
| fi2010_neural_full_grid | archived_valid | no |
| fi2010_neural_proper_training_subset | partial_real | no |
| ssl_failure_analysis_report | complete_real | no |
| fi2010_ssl_v2_benchmark | complete_real | no |
| ssl_v2_analysis_report | complete_real | no |
| fi2010_figures | partial_real | no |
| execution_v3_outputs | archived_valid | no |
| execution_v3_analysis_report | complete_real | no |
| execution_centrepiece_report | archived_valid | no |
| feature_registry_audit_outputs | optional_missing | no |
| feature_ablation_outputs | partial_real | no |
| feature_ablation_analysis_report | complete_real | no |
| ablation_figures | complete_real | no |
| final_empirical_report | complete_real | no |
| synthetic_lob_extension_report | archived_valid | no |
| binance_l2_extension_report | partial_real | no |
| project_audit_archive | unknown_staleness | no |

## Results

- Best classical result: gradient_boosting macro-F1 0.4654 from stored artefacts.
- Best neural full-grid result: matrix_transformer macro-F1 0.4922 from stored artefacts.
- SSL comparison: no broad SSL improvement claim; see claim_audit.md for macro-F1=unsupported, calibration=unsupported.
- SSL-v2: scoped predictive improvement is supported only for the exact stored scope: folds 1, 2, 3, 4, 5, horizons 10, 50, seeds 0, 1, 2, lookbacks 50; calibration=supported and broad SSL remains unsupported.
- Execution-v3: complete real artefacts retained (generated at an older commit).
- Execution centrepiece: complete real artefacts retained (generated at an older commit).
- Feature ablations: partial_real.
- Figures: partial_real.

## Limitations

- Smoke diagnostics are labelled as smoke diagnostics and are not empirical evidence.
- Full-grid neural artefacts are present, but public wording must quote stored scope and metrics exactly.
- SSL improvement language is blocked unless real aggregate deltas support it.
- SSL-v2 predictive and calibration claims are limited to the exact stored scope: folds 1, 2, 3, 4, 5, horizons 10, 50, seeds 0, 1, 2, lookbacks 50; calibration=supported and broad SSL improvement remains unsupported.
- Execution-v3 metrics are offline proxy diagnostics, not deployed execution results.
- FI-2010 snapshot features do not expose event-level order flow or queue position.

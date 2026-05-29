# README Result Snapshot

## Current Evidence Status

| component | status | smoke |
| --- | --- | --- |
| fi2010_classical_benchmarks | complete_real | no |
| fi2010_ssl_runner_outputs | missing | no |
| fi2010_neural_full_grid | complete_real | no |
| fi2010_figures | partial_real | no |
| execution_v3_outputs | complete_real | no |
| feature_registry_audit_outputs | missing | no |
| feature_ablation_outputs | partial_real | no |
| ablation_figures | complete_real | no |
| final_empirical_report | complete_real | no |
| project_audit_archive | unknown_staleness | no |

## Results

- Best classical result: gradient_boosting macro-F1 0.4654 from stored artefacts.
- Best neural full-grid result: matrix_transformer macro-F1 0.4922 from stored artefacts.
- SSL comparison: no broad SSL improvement claim; see claim_audit.md for macro-F1=unsupported, calibration=unsupported.
- Execution-v3: complete real artefacts present.
- Feature ablations: partial_real.
- Figures: partial_real.

## Limitations

- Smoke diagnostics are labelled as smoke diagnostics and are not empirical evidence.
- Full-grid neural artefacts are present, but public wording must quote stored scope and metrics exactly.
- SSL improvement language is blocked unless real aggregate deltas support it.
- Execution-v3 metrics are offline proxy diagnostics, not deployed execution results.
- FI-2010 snapshot features do not expose event-level order flow or queue position.

# README Result Snapshot

## Current Evidence Status

| component | status | smoke |
| --- | --- | --- |
| fi2010_classical_benchmarks | stale | no |
| fi2010_ssl_runner_outputs | missing | no |
| fi2010_neural_full_grid | stale | no |
| fi2010_neural_proper_training_subset | partial_real | no |
| fi2010_figures | partial_real | no |
| execution_v3_outputs | stale | no |
| feature_registry_audit_outputs | missing | no |
| feature_ablation_outputs | stale | no |
| ablation_figures | complete_real | no |
| final_empirical_report | complete_real | no |
| project_audit_archive | unknown_staleness | no |

## Results

- Best classical result: not cleanly supported (stale).
- Best neural full-grid result: not cleanly supported (stale).
- SSL comparison: no broad SSL improvement claim; see claim_audit.md for macro-F1=unsupported, calibration=unsupported.
- Execution-v3: stale.
- Feature ablations: stale.
- Figures: partial_real.

## Limitations

- Smoke diagnostics are labelled as smoke diagnostics and are not empirical evidence.
- Missing real full-grid artefacts mean no full-grid empirical result is claimed.
- SSL improvement language is blocked unless real aggregate deltas support it.
- Execution-v3 metrics are offline proxy diagnostics, not deployed execution results.
- FI-2010 snapshot features do not expose event-level order flow or queue position.

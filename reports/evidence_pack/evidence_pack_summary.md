# ChronosLOB Evidence Pack Summary

This pack inventories stored artefacts and separates smoke diagnostics from real evidence.

`complete_real` means required non-smoke artefacts are present and pass completion checks.
`partial_real` means real artefacts exist but the scope is incomplete, mixed or has explicit skipped diagnostics.
Smoke rows remain code-path checks only.

| artefact | status | smoke | completed | failed | notes |
| --- | --- | --- | --- | --- | --- |
| fi2010_classical_benchmarks | complete_real | no | 60 | 0 | Generated timestamp is present. |
| fi2010_ssl_runner_outputs | missing | no |  |  | Artefact path is missing. |
| fi2010_neural_full_grid | complete_real | no | 135 | 0 | Generated timestamp is present. |
| fi2010_figures | partial_real | no | 17 |  | Generated timestamp is present. |
| execution_v3_outputs | complete_real | no | 4800519 | 0 | Generated timestamp is present. |
| feature_registry_audit_outputs | missing | no |  |  | The CLI feature audit is read-only unless a caller stores its output. Artefact path is missing. |
| feature_ablation_outputs | partial_real | no | 840 | 0 | Generated timestamp is present. |
| ablation_figures | complete_real | no | 6 |  | Generated timestamp is present. |
| final_empirical_report | complete_real | no |  |  | Generated timestamp is present. |
| project_audit_archive | unknown_staleness | no |  |  | No commit, input-hash or timestamp evidence was available for staleness. |

Smoke-test artefacts are code-path diagnostics only. They are not empirical evidence.
Generated evidence-pack files should be regenerated from artefacts rather than hand-edited.

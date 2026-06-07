# ChronosLOB Evidence Pack Summary

This pack inventories stored artefacts and separates smoke diagnostics from real evidence.
Completeness and freshness are tracked separately, so a valid retained summary is never shown as broken purely because its heavy compute was produced at an older commit.

Status vocabulary:

- `complete_real`: required non-smoke artefacts are present and pass completion checks; freshness is tracked separately.
- `archived_valid`: content-complete evidence generated at an older commit, or whose heavy raw predictions/checkpoints were intentionally removed; retained summaries match.
- `partial_real`: real artefacts exist, but the scope is incomplete, mixed or has explicit skipped diagnostics.
- `optional_missing`: an optional artefact was never stored; no core claim depends on it.
- `obsolete_superseded`: a legacy artefact is superseded by newer matched evidence.
- `stale`: retained content changed and no longer matches its recorded hash, or an input is newer than its derived output; this needs recomputation.
- `unknown_staleness`: not enough hash or timestamp evidence to compute freshness.
- `smoke_test_only`: code-path diagnostics only; never empirical evidence.

The freshness column records `fresh`, `archived`, `stale`, `unknown` or `absent` independently of completeness.

| artefact | status | freshness | smoke | completed | failed | notes |
| --- | --- | --- | --- | --- | --- | --- |
| fi2010_classical_benchmarks | archived_valid | archived | no | 60 | 0 | Recorded git commit is older than the current repository commit, but retained hashes, files and summaries are consistent; the artefact is summary-valid at its generating commit. |
| fi2010_ssl_runner_outputs | obsolete_superseded | absent | no |  |  | Legacy standalone SSL runner output. The matched full-grid SSL comparison and the SSL-v2 benchmark supersede it; its absence does not weaken the retained SSL evidence. Superseded by fi2010_neural_full_grid, fi2010_ssl_v2_benchmark; legacy artefact intentionally not retained, so its absence does not weaken the matched evidence. |
| fi2010_neural_full_grid | archived_valid | archived | no | 135 | 0 | Recorded git commit is older than the current repository commit, but retained hashes, files and summaries are consistent; the artefact is summary-valid at its generating commit. |
| fi2010_neural_proper_training_subset | partial_real | archived | no | 6 | 0 | Recorded git commit is older than the current repository commit, but retained hashes, files and summaries are consistent; the artefact is summary-valid at its generating commit. |
| ssl_failure_analysis_report | complete_real | fresh | no | 4 |  | Hash/commit staleness checks passed. |
| fi2010_ssl_v2_benchmark | complete_real | fresh | no | 60 | 0 | Hash/commit staleness checks passed. |
| ssl_v2_analysis_report | complete_real | fresh | no |  | 0 | Hash/commit staleness checks passed. |
| fi2010_figures | partial_real | fresh | no | 17 |  | Hash/commit staleness checks passed. |
| execution_v3_outputs | archived_valid | archived | no | 4800519 | 0 | 135 recorded input artefact(s) were intentionally removed (heavy raw predictions, checkpoints or ignored per-run details); retained summaries and manifests are consistent. Retained hashed files still match their recorded hashes. The generating commit also differs from the current repository commit. |
| execution_v3_analysis_report | complete_real | fresh | no | 6 |  | Hash/commit staleness checks passed. |
| execution_centrepiece_report | archived_valid | archived | no | 1 |  | Recorded git commit is older than the current repository commit, but retained hashes, files and summaries are consistent; the artefact is summary-valid at its generating commit. |
| feature_registry_audit_outputs | optional_missing | absent | no |  |  | Optional artefact. The CLI feature audit is read-only unless a caller stores its output; no public claim depends on a stored copy. Optional artefact path is absent; no core public claim depends on it. |
| feature_ablation_outputs | partial_real | archived | no | 2520 | 0 | Recorded git commit is older than the current repository commit, but retained hashes, files and summaries are consistent; the artefact is summary-valid at its generating commit. |
| feature_ablation_analysis_report | complete_real | fresh | no | 2580 | 0 | Hash/commit staleness checks passed. |
| ablation_figures | complete_real | fresh | no | 6 |  | Hash/commit staleness checks passed. |
| final_empirical_report | complete_real | fresh | no |  |  | Hash/commit staleness checks passed. |
| synthetic_lob_extension_report | archived_valid | archived | no | 12 | 0 | Recorded git commit is older than the current repository commit, but retained hashes, files and summaries are consistent; the artefact is summary-valid at its generating commit. |
| binance_l2_extension_report | partial_real | fresh | no | 3 |  | Hash/commit staleness checks passed. |
| project_audit_archive | unknown_staleness | unknown | no |  |  | No commit, input-hash or timestamp evidence was available for staleness. |

## Archived or summary-valid artefacts

These artefacts were produced by heavy compute at an earlier commit.
In some cases, large raw predictions or checkpoints were intentionally removed to keep the repository light. This is expected and is not a failure:

- The retained summaries, manifests and recorded hashes are consistent.
- No public claim depends on regenerating the removed raw files.
- `archived_valid` rows carry the same evidential weight as `complete_real`.
- A row becomes `stale` only when retained content actually changed or an input is newer than its output, which would require recomputation.

Recomputation commands for each artefact are listed in `reproduction_commands.md`.

| artefact | status | generating commit | note |
| --- | --- | --- | --- |
| fi2010_classical_benchmarks | archived_valid | a72d46f0a1d7 | Recorded git commit is older than the current repository commit, but retained hashes, files and summaries are consistent; the artefact is summary-valid at its generating commit. |
| fi2010_neural_full_grid | archived_valid | a72d46f0a1d7 | Recorded git commit is older than the current repository commit, but retained hashes, files and summaries are consistent; the artefact is summary-valid at its generating commit. |
| fi2010_neural_proper_training_subset | partial_real | ef0724fb55e5 | Recorded git commit is older than the current repository commit, but retained hashes, files and summaries are consistent; the artefact is summary-valid at its generating commit. |
| execution_v3_outputs | archived_valid | a72d46f0a1d7 | 135 recorded input artefact(s) were intentionally removed (heavy raw predictions, checkpoints or ignored per-run details); retained summaries and manifests are consistent. Retained hashed files still match their recorded hashes. The generating commit also differs from the current repository commit. |
| execution_centrepiece_report | archived_valid | 597b3599a3a4 | Recorded git commit is older than the current repository commit, but retained hashes, files and summaries are consistent; the artefact is summary-valid at its generating commit. |
| feature_ablation_outputs | partial_real | 21807a9b9217 | Recorded git commit is older than the current repository commit, but retained hashes, files and summaries are consistent; the artefact is summary-valid at its generating commit. |
| synthetic_lob_extension_report | archived_valid | 4377803a3d0d | Recorded git commit is older than the current repository commit, but retained hashes, files and summaries are consistent; the artefact is summary-valid at its generating commit. |

## Optional or superseded artefacts

These paths are intentionally absent and do not weaken any core claim:

- `fi2010_ssl_runner_outputs` (obsolete_superseded): Legacy standalone SSL runner output. The matched full-grid
  SSL comparison and the SSL-v2 benchmark supersede it; its absence does not weaken the retained SSL evidence.
  Superseded by fi2010_neural_full_grid, fi2010_ssl_v2_benchmark; legacy artefact intentionally not retained, so
  its absence does not weaken the matched evidence.
- `feature_registry_audit_outputs` (optional_missing): Optional artefact. The CLI feature audit is read-only
  unless a caller stores its output; no public claim depends on a stored copy. Optional artefact path is absent;
  no core public claim depends on it.

Smoke-test artefacts are code-path diagnostics only. They are not empirical evidence.
Generated evidence-pack files should be regenerated from artefacts rather than hand-edited.

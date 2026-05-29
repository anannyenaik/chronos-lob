# Public Writing Audit

Date: 2026-05-29

## Files Reviewed

- `README.md`
- `docs/EVIDENCE_PACK.md`
- `docs/EXECUTION_VALIDATION_V3.md`
- `docs/FEATURE_ABLATIONS.md`
- `docs/FIGURE_INDEX.md`
- `docs/MICROSTRUCTURE_FEATURES.md`
- `docs/PROJECT_STATUS.md`
- `reports/chronoslob_final_empirical_report.md`
- `reports/evidence_pack/evidence_pack_summary.md`
- `reports/evidence_pack/claim_audit.md`
- `reports/evidence_pack/readme_result_snapshot.md`
- `reports/real_run_completion_summary.md`
- Experiment READMEs and notes for neural full grid, feature ablations and execution-v3.
- Figure, execution-v3 and feature-ablation documentation.

## Findings

| Area | Issue | Edit made or recommendation |
| --- | --- | --- |
| README | Stale placeholder wording around SSL/full-grid evidence and reduced-scope neural evidence. | Rewritten around current statuses, matched-grid caveats and non-claims. |
| README | Result language previously mixed infrastructure with empirical findings. | Added compact evidence-status table and separated findings from non-claims. |
| Project status | Neural evidence still read as single-seed only. | Updated to distinguish completed one-epoch full grid from the separate 25-epoch reduced-scope benchmark. |
| Evidence pack | Generated public bullet files were framed too broadly. | Generator now writes `public_bullets_*` files and describes them as public release summaries, not personal-profile copy. |
| Claim audit | Mixed SSL rows could be interpreted as partial support for broad SSL improvement. | Audit now treats mixed broad SSL-improvement claims as unsupported. |
| Execution-v3 | Generated notes changed, making the manifest stale. | Regenerated execution-v3 and dependent figures. |
| Feature ablations | Partial scope needed clearer context. | Docs and final report now state the current folds/horizon/models and the unfinished scope. |
| Figures | Skipped regime diagnostics could look like missing results. | Figure docs now state regime plots are skipped because labels are unavailable. |
| Generated reports | `What This Proves` was too strong. | Final report generator now renders `What This Supports` and `What This Does Not Claim`. |
| Legacy reports | Older architecture/execution docs used PnL/backtest wording where proxy wording is clearer. | Reworded the most visible cases; remaining historical/generated occurrences are caveated in the claim scan. |

## Missing Caveats Added

- Smoke-test outputs are code-path diagnostics only.
- `partial_real` means real evidence exists but scope is incomplete or explicitly skipped.
- Execution-v3 is an offline execution-aware proxy diagnostic; cost-adjusted values are not PnL.
- `snapshot_order_flow_proxy` is a snapshot-delta proxy, not true event-level OFI.
- The one-epoch matched neural grid is not the same artefact as the 25-epoch reduced-scope neural benchmark.
- No SSL improvement is supported by the matched full grid.

## Generated Artefacts

Do not manually edit these after upstream artefacts change; regenerate them:

- `reports/chronoslob_final_empirical_report.md`
- `reports/chronoslob_final_empirical_report_summary.json`
- `reports/evidence_pack/*`
- `experiments/fi2010_execution_v3/*`
- `reports/figures/fi2010_neural_full_grid/*`
- `reports/figures/fi2010_feature_ablations/*`

Legacy local evidence-pack bullet files with old naming are no longer generated
by the evidence-pack builder and should be excluded from public release commits.

## Recommended Follow-Up

- Refresh the project-audit archive if clearing `unknown_staleness` is required.
- Keep the manual paper out of scope until the public report is intentionally authored.
- Broaden feature ablations only with new experiments, not writing edits.

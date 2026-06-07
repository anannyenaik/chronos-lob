# Evidence-Pack Trust Repair Plan

This note records why the evidence pack previously made valid retained artefacts
look broken, and how the classification was repaired so that public reviewers see
an accurate picture.

## Problem

The evidence pack collapsed two independent properties into a single status:

- completeness (is the scope complete, partial or smoke-only?), and
- freshness (does the generating commit and content still verify?).

Because of this, any artefact whose recorded generating commit differed from the
current repository commit was marked `stale`, and any intentionally removed heavy
raw file made an artefact look broken. Two optional legacy paths were marked
`missing`. None of these were real evidence problems: the retained summaries,
manifests and hashes were still valid.

This produced a reviewer-trust problem. A reader saw seven `stale` rows and two
`missing` rows and could reasonably assume the evidence was broken, even though
the summaries, reports and claim-audited artefacts were sound.

## Root causes

1. `_assess_staleness` returned `stale` as soon as the recorded git commit
   differed from the current commit, before any content check.
2. A recorded input/output hash whose file had been intentionally removed (heavy
   raw predictions and checkpoints) was treated as `stale` rather than archival.
3. Optional and superseded paths used the same `missing` status as genuinely
   required paths.
4. Infrastructure/existence claims required `complete_real` support, so an older
   generating commit silently downgraded them to `partially_supported`.

## Repair

Completeness and freshness are now separate. The status taxonomy was extended:

- `archived_valid`: content-complete evidence produced at an older commit, or
  whose heavy raw predictions/checkpoints were intentionally removed; retained
  summaries, manifests and hashes verify. It carries `complete_real` weight.
- `optional_missing`: an optional artefact was never stored; no core claim needs it.
- `obsolete_superseded`: a legacy artefact is superseded by newer matched evidence.

A new `freshness` column records `fresh`, `archived`, `stale`, `unknown` or
`absent` independently of completeness.

`_assess_staleness` now reports `stale` only when retained content actually
changed (a recorded hash no longer matches), a non-archival recorded hash path is
missing, or an input is newer than its derived output. A differing generating
commit, or an intentionally removed heavy raw prediction/checkpoint or ignored
per-run detail, is reported as `archived`. Existence claims are supported when
every required artefact is present and valid, including `archived_valid`,
`partial_real` and `unknown_staleness`.

The final report no longer records hashes of the regenerated evidence-pack
outputs, so rebuilding the pack no longer marks the report `stale`.

## Artefact reclassification (before vs after)

| artefact | before | after | reason |
| --- | --- | --- | --- |
| fi2010_classical_benchmarks | stale | archived_valid | older generating commit; summaries verify |
| fi2010_neural_full_grid | stale | archived_valid | older generating commit; summaries verify |
| execution_v3_outputs | stale | archived_valid | heavy raw predictions intentionally removed |
| synthetic_lob_extension_report | stale | archived_valid | older generating commit; summaries verify |
| fi2010_neural_proper_training_subset | stale | partial_real | older commit; scope intentionally partial |
| feature_ablation_outputs | stale | partial_real | older commit; scope intentionally partial |
| fi2010_ssl_v2_benchmark | partial_real | complete_real | folds 1-5, horizons 10/50, seed 0 closed; freshness tracked separately |
| binance_l2_extension_report | stale | partial_real | older commit; fixture-shaped replay |
| fi2010_ssl_runner_outputs | missing | obsolete_superseded | superseded by full grid and SSL-v2 |
| feature_registry_audit_outputs | missing | optional_missing | optional; no core claim needs it |
| project_audit_archive | unknown_staleness | unknown_staleness | documentation archive; no hash signals |
| ssl_failure_analysis_report | complete_real | complete_real | unchanged |
| execution_v3_analysis_report | complete_real | complete_real | unchanged |
| feature_ablation_analysis_report | complete_real | complete_real | unchanged |
| ablation_figures | complete_real | complete_real | unchanged |
| fi2010_figures | partial_real | partial_real | unchanged; skipped diagnostics |
| ssl_v2_analysis_report | partial_real | complete_real | scoped SSL-v2 analysis rebuilt from retained tables |
| final_empirical_report | complete_real | complete_real | rebuilt at the current commit |

Status counts before: complete_real=5, missing=2, partial_real=3, stale=7,
unknown_staleness=1. Current release counts: archived_valid=5, complete_real=7,
partial_real=4, obsolete_superseded=1, optional_missing=1,
unknown_staleness=1. No genuinely `stale` and no `missing` required artefacts
remain.

## Claim-audit effects

Core existence claims are no longer downgraded by an older generating commit or
by irrelevant optional artefacts. The following moved from `partially_supported`
to `supported`: reproducible platform, leakage-safe FI-2010 evaluation,
supervised-vs-SSL comparison, train-only SSL, execution-aware proxy diagnostics,
feature ablations, best classical baseline and confidence filtering.

Claim-status counts before: supported=14, partially_supported=11, unsupported=8,
forbidden=15, needs_real_evidence=2. Current release counts: supported=30,
partially_supported=3, unsupported=7, forbidden=18, needs_real_evidence=2.

## What stays unsupported or forbidden

The repair does not weaken any boundary:

- broad SSL improvement remains unsupported; the matched SSL-v1 full-grid
  predictive and calibration deltas remain mixed;
- the SSL-v2 benchmark is complete for folds 1-5, horizons 10/50, seeds 0-2 and
  lookback 50. Its mean predictive and calibration improvements are supported
  only for that exact stored scope, and the result is mixed by seed and horizon;
- synthetic real-market generalisation and Binance equity-market generalisation
  stay unsupported;
- profitability, tradable alpha, live trading, production execution simulation,
  state-of-the-art, foundation-model, true event-level order flow on FI-2010 and
  queue-position claims stay forbidden.

## What would require recomputation

Only a genuinely `stale` row needs recomputation. Per-artefact recomputation
commands are listed in `reports/evidence_pack/reproduction_commands.md`. Removed
heavy raw predictions, checkpoints and ignored per-run details can be regenerated
with those commands, but no public claim depends on regenerating them.

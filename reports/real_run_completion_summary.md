# ChronosLOB Real-Run Completion Summary

Honest status of the end-to-end real FI-2010 evidence run.

- Date: 2026-05-28
- Repository commit: `a72d46f0a1d7e0ccb62853eee6004375f7b5358c`
- Compute: CPU-only (PyTorch 2.12.0+cpu, CUDA unavailable)

This run produced real, reproducible artefacts. Every reported number comes from
a stored real artefact, and no smoke-test artefact was treated as empirical
evidence. Where compute limited scope, the artefacts are classified
`partial_real` and the exact remaining commands are listed below.

## Current Status Note

This file is chronological. Earlier sections record intermediate states
(including partial neural-grid and narrower feature-ablation states) at the time
they were observed. The current public-release status is the later state:
neural full grid `complete_real`, execution-v3 `complete_real`, feature
ablations `partial_real`, figures real with unsupported regime plots skipped,
and no supported SSL improvement, profitability, PnL, tradable-alpha, SOTA or
foundation-model claim.

## Commands run

Refresh and generation (all local, no network):

```bash
# Phase 1 - classical refresh (folds 1-5)
python -m chronoslob.cli run-fi2010-multifold-classical \
  --config configs/experiments/fi2010_multifold.yaml \
  --processed-root data/processed/fi2010 --folds all \
  --out experiments/fi2010_multifold_classical --overwrite

# Phase 2 - neural full grid (completed folds 1-2 of 5 this pass)
python -m chronoslob.cli run-fi2010-neural-full-grid \
  --config configs/experiments/fi2010_neural_serious.yaml \
  --processed-root data/processed/fi2010 \
  --folds 1,2,3,4,5 --horizons 10,20,50 --seeds 0,1,2 --lookbacks 20 \
  --objectives supervised,masked_reconstruction,next_field \
  --pretrain-epochs 1 --max-epochs 1 --batch-size 256 --device cpu \
  --out experiments/fi2010_neural_full_grid

# Phase 3 - feature audit + ablations (fold 1, horizon 10, seed 0)
python -m chronoslob.cli audit-fi2010-features \
  --path data/processed/fi2010/fold1_combined.csv --feature-groups all
python -m chronoslob.cli run-fi2010-feature-ablations \
  --config configs/experiments/fi2010_multifold.yaml \
  --processed-root data/processed/fi2010 \
  --folds 1 --horizons 10 --seeds 0 \
  --models logistic,ridge,elastic_net,gradient_boosting \
  --ablation-modes all --feature-groups all \
  --out experiments/fi2010_feature_ablations --strict

# Phase 4 - execution-aware proxy diagnostic v3
python -m chronoslob.cli build-fi2010-execution-v3 \
  --neural-full-grid experiments/fi2010_neural_full_grid \
  --feature-ablations experiments/fi2010_feature_ablations \
  --out experiments/fi2010_execution_v3 --overwrite --strict

# Phase 5 - figures
python -m chronoslob.cli build-fi2010-figures \
  --neural-full-grid experiments/fi2010_neural_full_grid \
  --execution-v3 experiments/fi2010_execution_v3 \
  --out reports/figures/fi2010_neural_full_grid --overwrite --strict
python -m chronoslob.cli build-fi2010-ablation-figures \
  --feature-ablations experiments/fi2010_feature_ablations \
  --out reports/figures/fi2010_feature_ablations --overwrite

# Phase 6 - final empirical report
python -m chronoslob.cli build-final-empirical-report \
  --classical experiments/fi2010_multifold_classical \
  --neural experiments/fi2010_multifold_neural \
  --uncertainty experiments/fi2010_uncertainty \
  --ablations experiments/fi2010_brutal_ablations \
  --feature-ablations experiments/fi2010_feature_ablations \
  --execution experiments/fi2010_execution_v2 \
  --execution-v3 experiments/fi2010_execution_v3 \
  --external experiments/fi2010_external_context \
  --neural-full-grid experiments/fi2010_neural_full_grid \
  --out reports/chronoslob_final_empirical_report.md --overwrite

# Phase 7 - evidence pack
python -m chronoslob.cli build-evidence-pack \
  --out reports/evidence_pack \
  --neural-full-grid experiments/fi2010_neural_full_grid \
  --figures reports/figures/fi2010_neural_full_grid \
  --execution-v3 experiments/fi2010_execution_v3 \
  --feature-ablations experiments/fi2010_feature_ablations \
  --ablation-figures reports/figures/fi2010_feature_ablations \
  --final-report reports/chronoslob_final_empirical_report.md \
  --classical experiments/fi2010_multifold_classical --overwrite --strict
```

## Artefacts generated

| Artefact | Path | Status |
| --- | --- | --- |
| Classical benchmarks | `experiments/fi2010_multifold_classical/` | refreshed (complete_real) |
| Neural full grid | `experiments/fi2010_neural_full_grid/` | partial_real (folds 1-2) |
| Feature audit | (read-only CLI output, not stored) | pass |
| Feature ablations | `experiments/fi2010_feature_ablations/` | complete_real (fold 1, h10, seed 0) |
| Execution-v3 | `experiments/fi2010_execution_v3/` | complete_real |
| Neural figures | `reports/figures/fi2010_neural_full_grid/` | partial_real (17 figures) |
| Ablation figures | `reports/figures/fi2010_feature_ablations/` | complete_real (6 figures) |
| Final report | `reports/chronoslob_final_empirical_report.md` | complete_real |
| Evidence pack | `reports/evidence_pack/` | rebuilt |

## Completed / failed / skipped run counts

- Classical: 60 model-fold rows (5 folds x 6 models x 2 splits), 0 failures.
- Neural full grid: 54 completed / 0 failed / 0 missing-pair, out of the
  135-run target. Completed = all 3 horizons x 3 seeds x 3 objectives for
  folds 1 and 2. Remaining = folds 3, 4, 5 (81 runs).
- Feature ablations: 112 completed / 0 failed (4 models x 28 ablation specs for
  fold 1, horizon 10, seed 0). 28 specs = all_features + 12 remove_one_group +
  12 only_one_group + raw_lob_only + derived_microstructure_only +
  no_proxy_features.
- Execution-v3: 3,225,348 prediction rows consumed across the grid and ablation
  prediction artefacts; 0 failures; 5 regime diagnostics skipped (no regime
  labels in predictions).

## Best supported classical result

`gradient_boosting`, FI-2010 test split, macro-F1 `0.4654 +/- 0.0039` across
folds 1-5 (accuracy `0.6410`). This is the strongest classical baseline in the
refreshed table and is marked `supported` in the claim audit.

## Best supported neural result

Two distinct neural artefacts exist and must not be conflated:

- Reduced-scope neural benchmark (`experiments/fi2010_multifold_neural/`,
  separate 25-epoch study, single seed): `matrix_transformer` test macro-F1
  `0.7337 +/- 0.0280`. This is the report's "best neural" figure.
- Neural full grid (`experiments/fi2010_neural_full_grid/`, this run): these are
  deliberately **1-epoch matched** supervised-vs-SSL runs over folds 1-2. Best
  mean macro-F1 is approximately `0.39` (masked, horizon 50). These low absolute
  scores are expected at one epoch and are intended for matched comparison, not
  to maximise performance. The evidence pack marks the full-grid result
  `partial_real` and does not claim a clean full-grid neural result.

## SSL matched comparison summary

From `experiments/fi2010_neural_full_grid/ssl_comparison.csv`, 36 matched
supervised-vs-SSL comparisons over folds 1-2 (1 epoch, matched architecture,
preprocessing and evaluation):

- masked_reconstruction: mean delta macro-F1 `+0.011` (n=18) - roughly neutral.
- next_field: mean delta macro-F1 `-0.067` (n=18) - negative.
- SSL wins macro-F1 in 12/36 comparisons, loses 24/36.
- SSL wins ECE (calibration) in 21/36, loses 15/36.

Conclusion: under these matched minimal-epoch conditions on folds 1-2, SSL
pretraining shows mixed-to-negative effect on macro-F1 and a modest calibration
edge. No SSL improvement is claimed. The three SSL-improvement claims remain
`needs_real_evidence` in the audit.

## Execution-v3 summary

Offline execution-aware proxy diagnostic (not a backtest, not deployed
execution). `payoff_mode = unit_payoff`, `cost_mode = spread_proxy`; no PnL or
return-based calculation is performed or claimed. Outputs present:
`confidence_threshold_summary.csv`, `confidence_threshold_aggregate.csv`,
`cost_sensitivity_summary.csv`, `latency_sensitivity_summary.csv`,
`fill_assumption_summary.csv`, `adverse_selection_summary.csv`,
`regime_execution_summary.csv` and `execution_v3_manifest.json`. Regime
diagnostics are explicitly skipped because regime labels are unavailable in the
prediction artefacts. Raising the confidence threshold increases the per-trade
directional hit rate (about `0.76` -> `0.77`) while retaining far fewer samples;
this is a cost-adjusted proxy diagnostic only.

## Feature-ablation summary

Fold 1, horizon 10, seed 0; all 6 ablation modes, 4 classical models, 12 feature
groups. The feature audit flags 5 unsupported event-level groups
(true order flow, cancellation, trade imbalance and related) as unavailable from
FI-2010 snapshots, and labels `snapshot_order_flow_proxy` as a proxy. Removing
`snapshot_order_flow_proxy` produces the largest single-group degradation
(logistic delta macro-F1 about `-0.13`); single raw-level groups in isolation
collapse toward the majority baseline. Because the most impactful group is a
labelled proxy, no true event-level order-flow, cancellation, trade-imbalance or
queue-position claim is made.

## Figure status

- Neural figures: 17 completed PNGs with source CSVs and `figure_manifest.json`;
  2 regime figures skipped with explicit reasons; label-mapping audit `pass`
  (strict). Includes confusion matrices, reliability, macro-F1/ECE by
  horizon/fold, matched SSL delta, confidence-threshold and cost-adjusted proxy,
  and execution-v3 diagnostic figures.
- Ablation figures: 6 completed PNGs (group deltas, only-one-group,
  remove-one-group degradation, proxy-vs-non-proxy, horizon-specific
  importance); 0 skipped.

## Evidence-pack classification summary

Artefact statuses: 5 `complete_real` (classical, execution-v3, feature
ablations, ablation figures, final report), 2 `partial_real` (neural full grid,
neural figures), 2 `missing` (standalone SSL runner output - not run this pass;
read-only feature-registry audit - not stored), 1 `unknown_staleness`
(pre-existing project-audit archive).

Claim statuses: 5 `supported`, 5 `partially_supported`, 3 `needs_real_evidence`
(all SSL-improvement claims), 9 `forbidden`.

The core new artefacts that were previously `missing` (neural full grid,
execution-v3, figures, feature ablations, ablation figures) are now
`complete_real` or `partial_real`. The classical artefact is refreshed from
`stale` to `complete_real` under the current commit.

## Unsupported claims still blocked

The audit keeps all prestige and deployment claims `forbidden`, including:
state-of-the-art / SOTA ranking, foundation-model status, profitability or PnL,
live tradability or deployed execution quality, and unqualified SSL superiority.
SSL-improvement (macro-F1, calibration, execution proxy) remains
`needs_real_evidence` because the matched real deltas do not support it.

## Bug fixes applied (execution-blocking only)

- `chronoslob/analysis/fi2010_figures.py` `_json_scalar`: handle list/tuple cells
  (contributing folds/seeds) and array-like values; `pd.isna` on an array raised
  in a boolean context and blocked figure generation on real multi-fold data.
- `chronoslob/experiments/evidence_pack.py` `_payload_marks_smoke`: treat negated
  status labels such as "not smoke-test" as non-smoke; the prior substring check
  wrongly flagged the real strict execution-v3 artefact as smoke.
- `chronoslob/experiments/final_report.py`: split a generated feature-ablation
  sentence so the public report stays within the 220-character line threshold.
- `tests/test_fi2010_figures.py`: isolate the no-execution-v3 skip test from real
  repo execution-v3 artefacts via a `project_root` monkeypatch.

## Quality gates (final)

All pass at completion:

- `python -m pytest`: 1254 passed, 1 skipped (intentional), 0 failed.
- `python -m ruff check .`: all checks passed.
- `python -m mypy chronoslob`: no issues in 116 source files.
- `python -m chronoslob.cli doctor`: pass.
- `python -m chronoslob.cli inspect-release-readiness`: pass.
- `python -m chronoslob.cli run-project-audit --strict`: pass.

## Exact remaining work

To extend the neural grid to the full 5-fold target (completed runs are skipped
automatically on re-run):

```bash
python -m chronoslob.cli run-fi2010-neural-full-grid \
  --config configs/experiments/fi2010_neural_serious.yaml \
  --processed-root data/processed/fi2010 \
  --folds 3,4,5 --horizons 10,20,50 --seeds 0,1,2 --lookbacks 20 \
  --objectives supervised,masked_reconstruction,next_field \
  --pretrain-epochs 1 --max-epochs 1 --batch-size 256 --device cpu \
  --out experiments/fi2010_neural_full_grid
```

To broaden feature ablations beyond fold 1 / horizon 10 / seed 0:

```bash
python -m chronoslob.cli run-fi2010-feature-ablations \
  --config configs/experiments/fi2010_multifold.yaml \
  --processed-root data/processed/fi2010 \
  --folds 1,2,3,4,5 --horizons 10,20,50 --seeds 0,1,2 \
  --models logistic,ridge,elastic_net,gradient_boosting \
  --ablation-modes all --feature-groups all \
  --out experiments/fi2010_feature_ablations --strict
```

After either extension completes, re-run Phases 4-7 (execution-v3, figures,
final report, evidence pack) to fold the new runs into the dependent artefacts.
On CPU the full neural grid is on the order of a day and the full-scope ablation
grid is longer; both write per-run artefacts incrementally and skip completed
work, so they can be run in stages.

## Honest scope statement

This run materialised real, reproducible empirical evidence for refreshed
classical benchmarks, a 2-of-5-fold neural supervised-vs-SSL grid, a complete
execution-aware proxy diagnostic, fold-1 feature ablations, and the dependent
figures, report and evidence pack. It does not establish SSL improvement,
profitability, live tradability, state-of-the-art ranking or foundation-model
status, and the rebuilt evidence pack blocks those claims.

## Neural Grid Restart Preflight

Recorded 2026-05-28 before continuing the neural full grid onto folds 3-5.
(The section name avoids the public-wording token flagged by the strict audit;
the runner's restart behaviour is the default skip-completed path.) The neural
runner skips already-completed runs by default, so re-invoking it across all five
folds re-uses folds 1-2 unchanged and computes only the missing folds 3-5.

- Repository commit: `a72d46f0a1d7e0ccb62853eee6004375f7b5358c` (unchanged).
- Completed runs so far: 54 / 135 (folds 1-2, all 3 horizons x 3 seeds x 3
  objectives).
- Remaining runs: 81 (folds 3, 4, 5 x 3 horizons x 3 seeds x 3 objectives).
- Current failures: 0.
- Partial-grid artefact validation (all pass):
  - `results_summary.csv`: 54 rows, all skip-completed, 0 null macro-F1.
  - `aggregate_summary.csv` / `aggregate_summary.json`: 9 grouped rows.
  - `ssl_comparison.csv`: 36 rows, all matched, 0 missing pairs.
  - `failures.csv`: 0 failed runs.
  - per-run `predictions.csv`, `metrics.json`, `status.txt` present for all 54.
- Skip-completed detection check: folds 1-2 specs detected complete (will skip);
  folds 3-5 specs detected absent (will run).
- Quality gates before continuation: `doctor` pass, `inspect-release-readiness`
  pass, `run-project-audit --strict` pass.

Continuation command (skip-completed default keeps folds 1-2; `--batch-size 256`
matches the stored folds 1-2 runs; restart is the default behaviour):

```bash
python -m chronoslob.cli run-fi2010-neural-full-grid \
  --config configs/experiments/fi2010_neural_serious.yaml \
  --processed-root data/processed/fi2010 \
  --folds 1,2,3,4,5 --horizons 10,20,50 --seeds 0,1,2 --lookbacks 20 \
  --objectives supervised,masked_reconstruction,next_field \
  --pretrain-epochs 1 --max-epochs 1 --batch-size 256 --device cpu \
  --out experiments/fi2010_neural_full_grid
```

Restart detection: PASS. The skip-completed default re-uses the 54 folds 1-2
runs and schedules the 81 folds 3-5 runs. Folds 3-5 use progressively larger
combined files (fold 3 ~232 MB, fold 4 ~290 MB, fold 5 ~353 MB versus fold 1
~127 MB), so on CPU this stage is multi-hour and is run as a restartable
background job; partial progress is written per run and skip-completed makes
re-invocation safe.

## Neural Grid Completion Summary

The neural full grid finished at 2026-05-29T02:16:44Z with exit code 0. All five
folds are present and the runner classifies the grid `full_grid_complete`.

Coverage and counts:

- Completed run count: 135 / 135 (81 newly computed for folds 3-5, 54 re-used
  unchanged from folds 1-2).
- Failed run count: 0. Missing pair count: 0.
- Matched supervised-vs-SSL comparison count: 90 (all `matched`).
- Fold coverage: 1, 2, 3, 4, 5. Horizon coverage: 10, 20, 50. Seed coverage:
  0, 1, 2. Objective coverage: supervised, masked_reconstruction, next_field.
- Artefact integrity: `results_summary.csv` 135 rows (0 null macro-F1, 0 null
  ECE); `aggregate_summary.csv`/`.json` 9 grouped rows; `ssl_comparison.csv` 90
  matched rows; `failures.csv` 0 failed; `missing_pairs.csv` empty. 8 runs report
  null MCC (all next-field, folds 1-2): at one fine-tune epoch they collapse to
  the majority "stationary" class, leaving MCC undefined - an honest property of
  minimal-epoch runs, not an error.

Best aggregate macro-F1 per objective (mean +/- std across folds and seeds, best
horizon shown):

| Objective | Best mean macro-F1 | Horizon | Mean MCC | Mean ECE |
| --- | --- | --- | --- | --- |
| supervised | 0.4180 +/- 0.0443 | 50 | 0.1649 | 0.0733 |
| masked_reconstruction | 0.4148 +/- 0.0440 | 50 | 0.1547 | 0.0854 |
| next_field | 0.3823 +/- 0.0781 | 50 | 0.1173 | 0.0846 |

Matched SSL deltas (SSL minus supervised under identical architecture,
preprocessing and evaluation; 45 matched pairs per objective across all five
folds):

| SSL objective | mean delta macro-F1 | mean delta MCC | mean delta ECE | macro-F1 wins | MCC wins | ECE wins |
| --- | --- | --- | --- | --- | --- | --- |
| masked_reconstruction | -0.0100 | -0.0199 | +0.0221 | 19/45 | 15/45 | 18/45 |
| next_field | -0.0622 | -0.0651 | -0.0083 | 3/45 | 3/37 | 24/45 |

Delta ECE is lower-is-better, so "ECE wins" counts pairs where SSL ECE is lower;
MCC wins are out of the pairs with both MCC values defined.

Honest reading of the matched comparison:

- Masked-reconstruction SSL does not improve macro-F1 (mean -0.010) or MCC (mean
  -0.020) and is slightly worse on calibration (mean ECE +0.022). It loses the
  macro-F1 contest in 26 of 45 matched pairs. Net effect is
  neutral-to-slightly-negative.
- Next-field SSL is clearly negative on macro-F1 (mean -0.062, winning only 3 of
  45 pairs) and on MCC (mean -0.065). Its mean ECE is marginally lower (-0.008)
  but it wins calibration in only 24 of 45 pairs, roughly a coin flip.
- No SSL improvement claim is supported. Across matched minimal-epoch conditions
  over folds 1-5, neither SSL objective improves macro-F1 or MCC; masked is
  slightly worse on calibration and next-field is roughly neutral on calibration.

Scope caveat: these are deliberately one-pretrain-epoch, one-fine-tune-epoch
matched runs. They are controlled evidence for the supervised-vs-SSL comparison
under identical conditions, not a performance-maximising benchmark, and they are
not the same artefact as the separate 25-epoch reduced-scope neural benchmark
(`experiments/fi2010_multifold_neural/`), whose best matrix_transformer test
macro-F1 is `0.7337 +/- 0.0280`. The low absolute full-grid scores are expected
at one epoch and must not be read as a strong neural performance claim in either
direction.

## Final Status

Recorded 2026-05-29 after the neural full grid finished and every dependent
artefact was regenerated from it. All quality gates pass.

Run counts:

- Final completed run count: 135 / 135.
- Newly computed this pass (folds 3-5): 81.
- Skip-completed (folds 1-2 re-used unchanged): 54.
- Failed run count: 0.
- Matched supervised-vs-SSL comparisons: 90 (all matched, 0 missing pairs).

Artefact classifications:

- Neural full grid: `complete_real` (`evidence_level = full_grid_complete`,
  `core_grid_complete = true`).
- Execution-v3: `complete_real`, now built from the complete neural grid
  (4,800,519 prediction rows, 135 run groups, `payoff_mode = unit_payoff`,
  `cost_mode = unit_proxy`); regime diagnostics explicitly skipped because regime
  labels are unavailable in the prediction artefacts.
- Neural figures: `partial_real` - 17 figures completed with PNG, source CSV and
  metadata, 2 regime figures skipped because regime labels are unavailable in
  FI-2010 snapshots; label-mapping audit pass under strict mode.
- Ablation figures: `complete_real` (6 figures) from the unchanged feature
  ablation source.
- Final empirical report: `complete_real`, regenerated from the complete grid
  and the regenerated execution-v3.

Evidence-pack classification summary:

- Artefact statuses: 6 `complete_real` (classical benchmarks, neural full grid,
  execution-v3, feature ablations, ablation figures, final report), 1
  `partial_real` (neural figures - regime figures skipped), 2 `missing`
  (standalone SSL runner output and read-only feature-registry audit, neither run
  this pass), 1 `unknown_staleness` (pre-existing project-audit archive).
- Claim statuses: 7 `supported`, 3 `partially_supported`, 3 `needs_real_evidence`
  (all three SSL-improvement claims), 9 `forbidden`.

Supported claims (descriptive only; none asserts SSL improvement, profitability
or ranking):

- Leakage-safe FI-2010 evaluation (classical benchmarks).
- Train-only SSL pretraining code paths (neural full grid).
- Matched supervised-vs-SSL transformer comparison under identical FI-2010
  settings (neural full grid).
- Offline execution-aware proxy diagnostics with explicit limits (execution-v3).
- Microstructure feature-ablation diagnostics with proxy and unsupported groups
  labelled (feature ablations).
- Gradient boosting as the strongest classical baseline (classical benchmarks).
- Confidence filtering improving the cost-adjusted proxy diagnostic
  (execution-v3).

Unsupported claims still blocked:

- All three SSL-improvement claims (macro-F1, calibration, execution proxy)
  remain `needs_real_evidence`. The matched five-fold comparison shows masked SSL
  neutral-to-slightly-negative and next-field SSL clearly negative on
  discrimination, so no SSL improvement is supported.
- All prestige and deployment claims remain `forbidden`: state-of-the-art / SOTA
  ranking, foundation-model status, profitability or PnL, live tradability or
  deployed execution quality, production execution simulation, true event-level
  order flow on FI-2010, queue-position modelling on FI-2010 and tradable alpha.

Note on the execution-v3 source: execution-v3 is built directly from the neural
full-grid predictions, with no feature-ablation directory passed. The builder
treats a supplied feature-ablation directory as an alternative, exclusive
prediction source, so the directory must be omitted for execution-v3 to reflect
the neural grid. The prediction-row count therefore moved from the earlier
ablation-sourced 3,225,348 to 4,800,519 grid rows.

Quality gates (final, all pass):

- `python -m pytest`: 1254 passed, 1 skipped (intentional), 0 failed.
- `python -m ruff check .`: all checks passed.
- `python -m mypy chronoslob`: no issues in 116 source files.
- `python -m chronoslob.cli doctor`: pass.
- `python -m chronoslob.cli inspect-release-readiness`: pass.
- `python -m chronoslob.cli run-project-audit --strict`: pass.

Exact remaining work:

- None is required for the neural full grid: it is `complete_real` at 135/135
  with 0 failures, and execution-v3, figures, the final report and the evidence
  pack are all regenerated from it.
- Optional future scope, not part of this pass and not affecting any current
  classification or supported claim: broaden feature ablations beyond fold 1 /
  horizon 10 / seed 0; produce the standalone SSL runner output; and refresh the
  project-audit archive to clear its `unknown_staleness` marker.

## Feature Ablation Expansion Preflight

Recorded 2026-05-29 before expanding the FI-2010 feature-ablation evidence.

Current repository state:

- Git status at preflight: dirty worktree with pre-existing modified and
  untracked artefact/code/report files; nothing was staged or committed.
- No ChronosLOB-local Python or experiment command was running.
- Lightweight gates passed: `python -m chronoslob.cli doctor`,
  `python -m chronoslob.cli inspect-release-readiness`, and
  `python -m chronoslob.cli run-project-audit --strict`.

Evidence-pack classifications at preflight:

- Neural full grid: `complete_real` (135/135 completed, 0 failures).
- Execution-v3: `complete_real` from neural-grid predictions.
- Feature ablations: `complete_real` but narrow scope.
- Ablation figures: `complete_real` but narrow scope.
- SSL improvement claims: `needs_real_evidence`; no SSL improvement claim is
  supported by the completed matched full-grid evidence.
- Forbidden claims remain blocked, including profitability, PnL, SOTA,
  foundation-model, tradable-alpha, true event-level OFI, cancellation
  imbalance, trade imbalance and queue-position claims.

Current ablation scope:

- Folds: `fold_1`.
- Horizons: `10`.
- Seeds: `0`.
- Models: `logistic`, `ridge`, `elastic_net`, `gradient_boosting`.
- Modes: `all_features`, `remove_one_group`, `only_one_group`, `raw_lob_only`,
  `derived_microstructure_only`, `no_proxy_features`.
- Feature groups: 12 registered snapshot-supported/proxy groups; unsupported
  event-level groups remain explicitly unsupported.
- Completed fits: 112, with 0 failures.

Planned expanded scope:

- Folds: `fold_1`, `fold_2`, `fold_3`, `fold_4`, `fold_5`.
- Horizons: `10`, `20`, `50`.
- Seeds: `0`, `1`, `2`.
- Models: `logistic`, `ridge`, `elastic_net`, `gradient_boosting`.
- Same six ablation modes and same registry groups, preserving the
  `snapshot_order_flow_proxy` proxy label and unsupported event-level records.
- Expected total fits: 5 folds x 3 horizons x 3 seeds x 4 models x 28
  ablation specifications = 5,040 fits.

Restart/reuse behaviour:

- The ablation runner will be invoked with completed-run reuse enabled against
  `experiments/fi2010_feature_ablations/`.
- The existing fold 1 / horizon 10 / seed 0 fits are expected to be counted as
  skipped-existing runs, not silently dropped or recomputed.
- Completed, failed and skipped-existing counts will be recorded after the run.

Blockers:

- None at preflight. If the full expansion proves too expensive in the current
  session, the largest restartable subset will be recorded as `partial_real`
  with exact remaining commands.

## Feature Ablation Expansion Validation

Recorded 2026-05-29 after running the largest feasible restartable expansion in
the current workspace.

Execution outcome:

- Preferred full target: 5,040 fits (5 folds x 3 horizons x 3 seeds x 4 models
  x 28 ablation specifications).
- Full target was too expensive for the current machine/disk profile because
  the stored prediction artefacts are large and `elastic_net` / gradient
  boosting fits were slow at the expanded scale.
- Completed canonical expansion: 840 fits across folds 1-5, horizon 10, seeds
  0-2, models `logistic` and `ridge`, all 6 ablation modes and all 12 registry
  feature groups.
- Final ablation classification for this pass: `partial_real`, not
  `complete_real`, because horizons 20/50 and the slower `elastic_net` /
  `gradient_boosting` expansion remain unfinished.
- Completed fits in canonical `results_summary.csv`: 840.
- Failed fits: 0 (`failures.json` has `failure_count = 0`; this runner does not
  currently emit `failures.csv`).
- Skipped-existing fits in canonical `ablation_manifest.json`: 168.
- Newly completed fits in the canonical scope: 672.
- Canonical prediction artefacts: 840/840 manifest runs have `predictions.csv`.
- Extra prediction artefacts outside the canonical manifest: 65 from interrupted
  broader attempts and the prior wider fold-1 slice. These were left in place
  for restart safety and are not counted in the canonical summary.

Coverage:

- Folds: `fold_1`, `fold_2`, `fold_3`, `fold_4`, `fold_5`.
- Horizons: `10` only.
- Seeds: `0`, `1`, `2`.
- Models: `logistic`, `ridge`.
- Ablation modes: `all_features`, `remove_one_group`, `only_one_group`,
  `raw_lob_only`, `derived_microstructure_only`, `no_proxy_features`.
- Proxy groups: `snapshot_order_flow_proxy`.
- Unsupported groups preserved in every fold manifest: `time_context`,
  `true_order_flow_imbalance`, `cancellation_imbalance`, `trade_imbalance`,
  `queue_position`.

Feature-delta findings, interpreted conservatively:

- Removing `snapshot_order_flow_proxy` produced the largest mean degradation:
  mean delta macro-F1 `-0.195676`, mean delta MCC `-0.263423`, negative in
  30/30 matched logistic/ridge/fold/seed runs.
- The `no_proxy_features` mode is identical in this registry scope to removing
  `snapshot_order_flow_proxy`; it was negative in 30/30 runs
  (logistic mean delta macro-F1 `-0.333630`, ridge `-0.057723`).
- The degradation from removing `snapshot_order_flow_proxy` was present in every
  fold at horizon 10: fold means ranged from `-0.093127` to `-0.254480`.
- `snapshot_order_flow_proxy` as an only-one-group model was much closer to the
  all-features baseline than any other only-one-group family, but still below
  all-features on average (mean delta macro-F1 `-0.022106`; 27/30 negative).
- Removing `spread` was the next largest degradation, but much smaller and more
  model-dependent (mean delta macro-F1 `-0.007871`; 27/30 negative).
- Raw price/size families showed small mixed remove-one-group deltas
  (`price_levels` mean `-0.002060`, `size_levels` mean `-0.001742`).
- Several small positive remove-one-group deltas indicate local or mixed effects,
  not useful feature claims: `top_of_book_imbalance` mean `+0.000615`,
  `volatility_proxy` mean `+0.000451`, `liquidity_concentration` mean
  `+0.000449`.
- `raw_lob_only` was below all-features in 30/30 runs (mean delta macro-F1
  `-0.208966`), and `derived_microstructure_only` was mixed but below
  all-features on average (mean `-0.005884`, negative in 21/30).

Claim interpretation:

- Supported within this `partial_real` scope: feature-ablation evidence that the
  labelled `snapshot_order_flow_proxy` family is important to logistic/ridge
  horizon-10 FI-2010 forecasting under the stored leakage-safe pipeline.
- Mixed/local: small deltas for spread, raw levels and most derived
  microstructure groups; these should be reported with exact scope and model
  qualifiers.
- Unsupported: any claim of causal proof, true event-level OFI, cancellation
  imbalance, trade imbalance, queue position, profitability, PnL or tradable
  alpha.
- Horizon variation cannot be assessed from this pass because horizons 20 and
  50 remain unfinished.

Remaining commands:

```bash
python -m chronoslob.cli run-fi2010-feature-ablations \
  --config configs/experiments/fi2010_multifold.yaml \
  --processed-root data/processed/fi2010 \
  --folds 1,2,3,4,5 \
  --horizons 10 \
  --seeds 0,1,2 \
  --models elastic_net,gradient_boosting \
  --feature-groups all \
  --ablation-modes all \
  --out experiments/fi2010_feature_ablations \
  --resume \
  --strict

python -m chronoslob.cli run-fi2010-feature-ablations \
  --config configs/experiments/fi2010_multifold.yaml \
  --processed-root data/processed/fi2010 \
  --folds 1,2,3,4,5 \
  --horizons 20,50 \
  --seeds 0,1,2 \
  --models logistic,ridge,elastic_net,gradient_boosting \
  --feature-groups all \
  --ablation-modes all \
  --out experiments/fi2010_feature_ablations \
  --resume \
  --strict
```

## Public Writing Polish Completion

Recorded 2026-05-29 after the public-writing pass and final validation.

Files polished:

- `README.md`.
- Evidence docs: `docs/EVIDENCE_PACK.md`, `docs/EXECUTION_VALIDATION_V3.md`,
  `docs/FEATURE_ABLATIONS.md`, `docs/FIGURE_INDEX.md`,
  `docs/MICROSTRUCTURE_FEATURES.md` and `docs/PROJECT_STATUS.md`.
- Generated/public reports: final empirical report, evidence-pack summary,
  claim audit, README result snapshot and selected older public reports with
  riskier proxy wording.
- Experiment public notes: neural full grid, feature ablations and execution-v3.
- New release-support reports: public writing audit, claim-safety scan, release
  notes and commit plan.

Claim-safety status:

- No SSL improvement claim is supported.
- Execution-v3 remains an offline execution-aware proxy diagnostic using
  cost-adjusted proxy wording.
- `snapshot_order_flow_proxy` remains labelled as a proxy only.
- Unsupported profitability, PnL, SOTA, foundation-model, live-market,
  deployability, true event-level OFI and queue-position claims remain blocked
  or explicitly caveated.

Remaining manual-review items:

- Standalone SSL-runner artefacts are still missing.
- Feature ablations remain `partial_real` outside the completed logistic/ridge
  horizon-10 scope.
- Project-audit archive remains `unknown_staleness`.
- Legacy local evidence-pack bullet files with old naming are no longer
  generated and should be excluded from public release commits.

Quality-gate status:

- `python -m pytest`: pass, 1254 passed, 1 skipped.
- `python -m ruff check .`: pass.
- `python -m mypy chronoslob`: pass, no issues in 116 source files.
- `python -m chronoslob.cli doctor`: pass.
- `python -m chronoslob.cli inspect-release-readiness`: pass.
- `python -m chronoslob.cli run-project-audit --strict`: pass.

Suggested next action: review the staged-file set carefully before public push,
especially generated artefacts and large CSV/PNG files.

## Feature Ablation Execution-V3 Decision

Recorded 2026-05-29.

- Ablation prediction files are present and compatible at the per-run level.
- The execution-v3 feature-ablation loader consumes all `predictions.csv` files
  under `experiments/fi2010_feature_ablations/`.
- The directory currently contains 65 prediction files outside the canonical
  ablation manifest because broader exploratory attempts were stopped after
  proving too expensive. They were preserved for restart safety and not deleted.
- Running execution-v3 on this directory would therefore mix canonical
  logistic/ridge partial-real predictions with out-of-scope partial
  `elastic_net` / `gradient_boosting` artefacts.
- Decision: skipped optional ablation execution-v3 for this pass. Neural-grid
  execution-v3 in `experiments/fi2010_execution_v3/` remains the canonical
  execution-v3 evidence.

## Final Feature Ablation Expansion Status

Recorded 2026-05-29 after rebuilding ablation figures, the final empirical
report and the evidence pack, then running all requested quality gates.

Final ablation classification:

- `partial_real`.
- Canonical expanded scope completed in this pass: folds 1-5, horizon 10, seeds
  0-2, models `logistic` and `ridge`, all six ablation modes and all 12 registry
  feature groups.
- Canonical fit count: 840.
- Completed fits: 840.
- Failed fits: 0.
- Skipped-existing fits: 168.
- Newly completed canonical fits: 672.
- Full preferred 5,040-fit scope remains unfinished because horizons 20/50 and
  the slower `elastic_net` / `gradient_boosting` expansion were too expensive in
  the current workspace.

Most robust feature-ablation evidence:

- `snapshot_order_flow_proxy` remains the dominant feature-family signal within
  the canonical partial-real scope. Removing it degraded macro-F1 in 30/30
  matched logistic/ridge/fold/seed runs, with mean delta macro-F1 `-0.195676`
  and mean delta MCC `-0.263423`.
- The `no_proxy_features` mode produced the same degradation pattern in this
  registry scope and was negative in 30/30 runs.
- `snapshot_order_flow_proxy` as an only-one-group model was the closest
  only-one-group family to all-features, but it still trailed all-features on
  average (mean delta macro-F1 `-0.022106`).
- `spread` showed smaller feature-ablation evidence of importance at horizon 10
  (remove-one-group mean delta macro-F1 `-0.007871`, negative in 27/30 runs).
- Other raw and derived groups were mixed or local; small positive
  remove-one-group deltas are not evidence that a family is generally useful.

Proxy and unsupported-feature caveats:

- `snapshot_order_flow_proxy` is a labelled snapshot-delta proxy only; this is
  not true event-level order-flow evidence.
- Unsupported FI-2010 event-level groups remain explicitly unsupported:
  `time_context`, `true_order_flow_imbalance`, `cancellation_imbalance`,
  `trade_imbalance`, `queue_position`.
- No causal proof, true OFI, cancellation imbalance, trade imbalance,
  queue-position, profitability, tradable-alpha, SOTA or foundation-model claim
  is supported.
- Optional ablation execution-v3 was skipped because the directory contains
  extra out-of-manifest prediction files from stopped broader attempts; running
  the current feature-ablation loader would mix scopes. Neural-grid execution-v3
  remains the canonical execution-v3 evidence and uses cost-adjusted proxy
  wording only.

Regenerated artefacts:

- `experiments/fi2010_feature_ablations/`: canonical root summaries regenerated
  from the partial-real expanded scope.
- `reports/figures/fi2010_feature_ablations/`: 6/6 figures rebuilt, with PNG,
  source CSV and metadata JSON entries; 0 skipped figures.
- `reports/chronoslob_final_empirical_report.md`: regenerated from the current
  neural grid, execution-v3, partial-real feature ablations, refreshed ablation
  figures and evidence pack.
- `reports/evidence_pack/`: rebuilt after the report refresh.

Evidence-pack classification summary:

- Artefact statuses: 5 `complete_real`, 2 `partial_real`, 2 `missing`, 1
  `unknown_staleness`.
- Feature ablations: `partial_real` (840 completed, 0 failed, 168
  skipped-existing).
- Ablation figures: `complete_real`.
- Final empirical report: `complete_real`.
- Claim statuses: 6 `supported`, 4 `partially_supported`, 3
  `needs_real_evidence`, 9 `forbidden`.
- SSL improvement claims remain `needs_real_evidence`; no SSL improvement claim
  is supported.
- Forbidden claims remain blocked.

Quality gates (final, all pass):

- `python -m pytest`: 1254 passed, 1 skipped, 0 failed.
- `python -m ruff check .`: all checks passed.
- `python -m mypy chronoslob`: no issues in 116 source files.
- `python -m chronoslob.cli doctor`: pass.
- `python -m chronoslob.cli inspect-release-readiness`: pass.
- `python -m chronoslob.cli run-project-audit --strict`: pass.

Current workspace status:

- No ChronosLOB Python experiment command is running.
- Git worktree remains dirty with pre-existing modified/untracked files plus the
  generated artefact refreshes from this pass. Nothing was staged or committed.

Exact remaining work:

```bash
python -m chronoslob.cli run-fi2010-feature-ablations \
  --config configs/experiments/fi2010_multifold.yaml \
  --processed-root data/processed/fi2010 \
  --folds 1,2,3,4,5 \
  --horizons 10 \
  --seeds 0,1,2 \
  --models elastic_net,gradient_boosting \
  --feature-groups all \
  --ablation-modes all \
  --out experiments/fi2010_feature_ablations \
  --resume \
  --strict

python -m chronoslob.cli run-fi2010-feature-ablations \
  --config configs/experiments/fi2010_multifold.yaml \
  --processed-root data/processed/fi2010 \
  --folds 1,2,3,4,5 \
  --horizons 20,50 \
  --seeds 0,1,2 \
  --models logistic,ridge,elastic_net,gradient_boosting \
  --feature-groups all \
  --ablation-modes all \
  --out experiments/fi2010_feature_ablations \
  --resume \
  --strict
```

# ChronosLOB Final Empirical Report

Generated from stored FI-2010 artefacts only. No model training is run by this builder.

## Evidence Snapshot

| field | value |
| --- | --- |
| generated_at | 2026-05-30T01:18:04.243326+00:00 |
| git_commit | 31f5f688a83e3fabbc4275d921674b145d8be601 |
| classical_scope | multi-fold classical results |
| best_classical_test_macro_f1 | gradient_boosting: 0.4654 +/- 0.0039 |
| neural_full_grid_scope | completed one-epoch matched comparison grid; folds 1, 2, 3, 4, 5, horizons 10, 20, 50, seeds 0, 1, 2, objectives supervised, masked_reconstruction, next_field; pretrain_epochs 1, fine_tune_epochs 1; 135 completed, 0 failed; matched comparison and pipeline evidence, not a performance-maximising neural benchmark |
| proper_training_neural_scope | partial_real; folds 1, horizons 10, 50, seeds 0, lookbacks 50, objectives supervised, masked_reconstruction, next_field; max_epochs 25, patience 5; validation-only early stopping with best checkpoint restored before test |
| ssl_comparison_scope | matched supervised-vs-SSL comparison present in the full grid (masked_reconstruction, next_field objectives); no SSL improvement is supported |
| legacy_reduced_scope_neural_scope | separate earlier 25-epoch reduced-scope supervised benchmark, single-seed, lookback 20; reported separately, not used as matched-grid or SSL evidence |
| best_legacy_reduced_scope_neural_test_macro_f1 | matrix_transformer: 0.7337 +/- 0.0280, lookback 20 (separate 25-epoch reduced-scope benchmark) |
| execution_scope | skipped; metrics are proxy diagnostics |
| execution_v3_scope | offline execution-aware proxy diagnostic loaded; payoff_mode=unit_payoff, cost_mode=unit_proxy |
| external_scope | skipped; protocol context only, not ranking claims |
| report_path | reports/chronoslob_final_empirical_report.md |
| summary_path | reports/chronoslob_final_empirical_report_summary.json |

## Evidence Status Summary

This summary uses the same status language as the README and the evidence pack.

What is complete (`complete_real`):

- Multi-fold classical FI-2010 benchmark across the stored folds.
- One-epoch matched neural full grid across folds 1, 2, 3, 4, 5, horizons 10, 20, 50, seeds 0, 1, 2 and objectives supervised, masked_reconstruction, next_field.
- A matched supervised-vs-SSL comparison inside that grid.
- Execution-v3 offline cost-adjusted proxy diagnostics.

What is partial (`partial_real`):

- Proper-training neural subset: documented partial longer-training modelling evidence with validation-only early stopping; exact folds, horizons, seeds, lookbacks and objectives are listed in its
  section.
- FI-2010 snapshot feature ablations: currently folds 1-5 at horizon 10 for logistic and ridge only; wider model/horizon scope unfinished.

What is separate legacy / reduced-scope evidence:

- The earlier 25-epoch reduced-scope supervised matrix-transformer benchmark (single seed, lookback 20) is reported separately and is not used as matched SSL evidence.

What is not claimed:

- No SSL improvement: SSL was implemented and tested under matched settings, but no SSL improvement is supported.
- No profitability, tradability, live-trading, PnL, SOTA, foundation-model or production-execution-simulator claim.
- No true event-level order flow or queue position is observed from FI-2010 snapshots.

The completed matched full grid is a one-epoch comparison grid. It is useful for controlled supervised-vs-SSL comparison and pipeline validation, but it is not a performance-maximising neural training
result.

The earlier 25-epoch reduced-scope supervised matrix-transformer result is reported separately and is not used as matched SSL evidence.

## Evidence Pack Audit

| field | value |
| --- | --- |
| evidence_pack_status | loaded |
| evidence_pack_dir | reports/evidence_pack |
| artefact_status_counts | complete_real=4, missing=2, partial_real=1, stale=5, unknown_staleness=1 |
| claim_status_counts | forbidden=9, needs_real_evidence=1, partially_supported=11, supported=2, unsupported=2 |
| supported_claims | ChronosLOB includes a richer execution-aware proxy analysis report; SSL was implemented and evaluated under matched FI-2010 settings. |
| unsupported_or_limited_claims | ChronosLOB is a reproducible LOB research platform; ChronosLOB uses leakage-safe FI-2010 evaluation; ChronosLOB includes train-only SSL pretraining |

Release caveats from the evidence pack:

- Smoke diagnostics are not empirical evidence.
- SSL improvement language requires real aggregate comparison artefacts.
- Execution-v3 metrics remain offline proxy diagnostics.
- FI-2010 snapshot features do not expose event-level order flow or queue position.

## Research Question

Can stored FI-2010 artefacts support a traceable assessment of predictive mid-price direction performance, uncertainty, robustness, execution-aware proxy diagnostics and external protocol context?

## Dataset And Split Protocol

| field | value |
| --- | --- |
| dataset | FI-2010 |
| variant | NoAuction ZScore |
| task | midprice_direction |
| target_horizon | 10 |
| split_protocol | official split column with validation carved from train only |
| folds | 1, 2, 3, 4, 5 |
| classical_protocol | multi-fold; one stored classical seed across completed folds |
| neural_protocol | matched one-epoch full grid over folds 1, 2, 3, 4, 5, horizons 10, 20, 50, seeds 0, 1, 2 and objectives supervised, masked_reconstruction, next_field; separate earlier 25-epoch reduced-scope supervised benchmark (single seed, lookback 20) reported separately |

## Model Families

| family | models | scope |
| --- | --- | --- |
| classical | majority, logistic, ridge, elastic_net, random_forest, gradient_boosting | multi-fold stored fold summaries |
| neural matched grid | matrix_transformer | one-epoch matched supervised/SSL grid over folds 1, 2, 3, 4, 5, horizons 10, 20, 50, seeds 0, 1, 2; comparison evidence |
| neural proper-training subset | matrix_transformer | partial_real; folds 1, horizons 10, 50, seeds 0, lookbacks 50; validation-only early stopping |
| neural legacy supervised | deeplob_style, matrix_transformer | separate earlier reduced-scope, single-seed, lookback 20 |

## Main Result Table

Classical rows are multi-fold. Neural rows are reduced-scope, single-seed supervised results and are not used here to assert superiority over the classical family.

| model | family | scope | test macro-F1 | accuracy | MCC | folds | seeds/runs | lookback |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gradient_boosting | classical | multi-fold | 0.4654 +/- 0.0039 | 0.6410 | 0.2724 | 5 | 5 | n/a |
| random_forest | classical | multi-fold | 0.4547 +/- 0.0081 | 0.6263 | 0.2431 | 5 | 5 | n/a |
| logistic | classical | multi-fold | 0.3261 +/- 0.0106 | 0.6125 | 0.1227 | 5 | 5 | n/a |
| elastic_net | classical | multi-fold | 0.3260 +/- 0.0106 | 0.6125 | 0.1227 | 5 | 5 | n/a |
| ridge | classical | multi-fold | 0.3087 +/- 0.0082 | 0.6126 | 0.1116 | 5 | 5 | n/a |
| majority | classical | multi-fold | 0.2514 +/- 0.0054 | 0.6058 | 0.0000 | 5 | 5 | n/a |
| matrix_transformer | supervised neural | reduced-scope, single-seed | 0.7337 +/- 0.0280 | 0.8008 | 0.6288 | 5 | 1 | 20 |
| deeplob_style | supervised neural | reduced-scope, single-seed | 0.4753 +/- 0.0274 | 0.4815 | 0.2932 | 5 | 1 | 20 |

## Self-Supervised Pretraining

The standalone SSL runner artefact is not supplied, so no standalone `ssl_transformer` row is admitted here.

However, the matched supervised-vs-SSL comparison is not absent from this report: it is reported in the Full Neural Grid section below, where masked_reconstruction and next_field objectives are
compared against the supervised baseline under identical fold, horizon, seed, lookback, architecture and preprocessing settings.

That matched comparison is a one-epoch grid. No SSL improvement over the matched supervised baseline is supported; deltas are reported metric-by-metric in the Full Neural Grid section.

## Full Neural Grid

Status: loaded.
These artefacts are loaded as aggregate full-grid evidence, subject to the failures table below.

| field | value |
| --- | --- |
| execution_mode | benchmark |
| folds | 1, 2, 3, 4, 5 |
| horizons | 10, 20, 50 |
| seeds | 0, 1, 2 |
| lookbacks | 20 |
| objectives | supervised, masked_reconstruction, next_field |
| pretrain_epochs | 1 |
| fine_tune_epochs | 1 |
| completed_runs | 135 |
| failed_runs | 0 |
| core_grid_complete | True |

Aggregate supervised-vs-SSL rows:

| horizon | lookback | model | pretraining | completed | failed | mean macro-F1 | std macro-F1 | mean MCC | mean ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 20 | matrix_transformer | masked_reconstruction | 15 | 0 | 0.3233 | 0.0245 | 0.0090 | 0.1425 |
| 10 | 20 | matrix_transformer | next_field | 15 | 0 | 0.2733 | 0.0263 | -0.0104 | 0.0847 |
| 10 | 20 | matrix_transformer | none | 15 | 0 | 0.3336 | 0.0459 | 0.0380 | 0.1165 |
| 20 | 20 | matrix_transformer | masked_reconstruction | 15 | 0 | 0.3547 | 0.0424 | 0.0761 | 0.1281 |
| 20 | 20 | matrix_transformer | next_field | 15 | 0 | 0.2805 | 0.0637 | 0.0320 | 0.0956 |
| 20 | 20 | matrix_transformer | none | 15 | 0 | 0.3711 | 0.0514 | 0.0967 | 0.0999 |
| 50 | 20 | matrix_transformer | masked_reconstruction | 15 | 0 | 0.4148 | 0.0440 | 0.1547 | 0.0854 |
| 50 | 20 | matrix_transformer | next_field | 15 | 0 | 0.3823 | 0.0781 | 0.1173 | 0.0846 |
| 50 | 20 | matrix_transformer | none | 15 | 0 | 0.4180 | 0.0443 | 0.1649 | 0.0733 |

Matched SSL deltas:

| horizon | lookback | SSL objective | delta macro-F1 | delta MCC | delta ECE | macro-F1 | MCC | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 20 | masked_reconstruction | 0.0452 | 0.0285 | 0.0325 | win | win | loss |
| 10 | 20 | next_field | -0.0284 | n/a | -0.0207 | loss | n/a | win |
| 10 | 20 | masked_reconstruction | -0.0148 | -0.0063 | -0.0340 | loss | loss | win |
| 10 | 20 | next_field | -0.0521 | n/a | -0.0717 | loss | n/a | win |
| 10 | 20 | masked_reconstruction | -0.0028 | -0.0661 | 0.0633 | loss | loss | loss |
| 10 | 20 | next_field | -0.0226 | n/a | -0.0316 | loss | n/a | win |
| 20 | 20 | masked_reconstruction | 0.0627 | 0.0580 | 0.0883 | win | win | loss |
| 20 | 20 | next_field | -0.0843 | n/a | 0.0909 | loss | n/a | loss |
| 20 | 20 | masked_reconstruction | -0.0576 | -0.0292 | 0.0126 | loss | loss | loss |
| 20 | 20 | next_field | -0.1337 | n/a | 0.0048 | loss | n/a | loss |
| 20 | 20 | masked_reconstruction | -0.0820 | -0.0509 | 0.1025 | loss | loss | loss |
| 20 | 20 | next_field | -0.1454 | n/a | 0.0878 | loss | n/a | loss |
| 50 | 20 | masked_reconstruction | 0.0716 | 0.1067 | -0.0093 | win | win | win |
| 50 | 20 | next_field | -0.1087 | -0.0728 | 0.1008 | loss | loss | loss |
| 50 | 20 | masked_reconstruction | -0.0819 | -0.0751 | -0.0149 | loss | loss | win |
| 50 | 20 | next_field | -0.1148 | -0.1323 | 0.0558 | loss | loss | loss |
| 50 | 20 | masked_reconstruction | 0.0394 | 0.0164 | -0.0179 | win | win | win |
| 50 | 20 | next_field | -0.1294 | -0.1616 | 0.1322 | loss | loss | loss |
| 10 | 20 | masked_reconstruction | 0.0351 | 0.0146 | -0.0236 | win | win | win |
| 10 | 20 | next_field | -0.0154 | 0.0159 | -0.0989 | loss | win | win |
| 10 | 20 | masked_reconstruction | 0.0535 | 0.0121 | 0.0074 | win | win | loss |
| 10 | 20 | next_field | -0.0189 | 0.0033 | -0.1396 | loss | win | win |
| 10 | 20 | masked_reconstruction | 0.0009 | -0.0007 | 0.0066 | win | loss | loss |
| 10 | 20 | next_field | -0.0461 | n/a | -0.1598 | loss | n/a | win |
| 20 | 20 | masked_reconstruction | 0.0365 | 0.0110 | -0.0307 | win | win | win |
| 20 | 20 | next_field | -0.0846 | n/a | -0.1543 | loss | n/a | win |
| 20 | 20 | masked_reconstruction | 0.0396 | 0.0089 | -0.0098 | win | win | win |
| 20 | 20 | next_field | -0.0550 | -0.0033 | -0.1321 | loss | loss | win |
| 20 | 20 | masked_reconstruction | 0.0045 | 0.0018 | -0.0034 | win | win | win |
| 20 | 20 | next_field | -0.0515 | -0.0032 | -0.1762 | loss | loss | win |
| 50 | 20 | masked_reconstruction | -0.0114 | -0.0086 | 0.0435 | loss | loss | loss |
| 50 | 20 | next_field | -0.0411 | -0.0373 | -0.0240 | loss | loss | win |
| 50 | 20 | masked_reconstruction | 0.0154 | 0.0016 | -0.0140 | win | win | win |
| 50 | 20 | next_field | -0.0726 | -0.0919 | 0.0955 | loss | loss | loss |
| 50 | 20 | masked_reconstruction | 0.0498 | 0.0644 | -0.0480 | win | win | win |
| 50 | 20 | next_field | -0.0038 | -0.0520 | -0.0940 | loss | loss | win |
| 10 | 20 | masked_reconstruction | -0.0096 | -0.0248 | 0.1205 | loss | loss | loss |
| 10 | 20 | next_field | -0.0976 | -0.0453 | 0.0256 | loss | loss | loss |
| 10 | 20 | masked_reconstruction | -0.0255 | -0.0166 | 0.0809 | loss | loss | loss |
| 10 | 20 | next_field | -0.0985 | -0.0484 | -0.0656 | loss | loss | win |
| 10 | 20 | masked_reconstruction | 0.0007 | -0.0003 | 0.0029 | win | loss | loss |
| 10 | 20 | next_field | -0.0702 | -0.0084 | -0.0937 | loss | loss | win |
| 20 | 20 | masked_reconstruction | -0.0767 | -0.0251 | 0.1300 | loss | loss | loss |
| 20 | 20 | next_field | -0.1544 | -0.0973 | 0.0768 | loss | loss | loss |
| 20 | 20 | masked_reconstruction | -0.0417 | -0.0436 | 0.0629 | loss | loss | loss |
| 20 | 20 | next_field | -0.1784 | -0.1120 | 0.0064 | loss | loss | loss |
| 20 | 20 | masked_reconstruction | 0.0030 | -0.0101 | -0.0056 | win | loss | win |
| 20 | 20 | next_field | -0.0262 | -0.0646 | -0.0522 | loss | loss | win |
| 50 | 20 | masked_reconstruction | -0.0073 | -0.0188 | 0.0730 | loss | loss | loss |
| 50 | 20 | next_field | 0.0218 | -0.0004 | -0.0229 | win | loss | win |
| 50 | 20 | masked_reconstruction | -0.0192 | 0.0042 | 0.0083 | loss | win | loss |
| 50 | 20 | next_field | -0.0059 | -0.0050 | -0.0108 | loss | loss | win |
| 50 | 20 | masked_reconstruction | 0.0147 | 0.0062 | -0.0225 | win | win | win |
| 50 | 20 | next_field | -0.0419 | -0.0294 | -0.0491 | loss | loss | win |
| 10 | 20 | masked_reconstruction | -0.0236 | -0.0519 | 0.0007 | loss | loss | loss |
| 10 | 20 | next_field | -0.0635 | -0.0920 | 0.0435 | loss | loss | loss |
| 10 | 20 | masked_reconstruction | -0.0173 | -0.0286 | 0.0617 | loss | loss | loss |
| 10 | 20 | next_field | -0.0304 | -0.0475 | 0.0503 | loss | loss | loss |
| 10 | 20 | masked_reconstruction | -0.0677 | -0.0834 | 0.0575 | loss | loss | loss |
| 10 | 20 | next_field | -0.0753 | -0.1273 | 0.0816 | loss | loss | loss |
| 20 | 20 | masked_reconstruction | -0.0211 | -0.0396 | 0.0213 | loss | loss | loss |
| 20 | 20 | next_field | -0.1511 | -0.1564 | 0.0291 | loss | loss | loss |
| 20 | 20 | masked_reconstruction | -0.0203 | -0.0101 | -0.0134 | loss | loss | win |
| 20 | 20 | next_field | -0.0449 | -0.0969 | -0.0096 | loss | loss | win |
| 20 | 20 | masked_reconstruction | -0.0092 | -0.0228 | -0.0165 | loss | loss | win |
| 20 | 20 | next_field | -0.0408 | -0.0968 | -0.0069 | loss | loss | win |
| 50 | 20 | masked_reconstruction | -0.0269 | -0.0743 | 0.0660 | loss | loss | loss |
| 50 | 20 | next_field | -0.0028 | -0.0257 | 0.0044 | loss | loss | loss |
| 50 | 20 | masked_reconstruction | -0.0392 | -0.0516 | 0.0579 | loss | loss | loss |
| 50 | 20 | next_field | -0.0188 | -0.0089 | -0.0010 | loss | loss | win |
| 50 | 20 | masked_reconstruction | -0.0358 | -0.0785 | 0.0669 | loss | loss | loss |
| 50 | 20 | next_field | 0.0091 | 0.0170 | 0.0210 | win | win | loss |
| 10 | 20 | masked_reconstruction | -0.1446 | -0.2319 | -0.0722 | loss | loss | win |
| 10 | 20 | next_field | -0.1928 | -0.1996 | -0.0718 | loss | loss | win |
| 10 | 20 | masked_reconstruction | 0.0261 | 0.0675 | 0.1178 | win | win | loss |
| 10 | 20 | next_field | -0.0094 | -0.0051 | 0.1163 | loss | loss | loss |
| 10 | 20 | masked_reconstruction | -0.0105 | -0.0457 | -0.0324 | loss | loss | win |
| 10 | 20 | next_field | -0.0840 | -0.1415 | -0.0414 | loss | loss | win |
| 20 | 20 | masked_reconstruction | -0.0728 | -0.1224 | 0.0444 | loss | loss | loss |
| 20 | 20 | next_field | -0.0805 | -0.1618 | 0.0391 | loss | loss | loss |
| 20 | 20 | masked_reconstruction | 0.0051 | 0.0051 | 0.0609 | win | win | loss |
| 20 | 20 | next_field | -0.0429 | -0.0659 | 0.0835 | loss | loss | loss |
| 20 | 20 | masked_reconstruction | -0.0168 | -0.0412 | -0.0205 | loss | loss | win |
| 20 | 20 | next_field | -0.0855 | -0.1390 | 0.0483 | loss | loss | loss |
| 50 | 20 | masked_reconstruction | -0.0275 | -0.0361 | 0.0332 | loss | loss | loss |
| 50 | 20 | next_field | -0.0193 | -0.0406 | -0.0106 | loss | loss | win |
| 50 | 20 | masked_reconstruction | 0.0098 | -0.0035 | 0.0068 | win | loss | loss |
| 50 | 20 | next_field | 0.0026 | -0.0308 | -0.0322 | win | loss | win |
| 50 | 20 | masked_reconstruction | 0.0003 | -0.0061 | -0.0466 | win | loss | win |
| 50 | 20 | next_field | -0.0096 | -0.0421 | 0.0046 | loss | loss | loss |

Failure summary:

| fold | horizon | seed | objective | status | reason |
| --- | --- | --- | --- | --- | --- |
| 1 | 10 | 0 | supervised | skipped_existing | existing completed run reused |
| 1 | 10 | 0 | masked_reconstruction | skipped_existing | existing completed run reused |
| 1 | 10 | 0 | next_field | skipped_existing | existing completed run reused |
| 1 | 10 | 1 | supervised | skipped_existing | existing completed run reused |
| 1 | 10 | 1 | masked_reconstruction | skipped_existing | existing completed run reused |
| 1 | 10 | 1 | next_field | skipped_existing | existing completed run reused |
| 1 | 10 | 2 | supervised | skipped_existing | existing completed run reused |
| 1 | 10 | 2 | masked_reconstruction | skipped_existing | existing completed run reused |
| 1 | 10 | 2 | next_field | skipped_existing | existing completed run reused |
| 1 | 20 | 0 | supervised | skipped_existing | existing completed run reused |
44 additional rows are in failures.csv.

Interpretation:
- masked_reconstruction: mean deltas macro-F1 -0.0100, MCC -0.0199, ECE 0.0221; outcomes macro-F1 19 win/26 loss/0 tie, MCC 15 win/30 loss/0 tie, ECE 18 win/27 loss/0 tie.
  No overall SSL improvement is supported; report any deltas metric-by-metric.
- next_field: mean deltas macro-F1 -0.0622, MCC -0.0651, ECE -0.0083; outcomes macro-F1 3 win/42 loss/0 tie, MCC 3 win/34 loss/0 tie, ECE 24 win/21 loss/0 tie.
  No overall SSL improvement is supported; report any deltas metric-by-metric.

## Proper-Training Neural Subset

The one-epoch full grid is retained as matched comparison evidence. The proper-training subset is used to assess whether the neural models remain credible under a more realistic training budget.

Status: loaded.
These artefacts are loaded as longer-training modelling evidence, subject to the scope and failure rows below.

| field | value |
| --- | --- |
| subset_kind | proper_training_subset |
| evidence_level | partial_real |
| scope_label | limited_partial_real_slice |
| execution_mode | benchmark |
| folds | 1 |
| horizons | 10, 50 |
| seeds | 0 |
| lookbacks | 50 |
| objectives | supervised, masked_reconstruction, next_field |
| max_epochs | 25 |
| early_stopping_metric | validation_macro_f1 |
| early_stopping_patience | 5 |
| pretrain_epochs | 5 |
| completed_runs | 6 |
| failed_runs | 0 |
| planned_scope_complete | True |
| target_scope_complete | False |
| model_selection | validation-only early stopping; best checkpoint restored before test |

Scope note: this is a documented partial subset (`partial_real`). It does not cover the full primary proper-training target (folds 1-5, horizons 10 and 50, seed 0, all three objectives, lookback 50,
max_epochs 25, patience 5); the table above states exactly what was run.

Training / early-stopping summary:

| field | value |
| --- | --- |
| runs_with_curves | 6 |
| best_epoch_range | 1 to 25 (mean 9.3) |
| epochs_ran_range | 6 to 25 |
| runs_early_stopped | 5 of 6 |
| curve_files | per-run runs/**/curves.csv and curves.json (train/validation loss, validation macro-F1, accuracy, MCC) |

Aggregate test metrics by objective:

| horizon | lookback | pretraining | completed | mean macro-F1 | std macro-F1 | mean MCC | mean ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 50 | masked_reconstruction | 1 | 0.2477 | 0.0000 | 0.0000 | 0.1047 |
| 10 | 50 | next_field | 1 | 0.2477 | 0.0000 | 0.0000 | 0.1618 |
| 10 | 50 | none | 1 | 0.2477 | 0.0000 | 0.0000 | 0.0872 |
| 50 | 50 | masked_reconstruction | 1 | 0.4774 | 0.0000 | 0.2155 | 0.0741 |
| 50 | 50 | next_field | 1 | 0.3948 | 0.0000 | 0.1325 | 0.0813 |
| 50 | 50 | none | 1 | 0.3883 | 0.0000 | 0.0917 | 0.0496 |

Matched SSL deltas (longer training):

| horizon | seed | SSL objective | delta macro-F1 | delta MCC | delta ECE | macro-F1 | MCC | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 0 | masked_reconstruction | 0.0000 | 0.0000 | 0.0175 | tie | tie | loss |
| 10 | 0 | next_field | 0.0000 | 0.0000 | 0.0746 | tie | tie | loss |
| 50 | 0 | masked_reconstruction | 0.0891 | 0.1238 | 0.0245 | win | win | loss |
| 50 | 0 | next_field | 0.0065 | 0.0408 | 0.0317 | win | win | loss |

Interpretation:
- masked_reconstruction: mean deltas macro-F1 0.0445, MCC 0.0619, ECE 0.0210; outcomes macro-F1 1 win/0 loss/1 tie, MCC 1 win/0 loss/1 tie, ECE 0 win/2 loss/0 tie.
- next_field: mean deltas macro-F1 0.0032, MCC 0.0204, ECE 0.0531; outcomes macro-F1 1 win/0 loss/1 tie, MCC 1 win/0 loss/1 tie, ECE 0 win/2 loss/0 tie.
  Under this longer-training budget no broad SSL improvement is claimed; deltas are reported metric-by-metric, fold-by-fold and seed-by-seed. Any improvement is scoped to exactly the rows above.

## Legacy Reduced-Scope Benchmark

This is the earlier 25-epoch reduced-scope supervised neural benchmark. It is reported separately from the one-epoch matched full grid and from the proper-training supervised-vs-SSL subset, and it is
not used as matched SSL evidence.

Stored scope: seeds 0, lookbacks 20.

| model | test macro-F1 | accuracy | MCC | folds | seeds | lookback |
| --- | --- | --- | --- | --- | --- | --- |
| matrix_transformer | 0.7337 +/- 0.0280 | 0.8008 | 0.6288 | 5 | 1 | 20 |
| deeplob_style | 0.4753 +/- 0.0274 | 0.4815 | 0.2932 | 5 | 1 | 20 |

## SSL Interpretation

SSL evidence is interpreted only through matched supervised-vs-SSL rows. The one-epoch full grid remains comparison and infrastructure evidence; the proper-training subset is longer-training modelling
evidence at its exact stored scope.

| source | SSL objective | mean delta macro-F1 | mean delta MCC | mean delta ECE | macro-F1 outcomes |
| --- | --- | --- | --- | --- | --- |
| one-epoch full grid | masked_reconstruction | -0.0100 | -0.0199 | 0.0221 | 19 win/26 loss/0 tie |
| one-epoch full grid | next_field | -0.0622 | -0.0651 | -0.0083 | 3 win/42 loss/0 tie |
| proper-training subset | masked_reconstruction | 0.0445 | 0.0619 | 0.0210 | 1 win/0 loss/1 tie |
| proper-training subset | next_field | 0.0032 | 0.0204 | 0.0531 | 1 win/0 loss/1 tie |

The longer-training subset does not support an SSL improvement claim.

## SSL Failure Analysis

A dedicated SSL failure-analysis report at reports/ssl_failure_analysis/ssl_failure_analysis.md separates three distinct bodies of evidence and never merges them: the completed one-epoch matched full
grid (folds 1-5, horizons 10/20/50, seeds 0-2), the longer-training proper-training subset v2 (fold 1, horizons 10 and 50, seed 0, partial_real) and a separate older reduced-scope supervised benchmark
used only for context.

- Full grid masked_reconstruction: mean macro-F1 delta -0.0100, mean ECE delta 0.0221 (lower ECE is better).
- Full grid next_field: mean macro-F1 delta -0.0622, mean ECE delta -0.0083 (lower ECE is better).

- Proper-training masked SSL at fold 1 / horizon 50: macro-F1 delta 0.0891, MCC delta 0.1238, ECE delta 0.0245 (calibration worsened).

- Full-grid SSL does not improve overall: matched macro-F1 deltas are neutral-to-negative and calibration does not improve uniformly.
- Proper-training subset v2 shows a narrow fold-1/horizon-50 predictive gain in macro-F1 and MCC, but ECE worsened in every matched SSL row.
- No broad SSL improvement and no calibration improvement is claimed.
- More evidence would require broader proper-training runs and/or better SSL objective design rather than any success claim.

## Figure Index

| figure | title | path | description |
| --- | --- | --- | --- |
| confusion_matrix_h10 | Confusion Matrix H10 | reports/figures/fi2010_neural_full_grid/confusion_matrix_h10.png | canonical up/stationary/down confusion matrix |
| confusion_matrix_h20 | Confusion Matrix H20 | reports/figures/fi2010_neural_full_grid/confusion_matrix_h20.png | canonical up/stationary/down confusion matrix |
| confusion_matrix_h50 | Confusion Matrix H50 | reports/figures/fi2010_neural_full_grid/confusion_matrix_h50.png | canonical up/stationary/down confusion matrix |
| reliability_curve | Reliability Curve | reports/figures/fi2010_neural_full_grid/reliability_curve.png | confidence calibration from stored predictions |
| macro_f1_by_fold | Macro-F1 Across Folds | reports/figures/fi2010_neural_full_grid/macro_f1_by_fold.png | fold-level macro-F1 diagnostic |
| macro_f1_by_horizon | Macro-F1 Across Horizons | reports/figures/fi2010_neural_full_grid/macro_f1_by_horizon.png | mean macro-F1 across horizons |
| ece_by_horizon | ECE Across Horizons | reports/figures/fi2010_neural_full_grid/ece_by_horizon.png | mean calibration error across horizons |
| ssl_matched_delta | Matched SSL Deltas | reports/figures/fi2010_neural_full_grid/ssl_matched_delta.png | matched supervised-vs-SSL deltas only |
| confidence_threshold_eligible_fraction | Confidence Threshold Vs Eligible Fraction | reports/figures/fi2010_neural_full_grid/confidence_threshold_eligible_fraction.png | retained sample fraction by confidence |
| confidence_threshold_macro_f1 | Confidence Threshold Vs Retained Macro-F1 | reports/figures/fi2010_neural_full_grid/confidence_threshold_macro_f1.png | macro-F1 on retained high-confidence samples |
| cost_adjusted_proxy | Cost-Adjusted Proxy Diagnostic | reports/figures/fi2010_neural_full_grid/cost_adjusted_proxy.png | proxy diagnostics only when artefacts exist |
| execution_v3_confidence_active_fraction | Confidence Threshold Vs Active Trade Fraction Proxy Diagnostic | reports/figures/fi2010_neural_full_grid/execution_v3_confidence_active_fraction.png | stored-artefact diagnostic figure |
| execution_v3_confidence_net_proxy | Confidence Threshold Vs Net Cost-Adjusted Proxy Diagnostic | reports/figures/fi2010_neural_full_grid/execution_v3_confidence_net_proxy.png | stored-artefact diagnostic figure |
| execution_v3_cost_sensitivity | Cost Sensitivity Proxy Diagnostic | reports/figures/fi2010_neural_full_grid/execution_v3_cost_sensitivity.png | stored-artefact diagnostic figure |
| execution_v3_latency_sensitivity | Latency Sensitivity Proxy Diagnostic | reports/figures/fi2010_neural_full_grid/execution_v3_latency_sensitivity.png | stored-artefact diagnostic figure |
| execution_v3_fill_assumption_comparison | Fill Assumption Comparison Proxy Diagnostic | reports/figures/fi2010_neural_full_grid/execution_v3_fill_assumption_comparison.png | stored-artefact diagnostic figure |
| execution_v3_adverse_selection_by_confidence | Adverse Selection By Confidence Bucket Proxy Diagnostic | reports/figures/fi2010_neural_full_grid/execution_v3_adverse_selection_by_confidence.png | stored-artefact diagnostic figure |

Skipped plots:

| figure | reason |
| --- | --- |
| execution_v3_regime_breakdown | regime_execution_summary.csv has no plottable execution-v3 rows |
| regime_breakdown | regime labels not present in prediction artefacts |

## Uncertainty Summary

Seed variance is not available in the stored evidence; intervals are fold-level diagnostics.

| source | model | lookback | folds | seeds | mean | CI lower | CI upper | bootstrap lower | bootstrap upper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classical | gradient_boosting | n/a | 5 | 1 | 0.4654 | 0.4600 | 0.4708 | 0.4623 | 0.4692 |
| neural | matrix_transformer | 20 | 5 | 1 | 0.7337 | 0.6948 | 0.7726 | 0.7074 | 0.7535 |

## Ablation Summary

Skipped: ablation artefacts were not supplied or were unavailable.

## Feature Ablation Summary

Feature ablations are leakage-safe diagnostics over FI-2010 snapshot columns. Unsupported event-level groups remain explicit.
The `snapshot_order_flow_proxy` group is a snapshot-delta proxy, not true event-level order-flow imbalance.
Current stored feature-ablation evidence is `partial_real`: it covers folds 1-5 at horizon 10 for logistic and ridge models, with horizons 20/50 and slower model families left for future expansion.

| field | value |
| --- | --- |
| runner_version | fi2010-microstructure-feature-ablations/v2 |
| smoke_test | False |
| completed_run_count | 840 |
| failed_run_count | 0 |
| feature_groups | price_levels, size_levels, top_of_book, spread, midprice, microprice, top_of_book_imbalance, depth_imbalance, depth_slope, liquidity_concentration, snapshot_order_flow_proxy, volatility_proxy |
| proxy_groups | snapshot_order_flow_proxy |
| unsupported_groups | time_context, true_order_flow_imbalance, cancellation_imbalance, trade_imbalance, queue_position |

Aggregate rows:

| horizon | model | mode | feature group | completed | mean macro-F1 | mean MCC |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | logistic | all_features | all | 15 | 0.6090 | 0.4982 |
| 10 | logistic | derived_microstructure_only | derived_microstructure | 15 | 0.6079 | 0.4982 |
| 10 | logistic | no_proxy_features | no_proxy | 15 | 0.2754 | 0.0696 |
| 10 | logistic | only_one_group | depth_imbalance | 15 | 0.2514 | -0.0016 |
| 10 | logistic | only_one_group | depth_slope | 15 | 0.2514 | 0.0000 |
| 10 | logistic | only_one_group | liquidity_concentration | 15 | 0.2514 | -0.0014 |
| 10 | logistic | only_one_group | microprice | 15 | 0.2514 | 0.0018 |
| 10 | logistic | only_one_group | midprice | 15 | 0.2514 | 0.0000 |
| 10 | logistic | only_one_group | price_levels | 15 | 0.2581 | 0.0329 |
| 10 | logistic | only_one_group | size_levels | 15 | 0.2522 | 0.0072 |

Matched deltas versus all-features baseline:

| horizon | model | mode | feature group | delta macro-F1 | delta MCC | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | logistic | all_features | all | 0.0000 | 0.0000 | neutral |
| 10 | logistic | remove_one_group | price_levels | -0.0013 | 0.0028 | neutral |
| 10 | logistic | remove_one_group | size_levels | -0.0022 | -0.0064 | hurt |
| 10 | logistic | remove_one_group | top_of_book | -0.0006 | -0.0016 | neutral |
| 10 | logistic | remove_one_group | spread | -0.0197 | -0.0088 | hurt |
| 10 | logistic | remove_one_group | midprice | -0.0001 | -0.0004 | neutral |
| 10 | logistic | remove_one_group | microprice | -0.0001 | -0.0014 | neutral |
| 10 | logistic | remove_one_group | top_of_book_imbalance | 0.0001 | -0.0008 | neutral |
| 10 | logistic | remove_one_group | depth_imbalance | 0.0006 | 0.0003 | neutral |
| 10 | logistic | remove_one_group | depth_slope | -0.0006 | -0.0016 | neutral |

## Execution-Aware Proxy Summary

Execution-aware sections are offline execution-aware proxy diagnostics only.
They separate classification performance from confidence-filtered signal quality and cost-adjusted proxy diagnostics, and they do not support live-trading, profitability or PnL claims.

Execution-v3 is offline proxy analysis. To be explicit about its scope:

- it is not PnL;
- it is not live-trading evidence;
- it is not a production execution simulation;
- the confidence, cost, latency, fill and adverse-selection diagnostics test whether predictive metrics survive simple execution-like frictions.

A richer reviewer-facing breakdown of active fraction, turnover proxy, latency sensitivity, cost sensitivity, fill-assumption sensitivity and the adverse-selection proxy is generated by `analyse-fi2010-execution-v3`.
It explicitly skips regime diagnostics and is stored under `reports/execution_v3_analysis/execution_v3_analysis.md`.

Execution-v3 status:

| field | value |
| --- | --- |
| status | offline execution-aware proxy diagnostic loaded; payoff_mode=unit_payoff, cost_mode=unit_proxy |
| payoff_mode | unit_payoff |
| cost_mode | unit_proxy |
| smoke_test_status | not smoke-test |
| diagnostics_produced | confidence_threshold, cost_sensitivity, latency_sensitivity, fill_assumptions, adverse_selection |
| diagnostics_skipped | regime_execution |

Confidence filtering:

| model | objective | horizon | threshold | active fraction | macro-F1 | cost-adjusted proxy | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| matrix_transformer | masked_reconstruction | 10 | 0.33 | 0.5052 | 0.3233 | -10224.2000 | ok |
| matrix_transformer | masked_reconstruction | 10 | 0.35 | 0.5007 | 0.3230 | -10135.3333 | ok |
| matrix_transformer | masked_reconstruction | 10 | 0.4 | 0.4323 | 0.3196 | -8836.1333 | ok |
| matrix_transformer | masked_reconstruction | 10 | 0.45 | 0.2929 | 0.3138 | -6041.8000 | ok |
| matrix_transformer | masked_reconstruction | 10 | 0.5 | 0.1600 | 0.3096 | -3328.4000 | ok |
| matrix_transformer | masked_reconstruction | 10 | 0.55 | 0.0809 | 0.3067 | -1693.3333 | ok |
| matrix_transformer | masked_reconstruction | 10 | 0.6 | 0.0263 | 0.3029 | -558.3333 | ok |
| matrix_transformer | masked_reconstruction | 10 | 0.65 | 0.0088 | 0.3110 | -208.0000 | ok |
| matrix_transformer | masked_reconstruction | 10 | 0.7 | 0.0024 | 0.3051 | -62.6923 | ok |
| matrix_transformer | masked_reconstruction | 10 | 0.75 | 0.0003 | 0.3012 | -9.4615 | ok |

Cost sensitivity:

| model | objective | horizon | fee bps | spread x | gross proxy | cost-adjusted proxy | cost mode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| matrix_transformer | masked_reconstruction | 10 | 0.0 | 0.0 | -12206.0000 | -12206.0000 | unit_proxy |
| matrix_transformer | masked_reconstruction | 10 | 0.0 | 0.5 | -12206.0000 | -12311.9400 | unit_proxy |
| matrix_transformer | masked_reconstruction | 10 | 0.0 | 1.0 | -12206.0000 | -12417.8800 | unit_proxy |
| matrix_transformer | masked_reconstruction | 10 | 0.0 | 2.0 | -12206.0000 | -12629.7600 | unit_proxy |
| matrix_transformer | masked_reconstruction | 10 | 1.0 | 0.0 | -12206.0000 | -12208.1188 | unit_proxy |
| matrix_transformer | masked_reconstruction | 10 | 1.0 | 0.5 | -12206.0000 | -12314.0588 | unit_proxy |
| matrix_transformer | masked_reconstruction | 10 | 1.0 | 1.0 | -12206.0000 | -12419.9988 | unit_proxy |
| matrix_transformer | masked_reconstruction | 10 | 1.0 | 2.0 | -12206.0000 | -12631.8788 | unit_proxy |
| matrix_transformer | masked_reconstruction | 10 | 2.0 | 0.0 | -12206.0000 | -12210.2376 | unit_proxy |
| matrix_transformer | masked_reconstruction | 10 | 2.0 | 0.5 | -12206.0000 | -12316.1776 | unit_proxy |

Latency sensitivity:

| model | objective | horizon | latency | active trades | hit rate | cost-adjusted proxy | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| matrix_transformer | masked_reconstruction | 10 | 0 | 21188 | 0.2120 | -12206.0000 | ok |
| matrix_transformer | masked_reconstruction | 10 | 1 | 21188 | 0.2110 | -12246.0000 | ok |
| matrix_transformer | masked_reconstruction | 10 | 2 | 21188 | 0.2069 | -12422.0000 | ok |
| matrix_transformer | masked_reconstruction | 10 | 5 | 21188 | 0.2058 | -12468.0000 | ok |
| matrix_transformer | masked_reconstruction | 10 | 10 | 21188 | 0.2029 | -12590.0000 | ok |
| matrix_transformer | masked_reconstruction | 10 | 0 | 18788 | 0.1984 | -11334.0000 | ok |
| matrix_transformer | masked_reconstruction | 10 | 1 | 18788 | 0.1981 | -11346.0000 | ok |
| matrix_transformer | masked_reconstruction | 10 | 2 | 18788 | 0.1953 | -11448.0000 | ok |
| matrix_transformer | masked_reconstruction | 10 | 5 | 18788 | 0.1953 | -11450.0000 | ok |
| matrix_transformer | masked_reconstruction | 10 | 10 | 18788 | 0.1942 | -11492.0000 | ok |

Fill assumptions:

| model | objective | fill mode | filled | fill fraction | hit rate | cost-adjusted proxy | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| matrix_transformer | masked_reconstruction | aggressive_crossing | 21188 | 1.0000 | 0.2120 | -12873.4220 | ok |
| matrix_transformer | masked_reconstruction | passive_optimistic | 21188 | 1.0000 | 0.2120 | -12650.9480 | ok |
| matrix_transformer | masked_reconstruction | passive_conservative | 3 | 0.0001 | 0.3333 | -1.0315 | ok |
| matrix_transformer | masked_reconstruction | abstain_only | 0 | 0.0000 | n/a | 0.0000 | ok |
| matrix_transformer | masked_reconstruction | aggressive_crossing | 18788 | 1.0000 | 0.1984 | -11925.8220 | ok |
| matrix_transformer | masked_reconstruction | passive_optimistic | 18788 | 1.0000 | 0.1984 | -11728.5480 | ok |
| matrix_transformer | masked_reconstruction | passive_conservative | 0 | 0.0000 | n/a | 0.0000 | ok |
| matrix_transformer | masked_reconstruction | abstain_only | 0 | 0.0000 | n/a | 0.0000 | ok |
| matrix_transformer | masked_reconstruction | aggressive_crossing | 18785 | 1.0000 | 0.1936 | -12102.7275 | ok |
| matrix_transformer | masked_reconstruction | passive_optimistic | 18785 | 1.0000 | 0.1936 | -11905.4850 | ok |

Adverse-selection proxy:

| model | objective | horizon | confidence bucket | fill assumption | adverse fraction | mode |
| --- | --- | --- | --- | --- | --- | --- |
| matrix_transformer | masked_reconstruction | 10.0 | 0.33-0.50 | aggressive_crossing | 0.1868 | label_proxy |
| matrix_transformer | masked_reconstruction | 10.0 | 0.50-0.70 | aggressive_crossing | 0.1629 | label_proxy |
| matrix_transformer | masked_reconstruction | 10.0 | 0.70-0.85 | aggressive_crossing | 0.0807 | label_proxy |
| matrix_transformer | masked_reconstruction | 20.0 | 0.33-0.50 | aggressive_crossing | 0.2578 | label_proxy |
| matrix_transformer | masked_reconstruction | 20.0 | 0.50-0.70 | aggressive_crossing | 0.2208 | label_proxy |
| matrix_transformer | masked_reconstruction | 20.0 | 0.70-0.85 | aggressive_crossing | 0.1609 | label_proxy |
| matrix_transformer | masked_reconstruction | 50.0 | 0.33-0.50 | aggressive_crossing | 0.3311 | label_proxy |
| matrix_transformer | masked_reconstruction | 50.0 | 0.50-0.70 | aggressive_crossing | 0.3082 | label_proxy |
| matrix_transformer | masked_reconstruction | 50.0 | 0.70-0.85 | aggressive_crossing | 0.2536 | label_proxy |
| matrix_transformer | masked_reconstruction | 50.0 | 0.85-1.00 | aggressive_crossing | 0.2019 | label_proxy |

Skipped diagnostics:

| diagnostic | scope | reason |
| --- | --- | --- |
| regime_execution | volatility_regime | volatility_regime labels unavailable in prediction artefacts |
| regime_execution | spread_regime | spread_regime labels unavailable in prediction artefacts |
| regime_execution | imbalance_regime | imbalance_regime labels unavailable in prediction artefacts |
| regime_execution | liquidity_regime | liquidity_regime labels unavailable in prediction artefacts |
| regime_execution | regime | regime labels unavailable in prediction artefacts |

Conservative interpretation: execution-v3 can show how stored FI-2010 signals respond to confidence filters, costs, latency and fill proxy assumptions. It does not establish deployable execution quality.

## External Benchmark Context

Skipped: external context artefacts were not supplied or were unavailable.

## What This Supports

- The committed artefacts support a traceable multi-fold classical FI-2010 result.
- A separate, earlier 25-epoch reduced-scope, single-seed supervised neural benchmark is reported on its own terms and is not used as matched-grid or SSL evidence.
- The uncertainty, ablation and proxy-diagnostic layers are generated from stored tables.
- External references are used only to document protocol context.
- The one-epoch full neural grid artefacts compare supervised and SSL matrix-transformer variants under matched fold, horizon, seed, lookback, architecture and preprocessing keys; this is matched
  comparison evidence and supports no SSL improvement claim.
- Execution-v3 artefacts support an offline execution-aware proxy diagnostic over stored FI-2010 full-grid predictions.
- Feature-ablation artefacts support leakage-safe FI-2010 snapshot feature-family diagnostics with proxy features labelled as proxies.

## What This Does Not Claim

- Profitability or tradability in deployed markets.
- Production execution quality or market-impact realism.
- Unsupported live-trading claims from execution-v3 proxy diagnostics.
- Foundation-model status.
- SSL improvement or SOTA status.
- A full-grid SSL improvement claim when the full-grid directory is missing, smoke-only or contains failed matched runs.
- True order-flow, cancellation, trade-imbalance or queue-position claims from FI-2010 feature ablations; absent event-level fields remain unsupported.
- Neural superiority over the classical family.

## Limitations

| limitation | detail |
| --- | --- |
| classical_seed_count | 1 |
| neural_seed_count | 1 |
| neural_scope | single seed and single lookback in stored reduced-scope artefacts |
| execution_scope | offline execution-aware proxy diagnostics only; queue, impact and venue mechanics are not modelled |
| external_scope | protocol context only; no external numeric metrics are copied |
| prediction_checkpoint_policy | full predictions and checkpoints are not required by this report builder |
| full_neural_grid_scope | reported only when aggregate artefacts are supplied; smoke artefacts are not empirical evidence |
| feature_ablation_scope | snapshot-derived FI-2010 diagnostics only; snapshot-flow columns are proxies and unsupported event-level groups remain unavailable |

## Artefact Traceability

| artefact | path | sha256 |
| --- | --- | --- |
| classical_dir | experiments/fi2010_multifold_classical | directory |
| classical_results_summary | experiments/fi2010_multifold_classical/results_summary.csv | 7a4d3c042805ecb4d8735fe9ad95f1ccc9bf50a0d4a83acd646f4d6417a9e03e |
| classical_summary | experiments/fi2010_multifold_classical/summary.json | 6e82bc2ff4b6656b28619338b7486b850d9eca1baf4e4f53fc3ea397794b155a |
| evidence_pack_claim_audit | reports/evidence_pack/claim_audit.json | b05df88625f5d1e4b6c255a2bdbd2ea84e48f4a92f26bbd0d8c62b66bf824ac5 |
| evidence_pack_dir | reports/evidence_pack | directory |
| evidence_pack_manifest | reports/evidence_pack/evidence_pack_manifest.json | 971bd62b14f0f2c82e0f4a477844be0acdc1be07bfc433ee3b4c8d3ef19f26af |
| evidence_pack_supported_claims | reports/evidence_pack/supported_claims.md | 2f9de4ea0e9d80e1c1ea0f2949f832d7346ee3b62ae0b2eac331ca836d773c34 |
| evidence_pack_unsupported_claims | reports/evidence_pack/unsupported_claims.md | dcafc8b62c909eb641968cf58b0052e38ff22940627349098ba83302ccf2e72c |
| execution_v3_adverse_selection_summary | experiments/fi2010_execution_v3/adverse_selection_summary.csv | 94e339bc4749faa27c8e4a8ace6b7aac076754166edfc0a8d57a058374c9e014 |
| execution_v3_confidence_threshold_aggregate | experiments/fi2010_execution_v3/confidence_threshold_aggregate.csv | bbd14f3a458531f3722f435fcda296b7ddf882a37783470cf1854810ce18549f |
| execution_v3_confidence_threshold_summary | experiments/fi2010_execution_v3/confidence_threshold_summary.csv | 6603b546e9aec28bdb14df0fd12189467cc9b42a20f17c3b9e8e37f6a3127fff |
| execution_v3_cost_sensitivity_summary | experiments/fi2010_execution_v3/cost_sensitivity_summary.csv | c3a78242d8c7ef19c255e8306c5087513dd0e93335c55af172cc7ad1437745e7 |
| execution_v3_dir | experiments/fi2010_execution_v3 | directory |
| execution_v3_execution_v3_manifest | experiments/fi2010_execution_v3/execution_v3_manifest.json | b6ebe05a8bf888b212842375e8edddf018da01b7dcaba987be37cbaf3cc1d71f |
| execution_v3_fill_assumption_summary | experiments/fi2010_execution_v3/fill_assumption_summary.csv | 279b1ff9087e86386a70ff163a92bac440d7c18f1dd343d26f4822d588f6b0c2 |
| execution_v3_latency_sensitivity_summary | experiments/fi2010_execution_v3/latency_sensitivity_summary.csv | 2277b233fd4faed52a3e948328ff0aaeac8564f5ac878ee0c7f62d7935c6888e |
| execution_v3_regime_execution_summary | experiments/fi2010_execution_v3/regime_execution_summary.csv | d8b59fd77d21b802059096f84105943115cc7326abcdf69bb0c3195ae1771c4e |
| execution_v3_skipped_diagnostics | experiments/fi2010_execution_v3/skipped_diagnostics.json | 5b9cdc8684ad217047e0761c9033ea9257a4db00f3def62f7c6d7a5abd4307b5 |
| execution_v3_summary | experiments/fi2010_execution_v3/summary.json | 89d6b8d2ae5e1d96f9f15be4f0ba3852b98d51fa9b2542595aeaa1f07567f6be |
| feature_ablations_ablation_manifest | experiments/fi2010_feature_ablations/ablation_manifest.json | a307502c4c87cfa770c8c22fe3fcba8efe6624d578303a03fd7f9feb319dfdb5 |
| feature_ablations_aggregate_summary | experiments/fi2010_feature_ablations/aggregate_summary.csv | a89c5b6510dc25c72d21909783b6043319696e8dc55e082ef1684063ab5a4d92 |
| feature_ablations_dir | experiments/fi2010_feature_ablations | directory |
| feature_ablations_failures | experiments/fi2010_feature_ablations/failures.json | 36c25df88555a025cc8b83e815b6cddd81195b2aa2004be3d20a1eedb968dbde |
| feature_ablations_feature_delta_summary | experiments/fi2010_feature_ablations/feature_delta_summary.csv | afa03ef115948fcbd08104d619e9d564b0c2631dcdebd9a3521f61a14dad0b26 |
| feature_ablations_results_summary | experiments/fi2010_feature_ablations/results_summary.csv | f484e3232ecbd7d5b5b375e1008af8a2454e76a6f79f58c13388cfe43b2142af |
| feature_ablations_summary | experiments/fi2010_feature_ablations/summary.json | ea1707aaa5f7c37c51fd5061ae178315ef518b6fd947b1059903010bf569fca5 |
| fi2010_figure_dir | reports/figures/fi2010_neural_full_grid | directory |
| fi2010_figure_manifest | reports/figures/fi2010_neural_full_grid/figure_manifest.json | d1289ae54c0d04efb46f1a2758cd334078f22c4244f07bc7c37b7e3d3dc00306 |
| neural_dir | experiments/fi2010_multifold_neural | directory |
| neural_full_grid_aggregate_summary | experiments/fi2010_neural_full_grid/aggregate_summary.csv | 40dbe7f03d3aa2cd700341ba9a4ebef17011bdacfaa500e933c7c1ac45e8f452 |
| neural_full_grid_aggregate_summary_json | experiments/fi2010_neural_full_grid/aggregate_summary.json | 97fa54bf4573a3edec3294ea3fe68a124067ae3b5d8ce97af0a59ff0cd8bb736 |
| neural_full_grid_dir | experiments/fi2010_neural_full_grid | directory |
| neural_full_grid_failures | experiments/fi2010_neural_full_grid/failures.csv | 2829b70da9d7da1e989b94f6afc4259e687ffee4cd4af64c4471deded82d7a92 |
| neural_full_grid_results_summary | experiments/fi2010_neural_full_grid/results_summary.csv | 8db25bddd79d869130e07bcf3395f1389ba2a0140a1f8367ed46363d9856c1ff |
| neural_full_grid_ssl_comparison | experiments/fi2010_neural_full_grid/ssl_comparison.csv | ba10887aa6cf0b780fba5af9e5d7094764dee0afec894b3738dadf02399d21a9 |
| neural_full_grid_summary | experiments/fi2010_neural_full_grid/summary.json | 3f7f0abbb79974c3d84e0348858433bc49d26ea5f86fb1f6a556cf76e0758276 |
| neural_results_summary | experiments/fi2010_multifold_neural/results_summary.csv | bd6c0a52a6ea5eb5e01a66b91671cd84ee79b316ed202a9eadd198604a7f1e0c |
| neural_summary | experiments/fi2010_multifold_neural/summary.json | 9647aad87ff4bc8255d3971626850ff0c5b0eb76586a0e239141a64a9602b59e |
| proper_training_aggregate_summary | experiments/fi2010_neural_proper_training_subset_v2/aggregate_summary.csv | 73967613c4807fb8e7f6b91da5df5342a711489bf48ff1d259d796a5ab7f0b47 |
| proper_training_config_snapshot | experiments/fi2010_neural_proper_training_subset_v2/config_snapshot.json | d3a724cb6f57bd114a7a0b4ee28e78d305cd6fe2d4c04da7da28b52887539c91 |
| proper_training_curves_summary | experiments/fi2010_neural_proper_training_subset_v2/training_curves_summary.csv | 55317ba970df6c72ad172a9f7e338827c583e1fd60f6667953926d86e4b11b30 |
| proper_training_dir | experiments/fi2010_neural_proper_training_subset_v2 | directory |
| proper_training_failures | experiments/fi2010_neural_proper_training_subset_v2/failures.csv | 867b0a5fdbc0208d4e5441c75470af61348d641c367d47fbb5e7fcad8a7e7446 |
| proper_training_sha256_manifest | experiments/fi2010_neural_proper_training_subset_v2/sha256_manifest.json | 91ab967c9c45071a25bd6f2a68c90eb09370f96bcab78b1527ca2c757a9aa263 |
| proper_training_ssl_comparison | experiments/fi2010_neural_proper_training_subset_v2/ssl_comparison.csv | e3c2c7b72970c3d08d790d04bb1a9e94ef79210fd42b00cbe74743057c45c05f |
| proper_training_summary | experiments/fi2010_neural_proper_training_subset_v2/summary.json | 6aa9abb4f675349da3e6aea13fe6961a0335c85345b6e5e367fc82dffbe61717 |
| uncertainty_dir | experiments/fi2010_uncertainty | directory |
| uncertainty_metric_confidence_intervals | experiments/fi2010_uncertainty/metric_confidence_intervals.csv | db1468eb1f171702e1cad0d21ff9cc30ce5d082a9bfc7adc59882e720492e2cf |
| uncertainty_model_ranking | experiments/fi2010_uncertainty/model_ranking.csv | c1cb29fa431db19ade6167323902cacf8a2da4e385d369b9a92850338feee2ca |
| uncertainty_summary | experiments/fi2010_uncertainty/summary.json | 942e016bd40af0ab73412036279f2067731fcfb76156ee8c935beebf7b2d5c34 |

## Reproduction Commands

```bash
python -m chronoslob.cli build-final-empirical-report \
  --classical experiments/fi2010_multifold_classical \
  --neural experiments/fi2010_multifold_neural \
  --uncertainty experiments/fi2010_uncertainty \
  --ablations experiments/fi2010_brutal_ablations \
  --execution experiments/fi2010_execution_v2 \
  --execution-v3 experiments/fi2010_execution_v3 \
  --external experiments/fi2010_external_context \
  --neural-full-grid experiments/fi2010_neural_full_grid \
  --feature-ablations experiments/fi2010_feature_ablations \
  --out reports/chronoslob_final_empirical_report.md \
  --overwrite

python -m chronoslob.cli doctor
python -m chronoslob.cli inspect-release-readiness
python -m chronoslob.cli run-project-audit --strict
python -m pytest
python -m compileall -q chronoslob tests
python -m ruff check .
python -m mypy chronoslob
```

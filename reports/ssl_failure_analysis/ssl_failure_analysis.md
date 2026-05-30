# SSL Failure Analysis

Builder version `phase-l/ssl-failure-analysis/v1`.

This report explains what the FI-2010 self-supervised (SSL) objectives did and did not achieve
across the completed evidence. It is generated from retained lightweight summary tables only. The
heavy raw per-run prediction files and encoder checkpoints are not required and are not read.

## Evidence Sources

Three distinct bodies of evidence are kept separate and never merged:

- One-epoch matched full grid: folds 1-5, horizons 10/20/50, seeds 0-2, objectives supervised /
  masked_reconstruction / next_field. This is matched comparison and infrastructure evidence, not a
  tuned-training result.
- Proper-training subset v2: fold 1, horizons 10 and 50, seed 0, lookback 50, SSL pretrain 5 epochs,
  max 25 epochs, patience 5, CPU. Evidence level `partial_real`.
- A separate older reduced-scope supervised benchmark, used only for context and never as SSL
  evidence.

## One-Epoch Full Grid

Matched supervised-vs-SSL pairs analysed: 90.

Mean matched deltas by objective (SSL minus supervised). Positive macro-F1 / MCC is better; for ECE
a win is a lower value.

| objective | pairs | mean d-macroF1 | mean d-MCC | mean d-ECE | macroF1 w/t/l | ECE w/t/l |
| --- | --- | --- | --- | --- | --- | --- |
| masked_reconstruction | 45 | -0.0100 | -0.0199 | +0.0221 | 19/0/26 | 18/0/27 |
| next_field | 45 | -0.0622 | -0.0651 | -0.0083 | 3/0/42 | 24/0/21 |

Interpretation: the one-epoch grid is matched comparison and infrastructure evidence. It does not
support a broad SSL improvement claim. Masked reconstruction is neutral-to-slightly-negative overall
and next-field is clearly negative overall; ECE does not support a calibration-improvement claim.

Full-grid macro-F1 delta by horizon:

| objective | horizon | mean d-macroF1 |
| --- | --- | --- |
| masked_reconstruction | 10 | -0.0103 |
| masked_reconstruction | 20 | -0.0165 |
| masked_reconstruction | 50 | -0.0032 |
| next_field | 10 | -0.0603 |
| next_field | 20 | -0.0906 |
| next_field | 50 | -0.0357 |

Full-grid macro-F1 delta by fold:

| objective | fold | mean d-macroF1 |
| --- | --- | --- |
| masked_reconstruction | 1 | -0.0022 |
| masked_reconstruction | 2 | +0.0249 |
| masked_reconstruction | 3 | -0.0180 |
| masked_reconstruction | 4 | -0.0290 |
| masked_reconstruction | 5 | -0.0257 |
| next_field | 1 | -0.0910 |
| next_field | 2 | -0.0432 |
| next_field | 3 | -0.0724 |
| next_field | 4 | -0.0465 |
| next_field | 5 | -0.0579 |

Full-grid macro-F1 delta by seed:

| objective | seed | mean d-macroF1 |
| --- | --- | --- |
| masked_reconstruction | 0 | -0.0114 |
| masked_reconstruction | 1 | -0.0112 |
| masked_reconstruction | 2 | -0.0074 |
| next_field | 0 | -0.0735 |
| next_field | 1 | -0.0583 |
| next_field | 2 | -0.0549 |

Any positive cells are isolated rather than consistent across folds, horizons and seeds, so they do
not support a general SSL improvement claim.

## Proper-Training Subset v2

Exact scope: fold 1, horizons 10 and 50, seed 0, lookback 50, evidence level `partial_real`. Matched
supervised-vs-SSL pairs: 4.

Supervised baseline by horizon:

| horizon | macro-F1 | MCC | ECE |
| --- | --- | --- | --- |
| 10 | 0.2477 | 0.0000 | 0.0872 |
| 50 | 0.3883 | 0.0917 | 0.0496 |

Matched SSL deltas by horizon (SSL minus supervised):

| horizon | objective | d-macroF1 | d-MCC | d-ECE |
| --- | --- | --- | --- | --- |
| 10 | masked_reconstruction | 0.0000 | 0.0000 | +0.0175 |
| 10 | next_field | 0.0000 | 0.0000 | +0.0746 |
| 50 | masked_reconstruction | +0.0891 | +0.1238 | +0.0245 |
| 50 | next_field | +0.0065 | +0.0408 | +0.0317 |

At horizon 10 every objective collapses to the same stationary-majority prediction, so masked and
next-field tie supervised on macro-F1 and MCC while both worsen ECE. At horizon 50 masked
reconstruction improves macro-F1 and MCC but still worsens ECE, and next-field shows a small
macro-F1 / MCC gain with worse ECE.

The proper-training subset provides narrow partial evidence that masked SSL can improve
fold-1/horizon-50 predictive metrics under this configuration, but calibration worsened and the
scope is too small for a broad SSL improvement claim.

## Training-Curve Diagnostics

Runs summarised: 6. Trained beyond epoch 1: 6. Early-stopped: 5. Mean best epoch: 9.33.

| objective | horizon | epochs | best_epoch | best_val_macroF1 | early_stop | test_macroF1 |
| --- | --- | --- | --- | --- | --- | --- |
| supervised | 10 | 6 | 1 | 0.2547 | yes | 0.2477 |
| masked_reconstruction | 10 | 6 | 1 | 0.2547 | yes | 0.2477 |
| next_field | 10 | 6 | 1 | 0.2547 | yes | 0.2477 |
| supervised | 50 | 14 | 9 | 0.4140 | yes | 0.3883 |
| masked_reconstruction | 50 | 25 | 25 | 0.4842 | no | 0.4774 |
| next_field | 50 | 24 | 19 | 0.4513 | yes | 0.3948 |

Horizon-10 runs early-stop at the first epoch on the stationary-majority solution. Horizon-50 runs
train longer; masked reconstruction used the full budget while supervised and next-field
early-stopped. SSL fine-tuning therefore converged differently from supervised training only at
horizon 50.

## Claim Assessment

| claim | status | scope |
| --- | --- | --- |
| ssl_implemented_and_evaluated | supported | matched supervised-vs-SSL comparison rows in the stored artefacts |
| broad_ssl_improvement | unsupported | one-epoch full grid matched rows (folds 1-5, horizons 10/20/50, seeds 0-2) |
| ssl_calibration_improvement | unsupported | all matched SSL rows across the full grid and proper-training subset |
| proper_training_h50_predictive_improvement | partially_supported | proper-training subset, fold 1, horizon 50, seed 0, lookback 50, evidence level partial_real |

- `ssl_implemented_and_evaluated` (supported): Matched supervised-vs-SSL comparison rows are present,
  so the implementation-and-evaluation claim is supported.
- `broad_ssl_improvement` (unsupported): The completed one-epoch full grid does not support a broad
  SSL improvement claim: matched macro-F1 deltas are neutral-to-negative and calibration is not
  uniformly improved.
- `ssl_calibration_improvement` (unsupported): Calibration (ECE) did not improve uniformly; every
  matched proper-training SSL row worsened ECE, so no calibration improvement is claimed.
- `proper_training_h50_predictive_improvement` (partially_supported): Masked SSL improved macro-F1 and
  MCC at fold 1 / horizon 50, but calibration worsened and the scope is a single partial_real slice.

## Figures

| figure | title | path |
| --- | --- | --- |
| ssl_macro_f1_delta_by_horizon | SSL macro-F1 delta by horizon | reports/ssl_failure_analysis/ssl_macro_f1_delta_by_horizon.png |
| ssl_mcc_delta_by_horizon | SSL MCC delta by horizon | reports/ssl_failure_analysis/ssl_mcc_delta_by_horizon.png |
| ssl_ece_delta_by_horizon | SSL ECE delta by horizon | reports/ssl_failure_analysis/ssl_ece_delta_by_horizon.png |
| best_epoch_by_objective | Best epoch by objective (proper-training) | reports/ssl_failure_analysis/best_epoch_by_objective.png |

## What This Does Not Claim

This analysis does not claim that SSL improves ChronosLOB overall, that SSL improves calibration, or
anything about profitability, live trading or tradable signal. It is a diagnostic over stored
FI-2010 metrics. More evidence would require broader proper-training runs and/or better SSL
objective design rather than any success claim.

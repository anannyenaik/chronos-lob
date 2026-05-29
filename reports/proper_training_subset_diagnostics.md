# Proper-Training Subset Diagnostics

Generated from `experiments/fi2010_neural_proper_training_subset_v2/`.

## Scope

- Evidence level: `partial_real`
- Scope label: `limited_partial_real_slice`
- Folds: [1]
- Horizons: [10, 50]
- Seeds: [0]
- Lookbacks: [50]
- Objectives: ['supervised', 'masked_reconstruction', 'next_field']
- Pretrain epochs: 5
- Max epochs: 25
- Patience: 5
- Completed runs: 6 of 6
- Failed runs: 0

This is a completed partial scope. The attempted folds 1-3 fallback was
reduced because CPU-only runtime made the full fallback slice too expensive
in this pass.

## Early Stopping

- Epochs ran: 6 to 25
- Best epochs: 1 to 25
- Mean best epoch: 9.33
- Runs trained beyond epoch 1: 6 of 6
- Runs early-stopped: 5 of 6
- Best checkpoints are restored in-memory before test evaluation by the
  shared torch training loop; per-run metadata records
  `best_checkpoint_restored_before_test: true`.

## Curve Diagnostics

| run_id | objective | horizon | epochs | best_epoch | first_train_loss | last_train_loss | first_validation_loss | last_validation_loss | first_validation_macro_f1 | best_validation_macro_f1 | last_validation_macro_f1 | early_stop_marked |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fold_1__h10__seed_0__lb50__supervised | supervised | 10 | 6 | 1 | 1.0858 | 0.8908 | 0.9624 | 0.9390 | 0.2547 | 0.2547 | 0.2547 | True |
| fold_1__h10__seed_0__lb50__masked_reconstruction | masked_reconstruction | 10 | 6 | 1 | 1.0008 | 0.8778 | 0.9466 | 0.9209 | 0.2547 | 0.2547 | 0.2547 | True |
| fold_1__h10__seed_0__lb50__next_field | next_field | 10 | 6 | 1 | 0.9654 | 0.9598 | 0.9389 | 0.9285 | 0.2547 | 0.2547 | 0.2547 | True |
| fold_1__h50__seed_0__lb50__supervised | supervised | 50 | 14 | 9 | 1.1261 | 0.9861 | 1.1061 | 1.0856 | 0.1707 | 0.4140 | 0.4004 | True |
| fold_1__h50__seed_0__lb50__masked_reconstruction | masked_reconstruction | 50 | 25 | 25 | 1.1535 | 0.8989 | 1.1133 | 1.0549 | 0.1802 | 0.4842 | 0.4842 | False |
| fold_1__h50__seed_0__lb50__next_field | next_field | 50 | 24 | 19 | 1.2159 | 0.9807 | 1.1228 | 1.0872 | 0.1666 | 0.4513 | 0.4080 | True |

No objective collapsed to non-finite losses. Horizon 10 supervised and
next-field converged to the stationary-majority-style solution, while masked
reconstruction produced non-zero down/up class F1. Horizon 50 trained longer
and produced non-zero class F1 across all objectives.

## Test Metrics

Best supervised result by horizon:

| horizon | macro_f1 | mcc | ece | brier_score | nll |
| --- | --- | --- | --- | --- | --- |
| 10.0000 | 0.2477 | 0.0000 | 0.0872 | 0.5815 | 0.9850 |
| 50.0000 | 0.3883 | 0.0917 | 0.0496 | 0.6660 | 1.0976 |

Matched SSL deltas:

| horizon | ssl_objective | delta_macro_f1 | delta_mcc | delta_ece | delta_brier_score | delta_nll | macro_f1_outcome | mcc_outcome | ece_outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | masked_reconstruction | 0.0000 | 0.0000 | 0.0175 | -0.0053 | -0.0043 | tie | tie | loss |
| 10 | next_field | 0.0000 | 0.0000 | 0.0746 | 0.0394 | 0.0919 | tie | tie | loss |
| 50 | masked_reconstruction | 0.0891 | 0.1238 | 0.0245 | -0.0413 | -0.0610 | win | win | loss |
| 50 | next_field | 0.0065 | 0.0408 | 0.0317 | -0.0115 | -0.0167 | win | win | loss |

### masked_reconstruction

- Mean delta macro-F1: 0.0445
- Mean delta MCC: 0.0619
- Mean delta ECE: 0.0210
- Mean delta Brier: -0.0233
- Mean delta NLL: -0.0326
- Macro-F1 wins/ties/losses: 1 / 1 / 0
- MCC wins/ties/losses: 1 / 1 / 0
- ECE wins/ties/losses: 0 / 0 / 2

### next_field

- Mean delta macro-F1: 0.0032
- Mean delta MCC: 0.0204
- Mean delta ECE: 0.0531
- Mean delta Brier: 0.0140
- Mean delta NLL: 0.0376
- Macro-F1 wins/ties/losses: 1 / 1 / 0
- MCC wins/ties/losses: 1 / 1 / 0
- ECE wins/ties/losses: 0 / 0 / 2

## Interpretation

Masked reconstruction improved macro-F1 and MCC on both horizons in this fold,
but worsened ECE on both horizons. Next-field tied macro-F1 and MCC at horizon
10, improved macro-F1 and MCC at horizon 50, and worsened ECE on both horizons.

Because this is one fold only and calibration worsened in every matched row, it
supports only a narrow partial-real observation, not a broad SSL improvement
claim.

Figures were not generated in this pass; the per-run curve CSVs and predictions
are stored under `experiments/fi2010_neural_proper_training_subset_v2/runs/`
for future plotting.

# FI-2010 Uncertainty Notes

Fold variance is available because the classical and neural runners store per-fold metric tables.
Neural seed variance is not fully measured. The stored neural evidence covers a single seed per (model, fold, lookback). Cross-seed variance therefore remains future work unless an additional seed run is recorded.
Classical seed variance is not available; the classical runner records a single seed per (model, fold).
Confidence intervals use a Student-t two-sided interval at 0.95 together with a percentile bootstrap using 1000 iterations and seed=0. Fold is the unit of variance.
Comparisons against the baseline `gradient_boosting` are paired per fold.
Classical comparisons are based on 5 FI-2010 folds.
Neural comparisons are based on 5 FI-2010 folds.
The execution proxy summary and calibration summary in the upstream multi-fold directories remain diagnostic and are not live tradability claims.

## Classical test macro-F1 with confidence intervals

  - gradient_boosting: mean=0.4654 [0.4600, 0.4708] across 5 folds
  - random_forest: mean=0.4547 [0.4434, 0.4659] across 5 folds
  - logistic: mean=0.3261 [0.3114, 0.3408] across 5 folds
  - elastic_net: mean=0.3260 [0.3113, 0.3407] across 5 folds
  - ridge: mean=0.3087 [0.2972, 0.3201] across 5 folds
  - majority: mean=0.2514 [0.2439, 0.2590] across 5 folds

## Neural test macro-F1 with confidence intervals

  - matrix_transformer (lookback=20): mean=0.7337 [0.6948, 0.7726] across 5 folds
  - deeplob_style (lookback=20): mean=0.4753 [0.4372, 0.5133] across 5 folds

The neural numbers above remain reduced-scope, single-seed evidence unless additional seed runs are recorded; do not interpret them as cross-seed validated.

## Paired fold differences vs `gradient_boosting` (test macro-F1)

  - random_forest: mean diff=-0.0107 [-0.0191, -0.0024], wins=0/losses=5/ties=0 across 5 folds
  - logistic: mean diff=-0.1393 [-0.1535, -0.1251], wins=0/losses=5/ties=0 across 5 folds
  - elastic_net: mean diff=-0.1394 [-0.1535, -0.1253], wins=0/losses=5/ties=0 across 5 folds
  - ridge: mean diff=-0.1567 [-0.1675, -0.1459], wins=0/losses=5/ties=0 across 5 folds
  - majority: mean diff=-0.2140 [-0.2209, -0.2071], wins=0/losses=5/ties=0 across 5 folds

## Combined ranking (test macro-F1)

  1. matrix_transformer (lookback=20) (neural): mean=0.7337 [0.6948, 0.7726] over 5 folds
  2. deeplob_style (lookback=20) (neural): mean=0.4753 [0.4372, 0.5133] over 5 folds
  3. gradient_boosting (classical): mean=0.4654 [0.4600, 0.4708] over 5 folds
  4. random_forest (classical): mean=0.4547 [0.4434, 0.4659] over 5 folds
  5. logistic (classical): mean=0.3261 [0.3114, 0.3408] over 5 folds
  6. elastic_net (classical): mean=0.3260 [0.3113, 0.3407] over 5 folds
  7. ridge (classical): mean=0.3087 [0.2972, 0.3201] over 5 folds
  8. majority (classical): mean=0.2514 [0.2439, 0.2590] over 5 folds

## Reading the artefacts

- `metric_confidence_intervals.csv`: per-model, per-split, per-metric mean, std, standard error and Student-t plus percentile-bootstrap confidence intervals.
  Missing probability metrics (for example `ridge` Brier and ECE) are dropped and tracked via `n_missing`.
- `paired_model_comparisons.csv`: paired fold-level mean differences between each candidate model and the baseline.
- `rank_stability.csv`: how often each model is best per fold and the per-model mean rank across folds.
- `model_ranking.csv`: the combined classical+neural ranking on test macro-F1, ordered by mean, with the same confidence interval as the per-metric table.
- `summary.json`: the inputs, parameters, models, folds and artefact paths used by this run.

## What this analysis does not claim

- It does not establish profitability, market-beating performance or live tradability.
- It does not promote any model to foundation-model status or state-of-the-art status.
- It does not report self-supervised pretraining results; that ablation remains gated upstream.
- Neural superiority over the classical baseline must not be asserted without the caveats above.

# Statistical Uncertainty Analysis

## Purpose

This layer quantifies fold-level variance for the FI-2010 multi-fold
classical and supervised neural runs already on disk. It does not
re-run any model, does not require full prediction rows or
checkpoints, and is intended as a diagnostic surface around the existing
multi-fold evidence.

The layer is intentionally narrow: it tests whether observed
differences between models are stable across the five FI-2010 folds.
Where a metric is missing for a model (for example `ridge` cannot
report `brier_score` or `ece`), it is dropped from that model's
per-metric summary and tracked via a `n_missing` column.

## Inputs

Two stored artefact tables, produced by the multi-fold runners under
`chronoslob/experiments/`:

- `experiments/fi2010_multifold_classical/results_by_fold.csv`
- `experiments/fi2010_multifold_neural/results_by_fold_seed.csv`

Either input may be omitted. At least one must be present.

## Outputs

The analyser writes the following artefacts under the chosen `--out`
directory. All artefacts are small and committable.

- `summary.json`: inputs, parameters, models, folds and artefact
  paths used by the run.
- `metric_confidence_intervals.csv`: per-model, per-split, per-metric
  mean, std, standard error and confidence intervals.
- `paired_model_comparisons.csv`: paired fold-level mean differences
  between every candidate model and the baseline.
- `rank_stability.csv`: how often each model is best per fold and the
  per-model mean rank across folds.
- `model_ranking.csv`: the combined classical+neural ranking on the
  test split for macro-F1, ordered by mean, with the same confidence
  interval as the per-metric table.
- `uncertainty_notes.md`: human-readable notes that record the
  variance caveats below.

## Confidence interval method

For each `(model, split, metric)` grouping the analyser collapses any
multi-seed entries to a per-fold mean so that fold is the unit of
variance, then reports:

- Sample mean, sample standard deviation (`ddof=1`) and standard error.
- A two-sided Student-t confidence interval at the chosen `--ci-level`
  (default `0.95`). Critical values are taken from a small static table
  that covers `df = 1..10` for confidence levels `0.80`, `0.90`, `0.95`
  and `0.99`; larger samples fall back to the normal approximation.
- A percentile bootstrap interval of the mean over folds, using the
  configured `--bootstrap-iterations` (default `1000`) and
  `--bootstrap-seed` (default `0`). The resampler uses
  `numpy.random.default_rng` so the output is deterministic given the
  seed.

## Paired comparison method

Per metric and per split, candidate models are aligned with the
baseline by `fold_id`. For each candidate, the analyser records:

- The fold-level mean difference (`candidate − baseline`).
- The sample standard deviation of differences (`ddof=1`) and the
  matching Student-t confidence interval on the mean difference.
- Wins, losses and ties counts across the paired folds. Wins count
  folds where the candidate strictly outperformed the baseline on the
  metric, and losses count the reverse. For lower-is-better metrics
  (`brier_score`, `ece`) the sign of the difference reflects the
  raw direction and is not flipped.

## Rank stability

For each `(metric, split)` the analyser ranks all models per fold,
using ascending order for `brier_score` and `ece` and descending order
otherwise. It records per model: how often it was the best on a given
fold, the fraction of folds where it was best, the mean rank across
folds, and the sample standard deviation of ranks.

## Limitations

- Fold variance is available, but the stored classical runner uses a
  single seed and the stored neural runner uses a single seed, single
  lookback and reduced `max_epochs`. Cross-seed and multi-lookback
  variance for the supervised neural models therefore remains future
  work unless an additional seed run is recorded.
- The neural ranking is reduced-scope evidence. Matrix-transformer
  performance shown in the artefacts must be read as single-seed,
  single-lookback evidence, not as a cross-seed validated finding.
- Comparisons are based on five FI-2010 folds. The fold count is the
  only source of variance available here; with only five paired folds
  the confidence intervals are wide for many metrics.
- The execution proxy summary and calibration summary in the upstream
  multi-fold directories remain diagnostic and are not live
  tradability claims.

## Exact command

```bash
python -m chronoslob.cli analyse-fi2010-uncertainty \
  --classical experiments/fi2010_multifold_classical \
  --neural experiments/fi2010_multifold_neural \
  --out experiments/fi2010_uncertainty \
  --baseline gradient_boosting \
  --overwrite
```

The output directory must either not exist or be empty unless
`--overwrite` is passed.

## Claim boundaries

This analysis is diagnostic. It does not assert profitability, does not
assert live tradability, does not assert that any model beats the
market, does not promote any model to foundation-model status or to
state-of-the-art status, and does not report any self-supervised
result. Any reported neural advantage over the classical baseline must
be read together with the reduced-scope, single-seed caveat above.

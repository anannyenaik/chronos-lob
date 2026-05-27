# FI-2010 Brutal Ablations

## Purpose

The brutal ablation layer stress-tests where the supervised FI-2010
signal survives and where it breaks. It does not introduce a new model.
It reuses the prepared multi-fold folds and the stored classical and
neural evidence, then refits only a fast linear baseline where a real
refit is required. The goal is robustness evidence, not a new headline
number.

This layer is diagnostic. It makes no profitability or live tradability
claim, no leading-benchmark claim and no foundation-model claim, it
reports no self-supervised pretraining result, and it does not assert
neural superiority over the classical baseline.

## Ablation families

| Family | What it varies | How it is computed |
| --- | --- | --- |
| `feature_groups` | The feature subset fed to the model | Refit a fast linear baseline per group on the real folds |
| `model_class` | The model class | Reuse the stored classical per-fold metric table |
| `lookback` | The supervised neural lookback window | Skipped by default; opt in with `--neural-lookbacks` |
| `horizon` | The prediction horizon label column | Refit the linear baseline per configured horizon |
| `calibration` | Reliability and the confidence threshold | Reuse stored reliability and threshold proxy tables |
| `execution` | Cost and latency assumptions | Reuse stored execution proxy tables |

The feature groups are derived from the order book column names:

- `all_features` - every selected feature column.
- `top_of_book_only` - level-1 price and quantity columns only.
- `depth_features` - price and quantity columns at levels two and deeper.
- `liquidity_depth_features` - all quantity (volume) columns.
- `price_only` - all price columns (the price-level / return-proxy group).
- `labels_excluded` - identical to `all_features`; a leakage-control group
  that confirms label columns never enter the feature matrix.

The feature-group and horizon families refit a fast linear baseline so
the default run stays inexpensive across all five folds. The strongest
classical model is covered separately by the `model_class` family, which
reuses the stored gradient-boosting and tree evidence without retraining.

## Exact command

```bash
python -m chronoslob.cli run-fi2010-brutal-ablations \
  --config configs/experiments/fi2010_multifold.yaml \
  --neural-config configs/experiments/fi2010_neural_serious.yaml \
  --processed-root data/processed/fi2010 \
  --classical experiments/fi2010_multifold_classical \
  --neural experiments/fi2010_multifold_neural \
  --out experiments/fi2010_brutal_ablations \
  --overwrite
```

Useful subsets:

- `--families feature_groups,model_class` - run a subset of families.
- `--folds fold_1,fold_2` - restrict to specific folds.
- `--models gradient_boosting,matrix_transformer` - classical names drive
  the fit families; neural names drive the lookback family.
- `--neural-lookbacks 20,50 --max-epochs 5` - execute the lookback sweep.
- `--dry-run` - resolve the plan and write nothing.

## Artefacts

All artefacts are written under `--out` and are small enough to keep:

- `summary.json` - inputs, families run and skipped, folds, counts and
  claim boundaries.
- `ablation_results.csv` - the unified long-format result table.
- `ablation_summary.csv` - per-family aggregate means and the spread of
  the delta across folds.
- `skipped_ablations.json` - every skipped ablation with its reason.
- `feature_group_ablation.csv`, `model_class_ablation.csv`,
  `lookback_ablation.csv`, `horizon_ablation.csv`,
  `calibration_threshold_ablation.csv` and
  `execution_cost_latency_ablation.csv` - the per-family tables.
- `ablation_notes.md` - concise notes including the strongest and weakest
  feature-group findings.

Every result row carries `ablation_family`, `ablation_name`, `fold_id`,
`model_name`, `split`, `metric_name`, `metric_value`, `baseline_value`,
`delta_vs_baseline`, `status` and `skip_reason`. Full prediction rows and
model checkpoints are never written by default.

## How skipped ablations are recorded

Skipped ablations are never silently dropped. Each one is written as a
row with `status` set to `skipped`, an empty metric value and a populated
`skip_reason`, and is also listed in `skipped_ablations.json`. A family
that has no inputs (for example a missing stored directory or a missing
fold CSV) records a single skip row with the reason rather than failing
the whole run. The neural lookback sweep is skipped by default with a
reason that explains how to opt in.

## How to interpret deltas

`delta_vs_baseline` is `metric_value - baseline_value` for the family
baseline:

- `feature_groups` baseline is `all_features`. A large negative delta
  means the subset loses signal relative to the full feature set.
- `model_class` baseline is `gradient_boosting`. A negative delta means
  the simpler model class is weaker on that fold and metric.
- `horizon` baseline is the configured target horizon. A positive delta
  marks a horizon that is easier for the baseline.
- `calibration` and `execution` deltas are measured against the zero
  threshold, zero cost and zero latency reference.

The unit of variance is the fold. `ablation_summary.csv` reports the mean
delta and its standard deviation across folds so a finding that depends
on a single fold is visible.

## Execution metrics are proxies

The `execution` family and the confidence-threshold rows of the
`calibration` family are simplified proxy diagnostics under stated cost
and latency assumptions. They are not a backtest, they do not model order
placement and they do not imply tradability. They show how a confidence
gate trades coverage for hit rate and how the proxy net return responds
to cost and latency, nothing more.

## Boundaries

- Diagnostic only; no profitability or live tradability claim is made.
- No foundation-model or leading-benchmark claim.
- No self-supervised pretraining result is reported.
- Neural superiority over the classical baseline is not asserted.

# Feature-Group Ablation

Ablation type: `feature_group`

## Purpose

Restrict the feature matrix to a column-name pattern group; preprocessing stays fit on train rows only. The runner fails cleanly when too few columns match; the `all` group is the baseline child.

## What Changed

- `feature_top_of_book`: Feature subset restricted to top-of-book price and quantity columns through deterministic column-name patterns; preprocessing remains fit on train rows only.
  - parameters: `feature_patterns=['bid_price_1', 'ask_price_1', 'bid_quantity_1', 'ask_quantity_1']`
- `feature_imbalance`: Feature subset restricted to imbalance and microprice columns through column-name patterns. Skipped when too few matching columns exist on the supplied data.
  - parameters: `feature_patterns=['*imbalance*', '*microprice*']`
- `feature_depth_liquidity`: Feature subset restricted to depth and liquidity columns (bid/ask quantity levels) through column-name patterns.
  - parameters: `feature_patterns=['bid_quantity_*', 'ask_quantity_*']`

## Held Fixed

- base FI-2010 benchmark preparation config, data path, seed and split design
- preprocessing fit on train rows only
- model registry (no model family added by ablations)
- experiment artefact contract for each child experiment (validated before this report was written)

## Artefacts Used

- `ablation_summary.json`
- `ablation_results.csv`
- `ablation_manifest.json`
- `experiments/feature_top_of_book`
- `experiments/feature_depth_liquidity`

## Status

- `feature_top_of_book`: run (experiment: `experiments/feature_top_of_book`)
- `feature_imbalance`: skipped - ValueError: paper experiment feature_patterns produced no matching feature columns; patterns: ['*imbalance*', '*microprice*']
- `feature_depth_liquidity`: run (experiment: `experiments/feature_depth_liquidity`)

## Key Metric Summary

| ablation | model | metric | value | source |
| --- | --- | --- | --- | --- |
| feature_top_of_book | majority | accuracy | 0.628615 | experiments/feature_top_of_book |
| feature_top_of_book | majority | macro_f1 | 0.257321 | experiments/feature_top_of_book |
| feature_top_of_book | logistic | accuracy | 0.628615 | experiments/feature_top_of_book |
| feature_top_of_book | logistic | macro_f1 | 0.257321 | experiments/feature_top_of_book |
| feature_top_of_book | deeplob_style | accuracy | 0.628615 | experiments/feature_top_of_book |
| feature_top_of_book | deeplob_style | macro_f1 | 0.257321 | experiments/feature_top_of_book |
| feature_depth_liquidity | majority | accuracy | 0.628615 | experiments/feature_depth_liquidity |
| feature_depth_liquidity | majority | macro_f1 | 0.257321 | experiments/feature_depth_liquidity |
| feature_depth_liquidity | logistic | accuracy | 0.628615 | experiments/feature_depth_liquidity |
| feature_depth_liquidity | logistic | macro_f1 | 0.257321 | experiments/feature_depth_liquidity |
| feature_depth_liquidity | deeplob_style | accuracy | 0.628615 | experiments/feature_depth_liquidity |
| feature_depth_liquidity | deeplob_style | macro_f1 | 0.257321 | experiments/feature_depth_liquidity |
| baseline | majority | accuracy | 0.628615 | experiments/baseline |
| baseline | majority | macro_f1 | 0.257321 | experiments/baseline |
| baseline | logistic | accuracy | 0.623310 | experiments/baseline |
| baseline | logistic | macro_f1 | 0.351469 | experiments/baseline |
| baseline | deeplob_style | accuracy | 0.567174 | experiments/baseline |
| baseline | deeplob_style | macro_f1 | 0.352432 | experiments/baseline |

## Warnings And Limitations

- Ablation rows do not claim profitability, tradable alpha or live execution. Execution-aware values are simplified proxy assumptions.
- Aggregated values come directly from stored child-experiment artefacts and are not edited after the run.

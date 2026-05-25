# Lookback-Window Ablation

Ablation type: `lookback`

## Purpose

Vary only the neural lookback window. Classical baselines do not consume windows and therefore record the same metric values across this ablation; lookback effects are only meaningful when neural models are included.

## What Changed

- `lookback_2`: Neural lookback window of 2 rows; applies only to neural paper-runner models that consume windows.
  - parameters: `neural_settings.lookback=2`
- `lookback_4`: Neural lookback window of 4 rows; applies only to neural paper-runner models that consume windows.
  - parameters: `neural_settings.lookback=4`

## Held Fixed

- base FI-2010 benchmark preparation config, data path, seed and split design
- preprocessing fit on train rows only
- model registry (no model family added by ablations)
- experiment artefact contract for each child experiment (validated before this report was written)

## Artefacts Used

- `ablation_summary.json`
- `ablation_results.csv`
- `ablation_manifest.json`
- `experiments/lookback_2`
- `experiments/lookback_4`

## Status

- `lookback_2`: run (experiment: `experiments/lookback_2`)
- `lookback_4`: run (experiment: `experiments/lookback_4`)

## Key Metric Summary

| ablation | model | metric | value | source |
| --- | --- | --- | --- | --- |
| lookback_2 | majority | accuracy | 0.628615 | experiments/lookback_2 |
| lookback_2 | majority | macro_f1 | 0.257321 | experiments/lookback_2 |
| lookback_2 | logistic | accuracy | 0.623310 | experiments/lookback_2 |
| lookback_2 | logistic | macro_f1 | 0.351469 | experiments/lookback_2 |
| lookback_2 | deeplob_style | accuracy | 0.682585 | experiments/lookback_2 |
| lookback_2 | deeplob_style | macro_f1 | 0.499404 | experiments/lookback_2 |
| lookback_4 | majority | accuracy | 0.628615 | experiments/lookback_4 |
| lookback_4 | majority | macro_f1 | 0.257321 | experiments/lookback_4 |
| lookback_4 | logistic | accuracy | 0.623310 | experiments/lookback_4 |
| lookback_4 | logistic | macro_f1 | 0.351469 | experiments/lookback_4 |
| lookback_4 | deeplob_style | accuracy | 0.673029 | experiments/lookback_4 |
| lookback_4 | deeplob_style | macro_f1 | 0.470084 | experiments/lookback_4 |

## Warnings And Limitations

- Ablation rows do not claim profitability, tradable alpha or live execution. Execution-aware values are simplified proxy assumptions.
- Aggregated values come directly from stored child-experiment artefacts and are not edited after the run.

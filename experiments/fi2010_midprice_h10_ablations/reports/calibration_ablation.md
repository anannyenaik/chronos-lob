# Calibration-bin Ablation

Ablation type: `calibration_bins`

## Purpose

Vary only the calibration bin count used to compute reliability evidence from stored predictions; everything else is held fixed.

## What Changed

- `calibration_bins_5`: Reliability bins computed with 5 bins instead of the configured default.
  - parameters: `calibration.n_bins=5`
- `calibration_bins_10`: Reliability bins computed with 10 bins so the bin resolution can be compared against coarser settings.
  - parameters: `calibration.n_bins=10`

## Held Fixed

- base FI-2010 benchmark preparation config, data path, seed and split design
- preprocessing fit on train rows only
- model registry (no model family added by ablations)
- experiment artefact contract for each child experiment (validated before this report was written)

## Artefacts Used

- `ablation_summary.json`
- `ablation_results.csv`
- `ablation_manifest.json`
- `experiments/calibration_bins_5`
- `experiments/calibration_bins_10`

## Status

- `calibration_bins_5`: run (experiment: `experiments/calibration_bins_5`)
- `calibration_bins_10`: run (experiment: `experiments/calibration_bins_10`)

## Key Metric Summary

| ablation | model | metric | value | source |
| --- | --- | --- | --- | --- |
| calibration_bins_5 | majority | accuracy | 0.628615 | experiments/calibration_bins_5 |
| calibration_bins_5 | majority | macro_f1 | 0.257321 | experiments/calibration_bins_5 |
| calibration_bins_5 | logistic | accuracy | 0.623310 | experiments/calibration_bins_5 |
| calibration_bins_5 | logistic | macro_f1 | 0.351469 | experiments/calibration_bins_5 |
| calibration_bins_5 | deeplob_style | accuracy | 0.567174 | experiments/calibration_bins_5 |
| calibration_bins_5 | deeplob_style | macro_f1 | 0.352432 | experiments/calibration_bins_5 |
| calibration_bins_10 | majority | accuracy | 0.628615 | experiments/calibration_bins_10 |
| calibration_bins_10 | majority | macro_f1 | 0.257321 | experiments/calibration_bins_10 |
| calibration_bins_10 | logistic | accuracy | 0.623310 | experiments/calibration_bins_10 |
| calibration_bins_10 | logistic | macro_f1 | 0.351469 | experiments/calibration_bins_10 |
| calibration_bins_10 | deeplob_style | accuracy | 0.567174 | experiments/calibration_bins_10 |
| calibration_bins_10 | deeplob_style | macro_f1 | 0.352432 | experiments/calibration_bins_10 |

## Warnings And Limitations

- Ablation rows do not claim profitability, tradable alpha or live execution. Execution-aware values are simplified proxy assumptions.
- Aggregated values come directly from stored child-experiment artefacts and are not edited after the run.

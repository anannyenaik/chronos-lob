# Cost-Sensitivity Ablation

Ablation type: `cost_bps`

## Purpose

Vary only the per-trade cost assumption used by the simplified execution-aware sensitivity analysis; everything else is held fixed. These are explicit proxy assumptions, not live-execution results.

## What Changed

- `cost_0bps`: Execution-aware sensitivity reported with a single 0 bps cost level so the net signal proxy equals the gross proxy.
  - parameters: `execution_sensitivity.cost_bps=[0.0]`
- `cost_1bps`: Execution-aware sensitivity reported with a single 1 bps cost level under explicit cost assumptions.
  - parameters: `execution_sensitivity.cost_bps=[1.0]`

## Held Fixed

- base FI-2010 benchmark preparation config, data path, seed and split design
- preprocessing fit on train rows only
- model registry (no model family added by ablations)
- experiment artefact contract for each child experiment (validated before this report was written)

## Artefacts Used

- `ablation_summary.json`
- `ablation_results.csv`
- `ablation_manifest.json`
- `experiments/cost_0bps`
- `experiments/cost_1bps`

## Status

- `cost_0bps`: run (experiment: `experiments/cost_0bps`)
- `cost_1bps`: run (experiment: `experiments/cost_1bps`)

## Key Metric Summary

| ablation | model | metric | value | source |
| --- | --- | --- | --- | --- |
| cost_0bps | majority | accuracy | 0.591166 | experiments/cost_0bps |
| cost_0bps | majority | macro_f1 | 0.247687 | experiments/cost_0bps |
| cost_0bps | logistic | accuracy | 0.594343 | experiments/cost_0bps |
| cost_0bps | logistic | macro_f1 | 0.332821 | experiments/cost_0bps |
| cost_0bps | deeplob_style | accuracy | 0.591895 | experiments/cost_0bps |
| cost_0bps | deeplob_style | macro_f1 | 0.252138 | experiments/cost_0bps |
| cost_0bps | transformer | accuracy | 0.488931 | experiments/cost_0bps |
| cost_0bps | transformer | macro_f1 | 0.430628 | experiments/cost_0bps |
| cost_1bps | majority | accuracy | 0.591166 | experiments/cost_1bps |
| cost_1bps | majority | macro_f1 | 0.247687 | experiments/cost_1bps |
| cost_1bps | logistic | accuracy | 0.594343 | experiments/cost_1bps |
| cost_1bps | logistic | macro_f1 | 0.332821 | experiments/cost_1bps |
| cost_1bps | deeplob_style | accuracy | 0.591895 | experiments/cost_1bps |
| cost_1bps | deeplob_style | macro_f1 | 0.252138 | experiments/cost_1bps |
| cost_1bps | transformer | accuracy | 0.488931 | experiments/cost_1bps |
| cost_1bps | transformer | macro_f1 | 0.430628 | experiments/cost_1bps |

## Warnings And Limitations

- Ablation rows do not present trading or live-execution claims. Execution-aware values are simplified proxy assumptions.
- Aggregated values come directly from stored child-experiment artefacts and are not edited after the run.

# Latency-Sensitivity Ablation

Ablation type: `latency_steps`

## Purpose

Vary only the row-step latency assumption used by the simplified execution-aware sensitivity analysis; everything else is held fixed. These are explicit proxy assumptions, not live-execution results.

## What Changed

- `latency_0`: Execution-aware sensitivity with a single 0-row latency assumption.
  - parameters: `execution_sensitivity.latency_steps=[0]`
- `latency_1`: Execution-aware sensitivity with a 1-row latency assumption: realised proxy returns are pessimistically shifted by one row.
  - parameters: `execution_sensitivity.latency_steps=[1]`

## Held Fixed

- base FI-2010 benchmark preparation config, data path, seed and split design
- preprocessing fit on train rows only
- model registry (no model family added by ablations)
- experiment artefact contract for each child experiment (validated before this report was written)

## Artefacts Used

- `ablation_summary.json`
- `ablation_results.csv`
- `ablation_manifest.json`
- `experiments/latency_0`
- `experiments/latency_1`

## Status

- `latency_0`: run (experiment: `experiments/latency_0`)
- `latency_1`: run (experiment: `experiments/latency_1`)

## Key Metric Summary

| ablation | model | metric | value | source |
| --- | --- | --- | --- | --- |
| latency_0 | majority | accuracy | 0.591166 | experiments/latency_0 |
| latency_0 | majority | macro_f1 | 0.247687 | experiments/latency_0 |
| latency_0 | logistic | accuracy | 0.594343 | experiments/latency_0 |
| latency_0 | logistic | macro_f1 | 0.332821 | experiments/latency_0 |
| latency_0 | deeplob_style | accuracy | 0.591895 | experiments/latency_0 |
| latency_0 | deeplob_style | macro_f1 | 0.252138 | experiments/latency_0 |
| latency_0 | transformer | accuracy | 0.488931 | experiments/latency_0 |
| latency_0 | transformer | macro_f1 | 0.430628 | experiments/latency_0 |
| latency_1 | majority | accuracy | 0.591166 | experiments/latency_1 |
| latency_1 | majority | macro_f1 | 0.247687 | experiments/latency_1 |
| latency_1 | logistic | accuracy | 0.594343 | experiments/latency_1 |
| latency_1 | logistic | macro_f1 | 0.332821 | experiments/latency_1 |
| latency_1 | deeplob_style | accuracy | 0.591895 | experiments/latency_1 |
| latency_1 | deeplob_style | macro_f1 | 0.252138 | experiments/latency_1 |
| latency_1 | transformer | accuracy | 0.488931 | experiments/latency_1 |
| latency_1 | transformer | macro_f1 | 0.430628 | experiments/latency_1 |

## Warnings And Limitations

- Ablation rows do not present trading or live-execution claims. Execution-aware values are simplified proxy assumptions.
- Aggregated values come directly from stored child-experiment artefacts and are not edited after the run.

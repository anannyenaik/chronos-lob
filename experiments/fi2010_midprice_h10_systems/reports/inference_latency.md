# Inference Latency

## Purpose

Measure tiny CPU neural forward-pass latency per window.

## Measurement Method

The benchmark builds a small supervised matrix transformer and split-local sequence windows using existing dataset utilities, then times repeated CPU forward passes with gradients disabled.

## Input Data Source

- path: `data/processed/fi2010/fold1_combined.csv`
- data source kind: `local_file`
- benchmark set: `standard`

## Metrics

| Metric | Value | Unit | Status | Warning |
| --- | ---: | --- | --- | --- |
| windows_measured | 191970 | windows | run |  |
| elapsed_seconds | 0.409144 | seconds | run |  |
| latency_seconds_per_window | 2.13129e-06 | seconds/window | run |  |
| latency_ms_per_window | 0.00213129 | ms/window | run |  |

## Limitations

- The model is not trained for this benchmark section.
- This is inference-path latency, not a production latency claim.

## Smoke Fixture Measurement

No. The run used a local benchmark path supplied to this command. Interpret metrics only with the recorded environment and input provenance.

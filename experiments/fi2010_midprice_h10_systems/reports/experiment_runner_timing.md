# Experiment-Runner Timing

## Purpose

Measure wall-clock time for a small paper-runner invocation.

## Measurement Method

The benchmark runs `run_paper_experiment` into `child_experiments/paper_runner_timing` and validates the child experiment artefact directory before reporting timing metrics.

## Input Data Source

- path: `data/processed/fi2010/fold1_combined.csv`
- data source kind: `local_file`
- benchmark set: `standard`

## Metrics

| Metric | Value | Unit | Status | Warning |
| --- | ---: | --- | --- | --- |
| elapsed_seconds | 12.5619 | seconds | run |  |
| models_requested | 2 | models | run |  |
| models_run | 2 | models | run |  |
| prediction_rows | 76794 | rows | run |  |
| artefact_count | 15 | files | run |  |

## Limitations

- The child run is a real artefact-producing runner output.
- Fixture child outputs remain smoke artefacts and are not benchmark evidence.

## Smoke Fixture Measurement

No. The run used a local benchmark path supplied to this command. Interpret metrics only with the recorded environment and input provenance.

# Loader Throughput

## Purpose

Measure local FI-2010 load timing and row throughput.

## Measurement Method

The benchmark calls the existing FI-2010 loader once with the supplied config and local data path, timing only the load call with `time.perf_counter`.

## Input Data Source

- path: `data/processed/fi2010/fold1_combined.csv`
- data source kind: `local_file`
- benchmark set: `standard`

## Metrics

| Metric | Value | Unit | Status | Warning |
| --- | ---: | --- | --- | --- |
| elapsed_seconds | 2.08321 | seconds | run |  |
| rows_loaded | 77909 | rows | run |  |
| rows_per_second | 37398.5 | rows/second | run |  |

## Limitations

- Throughput is local to the recorded machine, Python environment and input file.
- Fixture timings validate code paths only and must not be compared with real runs.

## Smoke Fixture Measurement

No. The run used a local benchmark path supplied to this command. Interpret metrics only with the recorded environment and input provenance.

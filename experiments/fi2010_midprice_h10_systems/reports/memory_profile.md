# Memory Profile

## Purpose

Measure a Python-level peak memory profile for feature generation.

## Measurement Method

The benchmark runs normalised matrix feature preparation under `tracemalloc` and records the peak traced Python allocation for that section.

## Input Data Source

- path: `data/processed/fi2010/fold1_combined.csv`
- data source kind: `local_file`
- benchmark set: `standard`

## Metrics

| Metric | Value | Unit | Status | Warning |
| --- | ---: | --- | --- | --- |
| peak_memory_bytes | 91360668 | bytes | run |  |
| peak_memory_mb | 87.1283 | MiB | run |  |
| section_measured | 1 | feature_generation_section | run |  |

## Limitations

- `tracemalloc` does not capture every native allocation made by extensions.
- Treat this as a local resource profile, not a full system memory audit.

## Smoke Fixture Measurement

No. The run used a local benchmark path supplied to this command. Interpret metrics only with the recorded environment and input provenance.

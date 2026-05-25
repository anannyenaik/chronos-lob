# Feature-Generation Speed

## Purpose

Measure normalised FI-2010 matrix feature-frame preparation.

## Measurement Method

The benchmark prepares a numeric FI-2010 matrix feature frame directly from the loaded dataset and reports rows and generated feature values. Raw order-book snapshots are not reconstructed from normalised values.

## Input Data Source

- path: `data/processed/fi2010/fold1_combined.csv`
- data source kind: `local_file`
- benchmark set: `standard`

## Metrics

| Metric | Value | Unit | Status | Warning |
| --- | ---: | --- | --- | --- |
| elapsed_seconds | 0.156638 | seconds | run | feature_generation_speed measured normalised FI-2010 matrix feature throughput; raw order-book snapshot reconstruction was not used |
| rows_processed | 77909 | rows | run | feature_generation_speed measured normalised FI-2010 matrix feature throughput; raw order-book snapshot reconstruction was not used |
| features_generated | 11218896 | feature_values | run | feature_generation_speed measured normalised FI-2010 matrix feature throughput; raw order-book snapshot reconstruction was not used |
| rows_per_second | 497381 | rows/second | run | feature_generation_speed measured normalised FI-2010 matrix feature throughput; raw order-book snapshot reconstruction was not used |
| features_per_second | 7.16229e+07 | feature_values/second | run | feature_generation_speed measured normalised FI-2010 matrix feature throughput; raw order-book snapshot reconstruction was not used |

## Limitations

- Feature semantics are unchanged; labels are not introduced as features.
- The tiny fixture has too few rows to represent production-size workloads.

## Smoke Fixture Measurement

No. The run used a local benchmark path supplied to this command. Interpret metrics only with the recorded environment and input provenance.

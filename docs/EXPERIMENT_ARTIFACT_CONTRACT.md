# Experiment Artefact Contract

ChronosLOB experiment artefacts provide a strict on-disk contract for future
reproducible benchmark evidence. The contract separates configuration, local
data provenance, machine-readable metrics, row-level evidence and written
method notes so a completed run can be inspected without rerunning training.

The contract is a validation layer only. It does not download data, run
models, create plots or turn synthetic fixtures into benchmark evidence.

## Required Files

Every completed experiment directory must contain:

- `config.yaml`: full experiment configuration, including seed, split, label
  horizon, model settings, calibration settings and execution assumptions.
- `data_manifest.json`: local data provenance, source kind, source path,
  checksum where applicable, row counts where known, label name, horizon and
  split name.
- `results.json`: machine-readable results with predictive, calibration and
  execution evidence streams kept separate.
- `model_card.md`: concise methodology, data scope, leakage controls, known
  limitations and claim boundaries.

## Optional Evidence Files

The validator reports warnings when expected evidence files are absent, but
their absence does not invalidate a schema-only directory:

- `predictions.csv` or `predictions.parquet`
- `calibration_bins.csv`
- `execution_sensitivity.csv`
- `plots/reliability_curve.png`
- `plots/cost_sensitivity.png`
- `plots/confusion_matrix.png`
- `plots/regime_breakdown.png`

`predictions.csv` and `predictions.parquet` are alternatives. Either file
satisfies the prediction artefact expectation.

Plot artefacts are emitted by the Phase G plot builder
(`build-paper-plots` or `run-paper-experiment --build-plots`). Each
plot is generated from a specific stored artefact:

- `plots/reliability_curve.png` is derived from `calibration_bins.csv`.
- `plots/cost_sensitivity.png` is derived from
  `execution_sensitivity.csv`.
- `plots/confusion_matrix.png` is derived from `confusion_matrix.json`.
- `plots/regime_breakdown.png` is only generated when genuine regime
  data is available in stored artefacts; otherwise the plot is
  skipped with a clear warning and is not fabricated.

A `plot_summary.json` artefact written next to the experiment root
records the experiment directory, builder version, plots written,
plots skipped and any warnings produced during plot generation.

## Traceability Expectations

Any future benchmark metric should trace to:

- a versioned config
- a local data manifest
- a seed
- a temporal split definition
- a code commit when available
- stored predictions when row-level metrics are reported
- calibration artefacts when calibration metrics are reported
- execution-assumption artefacts when cost or latency sensitivity is reported

JSON artefacts use strict schemas. Datetimes must be timezone-aware, horizons
must be positive, seeds must be non-negative, model names must be non-empty
and metric values must be finite.

## Benchmark Evidence Boundary

Benchmark evidence means metrics produced by a completed, documented run on
locally supplied data, with matching config, manifest, results and supporting
artefacts.

Synthetic contract fixtures under `tests/fixtures/` do not count as benchmark
evidence. They exist only to exercise schema validation, CLI inspection and
failure handling.

## Inspecting A Directory

Use the read-only inspection command:

```bash
python -m chronoslob.cli inspect-experiment-artifacts --experiment tests/fixtures/experiments/minimal_valid_experiment
```

The command prints required artefacts, optional artefacts, validation status
and warnings. It writes no outputs, runs no training and performs no network
calls.

# FI-2010 Benchmark Preparation

ChronosLOB supports preparation of an FI-2010 mid-price direction
benchmark experiment from a user-supplied local file. The repository
does not ship FI-2010 data and does not download it. Users are
responsible for obtaining and licensing any benchmark copy they use.
See [FI2010_DATA_ACQUISITION.md](FI2010_DATA_ACQUISITION.md) for the
runbook covering the official Fairdata/Etsin source, the ignored local
layout, manual download steps, checksum and size verification, the
`.txt` matrix to CSV conversion path and the difference between tiny
fixture smoke runs and real benchmark runs.

This document describes the preparation step only. It is not a model
runner. The paper experiment runner consumes the artefacts produced here
to train models and write benchmark results.

## What The Preparation Step Does

`prepare-fi2010-benchmark` loads a local FI-2010-style file through the
existing loader, validates it through the existing FI-2010 validator,
constructs a horizon-10 mid-price direction label distribution from the
existing benchmark labels, computes a deterministic split summary and
writes a small set of
preparation artefacts to a chosen output directory.

It writes:

- `preparation_summary.json`: top-level preparation record with
  experiment metadata, data path, output directory, artefact map and
  warnings.
- `data_manifest.json`: portable data provenance under the experiment
  artefact contract, including row count, label name, horizon, split
  name, dataset variant and SHA-256 checksum of the source file.
- `label_summary.json`: distinct label classes, per-class counts and
  per-class proportions for the configured target column.
- `split_summary.json`: train/validation/test row counts and contiguous
  index bounds. For the real combined FI-2010 fold this records the
  official train/test split and the internal validation tail carved
  from official train rows.
- `validation_summary.json`: combined FI-2010 dataset validation and
  label-frame validation status.
- `config.yaml`: a copy of the source config so the preparation output
  is self-describing.

## What The Preparation Step Does Not Do

- It does not download FI-2010, FI-2010 mirrors or any market data.
- It does not commit FI-2010 data to the repository.
- It does not train any model.
- It does not write `results.json`, `predictions.csv`, calibration
  bins, execution sensitivity records or any plot.
- It does not fit scalers, quantile boundaries, calibrators or
  model-selection choices on validation or test data.
- It does not perform live trading, broker integration or order
  placement.
- It does not perform network calls.

## Configuration

The default preparation config lives at
`configs/experiments/fi2010_midprice_h10.yaml`. The `local_data_path`
field is a template value: supply the real local file with
`--data-path`, or replace the config field before a paper experiment
run.

The config now supports two split methods. The generic temporal method
keeps the earlier row-order train/validation/test behaviour. The
`official_column` method is used for the real FI-2010 combined CSV:
rows with `split=train` form the official training partition, validation
is carved from the tail of that partition only, and rows with
`split=test` form the held-out test partition. Official test rows are
not used for preprocessing, fitting, validation or model-selection
decisions.

## Smoke Command

The bundled FI-2010-like fixture under `tests/fixtures/fi2010` exists
only to exercise the preparation path. It is not the canonical benchmark
and does not represent benchmark performance.

```bash
python -m chronoslob.cli prepare-fi2010-benchmark \
  --config configs/experiments/fi2010_midprice_h10.yaml \
  --data-path tests/fixtures/fi2010/tiny_fi2010_like.csv \
  --out runs/fi2010_midprice_h10_prepare
```

`runs/` is in `.gitignore`, so preparation outputs are not committed.

## Difference From A Completed Experiment

A completed experiment under the artefact contract additionally
contains `results.json`, `model_card.md` and (optionally) row-level
predictions, calibration bins, execution sensitivity records and
plots. A preparation directory only contains preparation artefacts and
is intentionally not a complete experiment directory.

The validator at `inspect-experiment-artifacts` will report missing
required artefacts when run against a preparation directory because
`results.json` and `model_card.md` are intentionally absent.

## Use In Paper Experiments

The paper experiment runner:

- consume `data_manifest.json` and `split_summary.json` as part of the
  experiment provenance,
- train the model list referenced in the config under the same split
  policy,
- write `results.json`, `predictions.*`, `model_card.md` and other
  evidence artefacts into the selected output directory.

## Limitations Of FI-2010 As A Benchmark

FI-2010 is a normalised, snapshot-style benchmark. It is widely used
in the limit-order-book literature but has known limitations: it is
pre-normalised so true prices are not recoverable from the matrix, it
covers a fixed set of instruments and time periods, and the existing
benchmark labels follow a specific labelling protocol that should be
disclosed in any reported metric. Any claim using FI-2010 must
document the label horizon, the train/test split and the loader
configuration used.

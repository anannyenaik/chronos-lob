# Experiment Registry

Phase 5 adds a lightweight experiment registry skeleton. It records what was
attempted; it does not report performance, create model artefacts or imply that
any forecasting model exists.

## Metadata Captured

`chronoslob.training.experiment.ExperimentMetadata` records:

- run id and readable run name;
- timezone-aware UTC creation time;
- project name and phase;
- deterministic seed;
- current git commit when available;
- config path;
- input paths;
- output path;
- optional notes.

The git commit lookup is best effort. If git is unavailable, metadata creation
continues with `git_commit` set to `null`.

## Run Directory Layout

`initialise_experiment_run` creates a run directory under the chosen root with:

- `metadata.json`;
- `configs/`;
- `artifacts/`;
- `logs/`;
- `tables/`.

If a local config path is supplied, the file is copied into `configs/` and the
metadata records the copied path. Generated run directories belong under
`runs/`, which is ignored by git.

## Config Capture

Configs in `configs/experiments/` should describe data paths, split settings,
seeds, leakage controls and output locations. Phase 5 includes
`fi2010_split_audit.yaml` as a split-audit example. It contains no model section
and no result target.

## What Is Not Recorded Yet

Models, training loops, checkpoints, calibration reports, forecasts, backtests
and metric tables do not exist in this phase. The registry therefore does not
create or expect those artefacts.

## No Fake Metrics Policy

No manually invented metrics, plots, result tables or notebooks should be added.
Any future reported result must be generated from a reproducible run with a
captured config, seed, code version, input paths and output paths.

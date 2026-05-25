# Paper Ablation Suite

The paper ablation suite runs controlled robustness and assumption
sensitivity checks on top of the existing paper experiment runner. It
answers where a stored signal is stable under changed assumptions and
where it breaks, without adding new model families or inventing
missing evidence.

The suite is local-only. It requires an explicit local FI-2010-style
data path, never downloads data and writes traceable artefacts for
each child experiment that genuinely runs.

## Why Ablations Exist

A single paper experiment records one configuration. The ablation
suite reruns that same experiment while changing one assumption at a
time:

- calibration-bin count
- execution cost assumption
- latency assumption
- horizon and matching label column
- neural lookback window
- feature group
- SSL pretraining status, recorded as skipped until runner support is
  traceable

This keeps robustness analysis separate from model training logic. All
fitted models still come from `run-paper-experiment`.

## Supported Sets

`smoke` is the lightweight validation set for tests and synthetic
fixture checks. It runs:

- `baseline`
- `calibration_bins_5`
- `cost_0bps`
- `cost_1bps`
- `ssl_pretraining_ablation`, recorded as skipped

`standard` is intended for real local FI-2010 runs. It extends the
smoke set with additional calibration, cost, latency, horizon,
lookback and feature-group ablations. It is still config-driven: it
uses the model list supplied on the command line and skips lookback
ablations when no neural model is requested.

## Supported Ablation Types

Calibration-bin ablations override `calibration.n_bins` and rerun the
paper experiment so reliability-bin artefacts are rebuilt from stored
held-out predictions.

Cost-sensitivity ablations override
`execution_sensitivity.cost_bps`. Latency-sensitivity ablations
override `execution_sensitivity.latency_steps`. These are simplified
proxy assumptions recorded in `execution_sensitivity.csv`; they are
not production execution results.

Horizon ablations override `horizon` and the matching `label_name`
where the supplied FI-2010-style file has that label column.

Lookback ablations override `neural_settings.lookback` and apply only
to requested neural paper-runner models. Classical-only runs record
these ablations as skipped.

Feature-group ablations use deterministic column-name patterns after
the runner has excluded labels, split columns, timestamp columns and
future-label-like columns. Supported groups are `all`,
`top_of_book`, `imbalance` and `depth_liquidity`. If too few matching
features exist, the ablation is skipped with a clear warning rather
than producing a child experiment. The `all` group is represented by
the baseline child experiment.

SSL pretraining is always skipped in this phase. The reason is: no
traceable runner support for SSL pretraining/fine-tuning yet. Before
enabling it, the runner needs train-only pretraining, a stored
pretraining config, a checkpoint or weight-transfer trace, a
fine-tuning config and held-out evaluation.

## Output Layout

```text
runs/paper_ablation_smoke/
  ablation_summary.json
  ablation_results.csv
  ablation_manifest.json
  reports/
    calibration_ablation.md
    cost_sensitivity.md
    ssl_pretraining_ablation.md
  experiments/
    baseline/
    calibration_bins_5/
    cost_0bps/
    cost_1bps/
```

The `standard` set may write additional reports and child experiment
directories, depending on which ablations run successfully.

Skipped ablations appear in `ablation_summary.json`,
`ablation_results.csv`, `ablation_manifest.json` and the relevant
Markdown report, but they do not get fabricated child experiment
directories.

## Smoke Command

The bundled tiny file is a synthetic fixture smoke run only. It is not
benchmark evidence.

```bash
python -m chronoslob.cli run-paper-ablations \
  --config configs/experiments/fi2010_midprice_h10.yaml \
  --data-path tests/fixtures/fi2010/tiny_fi2010_like.csv \
  --out runs/paper_ablation_smoke \
  --models majority,logistic \
  --ablation-set smoke \
  --overwrite
```

## Real Local FI-2010 Usage

When a local FI-2010-style file is available, replace `--data-path`
with that file and choose a model list suitable for the run budget:

```bash
python -m chronoslob.cli run-paper-ablations \
  --config configs/experiments/fi2010_midprice_h10.yaml \
  --data-path /local/path/to/fi2010_normalised.csv \
  --out runs/paper_ablation_standard \
  --models majority,logistic,ridge,elastic_net,random_forest,gradient_boosting \
  --ablation-set standard \
  --overwrite
```

Neural models can be included explicitly, for example
`majority,deeplob_style,transformer`, but the suite does not force
long neural runs by default. The repository does not ship FI-2010 data
and does not download it.

## Use In Empirical Reports

`build-paper-report` can include a completed ablation directory through
`--ablations PATH`. The report uses `ablation_summary.json`,
`ablation_results.csv`, `ablation_manifest.json` and any stored ablation
Markdown reports. Run and skipped ablations are both shown; skipped SSL
pretraining status remains a status record, not a model result.

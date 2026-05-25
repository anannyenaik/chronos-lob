# Paper Reports

ChronosLOB includes an empirical report builder that turns stored paper
experiment, ablation and systems benchmark artefacts into one structured
Markdown report. The builder reads artefacts already on disk, validates the
required paper experiment contract and records unavailable optional evidence as
not available or skipped.

The report builder does not run models, build plots, download data or create
metrics that are absent from the supplied artefact directories.
It also normalises the Markdown structure it writes: headings, paragraphs,
tables and code fences are separated by blank lines so the report renders
cleanly on GitHub.

## Inputs

The required input is a completed paper experiment directory containing the
standard artefacts from `run-paper-experiment`:

- `config.yaml`
- `data_manifest.json`
- `results.json`
- `model_card.md`
- `runner_summary.json`
- optional evidence artefacts such as `predictions.csv`,
  `calibration_bins.csv`, `execution_sensitivity.csv` and `plots/`

Optional inputs are:

- a paper ablation directory from `run-paper-ablations`
- a systems benchmark directory from `run-system-benchmarks`

When an optional directory is not supplied, the corresponding report section is
kept and marked as not supplied.

## Outputs

`build-paper-report` writes:

- the Markdown empirical report at the requested `--out` path
- a companion JSON build summary named `<report_stem>_summary.json`

The summary JSON records the report path, input directories, sections written,
artefacts used, warnings and whether the inputs look like a fixture or smoke
run.

## Smoke Command

The bundled FI-2010-like fixture is for smoke validation only. Reports built
from it should stay under ignored output paths such as `runs/`.

```bash
python -m chronoslob.cli build-paper-report \
  --experiment runs/paper_experiment_plots_smoke \
  --ablations runs/paper_ablation_smoke \
  --systems runs/system_benchmark_smoke \
  --out runs/chronoslob_empirical_report_smoke.md \
  --overwrite
```

Inspect a generated report with:

```bash
python -m chronoslob.cli inspect-paper-report \
  --report runs/chronoslob_empirical_report_smoke.md
```

## Real Local FI-2010 Usage

For a real local FI-2010 workflow, first produce stored artefacts with the
paper experiment runner, ablation suite and systems benchmark suite using a
user-supplied local FI-2010-style file. Then build the report from those stored
directories, for example:

```bash
python -m chronoslob.cli build-paper-report \
  --experiment runs/fi2010_midprice_h10 \
  --ablations runs/paper_ablation_standard \
  --systems runs/system_benchmark_standard \
  --out reports/chronoslob_empirical_report.md \
  --overwrite
```

The repository does not ship FI-2010 data and does not fetch it. A report under
`reports/` should be based on real local benchmark artefacts, not fixture smoke
outputs.

## Report Contents

The generated report contains:

- dataset and provenance
- label construction
- leakage controls, official split-aware evaluation details and temporal
  validation metadata where applicable
- models
- predictive results
- calibration results
- execution-aware sensitivity and cost-aware signal quality
- ablations and robustness
- systems benchmarks
- a warning summary plus a detailed warning appendix, with repeated
  optional-plot and unsupported-SSL warnings grouped rather than repeated
  inline
- limitations
- reproducibility commands

All tables and links are derived from stored artefacts. Existing plot files are
referenced with relative Markdown links where possible; the builder does not
generate new plots.

# Reproducibility

ChronosLOB is designed as a reproducible research artefact. This page
records the canonical local validation path.

## Python and Installation

Use Python 3.11 or newer. CI runs on Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,torch]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,torch]"
```

The `torch` extra is optional for installation but required to run the
full test suite locally.

## Local Validation Commands

```bash
python -c "import chronoslob; print(chronoslob.__version__)"
python -m chronoslob.cli doctor
python -m chronoslob.cli inspect-release-readiness
python -m chronoslob.cli run-project-audit --strict
python -m pytest
python -m compileall -q chronoslob tests
python -m ruff check .
python -m mypy chronoslob
```

`make` targets exist for convenience but `make` may be unavailable on
Windows. The Python commands above are the canonical cross-platform
validation path.

## Determinism

Tests and CLI commands use explicit seeds where randomness is
involved. Torch code paths run on CPU by default and use deterministic
settings where practical. Financial time-series splitting is temporal
by default; random splits are not used for core experiments.

## Local Smoke Commands

These exercise infrastructure on bundled synthetic fixtures. Their
outputs validate code paths only; see
[SAFETY_AND_LIMITATIONS.md](SAFETY_AND_LIMITATIONS.md) for what these
outputs are not.

```bash
python -m chronoslob.cli inspect-fi2010 --path tests/fixtures/fi2010/tiny_fi2010_like.csv
python -m chronoslob.cli run-baseline-smoke --path tests/fixtures/fi2010/tiny_fi2010_like.csv
python -m chronoslob.cli run-deeplob-smoke --path tests/fixtures/fi2010/tiny_fi2010_like.csv --lookback 2 --epochs 1
python -m chronoslob.cli inspect-event-log --path tests/fixtures/event_logs/synthetic_snapshots.jsonl
python -m chronoslob.cli run-transformer-smoke --path tests/fixtures/event_logs/synthetic_snapshots.jsonl
python -m chronoslob.cli run-ssl-smoke --path tests/fixtures/event_logs/synthetic_snapshots.jsonl
python -m chronoslob.cli run-multitask-smoke --path tests/fixtures/event_logs/synthetic_snapshots.jsonl
python -m chronoslob.cli run-calibration-smoke
python -m chronoslob.cli run-execution-validation-smoke
python -m chronoslob.cli run-robustness-analysis-smoke
python -m chronoslob.cli run-paper-experiment --config configs/experiments/fi2010_midprice_h10.yaml --data-path tests/fixtures/fi2010/tiny_fi2010_like.csv --out runs/paper_experiment_plots_smoke --models majority,logistic,deeplob_style,transformer --overwrite --build-plots
python -m chronoslob.cli run-paper-ablations --config configs/experiments/fi2010_midprice_h10.yaml --data-path tests/fixtures/fi2010/tiny_fi2010_like.csv --out runs/paper_ablation_smoke --models majority,logistic --ablation-set smoke --overwrite
python -m chronoslob.cli run-system-benchmarks --config configs/experiments/fi2010_midprice_h10.yaml --data-path tests/fixtures/fi2010/tiny_fi2010_like.csv --out runs/system_benchmark_smoke --benchmark-set smoke --models majority,logistic --overwrite
python -m chronoslob.cli build-paper-report --experiment runs/paper_experiment_plots_smoke --ablations runs/paper_ablation_smoke --systems runs/system_benchmark_smoke --out runs/chronoslob_empirical_report_smoke.md --overwrite
```

## FI-2010 Benchmark Preparation

Real FI-2010 data is not committed to the repository and is never
downloaded automatically. See
[FI2010_BENCHMARK.md](FI2010_BENCHMARK.md) for the local-only
preparation step that produces a data manifest, label distribution
summary and temporal split summary from a user-supplied file before
any future paper experiment run.

## Paper Experiment Runner

The paper experiment runner consumes the same FI-2010 preparation
config and a local data path and writes a validated experiment
artefact directory in one command. Phase E supports `majority`,
`logistic`, `ridge`, `elastic_net`, `random_forest`,
`gradient_boosting`, `deeplob_style` and `transformer`. Multiple
models can be selected with a comma-separated `--models` list and the
`majority` baseline must always be included as the interpretable
floor. Neural settings are controlled by the `neural_settings` section
of `configs/experiments/fi2010_midprice_h10.yaml` and default to CPU
smoke-safe values.

Phase F additionally emits two evidence artefacts from stored
predictions: `calibration_bins.csv` (per-model reliability bins on
held-out test rows) and `execution_sensitivity.csv` (cost-aware
signal-quality rows over configured confidence thresholds, cost
levels in basis points and latency steps). Calibration and execution
sensitivity are configured under the `calibration` and
`execution_sensitivity` blocks of the same config. Synthetic fixture
smoke runs (under `tests/fixtures/fi2010`) exist only to exercise the
runner plumbing and are not benchmark evidence; in particular the
execution-sensitivity proxy is typically undefined on the tiny
fixture because the forward horizon exceeds the available rows. Real
benchmark evidence requires a local FI-2010 path and stored artefacts
produced by this runner. See
[PAPER_EXPERIMENTS.md](PAPER_EXPERIMENTS.md) for the full contract,
metric groups, leakage controls and the tiny fixture smoke command.

Phase G adds optional deterministic plot generation from the stored
artefacts. Pass `--build-plots` to `run-paper-experiment` to generate
plots in the same run, or invoke `build-paper-plots --experiment PATH`
on a completed directory. The plot builder writes
`plots/reliability_curve.png` from `calibration_bins.csv`,
`plots/cost_sensitivity.png` from `execution_sensitivity.csv`,
`plots/confusion_matrix.png` from `confusion_matrix.json`, and
`plots/regime_breakdown.png` only when genuine regime data is
available in stored artefacts. A `plot_summary.json` artefact
records plots written, plots skipped and any warnings. Synthetic
fixture smoke runs remain synthetic fixture smoke runs even with
plots attached; they are not benchmark evidence.
`inspect-paper-experiment --experiment PATH` prints a concise,
read-only summary of the directory (artefact validation, evidence
streams, row counts, plot inventory and fixture flag).

Phase H adds `run-paper-ablations` for controlled robustness analysis
and assumption sensitivity. The smoke set runs baseline,
calibration-bin and cost-sensitivity ablations on the synthetic
fixture and records SSL pretraining as skipped because there is no
traceable runner support for SSL pretraining/fine-tuning yet. The
command writes `ablation_summary.json`, `ablation_results.csv`,
`ablation_manifest.json`, Markdown reports and child paper experiment
directories only for ablations that genuinely run. Synthetic fixture
smoke runs are not benchmark evidence. See
[PAPER_ABLATIONS.md](PAPER_ABLATIONS.md) for the output layout and
the standard local FI-2010 usage pattern.

Phase I adds `run-system-benchmarks` for local systems benchmark
measurements. The command records loader throughput, feature-generation
speed, paper experiment-runner timing, inference latency and a small
resource profile, together with `environment.json` and input provenance.
Smoke runs on the bundled FI-2010-like fixture validate infrastructure
only and are not benchmark evidence. See
[SYSTEM_BENCHMARKS.md](SYSTEM_BENCHMARKS.md) for the output layout,
smoke command and local FI-2010 usage pattern.

Phase J adds `build-paper-report` for evidence-backed empirical reports.
The command reads stored paper experiment artefacts and optional ablation
and systems directories, then writes a Markdown report plus
`<report_stem>_summary.json`. Missing optional artefacts are marked as
not available or skipped. Fixture smoke reports should stay under ignored
paths such as `runs/`; a report under `reports/` should be based on real
local benchmark artefacts. See [PAPER_REPORTS.md](PAPER_REPORTS.md) for
the smoke command and local FI-2010 usage pattern.

## Evidence Archive

A local evidence archive of inventories, current command outputs and
Mermaid diagrams can be rebuilt with:

```bash
python -m chronoslob.cli build-report-archive
python -m chronoslob.cli inspect-report-archive
```

`--include-smoke-training` captures short synthetic training commands
in addition to the default lightweight inspections.

## Reporting Rule

Any reported metric must trace to a versioned config, data source,
seed, code commit and stored output. See
[SAFETY_AND_LIMITATIONS.md](SAFETY_AND_LIMITATIONS.md) for the broader
reporting boundaries.

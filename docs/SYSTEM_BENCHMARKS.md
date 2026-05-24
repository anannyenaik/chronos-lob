# Systems Benchmarks

ChronosLOB includes a local systems benchmark suite for measuring research
platform behaviour under explicit machine, Python and input-data conditions.
It is separate from forecast-quality experiments: it records loader
throughput, feature-generation speed, experiment-runner timing, inference
latency and a small resource profile.

The suite does not download FI-2010 data and does not report public
benchmark claims from the bundled fixture.

## Supported Sets

`smoke` is the fast validation set for tests and fixture checks. It is
intended to confirm that the benchmark infrastructure writes complete
artefacts and finite timing rows.

`standard` is intended for a user-supplied local FI-2010-style path. It
uses the same categories but remains controlled by the supplied config and
model list. It does not force long neural training by default.

Unsupported set names fail clearly before outputs are written.

## Output Layout

```text
runs/system_benchmark_smoke/
  system_benchmark_summary.json
  system_benchmark_results.csv
  environment.json
  reports/
    loader_throughput.md
    feature_generation_speed.md
    experiment_runner_timing.md
    inference_latency.md
    memory_profile.md
  child_experiments/
    paper_runner_timing/
```

Reports are written for measured categories and for explicit skipped
categories, with the skip reason recorded in both CSV and Markdown form.
The child experiment created for experiment-runner timing is a real
paper-runner output and is validated under the experiment artefact contract.

## Metrics Collected

- `loader_throughput`: elapsed seconds, rows loaded and rows per second.
- `feature_generation_speed`: elapsed seconds, rows processed, generated
  feature values and local throughput.
- `experiment_runner_timing`: elapsed seconds, models requested, models run,
  prediction rows and artefact count.
- `inference_latency`: tiny CPU forward-pass windows, elapsed seconds and
  inference latency per window when PyTorch is available.
- `memory_profile`: `tracemalloc` peak bytes and MiB for feature generation
  where the measurement can be isolated.

`system_benchmark_results.csv` uses deterministic ordering and records
skipped rows with a warning rather than missing or non-finite values.

## Smoke Versus Local Measurements

The bundled file under `tests/fixtures/fi2010` is a synthetic fixture for
smoke validation only. Measurements from that path are smoke measurements;
they are not benchmark evidence and are not representative of a local
FI-2010 benchmark path.

Local benchmark measurements require an explicit local FI-2010-style file
and should be interpreted only with `environment.json`, input provenance,
the selected benchmark set and the model list.

## Smoke Command

```bash
python -m chronoslob.cli run-system-benchmarks \
  --config configs/experiments/fi2010_midprice_h10.yaml \
  --data-path tests/fixtures/fi2010/tiny_fi2010_like.csv \
  --out runs/system_benchmark_smoke \
  --benchmark-set smoke \
  --models majority,logistic \
  --overwrite
```

Inspect the generated artefacts with:

```bash
python -m chronoslob.cli inspect-system-benchmarks \
  --benchmark runs/system_benchmark_smoke
```

## Local FI-2010 Usage

When a local FI-2010-style file is available, replace `--data-path` and
choose the benchmark set and model list that match the available run budget:

```bash
python -m chronoslob.cli run-system-benchmarks \
  --config configs/experiments/fi2010_midprice_h10.yaml \
  --data-path /local/path/to/fi2010_normalised.csv \
  --out runs/system_benchmark_standard \
  --benchmark-set standard \
  --models majority,logistic \
  --overwrite
```

The repository does not ship FI-2010 data and does not fetch it.

## Limitations

- Timings are local to the recorded environment, CPU, Python version and
  input file.
- Inference latency is a tiny CPU inference-path measurement, not a
  production-performance claim.
- The resource profile uses standard-library tracing and does not capture
  every native allocation from numerical libraries.
- The suite records systems measurements only; forecast quality, calibration
  quality and execution-assumption evidence remain in the paper experiment
  artefacts.

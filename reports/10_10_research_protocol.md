# 10/10 Research Protocol Note

This note is an implementation-facing companion to the public protocol in
[docs/RESEARCH_PROTOCOL.md](../docs/RESEARCH_PROTOCOL.md). It is written for
the repository maintainers, not as user-facing prose.

## Why The Upgrade Is Needed

The repository currently demonstrates strong research engineering rigour. It
does not yet demonstrate a multi-fold, multi-seed empirical study on FI-2010,
which is the standard expected for a 10/10 limit order book benchmark study.
This note records the gap and the ordered work required to close it without
weakening the existing leakage-safety, traceability or public-claim
boundaries.

## Current Evidence Baseline

The committed FI-2010 evidence covers a single official NoAuction ZScore
fold (fold 1) with majority, logistic, random forest, gradient boosting,
DeepLOB-style and normalised-matrix transformer baselines. Calibration and
execution-aware sensitivity are present, ablations are present and skipped
ablations carry explicit reasons. Systems benchmarks are present for the
recorded environment.

Authoritative pointers:

- [docs/EXPERIMENT_EVIDENCE_INDEX.md](../docs/EXPERIMENT_EVIDENCE_INDEX.md)
- [docs/REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md)
- [reports/chronoslob_empirical_report.md](chronoslob_empirical_report.md)

## Gap To 10/10

The remaining gap is empirical breadth and statistical credibility, not
infrastructure. Concretely:

- Only one official FI-2010 fold is evaluated; the protocol requires at
  least folds one through five.
- Each model is evaluated on a single seed; the protocol requires at least
  three seeds per configuration to support cross-seed variance reporting.
- Self-supervised pretraining remains an explicit skip; the protocol gates
  any self-supervised result on a traceable train-only pretraining and
  fine-tuning path inside the runner.
- The external benchmark comparison table is not yet built from stored
  multi-fold artefacts.

## Ordered Upgrade Plan

The upgrade is broken into stages. Each stage is independently verifiable and
must not regress earlier stages.

### Stage 1: Freeze the protocol layer

- Add [docs/RESEARCH_PROTOCOL.md](../docs/RESEARCH_PROTOCOL.md).
- Add this note.
- Add the multi-fold config skeleton at
  [configs/experiments/fi2010_multifold.yaml](../configs/experiments/fi2010_multifold.yaml).
- Wire light tests that verify these files exist, parse and stay within the
  documented claim boundaries.

Acceptance criteria:

- Both protocol files exist and are linked from the evidence index and
  reports README.
- The multi-fold config parses as YAML and exposes the required top-level
  keys.
- The forbidden-claims audit continues to pass.
- No raw data, processed FI-2010 CSV files or large artefacts are staged.

### Stage 2: Multi-fold execution evidence

- Add a multi-fold orchestration entry point that consumes the multi-fold
  config and runs classical models per prepared fold.
- Persist per-fold artefacts under
  `experiments/fi2010_multifold_classical/folds/fold_{n}/` following the
  classical multi-fold artefact contract.
- Persist a top-level multi-fold manifest summarising fold-level metrics.

Stage 2a (preparation layer, done):

- Add `chronoslob/experiments/fi2010_multifold.py` together with the
  `prepare-fi2010-multifold` and `inspect-fi2010-multifold` CLI commands.
- Produce one split-aware combined CSV per fold under the user-supplied
  `--processed-root`, plus per-fold manifests and a `summary.json` under
  `--out`. Combined CSVs and source matrices remain gitignored.
- Runbook: [docs/FI2010_MULTIFOLD_PROTOCOL.md](../docs/FI2010_MULTIFOLD_PROTOCOL.md).
- The preparation layer does not train models and does not produce
  multi-fold benchmark results; classical per-fold modelling is handled
  by Stage 2b.

Stage 2b (classical runner, implemented in this layer):

- Add `chronoslob/experiments/fi2010_multifold_runner.py` together with
  `run-fi2010-multifold-classical`.
- Run majority, logistic, ridge, elastic-net logistic, random forest and
  gradient boosting on prepared fold CSVs.
- Preserve the official split column, carve validation only from official
  train rows, fit preprocessing only on train rows and aggregate fold metrics.
- Emit calibration summaries and execution proxy summaries without writing
  full prediction rows by default.
- Real evidence from this stage is stored under
  [`experiments/fi2010_multifold_classical/`](../experiments/fi2010_multifold_classical/),
  covering the five official NoAuction ZScore folds at horizon
  `label_10` for the full classical baseline set with zero model failures.
  Full predictions and checkpoints are not written. Execution rows
  remain a simplified proxy under explicit cost assumptions and carry
  no live tradability claim.

Stage 2c (neural benchmark planning layer, implemented in this layer):

- Add `configs/experiments/fi2010_neural_serious.yaml`.
- Add `chronoslob/experiments/neural_benchmarking.py` and
  `inspect-fi2010-neural-plan`.
- Expand folds, seeds, supervised neural models and lookbacks into a
  deterministic plan without training.
- Define lightweight neural artefacts and per-run metadata for later runs.
- Keep checkpoint and full prediction writing disabled by default.

Stage 2d (neural benchmark execution layer, implemented in this layer):

- Add `chronoslob/experiments/fi2010_neural_runner.py` together with
  `run-fi2010-neural-benchmark`.
- Execute selected supervised neural folds, seeds, models and lookbacks
  from prepared fold CSVs.
- Preserve the official split column, carve validation only from official
  train rows, fit preprocessing only on train rows and evaluate final
  metrics on official test rows.
- Emit `summary.json`, `run_plan.csv`, `results_by_fold_seed.csv`,
  `results_summary.csv`, `training_summary.csv`,
  `model_capacity_summary.csv` and `model_failures.json`.
- Keep full prediction and checkpoint writing disabled by default.
- Reduced-scope evidence under
  [`experiments/fi2010_multifold_neural/`](../experiments/fi2010_multifold_neural/)
  covers all five official NoAuction ZScore folds for
  `deeplob_style` and `matrix_transformer` at horizon `label_10` on a single
  seed (`0`), single lookback (`20`) and `max_epochs=25`. Ten planned runs
  completed, zero failures. The full configured grid is not produced locally
  because CPU budget makes it impractical; cross-seed and multi-lookback
  variance is therefore not reported in this evidence.

Acceptance criteria:

- At least folds one and two are evaluated end-to-end on real data.
- Per-fold `results.json`, `calibration_bins.csv` and
  `execution_sensitivity.csv` exist.
- The multi-fold manifest reproduces the per-fold metric values from the
  per-fold artefacts.
- The neural plan inspection reports the run grid, device policy, output
  roots and smoke-versus-benchmark mode without writing outputs.
- A tiny supervised neural smoke run can execute from the CLI on prepared
  folds without launching the full configured grid.

### Stage 3: Cross-seed uncertainty

- Extend the multi-fold orchestrator to iterate over the configured seed
  list per fold.
- Persist per-seed artefacts and emit per-fold mean and standard deviation
  for each metric.

Stage 3a (statistical uncertainty layer, implemented in this layer):

- Add `chronoslob/experiments/statistics.py` together with the
  `analyse-fi2010-uncertainty` CLI command.
- Compute fold-level Student-t and percentile-bootstrap confidence
  intervals, paired fold-level comparisons against the
  `gradient_boosting` baseline, rank stability counts and a combined
  classical+neural ranking from the stored multi-fold tables under
  `experiments/fi2010_multifold_classical/` and
  `experiments/fi2010_multifold_neural/`.
- Persist these artefacts under
  [`experiments/fi2010_uncertainty/`](../experiments/fi2010_uncertainty/)
  with the caveats that the classical evidence is single-seed, the
  neural evidence is reduced-scope and single-seed, fold is the unit
  of variance, the execution proxy summary remains diagnostic only,
  and no profitability, live tradability, foundation-model or
  state-of-the-art claim is made.
- Cross-seed neural variance remains future work and is gated on the
  full Stage 3 seed sweep below.

Acceptance criteria:

- At least three seeds per fold per model on at least two folds.
- The aggregated metrics file documents the seed list, the per-seed values
  and the per-fold aggregates.

### Stage 4: Multi-fold ablation and systems coverage

- Run the ablation suite on every evaluated fold.
- Run the systems benchmark suite on the recorded environment with the
  multi-fold combined files.

Stage 4a (brutal ablation layer, implemented in this layer):

- Add `chronoslob/experiments/fi2010_brutal_ablations.py` together with
  the `run-fi2010-brutal-ablations` CLI command.
- Stress where the supervised signal survives and where it breaks across
  feature groups, model class, lookback, horizon, calibration threshold
  and execution cost/latency assumptions.
- The feature-group and horizon families refit a fast linear baseline on
  the real folds; the model-class, calibration and execution families
  reuse the stored multi-fold tables; the CPU-expensive neural lookback
  sweep is skipped by default and recorded with a reason.
- Write
  [`experiments/fi2010_brutal_ablations/`](../experiments/fi2010_brutal_ablations/)
  with per-family CSVs, an aggregate summary, a skipped-ablation record
  and concise notes. Execution numbers stay proxy diagnostics. See
  [docs/FI2010_BRUTAL_ABLATIONS.md](../docs/FI2010_BRUTAL_ABLATIONS.md).

Stage 4b (execution-aware evaluation v2, implemented in this layer):

- Add `chronoslob/experiments/execution_v2.py` together with the
  `run-fi2010-execution-v2` CLI command.
- Reuse the stored multi-fold classical, multi-fold neural and brutal
  ablation artefacts to make the forecasting-versus-tradability gap
  explicit through cost, latency, confidence, turnover, adverse-selection,
  fill and statistical-to-execution degradation proxies. No model is
  retrained and no full predictions or checkpoints are read.
- Write
  [`experiments/fi2010_execution_v2/`](../experiments/fi2010_execution_v2/)
  with `summary.json`, a per-scenario result table, the surface and
  summary CSVs, a skipped-diagnostic record, an assumptions file and
  concise notes. Neural runs ship no stored execution proxy rows, so their
  execution-aware diagnostics are recorded as explicit skips. Every metric
  stays a proxy diagnostic and no profitability or live tradability claim
  is made. See [docs/FI2010_EXECUTION_V2.md](../docs/FI2010_EXECUTION_V2.md).

Acceptance criteria:

- Every evaluated fold has a matching ablation directory and systems
  directory.
- Skipped ablations carry explicit reasons; nothing is silently omitted.

### Stage 5: External benchmark comparison

Stage 5a (context layer, implemented in this layer):

- Add [docs/FI2010_EXTERNAL_BENCHMARKS.md](../docs/FI2010_EXTERNAL_BENCHMARKS.md).
- Add
  [`reports/external_benchmark_context.md`](external_benchmark_context.md).
- Add the small structured context directory
  [`experiments/fi2010_external_context/`](../experiments/fi2010_external_context/)
  with `benchmark_context.json`, `protocol_comparison.csv` and
  `comparison_notes.md`.
- Document the comparison dimensions that decide whether an external
  FI-2010 or LOB-forecasting result is comparable: dataset variant,
  auction setting, normalisation, folds, horizon, label mapping, split
  protocol, metrics, preprocessing, model class and
  calibration/execution diagnostics.
- Record the current local classical and reduced-scope supervised neural
  result snapshot with the single-seed neural caveat and no SSL result.
- Do not copy external numeric paper metrics until exact source tables,
  folds, horizons and metric definitions are verified.

Future direct-comparison work:

- Verify external paper metrics from primary sources.
- Add numeric rows only when the local and external protocol dimensions
  match.
- Keep portfolio, execution and other non-forecasting objectives out of
  direct metric comparison unless the same objective is implemented locally.

Acceptance criteria:

- The public context document exists and states where direct metric
  comparison is valid or invalid.
- The structured artefact parses and contains no external numeric paper
  metrics.
- No external comparison appears without an explicit comparability caveat.

### Stage 6: Self-supervised pretraining gate

- Implement traceable train-only self-supervised pretraining followed by
  supervised fine-tuning inside the runner.
- Add a dedicated self-supervised ablation arm that uses the gate-compliant
  path.

Acceptance criteria:

- The self-supervised ablation produces a real child experiment directory
  with the same evidence streams used by the supervised baselines.
- All conditions in section 10 of the public protocol are satisfied.

### Stage 7: Result-first public surface

- Only after stages two through five are satisfied for at least two folds,
  rewrite the README and the high-level reports so the public surface
  leads with the multi-fold result.
- Keep this stage strictly behind the evidence stages.

Acceptance criteria:

- The README leads with multi-fold evidence pointers rather than fold-one
  evidence.
- Every result-first claim links to a per-fold artefact directory and the
  external comparison table.

## Hard Claim Boundaries

For every stage in this upgrade the same boundaries apply:

- The repository does not download FI-2010, does not commit raw archives,
  does not commit processed CSV files and does not commit prediction or
  intermediate matrix files.
- Reports may not describe the platform as suitable for live
  deployment, may not claim market-beating performance, may not claim
  foundation-model status and may not claim state-of-the-art
  performance.
- Reports may not claim self-supervised result parity until the public
  protocol gate is satisfied.
- Reports may not describe the platform as supporting live tradability.
- Every reported metric must trace to a versioned config, data source
  identifier, seed, split definition, code commit where available and
  stored output files.

## Public Surface Rule

The README and the public-facing reports remain framed around the current
single-fold evidence until at least stage two is complete on real data for
multiple folds. The result-first rewrite is explicitly gated behind that
multi-fold evidence and is not part of this protocol commit.

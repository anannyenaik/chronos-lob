# ChronosLOB v0.3.0 Alpha Candidate

Status: draft only. This candidate has not been published or tagged.

`v0.2.0` remains the published release. These notes describe proposed release
content from post-`v0.2.0` work on `main`.

## Central Finding

ChronosLOB remains centred on one result: forecasting metrics and
signal-quality diagnostics can diverge. Predictive and calibration metrics must
be read alongside confidence filtering, active fraction, turnover, cost,
latency and adverse-selection proxy diagnostics.

## Broader Proper-Training Neural Benchmark

The broader proper-training neural benchmark is now complete as post-v0.2.0
work. It covers 180 Hamilton Slurm cells across folds 1-5, seeds 0-2, lookbacks
20/50/100, horizons 10/50, and matrix-transformer plus DeepLOB-style model
families.

The matrix transformer has stronger mean macro-F1 and MCC than the DeepLOB-style
model in the retained benchmark, but with substantially higher variability and
weak lookback-100 behaviour. Confidence filtering improves retained-sample
metrics while reducing active fraction. The result supports a scoped benchmark
comparison, not a broad neural-superiority claim.

Across 90 runs per model:

| Model family | Mean macro-F1 | Mean MCC |
| --- | ---: | ---: |
| Matrix transformer | 0.6013 | 0.4294 |
| DeepLOB-style | 0.5133 | 0.3356 |

All 180 planned cells completed with zero failures. Validation-only early
stopping and best-checkpoint restore were used. The benchmark was executed as
staged Slurm jobs on Durham University Hamilton/NCC HPC.

## SSL-v2 Evidence

The retained SSL-v2 Hamilton multi-seed result covers folds 1-5, horizons 10/50,
seeds 0-2 and lookback 50. Across 30 matched comparison cells, the retained
analysis supports scoped mean predictive and calibration improvement for that
exact scope: mean macro-F1 `+0.0112`, MCC `+0.0259`, ECE `-0.0030` and Brier
score `-0.0235`. Results remain mixed by seed and horizon, so broad SSL
improvement is unsupported.

This multi-seed SSL-v2 evidence was included in the published `v0.2.0` evidence
line. The broader 180-cell supervised benchmark is the distinct post-`v0.2.0`
addition proposed for this candidate.

## Execution-Proxy Boundary

Execution-aware outputs are offline proxy diagnostics. They can describe
confidence-threshold, active-fraction, turnover, configured-cost, row-step
latency and adverse-selection sensitivity.

They do not establish PnL, realised profit, live-trading quality, tradable
alpha, production execution realism, venue-specific queue priority or market
impact. ChronosLOB makes no SOTA or foundation-model claim.

## Reproducibility And Provenance

The broader benchmark provenance is retained in:

- `experiments/fi2010_neural_proper_training_broader/hamilton_compute_provenance.json`
- `experiments/fi2010_neural_proper_training_broader/proper_neural_claim_assessment.json`
- `scripts/slurm/README_proper_neural.md`

Large checkpoints, raw predictions and cluster logs are excluded. GPU bitwise
reproducibility is not claimed.

Run the local validation gates:

```powershell
python -m pytest -q
python -m ruff check .
python -m mypy chronoslob
python -m chronoslob.cli doctor
python -m chronoslob.cli inspect-release-readiness
python -m chronoslob.cli run-project-audit --strict
git diff --check
```

## Deferred Work

- Write the manual paper.
- Add richer real event-level limit order book datasets where access and
  licensing allow.
- Extend non-linear feature-stability and execution modelling without
  broadening claims beyond retained evidence.

## Release Decision

Do not tag or publish `v0.3.0` from this draft. Publication requires explicit
approval after the documentation and retained evidence are reviewed.

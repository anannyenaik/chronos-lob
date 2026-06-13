# ChronosLOB: Forecasting Quality Is Not Trading-Signal Quality

ChronosLOB is a research platform for limit order book representation learning,
market-state forecasting, calibration and execution-aware validation. Every
layer is leakage-safe by construction, and forecasting, calibration,
feature-stability, event-level replay and execution-aware proxy diagnostics are
reported as separate, auditable evidence streams.

## Main Finding

Forecasting metrics and trading-signal diagnostics can diverge.

ChronosLOB's central evidence joins retained predictive and calibration
summaries to confidence filtering, active fraction, turnover, cost, latency and
adverse-selection proxy diagnostics.

![Forecasting versus signal quality](reports/execution_centrepiece/forecasting_vs_signal_quality.png)

This execution centrepiece is an offline diagnostic over stored artefacts. It
does not establish realised returns, profitability, tradability, live-market
execution quality or production execution realism.

## Evidence Map

| Evidence | Status | Public interpretation |
| --- | --- | --- |
| Leakage-safe FI-2010 classical benchmark | `archived_valid` | Retained folds 1-5 aggregate evidence. |
| Neural full grid (one-epoch matched) | `archived_valid` | Supervised/SSL comparison across folds 1-5, horizons 10/20/50 and seeds 0-2; not a performance-maximising neural benchmark. |
| Broader proper-training neural benchmark | `complete_real` | 180 supervised cells across folds 1-5, horizons 10/50, seeds 0-2, lookbacks 20/50/100 and two model families; results are mixed by model, lookback and horizon. |
| SSL-v2 benchmark and analysis | `complete_real` | Mean predictive and calibration improvements are supported only for the exact stored folds 1-5, horizons 10/50, seeds 0-2, lookback-50 scope. Results are mixed by seed and horizon; broad SSL improvement is unsupported. |
| Feature-stability analysis | `partial_real` plus `complete_real` analysis | Scoped FI-2010 snapshot-feature stability, not causal feature importance. |
| Execution centrepiece and execution-v3 | `archived_valid` | Offline execution-aware proxy diagnostics only. |
| Synthetic event-level replay | `archived_valid` | Controlled synthetic evidence only. |
| Binance Spot aggregated L2 replay | `partial_real` | Crypto L2 replay engineering path; committed sample is fixture-backed, not equity or predictive-success evidence. |

The repository is strong because its claims are audited against retained
artefacts and explicit limitations, not because it claims profitability.

## SSL-v2 Scope

The SSL-v2 benchmark is complete for the stored FI-2010 scope:
folds 1–5, horizons 10/50, seeds 0–2 and lookback 50.

Across 30 matched comparison cells, SSL-v2 has positive mean deltas for
macro-F1, MCC, ECE and Brier, supporting scoped predictive and calibration
improvement for this exact retained scope.

The evidence is mixed by seed and horizon, including negative mean macro-F1
deltas for seed 1 and horizon 50, so broad SSL improvement remains unsupported.

The one-epoch neural full grid is matched comparison evidence, not a
performance-maximising neural benchmark.

The broader proper-training neural benchmark completed all 180 supervised
cells. Across 90 runs per model, the matrix transformer has stronger overall
mean predictive and calibration metrics than the DeepLOB-style baseline, but
substantially higher variability and weak lookback-100 rows. DeepLOB-style is
steadier but lower overall. Results are mixed by model, lookback and horizon,
so no broad neural superiority is claimed.

This benchmark is post-`v0.2.0` work on `main`. `v0.2.0` remains the published
release and does not include the broader proper-training benchmark; a later
release would be required to publish it.

## Hamilton Compute Provenance

The seed-1 and seed-2 SSL-v2 refresh was executed as independent Slurm array
jobs on Durham University Hamilton/NCC HPC.

Retained summaries, provenance and claim assessments are committed; large
checkpoints, raw predictions and cluster logs are intentionally excluded.

GPU determinism warnings are documented, and bitwise reproducibility is not
claimed. See
[Hamilton compute provenance](reports/ssl_v2_analysis/hamilton_compute_provenance.json).

The broader proper-training neural benchmark was executed as Slurm jobs on
Durham University Hamilton/NCC HPC. Retained summaries and claim assessments
are committed; large checkpoints, raw predictions and cluster logs are
excluded. Its timing gate, staged arrays, final consolidation and GPU
determinism warning are recorded in
[proper-training Hamilton provenance](experiments/fi2010_neural_proper_training_broader/hamilton_compute_provenance.json).

## What This Is Not

- ChronosLOB is not financial advice.
- ChronosLOB is not live trading infrastructure.
- Not broker integration or automated order-placement software.
- Not evidence of profitability, tradable alpha or realised PnL.
- Not a production execution simulator.
- Not evidence of true event-level order flow or queue position from FI-2010.
- Not evidence that Binance crypto replay generalises to equity markets.
- Not a broad SSL-improvement or broad calibration-improvement claim.

## Claim Boundaries

- `snapshot_order_flow_proxy` is a labelled FI-2010 snapshot proxy, not true
  event-level order-flow imbalance.
- Feature ablations are feature-stability evidence, not causal importance.
- Synthetic replay is controlled synthetic evidence only.
- Binance diff-depth updates are aggregated level updates, not individual-order
  events.
- The manual paper is deferred.

## Public Artefacts

| Artefact | Link |
| --- | --- |
| Final empirical report | [reports/chronoslob_final_empirical_report.md](reports/chronoslob_final_empirical_report.md) |
| Evidence pack | [reports/evidence_pack/evidence_pack_summary.md](reports/evidence_pack/evidence_pack_summary.md) |
| Claim audit | [reports/evidence_pack/claim_audit.md](reports/evidence_pack/claim_audit.md) |
| Execution centrepiece | [reports/execution_centrepiece/execution_centrepiece.md](reports/execution_centrepiece/execution_centrepiece.md) |
| Execution proxy validity | [docs/EXECUTION_PROXY_VALIDITY.md](docs/EXECUTION_PROXY_VALIDITY.md) |
| Safety and limitations | [docs/SAFETY_AND_LIMITATIONS.md](docs/SAFETY_AND_LIMITATIONS.md) |
| Project status | [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) |
| v0.3.0 Alpha Candidate notes | [docs/RELEASE_NOTES_v0.3.0.md](docs/RELEASE_NOTES_v0.3.0.md) |
| Reproducibility guide | [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) |
| CLI reference | [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md) |
| Roadmap | [ROADMAP.md](ROADMAP.md) |

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,torch]"
python -m chronoslob.cli doctor
```

## Reproducibility

Build the public report from retained artefacts only:

```powershell
python -m chronoslob.cli build-final-empirical-report `
  --classical experiments/fi2010_multifold_classical `
  --neural experiments/fi2010_multifold_neural `
  --uncertainty experiments/fi2010_uncertainty `
  --ablations experiments/fi2010_brutal_ablations `
  --execution experiments/fi2010_execution_v2 `
  --execution-v3 experiments/fi2010_execution_v3 `
  --execution-centrepiece reports/execution_centrepiece `
  --external experiments/fi2010_external_context `
  --neural-full-grid experiments/fi2010_neural_full_grid `
  --proper-training experiments/fi2010_neural_proper_training_broader `
  --ssl-v2-analysis reports/ssl_v2_analysis `
  --feature-ablations experiments/fi2010_feature_ablations `
  --feature-ablation-analysis reports/feature_ablation_analysis `
  --evidence-pack reports/evidence_pack `
  --synthetic-lob reports/synthetic_lob_extension `
  --binance-l2 reports/binance_l2_extension `
  --out reports/chronoslob_final_empirical_report.md `
  --overwrite
```

Run the release gates:

```powershell
python -m pytest -q
python -m ruff check .
python -m mypy chronoslob
python -m chronoslob.cli doctor
python -m chronoslob.cli inspect-release-readiness
python -m chronoslob.cli run-project-audit --strict
git diff --check
```

The complete artefact-building command set is in
[reports/evidence_pack/reproduction_commands.md](reports/evidence_pack/reproduction_commands.md).

## Repository Structure

```text
chronoslob/  data, features, labels, models, training, diagnostics and analysis
configs/     YAML configs for data, models and experiments
docs/        protocols, evidence, reproducibility and safety documentation
reports/     public reports, evidence pack and generated figure artefacts
experiments/ retained FI-2010 evidence artefacts
tests/       deterministic tests and tiny synthetic fixtures
```

## Data And Licence Notes

No real exchange data, licensed data, private data, API keys or credentials are
committed. Tiny files under [tests/fixtures](tests/fixtures/) are synthetic and
exist only to exercise deterministic code paths.

Released under the [MIT Licence](LICENSE).

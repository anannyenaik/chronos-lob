# Final Empirical Report

This document describes the generated final FI-2010 empirical report.

## Purpose

`build-final-empirical-report` builds a coherent public Markdown report from
stored artefacts. It does not train models, download data or infer missing
metrics.

## Public Report Structure

The generated report is organised around:

1. Executive summary
2. Evidence map
3. Main finding: forecasting versus signal-quality gap
4. FI-2010 benchmark evidence
5. Supervised and SSL evidence
6. SSL-v2 scoped result
7. Feature-stability analysis
8. Execution-aware proxy diagnostics
9. Synthetic event-level replay
10. Binance L2 replay
11. Limitations
12. Reproducibility and artefacts
13. Deferred work

## Inputs

- `experiments/fi2010_multifold_classical/`
- `experiments/fi2010_multifold_neural/`
- `experiments/fi2010_uncertainty/`
- `experiments/fi2010_brutal_ablations/`
- `experiments/fi2010_execution_v2/`
- optional `experiments/fi2010_execution_v3/`
- optional `reports/execution_centrepiece/`
- `experiments/fi2010_external_context/`
- optional `experiments/fi2010_neural_full_grid/`
- optional `experiments/fi2010_neural_proper_training_broader/`
- optional `reports/ssl_v2_analysis/`
- optional `experiments/fi2010_feature_ablations/`
- optional `reports/feature_ablation_analysis/`
- optional `reports/evidence_pack/`
- optional `reports/synthetic_lob_extension/`
- optional `reports/binance_l2_extension/`
- optional FI-2010 figure manifest under
  `reports/figures/fi2010_neural_full_grid/` or the supplied full-grid directory

The classical, neural and uncertainty inputs are required. Ablation, execution
external-context, full neural grid and feature-ablation inputs are optional in
the builder API and are marked skipped when absent.

## Command

```bash
python -m chronoslob.cli build-final-empirical-report \
  --classical experiments/fi2010_multifold_classical \
  --neural experiments/fi2010_multifold_neural \
  --uncertainty experiments/fi2010_uncertainty \
  --ablations experiments/fi2010_brutal_ablations \
  --execution experiments/fi2010_execution_v2 \
  --execution-v3 experiments/fi2010_execution_v3 \
  --execution-centrepiece reports/execution_centrepiece \
  --external experiments/fi2010_external_context \
  --neural-full-grid experiments/fi2010_neural_full_grid \
  --proper-training experiments/fi2010_neural_proper_training_broader \
  --ssl-v2-analysis reports/ssl_v2_analysis \
  --feature-ablations experiments/fi2010_feature_ablations \
  --feature-ablation-analysis reports/feature_ablation_analysis \
  --evidence-pack reports/evidence_pack \
  --synthetic-lob reports/synthetic_lob_extension \
  --binance-l2 reports/binance_l2_extension \
  --out reports/chronoslob_final_empirical_report.md \
  --overwrite
```

## Outputs

- `reports/chronoslob_final_empirical_report.md`
- `reports/chronoslob_final_empirical_report_summary.json`

When `figure_manifest.json` is present, the report includes a Figure Index
section listing completed figure paths and skipped plots with reasons. Smoke-test
figures are labelled as diagnostics only.

When `--execution-v3` is supplied, the report includes execution-v3 status,
confidence filtering, cost sensitivity, latency sensitivity, fill assumptions,
adverse-selection proxy rows and skipped diagnostics. When it is missing, the
report explicitly states that no execution-v3 claim is made.

When `--execution-centrepiece` is supplied, the report includes the
forecasting-versus-signal-quality gap section near the top-level interpretation.
It uses retained proxy tables only and does not read raw predictions or realised
execution outcomes. See [EXECUTION_PROXY_VALIDITY.md](EXECUTION_PROXY_VALIDITY.md)
for the validity boundary.

When `--ssl-v2-analysis` is supplied, the report includes the scoped SSL-v2
interpretation.

The SSL-v2 benchmark is complete for the stored FI-2010 scope: folds 1-5,
horizons 10/50, seeds 0-2 and lookback 50. Across 30 matched comparison cells,
SSL-v2 has positive mean deltas for macro-F1, MCC, ECE and Brier, supporting
scoped predictive and calibration improvement for this exact retained scope.
The evidence is mixed by seed and horizon, including negative mean macro-F1
deltas for seed 1 and horizon 50, so broad SSL improvement remains unsupported.

When `--feature-ablations` is supplied, the report includes feature registry
status, unsupported FI-2010 groups, proxy warnings, aggregate ablation rows and
matched feature-delta rows. Smoke-test ablation artefacts are labelled as code
path checks rather than evidence.

When `--evidence-pack` is supplied, the report includes evidence-pack status,
claim audit counts, supported and unsupported claim summaries, and release
caveats.

When `--synthetic-lob` or `--binance-l2` is supplied, the report includes
clearly bounded extension sections. Binance L2 replay is reported as aggregated
depth-stream engineering evidence only, with explicit crypto-market,
fixture-data and non-predictive caveats.

## Traceability

The summary JSON records the generated timestamp, git commit when available,
input artefact paths, input file SHA-256 hashes, headline metrics, warnings and
skipped or missing sections.

## Claim Boundaries

The report states that classical results are multi-fold and horizon-10, while
neural results are split by scope: the one-epoch matched full grid is
multi-fold and multi-seed, the broader proper-training benchmark is
`complete_real` supervised evidence across 180 cells, and SSL-v2 predictive and
calibration improvements are limited to the exact stored multi-seed scope.
Execution-aware metrics are offline proxy diagnostics, external comparisons are
protocol context only, feature-ablation results are interpreted only when
matching non-smoke baselines exist, and full-grid neural results are claimed
only when non-smoke aggregate artefacts exist. Smoke-test full-grid,
execution-v3 and feature-ablation artefacts are labelled as code-path checks. No
profitability, tradability, live-trading, broad SSL superiority, broad
calibration improvement, true order-flow imbalance, cancellation imbalance,
trade imbalance, queue-position or SOTA claim is made.

## Limitations

The one-epoch neural full grid is matched comparison evidence, not a
performance-maximising neural benchmark. The broader proper-training benchmark
completed all 180 supervised cells. Results are mixed by model, lookback and
horizon, so no broad neural superiority is claimed.

The broader proper-training neural benchmark was executed as Slurm jobs on
Durham University Hamilton/NCC HPC. Retained summaries and claim assessments
are committed; large checkpoints, raw predictions and cluster logs are
excluded. GPU bitwise reproducibility is not claimed.

The benchmark is post-`v0.2.0` work on `main`. `v0.2.0` remains the published
release and does not include it; no `v0.3.0` release has been published.

The seed-1 and seed-2 SSL-v2 refresh was executed as independent Slurm array
jobs on Durham University Hamilton/NCC HPC. Retained summaries, provenance and
claim assessments are committed; large checkpoints, raw predictions and cluster
logs are intentionally excluded. GPU determinism warnings are documented, and
bitwise reproducibility is not claimed.

Execution outputs do not model queue priority, market impact or venue mechanics.
External benchmark papers are referenced for protocol context without importing
external numeric metrics. The manual paper is deferred.

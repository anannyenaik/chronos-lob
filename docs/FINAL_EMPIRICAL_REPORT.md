# Final Empirical Report

This document describes the generated final FI-2010 empirical report.

## Purpose

`build-final-empirical-report` builds a concise Markdown report from stored
artefacts. It does not train models, download data or infer missing metrics.

## Inputs

- `experiments/fi2010_multifold_classical/`
- `experiments/fi2010_multifold_neural/`
- `experiments/fi2010_uncertainty/`
- `experiments/fi2010_brutal_ablations/`
- `experiments/fi2010_execution_v2/`
- optional `experiments/fi2010_execution_v3/`
- `experiments/fi2010_external_context/`
- optional `experiments/fi2010_neural_full_grid/`
- optional `experiments/fi2010_feature_ablations/`
- optional `reports/evidence_pack/`
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
  --external experiments/fi2010_external_context \
  --neural-full-grid experiments/fi2010_neural_full_grid \
  --feature-ablations experiments/fi2010_feature_ablations \
  --evidence-pack reports/evidence_pack \
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

When `--feature-ablations` is supplied, the report includes feature registry
status, unsupported FI-2010 groups, proxy warnings, aggregate ablation rows and
matched feature-delta rows. Smoke-test ablation artefacts are labelled as code
path checks rather than evidence.

When `--evidence-pack` is supplied, the report includes evidence-pack status,
claim audit counts, supported and unsupported claim summaries, and release
caveats.

## Traceability

The summary JSON records the generated timestamp, git commit when available,
input artefact paths, input file SHA-256 hashes, headline metrics, warnings and
skipped or missing sections.

## Claim Boundaries

The report states that classical results are multi-fold, neural results are
reduced-scope and single-seed, execution-aware metrics are offline
execution-aware proxy diagnostics, external comparisons are protocol context
only, feature-ablation results are interpreted only when matching non-smoke
baselines exist, and full-grid neural results are claimed only when non-smoke
aggregate artefacts exist. Smoke-test full-grid, execution-v3 and
feature-ablation artefacts are labelled as code-path checks. No profitability,
tradability, live-trading, SSL superiority, true order-flow imbalance,
cancellation imbalance, trade imbalance, queue-position or SOTA claim is made.

## Limitations

Neural evidence is not multi-seed. Execution outputs do not model queues,
market impact or venue mechanics. External benchmark papers are referenced for
protocol context without importing external numeric metrics.

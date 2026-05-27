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
- `experiments/fi2010_external_context/`

The classical, neural and uncertainty inputs are required. Ablation, execution
and external-context inputs are optional in the builder API and are marked
skipped when absent.

## Command

```bash
python -m chronoslob.cli build-final-empirical-report \
  --classical experiments/fi2010_multifold_classical \
  --neural experiments/fi2010_multifold_neural \
  --uncertainty experiments/fi2010_uncertainty \
  --ablations experiments/fi2010_brutal_ablations \
  --execution experiments/fi2010_execution_v2 \
  --external experiments/fi2010_external_context \
  --out reports/chronoslob_final_empirical_report.md \
  --overwrite
```

## Outputs

- `reports/chronoslob_final_empirical_report.md`
- `reports/chronoslob_final_empirical_report_summary.json`

## Traceability

The summary JSON records the generated timestamp, git commit when available,
input artefact paths, input file SHA-256 hashes, headline metrics, warnings and
skipped or missing sections.

## Claim Boundaries

The report states that classical results are multi-fold, neural results are
reduced-scope and single-seed, execution-aware metrics are proxy diagnostics,
and external comparisons are protocol context only. No profitability,
tradability, SSL or SOTA claim is made.

## Limitations

Neural evidence is not multi-seed. Execution outputs do not model queues,
market impact or venue mechanics. External benchmark papers are referenced for
protocol context without importing external numeric metrics.

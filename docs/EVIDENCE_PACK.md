# Evidence Pack

The evidence pack is a reproducible release bundle for ChronosLOB experiment
artefacts. It inventories stored outputs, separates smoke diagnostics from real
empirical evidence, audits common public claims and writes conservative public
summaries.

It does not train models, create new results or write the manual paper.

## Build Command

```bash
python -m chronoslob.cli build-evidence-pack \
  --out reports/evidence_pack \
  --neural-full-grid experiments/fi2010_neural_full_grid \
  --figures reports/figures/fi2010_neural_full_grid \
  --execution-v3 experiments/fi2010_execution_v3 \
  --feature-ablations experiments/fi2010_feature_ablations \
  --feature-ablation-analysis reports/feature_ablation_analysis \
  --ablation-figures reports/figures/fi2010_feature_ablations \
  --final-report reports/chronoslob_final_empirical_report.md \
  --strict \
  --overwrite
```

Use `--allow-smoke-test --no-strict` only when building a diagnostic pack from
smoke artefacts.

## Outputs

The pack writes a manifest, Markdown summary, CSV inventory, claim audit,
supported and unsupported claim summaries, conservative public summary-bullet
files, reproduction commands, release checklist and README result snapshot under
the chosen output directory.

## Artefact Classification

Each artefact group is classified as:

- `missing`: expected evidence path or required evidence files are absent.
- `smoke_test_only`: explicit smoke metadata is present.
- `complete_real`: required non-smoke artefacts are present and completion checks pass.
- `partial_real`: real non-smoke artefacts exist, but the scope is incomplete,
  mixed or has explicit skipped diagnostics.
- `invalid`: metadata cannot be parsed or the artefact shape is unusable.
- `stale`: commit hashes, input hashes or source timestamps indicate drift.
- `unsupported`: the artefact path is intentionally not configured.
- `unknown_staleness`: there is not enough hash or timestamp evidence to call it clean.

Classification is based on stored summaries, manifests, status files, smoke
markers, run counts, hashes and timestamps where available.

## Claim Audit

The claim audit checks infrastructure claims, empirical-result claims and
blocked high-risk language. It emits one row per claim with:

- claim text
- status
- supporting artefacts
- required artefacts
- reason
- safe rewording

Empirical claims are supported only by real non-smoke aggregate artefacts. Smoke
diagnostics can show that a code path runs, but they are not empirical evidence.

## Smoke And Real Evidence

Smoke artefacts remain useful for release diagnostics, but the pack labels them
as `smoke_test_only`. The README snapshot and public bullet files must not turn
smoke outputs into result claims.

If real full-grid artefacts are missing, the snapshot says so directly. If SSL
deltas are mixed, negative, stale or unavailable, the pack avoids improvement
language.

## Forbidden Claims

The pack blocks claims about deployed trading, profitability, PnL, broad
benchmark leadership, foundation-model status, return claims without genuine
marked return calculations, and event-level FI-2010 concepts that the snapshot
data does not expose.

FI-2010 snapshot-derived features must remain described as snapshot diagnostics
or proxies. They do not establish true event-level order flow, cancellation
imbalance, trade imbalance or queue position.

Feature-ablation claim statuses also block causal feature-importance language.
The current `snapshot_order_flow_proxy` evidence is a scoped feature-stability
analysis over a labelled snapshot proxy, not an event-level or causal claim.

## Public Bullet Files

Two public summary-bullet files are generated:

- A conservative file that uses only fully supported infrastructure claims.
- A conditional file with stronger variants, explicit support conditions and
  safe fallbacks.

These files are generated from the claim audit and artefact inventory, not from
free-form result interpretation. They are public release notes, not personal
profile copy.

## Current Release Reading

At the current public-release point, the neural full grid and execution-v3 are
`complete_real`, feature ablations are `partial_real`, feature-ablation analysis
is `partial_real`, figure outputs are real with unsupported regime plots
skipped, and the manual paper has not yet been written. The matched full grid
supports the existence of a supervised-vs-SSL comparison, but it does not
support SSL improvement language.

A dedicated SSL analysis artefact (`ssl_failure_analysis_report`, built by
`analyse-fi2010-ssl-results` into `reports/ssl_failure_analysis/`) is recorded in
the inventory and reads retained lightweight comparison tables only. It supports
an implementation-and-evaluation claim and a narrow fold-1/horizon-50
predictive-metric observation in the partial proper-training subset, while broad
SSL improvement and SSL calibration improvement remain unsupported.

A dedicated execution-v3 analysis artefact (`execution_v3_analysis_report`, built
by `analyse-fi2010-execution-v3` into `reports/execution_v3_analysis/`) is also
recorded in the inventory and reads only the retained execution-v3 output tables.
It supports the `general.execution_proxy_analysis` infrastructure claim and
carries per-diagnostic claim statuses (cost, latency, fill and adverse-selection
sensitivity supported from real tables; regime diagnostics explicitly skipped).
All of its statements stay within offline execution-aware proxy diagnostics and
make no PnL, profitability or live-trading claim.

A dedicated feature-ablation stability artefact
(`feature_ablation_analysis_report`, built by
`analyse-fi2010-feature-ablations` into `reports/feature_ablation_analysis/`) is
recorded in the inventory. It reads retained lightweight tables from the
logistic/ridge expansion and the small gradient-boosting slice. Claim audit rows
support the exact scoped `snapshot_order_flow_proxy` findings and keep causal
feature-importance and true event-level order-flow claims forbidden.

## Preparing A Release

Before release:

- Review `git status --short`.
- Run the tests, lint and type checks.
- Run doctor, release readiness and strict project audit.
- Build or refresh real artefacts where needed.
- Rebuild figures, report and evidence pack after upstream artefacts change.
- Review stale and unknown-staleness inventory rows.
- Remove unsupported public claims.
- Keep the manual paper out of scope until the stored evidence is ready.
- Regenerate evidence-pack Markdown files from artefacts rather than manually
  editing generated outputs.

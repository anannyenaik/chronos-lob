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

Each artefact group has a completeness status and a separate freshness state.
This keeps valid retained summaries from looking broken just because their
generating commit is older than the current repository commit.

- `complete_real`: required non-smoke artefacts are present and completion checks pass.
- `archived_valid`: complete retained summaries/manifests remain content-valid,
  but the generating commit is older or heavy raw predictions/checkpoints were
  intentionally removed. This carries the same evidential weight as
  `complete_real`.
- `partial_real`: real non-smoke artefacts exist, but the scope is incomplete,
  mixed or has explicit skipped diagnostics.
- `optional_missing`: an optional artefact was never stored; no core public
  claim depends on it.
- `obsolete_superseded`: a legacy artefact is superseded by newer matched
  evidence.
- `smoke_test_only`: explicit smoke metadata is present.
- `missing`: expected required evidence path or required evidence files are absent.
- `invalid`: metadata cannot be parsed or the artefact shape is unusable.
- `stale`: retained content changed, a non-archival recorded hash path is
  missing, or an input is newer than its derived output.
- `unsupported`: the artefact path is intentionally not configured.
- `unknown_staleness`: there is not enough hash or timestamp evidence to call it clean.

The freshness column records `fresh`, `archived`, `stale`, `unknown` or `absent`.
Older generating commits are `archived` when retained hashes, files and summaries
are consistent. Missing non-heavy hashed inputs and hash mismatches remain
genuine `stale` evidence and require recomputation.

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

If required real full-grid artefacts are missing, the snapshot says so directly.
If SSL deltas are mixed, negative, genuinely stale or unavailable, the pack
avoids improvement language.

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

The synthetic event-level extension (`synthetic_lob_extension_report`) is
classified separately from FI-2010. Its claims are scoped to synthetic data:
`synthetic.event_level_pipeline`, `synthetic.event_level_features` and
`synthetic.regime_diagnostics` can be supported, while
`synthetic.real_market_event_level_generalisation` is always unsupported and
`synthetic.live_trading_or_profitability` and
`synthetic.fi2010_true_event_level_ofi` are forbidden. Synthetic results are
controlled stress-test evidence, never real-market evidence.

The Binance L2 replay extension (`binance_l2_extension_report`) is also
classified separately from FI-2010. It can support the offline
snapshot-plus-diff replay pipeline, update-continuity validation and local book
invariant checks. A fixture-only sample does not support a real captured stream
claim; a local captured Binance Spot snapshot/diff stream is required for that
claim. Predictive success, equity-market generalisation, individual-order trade
or cancellation recovery, live trading, profitability and queue-position claims
remain unsupported or forbidden.

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
`archived_valid`, feature ablations are `partial_real`, feature-ablation
analysis is `complete_real`, figure outputs are real with unsupported regime
plots skipped, and the manual paper has not yet been written. The matched full
grid supports the existence of a supervised-vs-SSL comparison, but it does not
support SSL improvement language.

A dedicated SSL analysis artefact (`ssl_failure_analysis_report`, built by
`analyse-fi2010-ssl-results` into `reports/ssl_failure_analysis/`) is recorded in
the inventory and reads retained lightweight comparison tables only. It supports
an implementation-and-evaluation claim and a narrow fold-1/horizon-50
predictive-metric observation in the partial proper-training subset, while broad
SSL improvement and SSL calibration improvement remain unsupported.

The SSL-v2 benchmark (`fi2010_ssl_v2_benchmark`) and analysis
(`ssl_v2_analysis_report`) are inventoried separately from SSL-v1. The current
stored scope is partial_real: fold 1, horizons 10/50, seed 0, lookback 50.
Claim rows support SSL-v2 implementation and scoped evaluation, while SSL-v2
predictive improvement, SSL-v2 calibration improvement and broad SSL improvement
remain unsupported.

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

A dedicated Binance L2 replay artefact (`binance_l2_extension_report`, built by
`replay-binance-l2-sample` into `reports/binance_l2_extension/`) is recorded in
the inventory when present. The committed fixture sample is `partial_real`
engineering evidence: it exercises the Binance-shaped replay path and claim
boundaries, but a user-supplied local capture is needed before the real
captured-stream claim is supported.

The legacy standalone SSL runner path is `obsolete_superseded` because the
matched neural full grid and SSL-v2 benchmark are the retained SSL evidence. The
stored feature-audit path is `optional_missing`; no core public claim depends on
that optional copy.

## Preparing A Release

Before release:

- Review `git status --short`.
- Run the tests, lint and type checks.
- Run doctor, release readiness and strict project audit.
- Build or refresh real artefacts where needed.
- Rebuild figures, report and evidence pack after upstream artefacts change.
- Review genuinely stale and unknown-staleness inventory rows. `archived_valid`,
  `optional_missing` and `obsolete_superseded` rows are expected release states
  when their notes explain the retained evidence.
- Remove unsupported public claims.
- Keep the manual paper out of scope until the stored evidence is ready.
- Regenerate evidence-pack Markdown files from artefacts rather than manually
  editing generated outputs.

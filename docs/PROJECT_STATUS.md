# Project Status

Current package version: `0.1.0`.

ChronosLOB is a research-engineering platform for limit order book
representation learning, short-horizon market-state forecasting, calibration
and offline execution-aware proxy diagnostics.

## Implemented

- Local loaders, schemas, validators and small synthetic fixtures.
- Past-only feature generation and explicit future-horizon labels with
  no-look-ahead checks.
- Temporal, walk-forward and purged or embargoed splitters; train-only
  preprocessing; metadata-only experiment registry.
- Classical, DeepLOB-style and transformer model code paths.
- Self-supervised masked-field and next-field objectives plus
  multi-task fine-tuning infrastructure.
- Calibration, uncertainty and confidence-filtering diagnostics.
- Execution-aware proxy diagnostics with explicit assumptions for fees, spread
  costs, latency, turnover, passive fill proxies and risk constraints.
- Transfer, regime, ablation and sensitivity analysis utilities.
- Local audit, release-readiness inspection and evidence-archive
  builder.
- Real FI-2010 multi-fold classical evidence, a completed matched neural
  supervised-vs-SSL grid and reduced-scope supervised neural benchmark artefacts
  under official split-aware evaluation, with calibration, uncertainty,
  ablation and execution-aware proxy diagnostics.
- Normalised FI-2010 matrix support for the supervised transformer
  paper-runner path.
- Release evidence-pack builder for artefact inventory, smoke/real separation,
  claim audit, conservative README snapshots and release checklists.

## Current Evidence

The current FI-2010 artefacts live under
[`experiments/fi2010_multifold_classical/`](../experiments/fi2010_multifold_classical/),
[`experiments/fi2010_multifold_neural/`](../experiments/fi2010_multifold_neural/),
[`experiments/fi2010_uncertainty/`](../experiments/fi2010_uncertainty/),
[`experiments/fi2010_brutal_ablations/`](../experiments/fi2010_brutal_ablations/),
[`experiments/fi2010_neural_full_grid/`](../experiments/fi2010_neural_full_grid/),
[`experiments/fi2010_execution_v3/`](../experiments/fi2010_execution_v3/),
[`experiments/fi2010_feature_ablations/`](../experiments/fi2010_feature_ablations/) and
[`experiments/fi2010_external_context/`](../experiments/fi2010_external_context/).
The generated final report is
[`reports/chronoslob_final_empirical_report.md`](../reports/chronoslob_final_empirical_report.md).
The release evidence-pack workflow is documented in
[EVIDENCE_PACK.md](EVIDENCE_PACK.md).

The strongest stored classical row is `gradient_boosting`; the strongest stored
reduced-scope supervised neural row is `matrix_transformer`. The completed
neural full grid compares supervised, masked-reconstruction SSL and next-field
SSL transformer variants across folds 1-5, horizons 10/20/50 and seeds 0-2.
That matched one-epoch grid does not support an SSL improvement claim and must
not be conflated with the separate 25-epoch reduced-scope neural benchmark. A
dedicated SSL analysis report (`reports/ssl_failure_analysis/`, built by
`analyse-fi2010-ssl-results`) reads retained lightweight comparison tables only
and records why SSL is not a broad success: the completed grid does not improve
overall, the only positive predictive-metric signal is narrow to fold 1, horizon
50 in the partial proper-training subset, and calibration worsened there.

Execution-v3 is `complete_real` as an offline cost-adjusted proxy diagnostic,
not PnL or live-trading evidence. Feature ablations are `partial_real`: the
current evidence supports a scoped statement that `snapshot_order_flow_proxy`
matters in the logistic/ridge horizon-10 setting, but it is not true
event-level OFI.

## Data Assumptions

- The repository ships no real exchange data; users provide any
  FI-2010 or public venue data locally.
- Synthetic fixtures exist only to exercise loaders, schemas, replay,
  features, labels and models on a tiny deterministic input.
- Crypto-style reconstruction examples are local engineering
  demonstrations and should not be treated as equivalent to
  equity-market behaviour.

## Current Limitations

- Execution-aware validation is an offline proxy diagnostic. It does not model
  venue-specific queue priority, production-grade partial-fill behaviour,
  market impact or PnL.
- The one-epoch matched neural full grid is complete for comparison, but it is
  not a performance-maximising neural benchmark.
- The standalone SSL runner output remains missing; the completed matched grid
  is the current SSL comparison evidence.
- Generalisation beyond FI-2010 requires additional documented experiment
  records.
- See [SAFETY_AND_LIMITATIONS.md](SAFETY_AND_LIMITATIONS.md) for the
  full scope statement.

## Next

Future work is focused on broader feature ablations, additional limit order book
dataset adapters where data access allows, richer execution modelling and
genuine regime analysis when explicit regime labels are available.

# Project Status

Current package version: `0.1.0`.

ChronosLOB is a research platform for limit order book representation
learning, short-horizon market-state forecasting, calibration and
execution-aware validation.

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
- Execution-aware validation with explicit assumptions for fees, spread
  costs, latency, turnover, passive fill proxies and risk constraints.
- Transfer, regime, ablation and sensitivity analysis utilities.
- Local audit, release-readiness inspection and evidence-archive
  builder.
- Real FI-2010 multi-fold classical evidence and reduced-scope supervised
  neural evidence under official split-aware evaluation, with calibration,
  uncertainty, ablation and execution-aware proxy diagnostics.
- Normalised FI-2010 matrix support for the supervised transformer
  paper-runner path.

## Current Evidence

The current FI-2010 artefacts live under
[`experiments/fi2010_multifold_classical/`](../experiments/fi2010_multifold_classical/),
[`experiments/fi2010_multifold_neural/`](../experiments/fi2010_multifold_neural/),
[`experiments/fi2010_uncertainty/`](../experiments/fi2010_uncertainty/),
[`experiments/fi2010_brutal_ablations/`](../experiments/fi2010_brutal_ablations/),
[`experiments/fi2010_execution_v2/`](../experiments/fi2010_execution_v2/) and
[`experiments/fi2010_external_context/`](../experiments/fi2010_external_context/).
The generated final report is
[`reports/chronoslob_final_empirical_report.md`](../reports/chronoslob_final_empirical_report.md).

The strongest stored classical row is `gradient_boosting`; the strongest
stored reduced-scope supervised neural row is `matrix_transformer`.
Neural evidence is single-seed and lookback 20. `ssl_transformer` is not
reported as a model result.

## Data Assumptions

- The repository ships no real exchange data; users provide any
  FI-2010 or public venue data locally.
- Synthetic fixtures exist only to exercise loaders, schemas, replay,
  features, labels and models on a tiny deterministic input.
- Crypto-style reconstruction examples are local engineering
  demonstrations and should not be treated as equivalent to
  equity-market behaviour.

## Current Limitations

- Execution-aware validation is a simplified research simulation. It
  does not model venue-specific queue priority, production-grade
  partial-fill behaviour or market impact.
- Reduced-scope neural evidence is not multi-seed and does not cover the full
  configured grid.
- `ssl_transformer` remains unsupported in the paper runner.
- Generalisation beyond FI-2010 requires additional documented experiment
  records.
- See [SAFETY_AND_LIMITATIONS.md](SAFETY_AND_LIMITATIONS.md) for the
  full scope statement.

## Next

Future work is focused on broader neural evidence, a traceable SSL
pretraining/fine-tuning runner, additional limit order book dataset adapters
where data access allows, richer execution modelling and genuine regime
analysis.

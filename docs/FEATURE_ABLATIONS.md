# FI-2010 Feature Ablations

`run-fi2010-feature-ablations` evaluates which FI-2010 snapshot feature
families contribute to classical results. The pipeline is classical-first and
does not block on neural runs.

The current stored artefact is `partial_real`: folds 1-5, horizon 10, seeds 0-2
and logistic/ridge models are covered; horizons 20/50 and the slower
`elastic_net` / `gradient_boosting` expansion remain future work. This scope is
broad enough to support scoped feature-family diagnostics, but not broad
feature conclusions across all horizons or model families.

## Command

```bash
python -m chronoslob.cli run-fi2010-feature-ablations \
  --folds 1 \
  --horizons 10 \
  --seeds 0 \
  --models logistic,ridge,elastic_net,gradient_boosting \
  --feature-groups all \
  --ablation-modes all \
  --out experiments/fi2010_feature_ablations \
  --smoke-test
```

Prepared FI-2010 fold CSVs can be supplied with `--config` and
`--processed-root`, or a single synthetic/local CSV with `--data-path`.

## Modes

| Mode | Meaning |
| --- | --- |
| `all_features` | all resolved selected groups |
| `remove_one_group` | all groups except one named group |
| `only_one_group` | one group at a time |
| `raw_lob_only` | raw price/size/top-of-book groups |
| `derived_microstructure_only` | derived, rolling and proxy groups |
| `no_proxy_features` | all non-proxy groups |

## Outputs

Root outputs:

- `results_summary.csv`
- `aggregate_summary.csv`
- `feature_delta_summary.csv`
- `failures.json`
- `summary.json`
- `ablation_manifest.json`
- `sha256_manifest.json`

Each run directory stores a config snapshot, selected feature groups, metrics,
predictions where available, status and SHA256 manifest.

## Interpretation

`feature_delta_summary.csv` compares each ablation to its matched
`all_features` baseline by fold, horizon, seed and model. Interpretations are
conservative:

- `helped`: ablation improved macro-F1 versus baseline.
- `hurt`: ablation reduced macro-F1 versus baseline.
- `neutral`: macro-F1 delta is very small.
- `insufficient evidence`: no matched all-features baseline.

Smoke-test artefacts prove the pipeline path only. They do not support empirical
claims.

Supported claim with real artefacts:

> ChronosLOB includes a leakage-safe FI-2010 microstructure feature registry and
> ablation pipeline, separating raw LOB, derived depth/imbalance/spread features
> and clearly labelled snapshot-flow proxies to evaluate which feature families
> contribute to forecasting, calibration and execution-aware diagnostics.

Current scoped finding:

> In the stored logistic/ridge horizon-10 ablation scope, removing
> `snapshot_order_flow_proxy` gives the clearest degradation. This supports a
> scoped proxy-feature importance statement only; it is not true event-level
> order-flow evidence.

Unsupported claims:

- true event-level order-flow imbalance from FI-2010 snapshots
- cancellation or trade imbalance without event messages
- queue position or fill priority
- profitability, tradability or live execution quality

Figures can be generated with:

```bash
python -m chronoslob.cli build-fi2010-ablation-figures \
  --feature-ablations experiments/fi2010_feature_ablations \
  --out reports/figures/fi2010_feature_ablations \
  --allow-smoke-test
```

## Evidence-Pack Use

`build-evidence-pack` reads `summary.json`, `ablation_manifest.json`,
`feature_delta_summary.csv` and `failures.json`. It supports only conservative
feature-ablation language unless real, non-smoke deltas name the exact feature
group, model, horizon, split and ablation mode. Unsupported event-level feature
concepts remain blocked even when snapshot proxy features are available.

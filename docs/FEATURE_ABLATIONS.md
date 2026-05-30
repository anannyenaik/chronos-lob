# FI-2010 Feature Ablations

`run-fi2010-feature-ablations` evaluates which FI-2010 snapshot feature
families contribute to classical results. The pipeline is classical-first and
does not block on neural runs.

The current stored artefact remains `partial_real`, but the scope is now broader
and more reviewer-useful:

- Main storage-light expansion: folds 1-5, horizons 10/20/50, seeds 0-2,
  logistic/ridge, all 12 supported registry groups and all 6 ablation modes.
- Small non-linear slice: folds 1-5, horizons 10/50, seed 0,
  `gradient_boosting`, key groups
  (`snapshot_order_flow_proxy`, `spread`, `depth_imbalance`,
  `top_of_book_imbalance`) and modes `all_features`, `remove_one_group`,
  `no_proxy_features`.
- Total analysed fits: 2,580 completed, 0 failed.
- Raw predictions and cached feature matrices were skipped.

This supports a scoped feature-stability analysis, not complete
feature-ablation evidence across all model families and horizons.

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
  --summary-only \
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
status and SHA256 manifest. Prediction files are optional and are not written in
the current storage-light real artefacts.

Storage-light options:

- `--summary-only` keeps the run storage-light and skips raw predictions and
  cached feature matrices.
- `--save-predictions --no-summary-only` writes row-level predictions for a
  targeted rerun when execution-aware ablation diagnostics are needed.
- `--save-heavy-artefacts --no-summary-only` writes cached feature matrices for
  debugging only.

The scoped stability analysis is generated with:

```bash
python -m chronoslob.cli analyse-fi2010-feature-ablations \
  --feature-ablations experiments/fi2010_feature_ablations \
  --extra-feature-ablations experiments/fi2010_feature_ablations_nonlinear_slice \
  --out reports/feature_ablation_analysis \
  --overwrite
```

It writes:

- `feature_ablation_analysis.md`
- `feature_delta_by_horizon.csv`
- `feature_delta_by_model.csv`
- `feature_delta_by_fold.csv`
- `feature_delta_by_seed.csv`
- `feature_group_stability.csv`
- `snapshot_order_flow_proxy_scope.csv`
- `feature_claim_assessment.json`
- `summary.json`
- `figure_manifest.json`

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

Current scoped findings:

> In the stored logistic/ridge scope, removing `snapshot_order_flow_proxy`
> degraded macro-F1 in all 90 matched remove-one-group rows across horizons
> 10/20/50. In the small `gradient_boosting` slice, removing the same group
> degraded macro-F1 in all 10 matched rows across horizons 10/50. This supports
> a horizon/model-specific effect for a labelled snapshot proxy only.

Required caveat:

> `snapshot_order_flow_proxy` is a labelled snapshot proxy derived from
> FI-2010 matrices. It should not be interpreted as true event-level
> order-flow imbalance.

Execution-aware ablation diagnostics require retained prediction-level outputs
or a targeted rerun.

Unsupported claims:

- true event-level order-flow imbalance from FI-2010 snapshots
- causal feature importance
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
`feature_delta_summary.csv`, `failures.json` and the generated stability
analysis under `reports/feature_ablation_analysis/`. It supports only
conservative feature-ablation language unless real, non-smoke deltas name the
exact feature group, model, horizon, fold/seed scope and ablation mode.
Unsupported event-level feature concepts and causal feature-importance claims
remain blocked even when snapshot proxy features are available.

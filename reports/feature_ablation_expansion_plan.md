# Feature Ablation Expansion Plan

Created: 2026-05-30

## Current Completed Scope

The retained FI-2010 feature-ablation evidence is `partial_real`.

- Directory: `experiments/fi2010_feature_ablations`
- Completed fits: 840 / 840 for the feasible stored slice
- Folds: `fold_1` to `fold_5`
- Horizons: 10
- Seeds: 0, 1, 2
- Models: `logistic`, `ridge`
- Feature groups: all 12 supported FI-2010 registry groups
- Ablation modes: `all_features`, `remove_one_group`, `only_one_group`, `raw_lob_only`, `derived_microstructure_only`, `no_proxy_features`
- Key scoped finding: removing `snapshot_order_flow_proxy` degraded macro-F1 in the matched horizon-10 logistic/ridge slice.
- Caveat: `snapshot_order_flow_proxy` is a labelled snapshot proxy, not true event-level OFI.

The current retained artefacts are storage-light: aggregate tables, per-run metrics/config/status files, manifests and reports remain; raw prediction files and cached heavy feature matrices are absent.

## Proposed Expansion Scope

Priority A is the selected first expansion because it directly answers whether the horizon-10 `snapshot_order_flow_proxy` result survives beyond horizon 10.

- Add horizons: 20 and 50
- Folds: `fold_1` to `fold_5`
- Seeds: 0, 1, 2
- Models: `logistic`, `ridge`
- Feature groups: all 12 supported FI-2010 registry groups
- Ablation modes: all 6 existing modes
- Expected new fits: 1,680
- Expected total retained fits after expansion: 2,520

This remains `partial_real` relative to the broader 5,040-fit full
feature-ablation target because `elastic_net` and `gradient_boosting` are not
included across every horizon, seed and fold. It should be reported as a
scoped feature-stability analysis, not complete feature-ablation evidence.

Priority B is a small non-linear slice if Priority A runtime remains reasonable.

- Model: `gradient_boosting`
- Folds: `fold_1` to `fold_5`
- Horizons: 10 and 50
- Seed: 0 only
- Key groups: `snapshot_order_flow_proxy`, `spread`, `depth_imbalance`, `top_of_book_imbalance`
- Suggested modes: `all_features`, `remove_one_group`, `no_proxy_features`
- Expected fits: 60

Priority C is already mostly covered by existing modes. `raw_lob_only`,
`derived_microstructure_only` and `all_features` give raw LOB only vs
engineered/proxy features vs combined comparisons without adding a new
artefact type.

## Storage Policy

The expansion should run in summary-heavy, prediction-light mode.

- Raw prediction files: skipped by default.
- Heavy cached feature matrices: skipped by default.
- Per-run `config_snapshot.json`, `feature_groups.json`, `metrics.json`, `status.json` and SHA256 manifests: preserved.
- Root `summary.json`, `results_summary.csv`, `aggregate_summary.csv`, `feature_delta_summary.csv`, `failures.json`, `ablation_manifest.json` and `sha256_manifest.json`: preserved.
- Final report and evidence pack must depend only on lightweight tables and manifests.
- Execution-aware ablation diagnostics require retained prediction-level outputs or a targeted rerun.

Current storage before expansion is about 7.5 MB under
`experiments/fi2010_feature_ablations`. With predictions and feature matrices
skipped, Priority A is expected to add tens of MB rather than GB. Saving raw
predictions for the full expansion is explicitly avoided.

## Compute Estimate

The broad 5,040-fit target should not be recomputed blindly. The first step is a small pilot:

- Folds: `fold_1`
- Horizon: 20
- Seed: 0
- Models: `logistic`, `ridge`
- Groups: `spread`, `snapshot_order_flow_proxy`
- Modes: `all_features`, `remove_one_group`, `no_proxy_features`
- Expected pilot fits: 8

The pilot will record runtime, output size, failures and whether summary-only mode works. Priority A proceeds only if the pilot confirms acceptable storage and runtime.

## Pilot Result

Pilot command:

```bash
python -m chronoslob.cli run-fi2010-feature-ablations \
  --config configs/experiments/fi2010_multifold.yaml \
  --processed-root data/processed/fi2010 \
  --folds 1 \
  --horizons 20 \
  --seeds 0 \
  --models logistic,ridge \
  --feature-groups spread,snapshot_order_flow_proxy \
  --ablation-modes all_features,remove_one_group,no_proxy_features \
  --out experiments/fi2010_feature_ablation_pilot \
  --strict \
  --summary-only
```

Pilot outcome:

- Completed fits: 8
- Failed fits: 0
- Runtime: 16.26 seconds
- Output size: 94,607 bytes
- Raw prediction files saved: no
- Cached feature matrices saved: no
- Summary-only mode: works
- The pilot wrote to a fresh output directory.

Priority A remains feasible if run in chunks. A naive fit-count extrapolation
from the pilot is about 57 minutes; scaling for the larger folds suggests
roughly one to two hours. The real run should therefore proceed fold-by-fold
or horizon-by-horizon with summary-only outputs retained.

## Fallback Scope

If Priority A is too slow, use this fallback:

- Horizons: 20 and 50
- Folds: `fold_1` to `fold_5`
- Seed: 0
- Models: `logistic`, `ridge`
- Feature groups: all 12 supported groups
- Ablation modes: all 6 modes
- Expected fits: 560

If that is still too expensive, run a clearly labelled partial slice and keep the evidence status `partial_real`.

## Claim Boundaries

Allowed language:

- feature-ablation evidence
- scoped feature-stability analysis
- `snapshot_order_flow_proxy`
- labelled snapshot proxy
- horizon/model-specific effect
- `partial_real`, if scope remains incomplete
- execution-aware proxy follow-up, if linked to execution-v3

Blocked language:

- true OFI
- causal driver
- proves order flow
- universally important
- trading value

## Completed Expansion

Priority A completed successfully.

- Added horizons: 20 and 50
- Folds: `fold_1` to `fold_5`
- Seeds: 0, 1, 2
- Models: `logistic`, `ridge`
- Feature groups: all 12 supported FI-2010 registry groups
- Ablation modes: all 6 existing modes
- New Priority A fits: 1,680
- Aggregated main feature-ablation fits: 2,520
- Failed main fits: 0
- Raw predictions saved: no
- Cached feature matrices saved: no
- Main output size after expansion: 20,495,099 bytes
- Main output size before expansion: 7,542,975 bytes

Priority B also completed as a small non-linear slice.

- Model: `gradient_boosting`
- Folds: `fold_1` to `fold_5`
- Horizons: 10 and 50
- Seed: 0
- Groups: `snapshot_order_flow_proxy`, `spread`, `depth_imbalance`,
  `top_of_book_imbalance`
- Modes: `all_features`, `remove_one_group`, `no_proxy_features`
- Completed fits: 60
- Failed fits: 0
- Raw predictions saved: no
- Cached feature matrices saved: no
- Output size: 679,378 bytes

Combined stability analysis:

- Directory: `reports/feature_ablation_analysis`
- Analysed fits: 2,580
- Evidence status: `partial_real`
- `snapshot_order_flow_proxy` horizon-10 logistic/ridge finding: supported
- Broader horizon 20/50 `snapshot_order_flow_proxy` finding: supported
- Non-linear model slice: supported for the small gradient-boosting slice
- Execution-aware ablation diagnostics: skipped as a future-work item because
  prediction-level outputs were not retained

Observed `snapshot_order_flow_proxy` remove-one-group deltas:

- Logistic/ridge horizon 10: 30/30 matched rows degraded macro-F1.
- Logistic/ridge horizons 20/50: 60/60 matched rows degraded macro-F1.
- Gradient boosting horizons 10/50: 10/10 matched rows degraded macro-F1.

These are horizon/model-specific effects for a labelled snapshot proxy. They
are not causal feature-importance evidence and are not true event-level
order-flow imbalance evidence.

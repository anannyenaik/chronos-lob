# FI-2010 Feature Ablations

This directory contains microstructure feature-ablation artefacts written by:

```bash
python -m chronoslob.cli run-fi2010-feature-ablations \
  --out experiments/fi2010_feature_ablations
```

## Current Scope

The current stored ablation evidence is `partial_real`: folds 1-5, horizon 10,
seeds 0-2, models `logistic` and `ridge`, all six ablation modes and all
snapshot-supported/proxy feature groups. Horizons 20/50 and the slower
`elastic_net` / `gradient_boosting` expansion remain future work.

Within this scope, removing `snapshot_order_flow_proxy` produces the clearest
degradation for logistic/ridge horizon-10 models. That feature is a
snapshot-delta proxy only; it is not true event-level order-flow imbalance.

Smoke-test outputs are pipeline diagnostics only. Real FI-2010 claims require
non-smoke artefacts with completed runs. Snapshot-flow columns are labelled as
`snapshot_order_flow_proxy`; this directory must not be used to claim true
event-level order flow, cancellation imbalance, trade imbalance or
queue-position evidence unless future data explicitly provides those event
fields.

# Config Inventory

This inventory lists local YAML configs by directory. Files containing `smoke` in the name are synthetic plumbing configs unless documented otherwise.

## `configs/data`

- `configs/data/binance_replay.yaml` - Local data loading or replay configuration. Synthetic smoke: `no`.
- `configs/data/event_log.yaml` - Local data loading or replay configuration. Synthetic smoke: `no`.
- `configs/data/fi2010.yaml` - Local data loading or replay configuration. Synthetic smoke: `no`.

## `configs/experiments`

- `configs/experiments/calibration_smoke.yaml` - Synthetic smoke or plumbing configuration. Synthetic smoke: `yes`.
- `configs/experiments/event_log_feature_audit.yaml` - Experiment or audit configuration. Synthetic smoke: `no`.
- `configs/experiments/event_multitask_smoke.yaml` - Synthetic smoke or plumbing configuration. Synthetic smoke: `yes`.
- `configs/experiments/event_ssl_smoke.yaml` - Synthetic smoke or plumbing configuration. Synthetic smoke: `yes`.
- `configs/experiments/event_tokenisation_smoke.yaml` - Synthetic smoke or plumbing configuration. Synthetic smoke: `yes`.
- `configs/experiments/event_transformer_smoke.yaml` - Synthetic smoke or plumbing configuration. Synthetic smoke: `yes`.
- `configs/experiments/execution_validation_smoke.yaml` - Synthetic smoke or plumbing configuration. Synthetic smoke: `yes`.
- `configs/experiments/feature_audit_fi2010.yaml` - Experiment or audit configuration. Synthetic smoke: `no`.
- `configs/experiments/fi2010_baseline_smoke.yaml` - Synthetic smoke or plumbing configuration. Synthetic smoke: `yes`.
- `configs/experiments/fi2010_deeplob_smoke.yaml` - Synthetic smoke or plumbing configuration. Synthetic smoke: `yes`.
- `configs/experiments/fi2010_split_audit.yaml` - Experiment or audit configuration. Synthetic smoke: `no`.
- `configs/experiments/fi2010_torch_dataset_smoke.yaml` - Synthetic smoke or plumbing configuration. Synthetic smoke: `yes`.
- `configs/experiments/full_audit_smoke.yaml` - Synthetic smoke or plumbing configuration. Synthetic smoke: `yes`.
- `configs/experiments/label_audit_fi2010.yaml` - Experiment or audit configuration. Synthetic smoke: `no`.
- `configs/experiments/public_release_readiness.yaml` - Experiment or audit configuration. Synthetic smoke: `no`.
- `configs/experiments/report_archive_smoke.yaml` - Synthetic report-archive build configuration. Synthetic smoke: `yes`.
- `configs/experiments/robustness_analysis_smoke.yaml` - Synthetic smoke or plumbing configuration. Synthetic smoke: `yes`.

## `configs/models`

- `configs/models/baselines.yaml` - Model architecture or baseline configuration. Synthetic smoke: `no`.
- `configs/models/deeplob.yaml` - Model architecture or baseline configuration. Synthetic smoke: `no`.
- `configs/models/multitask_transformer.yaml` - Model architecture or baseline configuration. Synthetic smoke: `no`.
- `configs/models/ssl_transformer.yaml` - Model architecture or baseline configuration. Synthetic smoke: `no`.
- `configs/models/transformer.yaml` - Model architecture or baseline configuration. Synthetic smoke: `no`.

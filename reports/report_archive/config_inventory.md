# Config Inventory

Local YAML configs grouped by directory. Files containing `smoke` in the name use bundled synthetic fixtures.

## `configs/data`

- `configs/data/binance_replay.yaml` - Local data loading or replay configuration. Uses synthetic fixture: `no`.
- `configs/data/event_log.yaml` - Local data loading or replay configuration. Uses synthetic fixture: `no`.
- `configs/data/fi2010.yaml` - Local data loading or replay configuration. Uses synthetic fixture: `no`.

## `configs/experiments`

- `configs/experiments/calibration_smoke.yaml` - Synthetic-fixture configuration. Uses synthetic fixture: `yes`.
- `configs/experiments/event_log_feature_audit.yaml` - Experiment or audit configuration. Uses synthetic fixture: `no`.
- `configs/experiments/event_multitask_smoke.yaml` - Synthetic-fixture configuration. Uses synthetic fixture: `yes`.
- `configs/experiments/event_ssl_smoke.yaml` - Synthetic-fixture configuration. Uses synthetic fixture: `yes`.
- `configs/experiments/event_tokenisation_smoke.yaml` - Synthetic-fixture configuration. Uses synthetic fixture: `yes`.
- `configs/experiments/event_transformer_smoke.yaml` - Synthetic-fixture configuration. Uses synthetic fixture: `yes`.
- `configs/experiments/execution_validation_smoke.yaml` - Synthetic-fixture configuration. Uses synthetic fixture: `yes`.
- `configs/experiments/feature_audit_fi2010.yaml` - Experiment or audit configuration. Uses synthetic fixture: `no`.
- `configs/experiments/fi2010_baseline_smoke.yaml` - Synthetic-fixture configuration. Uses synthetic fixture: `yes`.
- `configs/experiments/fi2010_deeplob_smoke.yaml` - Synthetic-fixture configuration. Uses synthetic fixture: `yes`.
- `configs/experiments/fi2010_split_audit.yaml` - Experiment or audit configuration. Uses synthetic fixture: `no`.
- `configs/experiments/fi2010_torch_dataset_smoke.yaml` - Synthetic-fixture configuration. Uses synthetic fixture: `yes`.
- `configs/experiments/full_audit_smoke.yaml` - Synthetic-fixture configuration. Uses synthetic fixture: `yes`.
- `configs/experiments/label_audit_fi2010.yaml` - Experiment or audit configuration. Uses synthetic fixture: `no`.
- `configs/experiments/public_release_readiness.yaml` - Experiment or audit configuration. Uses synthetic fixture: `no`.
- `configs/experiments/report_archive_smoke.yaml` - Evidence-archive build configuration. Uses synthetic fixture: `yes`.
- `configs/experiments/robustness_analysis_smoke.yaml` - Synthetic-fixture configuration. Uses synthetic fixture: `yes`.

## `configs/models`

- `configs/models/baselines.yaml` - Model architecture or baseline configuration. Uses synthetic fixture: `no`.
- `configs/models/deeplob.yaml` - Model architecture or baseline configuration. Uses synthetic fixture: `no`.
- `configs/models/multitask_transformer.yaml` - Model architecture or baseline configuration. Uses synthetic fixture: `no`.
- `configs/models/ssl_transformer.yaml` - Model architecture or baseline configuration. Uses synthetic fixture: `no`.
- `configs/models/transformer.yaml` - Model architecture or baseline configuration. Uses synthetic fixture: `no`.

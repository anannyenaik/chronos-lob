# Experiment Configs

Experiment configuration files combine data, features, labels, splitters, models,
metrics, seeds and output paths.

Every future experiment should be reproducible from a config in this directory or a
tracked derivative of one. Financial data experiments should use temporal splits by
default and must document label horizons and leakage controls.

Current examples cover feature audits, label audits and Phase 5 split validation.
`fi2010_split_audit.yaml` is a pre-modelling config for temporal split, purging,
embargo and train-only fitting checks; it contains no model or performance target.
`fi2010_baseline_smoke.yaml` documents a synthetic-fixture classical baseline
smoke run. `fi2010_torch_dataset_smoke.yaml` documents a synthetic-fixture
sequence DataLoader smoke run and `fi2010_deeplob_smoke.yaml` documents the
DeepLOB-style supervised neural smoke run from Phase 7B. None of these
smoke configs claim benchmark performance.

Do not create config files for fake or manually invented results.

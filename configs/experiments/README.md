# Experiment Configs

Experiment configuration files combine data, features, labels, splitters, models,
metrics, seeds and output paths.

Every future experiment should be reproducible from a config in this directory or a
tracked derivative of one. Financial data experiments should use temporal splits by
default and must document label horizons and leakage controls.

Current examples cover feature audits, label audits, split validation, classical
baselines, torch datasets, DeepLOB-style supervised smoke, event tokenisation,
transformer smoke, self-supervised smoke, multi-task smoke, calibration smoke,
execution-validation smoke, robustness-analysis smoke and full audit hardening.
None of these smoke configs claim benchmark performance or market evidence.

Do not create config files for fake or manually invented results.

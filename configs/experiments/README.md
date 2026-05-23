# Experiment Configs

Experiment configuration files combine data, features, labels, splitters,
models, metrics, seeds and output paths.

Every experiment should be reproducible from a config in this directory or a
tracked derivative of one. Financial data experiments use temporal splits by
default and document label horizons and leakage controls.

Current examples cover feature audits, label audits, split validation,
classical baselines, torch datasets, DeepLOB-style supervised checks, event
tokenisation, transformer, self-supervised and multi-task smoke
configurations, calibration, execution-validation, robustness-analysis and the
evidence-archive build.

Reported metrics must trace to a config, data source, seed, code commit and
stored output.

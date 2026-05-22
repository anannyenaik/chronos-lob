# Experiment Configs

Experiment configuration files combine data, features, labels, splitters, models,
metrics, seeds and output paths.

Every future experiment should be reproducible from a config in this directory or a
tracked derivative of one. Financial data experiments should use temporal splits by
default and must document label horizons and leakage controls.

Do not create config files for fake or manually invented results.

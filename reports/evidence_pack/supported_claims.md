# Supported Claims

- ChronosLOB uses leakage-safe FI-2010 evaluation
  - support: fi2010_classical_benchmarks
  - safe wording: ChronosLOB stores FI-2010 evaluation artefacts with train/test protocol metadata and leakage-control documentation.
- ChronosLOB includes train-only SSL pretraining
  - support: fi2010_neural_full_grid
  - safe wording: ChronosLOB includes code paths for train-only SSL pretraining; empirical claims require real non-smoke SSL artefacts.
- ChronosLOB compares supervised and SSL transformers
  - support: fi2010_neural_full_grid
  - safe wording: ChronosLOB includes infrastructure to compare supervised and SSL transformers under matched FI-2010 settings.
- ChronosLOB includes execution-aware proxy diagnostics
  - support: execution_v3_outputs
  - safe wording: ChronosLOB includes offline execution-aware proxy diagnostics with explicit limitations.
- Gradient boosting remained the strongest classical baseline
  - support: fi2010_classical_benchmarks
  - safe wording: Name the best classical model from the stored result table and include the metric, split and scope.
- Confidence filtering improved cost-adjusted proxy
  - support: execution_v3_outputs
  - safe wording: Report confidence-threshold proxy diagnostics with threshold, payoff mode, cost mode and retained-sample fraction.

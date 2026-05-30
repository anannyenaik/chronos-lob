# Supported Claims

- ChronosLOB includes a richer execution-aware proxy analysis report
  - support: execution_v3_analysis_report
  - safe wording: ChronosLOB includes a richer offline execution-aware proxy analysis covering confidence, turnover, cost, latency, fill and adverse-selection proxies; regime diagnostics are explicitly skipped.
- Feature-ablation infrastructure is available
  - support: feature_ablation_outputs
  - safe wording: ChronosLOB includes FI-2010 feature-ablation evidence infrastructure with proxy and unsupported groups labelled.
- SSL was implemented and evaluated under matched FI-2010 settings.
  - support: fi2010_neural_full_grid, ssl_failure_analysis_report
  - safe wording: SSL objectives were implemented and evaluated under matched settings.
- snapshot_order_flow_proxy remains important at horizon 10 for logistic/ridge
  - support: feature_ablation_analysis_report, feature_ablation_outputs
  - safe wording: State the exact feature group, horizons, models, folds and seeds; describe snapshot_order_flow_proxy as a labelled snapshot proxy.
- snapshot_order_flow_proxy importance survives horizons 20 and 50
  - support: feature_ablation_analysis_report, feature_ablation_outputs
  - safe wording: State the exact feature group, horizons, models, folds and seeds; describe snapshot_order_flow_proxy as a labelled snapshot proxy.
- Feature-ablation effects appear in a non-linear model slice
  - support: feature_ablation_analysis_report, feature_ablation_outputs
  - safe wording: State the exact feature group, horizons, models, folds and seeds; describe snapshot_order_flow_proxy as a labelled snapshot proxy.

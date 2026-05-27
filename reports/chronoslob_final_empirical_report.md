# ChronosLOB Final Empirical Report

Generated from stored FI-2010 artefacts only. No model training is run by this builder.

## Evidence Snapshot

| field | value |
| --- | --- |
| generated_at | 2026-05-27T14:51:31.360946+00:00 |
| git_commit | 55def5f3a3849f88a743bade04475b8251cd856e |
| classical_scope | multi-fold classical results |
| best_classical_test_macro_f1 | gradient_boosting: 0.4654 +/- 0.0039 |
| neural_scope | reduced-scope supervised neural, single-seed |
| best_neural_test_macro_f1 | matrix_transformer: 0.7337 +/- 0.0280, lookback 20 |
| execution_scope | proxy diagnostics loaded; metrics are proxy diagnostics |
| external_scope | protocol context loaded; protocol context only, not ranking claims |
| report_path | reports/chronoslob_final_empirical_report.md |
| summary_path | reports/chronoslob_final_empirical_report_summary.json |

## Research Question

Can stored FI-2010 artefacts support a traceable assessment of predictive mid-price direction performance, uncertainty, robustness, execution-aware proxy diagnostics and external protocol context?

## Dataset And Split Protocol

| field | value |
| --- | --- |
| dataset | FI-2010 |
| variant | NoAuction ZScore |
| task | midprice_direction |
| target_horizon | 10 |
| split_protocol | official split column; validation carved from train only |
| folds | 1, 2, 3, 4, 5 |
| classical_protocol | multi-fold; one stored classical seed across completed folds |
| neural_protocol | reduced-scope supervised neural; one seed and one lookback in stored artefacts |

## Model Families

| family | models | scope |
| --- | --- | --- |
| classical | majority, logistic, ridge, elastic_net, random_forest, gradient_boosting | multi-fold stored fold summaries |
| neural | deeplob_style, matrix_transformer | reduced-scope, single-seed, lookback 20 |

## Main Result Table

Classical rows are multi-fold. Neural rows are reduced-scope, single-seed supervised results and are not used here to assert superiority over the classical family.

| model | family | scope | test macro-F1 | accuracy | MCC | folds | seeds/runs | lookback |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gradient_boosting | classical | multi-fold | 0.4654 +/- 0.0039 | 0.6410 | 0.2724 | 5 | 5 | n/a |
| random_forest | classical | multi-fold | 0.4547 +/- 0.0081 | 0.6263 | 0.2431 | 5 | 5 | n/a |
| logistic | classical | multi-fold | 0.3261 +/- 0.0106 | 0.6125 | 0.1227 | 5 | 5 | n/a |
| elastic_net | classical | multi-fold | 0.3260 +/- 0.0106 | 0.6125 | 0.1227 | 5 | 5 | n/a |
| ridge | classical | multi-fold | 0.3087 +/- 0.0082 | 0.6126 | 0.1116 | 5 | 5 | n/a |
| majority | classical | multi-fold | 0.2514 +/- 0.0054 | 0.6058 | 0.0000 | 5 | 5 | n/a |
| matrix_transformer | supervised neural | reduced-scope, single-seed | 0.7337 +/- 0.0280 | 0.8008 | 0.6288 | 5 | 1 | 20 |
| deeplob_style | supervised neural | reduced-scope, single-seed | 0.4753 +/- 0.0274 | 0.4815 | 0.2932 | 5 | 1 | 20 |

## Uncertainty Summary

Seed variance is not available in the stored evidence; intervals are fold-level diagnostics.

| source | model | lookback | folds | seeds | mean | CI lower | CI upper | bootstrap lower | bootstrap upper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classical | gradient_boosting | n/a | 5 | 1 | 0.4654 | 0.4600 | 0.4708 | 0.4623 | 0.4692 |
| neural | matrix_transformer | 20 | 5 | 1 | 0.7337 | 0.6948 | 0.7726 | 0.7074 | 0.7535 |

## Ablation Summary

Stored ablations are diagnostic stress checks; skipped ablations remain explicit.

| family | stored rows | summary |
| --- | --- | --- |
| feature_groups | 180 | -0.0884 to 0.0000 test macro-F1 delta |
| model_class | 180 | -0.2191 to 0.0000 test macro-F1 delta |
| horizon | 90 | 0.0000 to 0.1454 test macro-F1 delta |
| calibration | 225 | ECE diagnostics |
| execution | 600 | cost/latency proxy diagnostics |

| field | value |
| --- | --- |
| families_run | feature_groups, horizon, model_class, calibration, execution |
| families_skipped | lookback |
| checkpoints_written | False |

| recorded skip |
| --- |
| ablation lookback_sweep: neural lookback sweep not requested; this is CPU-expensive, so pass --neural-lookbacks (and --max-epochs) to execute it |
| execution adverse_selection: neural runs ship no stored execution proxy rows, so an adverse-selection proxy cannot be computed |
| execution adverse_selection: neural runs ship no stored execution proxy rows, so an adverse-selection proxy cannot be computed |
| execution fill_assumption: neural runs ship no stored execution proxy rows, so a fill-assumption proxy cannot be computed |
| execution fill_assumption: neural runs ship no stored execution proxy rows, so a fill-assumption proxy cannot be computed |
| execution degradation: neural runs ship no stored execution proxy rows, so the execution side of the degradation cannot be computed |
| execution degradation: neural runs ship no stored execution proxy rows, so the execution side of the degradation cannot be computed |

## Execution-Aware Proxy Summary

Execution-aware metrics are proxy diagnostics only. They are not a backtest and make no profitability or tradability claim.

| model | source | status | test macro-F1 | base proxy | stress proxy | relative degradation |
| --- | --- | --- | --- | --- | --- | --- |
| elastic_net | classical | ok | 0.3260 | 19.6975 | 14.5384 | 0.2619 |
| gradient_boosting | classical | ok | 0.4654 | 13.0288 | 5.5975 | 0.5704 |
| logistic | classical | ok | 0.3261 | 19.6985 | 14.5426 | 0.2617 |
| majority | classical | ok | 0.2514 | 21.2862 | 16.2858 | 0.2349 |
| random_forest | classical | ok | 0.4547 | 9.8244 | 4.9636 | 0.4948 |
| deeplob_style | neural | skipped | 0.4753 | n/a | n/a | n/a |
| matrix_transformer | neural | skipped | 0.7337 | n/a | n/a | n/a |

| field | value |
| --- | --- |
| proxy_diagnostics | True |
| fill_assumption | full_fill_at_mid_no_queue |
| checkpoints_required | False |
| full_predictions_required | False |

## External Benchmark Context

External comparisons are protocol context, not ranking claims. No external numeric metrics are imported into this report.

| source | type | numeric metrics included |
| --- | --- | --- |
| Ntakaris et al. FI-2010 dataset and baselines | paper_and_dataset | False |
| Tsantekidis et al. stationary-feature LOB forecasting | paper | False |
| Zhang, Zohren and Roberts DeepLOB | paper | False |
| Wallbridge TransLOB | paper | False |
| Sangadiev et al. DeepFolio | paper | False |

## What This Proves

- The committed artefacts support a traceable multi-fold classical FI-2010 result.
- The committed artefacts support reduced-scope, single-seed supervised neural evidence.
- The uncertainty, ablation and proxy-diagnostic layers are generated from stored tables.
- External references are used only to document protocol context.

## What This Does Not Prove

- Profitability or tradability in deployed markets.
- Production execution quality or market-impact realism.
- Foundation-model status.
- SSL or SOTA performance.
- Neural superiority over the classical family.

## Limitations

| limitation | detail |
| --- | --- |
| classical_seed_count | 1 |
| neural_seed_count | 1 |
| neural_scope | single seed and single lookback in stored reduced-scope artefacts |
| execution_scope | proxy diagnostics only; queue, impact and venue mechanics are not modelled |
| external_scope | protocol context only; no external numeric metrics are copied |
| prediction_checkpoint_policy | full predictions and checkpoints are not required by this report builder |

## Artefact Traceability

| artefact | path | sha256 |
| --- | --- | --- |
| ablations_calibration_threshold_ablation | experiments/fi2010_brutal_ablations/calibration_threshold_ablation.csv | 0f7163d1c72deb1c336d6d029ea6476573bb3f5d1ae18b33ddf85c7cb73ce9d1 |
| ablations_dir | experiments/fi2010_brutal_ablations | directory |
| ablations_execution_cost_latency_ablation | experiments/fi2010_brutal_ablations/execution_cost_latency_ablation.csv | beed6a1e458328d06d0c0cbc7c9368150e9730ed5d60da0ea697ade2f03470b4 |
| ablations_feature_group_ablation | experiments/fi2010_brutal_ablations/feature_group_ablation.csv | f41a7bf4a6e280be21904bd9e6e91b99f6a367b95d2a6a53d65fd34a54a837d0 |
| ablations_horizon_ablation | experiments/fi2010_brutal_ablations/horizon_ablation.csv | 53b8642ac849de2ecb9da150108fca8fcb372d37822101f333f3c3ca39fdf6c2 |
| ablations_model_class_ablation | experiments/fi2010_brutal_ablations/model_class_ablation.csv | d73a3dfdb2d2ae68a4a7348e8757973d88b666da72cf38417833185eabcc8328 |
| ablations_skipped_ablations | experiments/fi2010_brutal_ablations/skipped_ablations.json | d3d4927131c2733e10c2efbcd77ea7e067439d4e26b9f389488daeb5cd9c0fd9 |
| ablations_summary | experiments/fi2010_brutal_ablations/summary.json | b2ada55d9802da545dc67435bf30a93357203c23bb92c55ae4e847d45cc0716d |
| classical_dir | experiments/fi2010_multifold_classical | directory |
| classical_results_summary | experiments/fi2010_multifold_classical/results_summary.csv | 7a4d3c042805ecb4d8735fe9ad95f1ccc9bf50a0d4a83acd646f4d6417a9e03e |
| classical_summary | experiments/fi2010_multifold_classical/summary.json | 7428cc7975c223f17c43be86fe4bdde0d1024c90522b663fa3778e60b8ec892c |
| execution_adverse_selection_summary | experiments/fi2010_execution_v2/adverse_selection_summary.csv | 57bd2c42e591e64bdc1e4be6aa3a2d8902f031178521d699f5e5dde9965d1be8 |
| execution_confidence_threshold_summary | experiments/fi2010_execution_v2/confidence_threshold_summary.csv | 7aca37c1b7030e6120f631e3d49ea96611b534bf81b9a2a212854cb8b4a9b669 |
| execution_degradation_summary | experiments/fi2010_execution_v2/degradation_summary.csv | 02c313f2bb64beb09f490e9f81f57f29978d4d0046de28d27818384c36c06df1 |
| execution_dir | experiments/fi2010_execution_v2 | directory |
| execution_fill_assumption_summary | experiments/fi2010_execution_v2/fill_assumption_summary.csv | cd5fe1dc63de3dc522defee688605069c4a5b3afc3627c52c56a2192fd9a5d6c |
| execution_skipped_diagnostics | experiments/fi2010_execution_v2/skipped_diagnostics.json | 4830309d22260be7dd982f704f8569cd5479e51937ae414e55ec02962cc36d40 |
| execution_summary | experiments/fi2010_execution_v2/summary.json | 39c9b74944fba54daf1a4d49b55af403514deb6a180ef1af7ccd88cedd66ae86 |
| execution_turnover_summary | experiments/fi2010_execution_v2/turnover_summary.csv | d961acc368b842bf22d48fefc1ecf45fde1b39962a5d5511efe37df39700393f |
| external_benchmark_context | experiments/fi2010_external_context/benchmark_context.json | 2e1f4e0c4914378f31dbf50b3bc03df33b9d4254ebeda0944e5a06225fe30c8e |
| external_dir | experiments/fi2010_external_context | directory |
| external_protocol_comparison | experiments/fi2010_external_context/protocol_comparison.csv | 756c3ce98cbd4e15e1afaa5a0a2d4007ea775d58b89dc5409c513e2a8cdd460c |
| neural_dir | experiments/fi2010_multifold_neural | directory |
| neural_results_summary | experiments/fi2010_multifold_neural/results_summary.csv | bd6c0a52a6ea5eb5e01a66b91671cd84ee79b316ed202a9eadd198604a7f1e0c |
| neural_summary | experiments/fi2010_multifold_neural/summary.json | 9647aad87ff4bc8255d3971626850ff0c5b0eb76586a0e239141a64a9602b59e |
| uncertainty_dir | experiments/fi2010_uncertainty | directory |
| uncertainty_metric_confidence_intervals | experiments/fi2010_uncertainty/metric_confidence_intervals.csv | db1468eb1f171702e1cad0d21ff9cc30ce5d082a9bfc7adc59882e720492e2cf |
| uncertainty_model_ranking | experiments/fi2010_uncertainty/model_ranking.csv | c1cb29fa431db19ade6167323902cacf8a2da4e385d369b9a92850338feee2ca |
| uncertainty_summary | experiments/fi2010_uncertainty/summary.json | 942e016bd40af0ab73412036279f2067731fcfb76156ee8c935beebf7b2d5c34 |

## Reproduction Commands

```bash
python -m chronoslob.cli build-final-empirical-report \
  --classical experiments/fi2010_multifold_classical \
  --neural experiments/fi2010_multifold_neural \
  --uncertainty experiments/fi2010_uncertainty \
  --ablations experiments/fi2010_brutal_ablations \
  --execution experiments/fi2010_execution_v2 \
  --external experiments/fi2010_external_context \
  --out reports/chronoslob_final_empirical_report.md \
  --overwrite

python -m chronoslob.cli doctor
python -m chronoslob.cli inspect-release-readiness
python -m chronoslob.cli run-project-audit --strict
python -m pytest
python -m compileall -q chronoslob tests
python -m ruff check .
python -m mypy chronoslob
```

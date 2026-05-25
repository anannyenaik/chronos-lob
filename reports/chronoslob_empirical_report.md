# ChronosLOB Empirical Report

## Abstract

This empirical report summarises stored artefacts for `fi2010_midprice_h10` on task `midprice_direction`.
It covers predictive results, calibration results, execution-aware sensitivity, ablation results and systems benchmarks where those artefacts were supplied.
The target research question is whether self-supervised order-book representations can improve short-horizon market-state forecasting.
Evidence is bounded by leakage-safe validation, calibration analysis and explicit execution assumptions.
This report only records evidence present on disk and does not claim profitability, deployability or live trading.
Successful model entries in the main experiment: majority, logistic, random_forest, gradient_boosting, deeplob_style.

## 1. Dataset and provenance

| field | value |
| --- | --- |
| dataset name | FI-2010 |
| dataset version | not available |
| dataset variant | midprice_direction_h10 |
| source kind | local_file |
| source path | data/processed/fi2010/fold1_combined.csv |
| source checksum | 91aef9f1923dfd87955dd5838f709ca198d8ae903cfd6baeb029fe827a0539d0 |
| row count | 77909 |
| event count | not available |
| feature count | 144 |
| runner data source kind | local_file |
| Python | 3.11.9 |
| platform | Windows-10-10.0.19045-SP0 |
| package version | 0.1.0 |

Limitations recorded in provenance: Preparation is local-only. The repository does not download FI-2010
and does not ship FI-2010 data. The tiny FI-2010-like fixture under
tests/fixtures/fi2010 exists only to exercise the preparation
plumbing and does not represent the canonical benchmark.

## 2. Label construction

| field | value |
| --- | --- |
| task name | midprice_direction |
| label name | label_10 |
| horizon | 10 |
| label source | fi2010_existing_labels |
| label mapping | not available |
| distinct classes | 1, 2, 3 |
| class counts | 1=15483, 2=47639, 3=14787 |

Leakage details: label construction is reported from the config snapshot and preparation artefacts. Any unavailable label detail is marked as not available rather than inferred.

## 3. Leakage controls and temporal validation

| field | value |
| --- | --- |
| split design | temporal |
| total rows | 77909 |
| train rows | 54536 |
| validation rows | 11687 |
| test rows | 11686 |
| train start/end | 0 to 54535 |
| validation start/end | 54536 to 66222 |
| test start/end | 66223 to 77908 |

Stored model metadata records train-only preprocessing or tokenisation for: deeplob_style standardisation, deeplob_style split-contained windows.
The experiment directory passed the required artefact validation contract before this report was written.

## 4. Models

| field | value |
| --- | --- |
| requested models | majority, logistic, random_forest, gradient_boosting, deeplob_style, transformer |
| successful models | majority, logistic, random_forest, gradient_boosting, deeplob_style |
| skipped models | transformer: ValidationError: 1 validation error for OrderBookLevel quantity   Value error, quantity must be non-negative; got -0.47933132 [type=value_error, input_value=-0.47933132, input_type=float]     For further information visit https://errors.pydantic.dev/2.13/v/value_error |

`deeplob_style` is reported as a compact DeepLOB-style supervised baseline in the stored runner metadata, not as an exact external-paper reproduction.
`ssl_transformer` is present only as planned or skipped metadata; no SSL model result is reported here.

## 5. Predictive results

The table below is populated only from `results.json`; missing metrics are marked as not available.

| model | split | horizon | accuracy | macro F1 | test count | class count test | warnings |
| --- | --- | --- | --- | --- | --- | --- | --- |
| majority | test | 10 | 0.628615 | 0.257321 | 11686 | 3 | none |
| logistic | test | 10 | 0.62331 | 0.351469 | 11686 | 3 | none |
| random_forest | test | 10 | 0.63161 | 0.409803 | 11686 | 3 | none |
| gradient_boosting | test | 10 | 0.645131 | 0.442036 | 11686 | 3 | none |
| deeplob_style | test | 10 | 0.567174 | 0.352432 | 11686 | 3 | none |

![Confusion matrix](experiments/fi2010_midprice_h10/plots/confusion_matrix.png)

## 6. Calibration results

`calibration_bins.csv` is present with 50 rows.

| model | split | ECE | Brier score | mean confidence | calibration rows | positive bins |
| --- | --- | --- | --- | --- | --- | --- |
| majority | test | 0.0256743 | 0.536689 | 0.602941 | 10 | 1 |
| logistic | test | 0.0411109 | 0.518987 | 0.587467 | 10 | 7 |
| random_forest | test | 0.0965356 | 0.521308 | 0.535226 | 10 | 7 |
| gradient_boosting | test | 0.031549 | 0.482168 | 0.61879 | 10 | 6 |
| deeplob_style | test | 0.136053 | 0.58772 | 0.431122 | 10 | 4 |

![Reliability curve](experiments/fi2010_midprice_h10/plots/reliability_curve.png)

## 7. Execution-aware sensitivity

`execution_sensitivity.csv` is present with 60 rows.

| field | value |
| --- | --- |
| confidence thresholds | 0, 0.5, 0.6, 0.7 |
| cost assumptions | 0, 1, 5 |
| latency assumptions | 0 |
| return proxy | ask_price_column=ask_price_1, bid_price_column=bid_price_1, horizon_offset=not available, kind=mid_forward_return |

| model | rows | thresholds | cost bps | latency steps | max eligible | net proxy min | net proxy max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deeplob_style | 12 | 0, 0.5, 0.6, 0.7 | 0, 1, 5 | 0 | 11676 | -20.6589 | 0 |
| gradient_boosting | 12 | 0, 0.5, 0.6, 0.7 | 0, 1, 5 | 0 | 11676 | -5.37553 | -0.0381572 |
| logistic | 12 | 0, 0.5, 0.6, 0.7 | 0, 1, 5 | 0 | 11676 | -6.51566 | -0.428625 |
| majority | 12 | 0, 0.5, 0.6, 0.7 | 0, 1, 5 | 0 | 11676 | -5.43746 | 0 |
| random_forest | 12 | 0, 0.5, 0.6, 0.7 | 0, 1, 5 | 0 | 11676 | -5.40135 | 0.0608286 |

![Cost sensitivity](experiments/fi2010_midprice_h10/plots/cost_sensitivity.png)

These rows are proxy sensitivity under explicit assumptions, not a production backtest or execution system for deployment.

## 8. Ablations and robustness

| field | value |
| --- | --- |
| ablation set | standard |
| ablations run | baseline, calibration_bins_5, cost_0bps, cost_1bps, calibration_bins_10, latency_0, latency_1, horizon_50, lookback_2, lookback_4, feature_top_of_book, feature_depth_liquidity |
| ablations skipped | ssl_pretraining_ablation, feature_imbalance |
| models requested | majority, logistic, deeplob_style, transformer |
| fixture run | no |

Ablation report paths:
- `reports/calibration_ablation.md`
- `reports/cost_sensitivity.md`
- `reports/feature_group_ablation.md`
- `reports/horizon_ablation.md`
- `reports/latency_sensitivity.md`
- `reports/lookback_window_ablation.md`
- `reports/ssl_pretraining_ablation.md`

| ablation | status | model | metric | value | source or warning |
| --- | --- | --- | --- | --- | --- |
| baseline | run | majority | accuracy | 0.6286154372753723 | experiments/baseline |
| baseline | run | majority | macro_f1 | 0.25732100322264256 | experiments/baseline |
| baseline | run | majority | expected_calibration_error | 0.025674260804784077 | experiments/baseline |
| baseline | run | logistic | accuracy | 0.6233099435221633 | experiments/baseline |
| baseline | run | logistic | macro_f1 | 0.35146855037871383 | experiments/baseline |
| baseline | run | logistic | expected_calibration_error | 0.041110897222318726 | experiments/baseline |
| baseline | run | deeplob_style | accuracy | 0.5671743967140168 | experiments/baseline |
| baseline | run | deeplob_style | macro_f1 | 0.35243154257775067 | experiments/baseline |
| baseline | run | deeplob_style | expected_calibration_error | 0.13605281684589038 | experiments/baseline |
| calibration_bins_5 | run | majority | accuracy | 0.6286154372753723 | experiments/calibration_bins_5 |
| calibration_bins_5 | run | majority | macro_f1 | 0.25732100322264256 | experiments/calibration_bins_5 |
| calibration_bins_5 | run | majority | expected_calibration_error | 0.025674260804784077 | experiments/calibration_bins_5 |
| calibration_bins_5 | run | logistic | accuracy | 0.6233099435221633 | experiments/calibration_bins_5 |
| calibration_bins_5 | run | logistic | macro_f1 | 0.35146855037871383 | experiments/calibration_bins_5 |
| calibration_bins_5 | run | logistic | expected_calibration_error | 0.041110897222318726 | experiments/calibration_bins_5 |
| calibration_bins_5 | run | deeplob_style | accuracy | 0.5671743967140168 | experiments/calibration_bins_5 |

SSL pretraining ablation status:
- `ssl_pretraining_ablation`: skipped; no traceable runner support for SSL pretraining/fine-tuning yet; ssl_transformer is not registered in the paper-runner model registry

## 9. Systems benchmarks

Systems benchmark directory: not supplied.

## 10. Failure cases and warnings

- optional artefact missing: plots/regime_breakdown.png
- requested model 'transformer' was skipped: ValidationError: 1 validation error for OrderBookLevel
quantity
  Value error, quantity must be non-negative; got -0.47933132 [type=value_error, input_value=-0.47933132, input_type=float]
    For further information visit https://errors.pydantic.dev/2.13/v/value_error
- regime breakdown skipped: no genuine regime-breakdown data is available in stored artefacts; not fabricating regimes from row numbers or timestamps
- ablation 'baseline': requested model 'transformer' was skipped: ValidationError: 1 validation error for OrderBookLevel
quantity
  Value error, quantity must be non-negative; got -0.47933132 [type=value_error, input_value=-0.47933132, input_type=float]
    For further information visit https://errors.pydantic.dev/2.13/v/value_error
- ablation 'calibration_bins_5': requested model 'transformer' was skipped: ValidationError: 1 validation error for OrderBookLevel
quantity
  Value error, quantity must be non-negative; got -0.47933132 [type=value_error, input_value=-0.47933132, input_type=float]
    For further information visit https://errors.pydantic.dev/2.13/v/value_error
- ablation 'cost_0bps': requested model 'transformer' was skipped: ValidationError: 1 validation error for OrderBookLevel
quantity
  Value error, quantity must be non-negative; got -0.47933132 [type=value_error, input_value=-0.47933132, input_type=float]
    For further information visit https://errors.pydantic.dev/2.13/v/value_error
- ablation 'cost_1bps': requested model 'transformer' was skipped: ValidationError: 1 validation error for OrderBookLevel
quantity
  Value error, quantity must be non-negative; got -0.47933132 [type=value_error, input_value=-0.47933132, input_type=float]
    For further information visit https://errors.pydantic.dev/2.13/v/value_error
- ablation 'calibration_bins_10': requested model 'transformer' was skipped: ValidationError: 1 validation error for OrderBookLevel
quantity
  Value error, quantity must be non-negative; got -0.47933132 [type=value_error, input_value=-0.47933132, input_type=float]
    For further information visit https://errors.pydantic.dev/2.13/v/value_error
- ablation 'latency_0': requested model 'transformer' was skipped: ValidationError: 1 validation error for OrderBookLevel
quantity
  Value error, quantity must be non-negative; got -0.47933132 [type=value_error, input_value=-0.47933132, input_type=float]
    For further information visit https://errors.pydantic.dev/2.13/v/value_error
- ablation 'latency_1': requested model 'transformer' was skipped: ValidationError: 1 validation error for OrderBookLevel
quantity
  Value error, quantity must be non-negative; got -0.47933132 [type=value_error, input_value=-0.47933132, input_type=float]
    For further information visit https://errors.pydantic.dev/2.13/v/value_error
- ablation 'horizon_50': requested model 'transformer' was skipped: ValidationError: 1 validation error for OrderBookLevel
quantity
  Value error, quantity must be non-negative; got -0.47933132 [type=value_error, input_value=-0.47933132, input_type=float]
    For further information visit https://errors.pydantic.dev/2.13/v/value_error
- ablation 'lookback_2': requested model 'transformer' was skipped: ValidationError: 1 validation error for OrderBookLevel
quantity
  Value error, quantity must be non-negative; got -0.47933132 [type=value_error, input_value=-0.47933132, input_type=float]
    For further information visit https://errors.pydantic.dev/2.13/v/value_error
- ablation 'lookback_4': requested model 'transformer' was skipped: ValidationError: 1 validation error for OrderBookLevel
quantity
  Value error, quantity must be non-negative; got -0.47933132 [type=value_error, input_value=-0.47933132, input_type=float]
    For further information visit https://errors.pydantic.dev/2.13/v/value_error
- ablation 'feature_top_of_book': requested model 'transformer' was skipped: ValidationError: 1 validation error for OrderBookLevel
quantity
  Value error, quantity must be non-negative; got -0.47933132 [type=value_error, input_value=-0.47933132, input_type=float]
    For further information visit https://errors.pydantic.dev/2.13/v/value_error
- ablation 'feature_imbalance' skipped: ValueError: paper experiment feature_patterns produced no matching feature columns; patterns: ['*imbalance*', '*microprice*']
- ablation 'feature_depth_liquidity': requested model 'transformer' was skipped: ValidationError: 1 validation error for OrderBookLevel
quantity
  Value error, quantity must be non-negative; got -0.47933132 [type=value_error, input_value=-0.47933132, input_type=float]
    For further information visit https://errors.pydantic.dev/2.13/v/value_error
- ablation 'baseline': optional artefact missing: plots/reliability_curve.png
- ablation 'baseline': optional artefact missing: plots/cost_sensitivity.png
- ablation 'baseline': optional artefact missing: plots/confusion_matrix.png
- ablation 'baseline': optional artefact missing: plots/regime_breakdown.png
- ablation 'calibration_bins_5': optional artefact missing: plots/reliability_curve.png
- ablation 'calibration_bins_5': optional artefact missing: plots/cost_sensitivity.png
- ablation 'calibration_bins_5': optional artefact missing: plots/confusion_matrix.png
- ablation 'calibration_bins_5': optional artefact missing: plots/regime_breakdown.png
- ablation 'cost_0bps': optional artefact missing: plots/reliability_curve.png
- ablation 'cost_0bps': optional artefact missing: plots/cost_sensitivity.png
- ablation 'cost_0bps': optional artefact missing: plots/confusion_matrix.png
- ablation 'cost_0bps': optional artefact missing: plots/regime_breakdown.png
- ablation 'cost_1bps': optional artefact missing: plots/reliability_curve.png
- ablation 'cost_1bps': optional artefact missing: plots/cost_sensitivity.png
- ablation 'cost_1bps': optional artefact missing: plots/confusion_matrix.png
- ablation 'cost_1bps': optional artefact missing: plots/regime_breakdown.png
- ablation 'ssl_pretraining_ablation': no traceable runner support for SSL pretraining/fine-tuning yet; ssl_transformer is not registered in the paper-runner model registry
- ablation 'calibration_bins_10': optional artefact missing: plots/reliability_curve.png
- ablation 'calibration_bins_10': optional artefact missing: plots/cost_sensitivity.png
- ablation 'calibration_bins_10': optional artefact missing: plots/confusion_matrix.png
- ablation 'calibration_bins_10': optional artefact missing: plots/regime_breakdown.png
- ablation 'latency_0': optional artefact missing: plots/reliability_curve.png
- ablation 'latency_0': optional artefact missing: plots/cost_sensitivity.png
- ablation 'latency_0': optional artefact missing: plots/confusion_matrix.png
- ablation 'latency_0': optional artefact missing: plots/regime_breakdown.png
- ablation 'latency_1': optional artefact missing: plots/reliability_curve.png
- ablation 'latency_1': optional artefact missing: plots/cost_sensitivity.png
- ablation 'latency_1': optional artefact missing: plots/confusion_matrix.png
- ablation 'latency_1': optional artefact missing: plots/regime_breakdown.png
- ablation 'horizon_50': optional artefact missing: plots/reliability_curve.png
- ablation 'horizon_50': optional artefact missing: plots/cost_sensitivity.png
- ablation 'horizon_50': optional artefact missing: plots/confusion_matrix.png
- ablation 'horizon_50': optional artefact missing: plots/regime_breakdown.png
- ablation 'lookback_2': optional artefact missing: plots/reliability_curve.png
- ablation 'lookback_2': optional artefact missing: plots/cost_sensitivity.png
- ablation 'lookback_2': optional artefact missing: plots/confusion_matrix.png
- ablation 'lookback_2': optional artefact missing: plots/regime_breakdown.png
- ablation 'lookback_4': optional artefact missing: plots/reliability_curve.png
- ablation 'lookback_4': optional artefact missing: plots/cost_sensitivity.png
- ablation 'lookback_4': optional artefact missing: plots/confusion_matrix.png
- ablation 'lookback_4': optional artefact missing: plots/regime_breakdown.png
- ablation 'feature_top_of_book': optional artefact missing: plots/reliability_curve.png
- ablation 'feature_top_of_book': optional artefact missing: plots/cost_sensitivity.png
- ablation 'feature_top_of_book': optional artefact missing: plots/confusion_matrix.png
- ablation 'feature_top_of_book': optional artefact missing: plots/regime_breakdown.png
- ablation 'feature_imbalance': ValueError: paper experiment feature_patterns produced no matching feature columns; patterns: ['*imbalance*', '*microprice*']
- ablation 'feature_depth_liquidity': optional artefact missing: plots/reliability_curve.png
- ablation 'feature_depth_liquidity': optional artefact missing: plots/cost_sensitivity.png
- ablation 'feature_depth_liquidity': optional artefact missing: plots/confusion_matrix.png
- ablation 'feature_depth_liquidity': optional artefact missing: plots/regime_breakdown.png
- ablation row ssl_pretraining_ablation: no traceable runner support for SSL pretraining/fine-tuning yet; ssl_transformer is not registered in the paper-runner model registry
- ablation row feature_imbalance: ValueError: paper experiment feature_patterns produced no matching feature columns; patterns: ['*imbalance*', '*microprice*']

## 11. Limitations

- Real benchmark evidence depends on a user-supplied local FI-2010-style file.
- FI-2010 is a fixed historical benchmark and may not represent other instruments, venues or regimes.
- Execution-aware sensitivity is a simplified proxy analysis with explicit costs and latency assumptions.
- There is no live trading, broker integration or order placement in this report.
- There is no production market impact model.
- SSL results are absent unless a stored SSL model result is genuinely present in the supplied artefacts.

## 12. Reproducibility commands

```bash
python -m chronoslob.cli prepare-fi2010-benchmark --config experiments/fi2010_midprice_h10/config.yaml --data-path data/processed/fi2010/fold1_combined.csv --out experiments/fi2010_midprice_h10/preparation
```

```bash
python -m chronoslob.cli run-paper-experiment --config experiments/fi2010_midprice_h10/config.yaml --data-path data/processed/fi2010/fold1_combined.csv --out experiments/fi2010_midprice_h10 --models majority,logistic,random_forest,gradient_boosting,deeplob_style,transformer --overwrite
```

```bash
python -m chronoslob.cli build-paper-plots --experiment experiments/fi2010_midprice_h10 --overwrite
```

```bash
python -m chronoslob.cli inspect-paper-experiment --experiment experiments/fi2010_midprice_h10
```

```bash
python -m chronoslob.cli run-paper-ablations --config configs/experiments/fi2010_midprice_h10.yaml --data-path data/processed/fi2010/fold1_combined.csv --out experiments/fi2010_midprice_h10_ablations --models majority,logistic,deeplob_style,transformer --ablation-set standard --overwrite
```

```bash
python -m chronoslob.cli build-paper-report --experiment experiments/fi2010_midprice_h10 --ablations experiments/fi2010_midprice_h10_ablations --out reports/chronoslob_empirical_report.md --overwrite
```

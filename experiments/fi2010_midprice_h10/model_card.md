# Model Card: fi2010_midprice_h10

Status: locally executed paper experiment run of the benchmark suite. Inspect the data manifest, split summary and limitations before treating the metrics as benchmark evidence.

## Experiment

- name: `fi2010_midprice_h10`
- task: `midprice_direction`
- horizon: 10
- label column: `label_10`
- split: `temporal` (temporal train/validation/test)
- seed: 0
- code commit: `b6bff37842b84a1abacd8c9da0ca4d8a07ab33f2`
- runner version: `phase-e/paper-experiment-runner/v1`

## Data

- dataset: `FI-2010`
- data source kind: `local_file`
- local source path: `data\processed\fi2010\fold1_combined.csv`
- source SHA-256: `91aef9f1923dfd87955dd5838f709ca198d8ae903cfd6baeb029fe827a0539d0`

## Split Design

- total rows loaded: 77909
- train rows: 54536
- validation rows: 11687
- test rows: 11686
- split is constructed by the deterministic temporal splitter; no shuffling, no stratification and no test-row use during preprocessing or model fitting.

## Models

- requested:
  - `majority`
  - `logistic`
  - `random_forest`
  - `gradient_boosting`
  - `deeplob_style`
  - `transformer`
- successfully run:
  - `majority` (type `majority_class`) on the `test` split with 11686 test rows
  - `logistic` (type `logistic_regression`) on the `test` split with 11686 test rows
  - `random_forest` (type `random_forest`) on the `test` split with 11686 test rows
  - `gradient_boosting` (type `gradient_boosting`) on the `test` split with 11686 test rows
  - `deeplob_style` (type `deeplob_style`) on the `test` split with 11686 test rows
- skipped:
  - `transformer`: ValidationError: 1 validation error for OrderBookLevel
quantity
  Value error, quantity must be non-negative; got -0.47933132 [type=value_error, input_value=-0.47933132, input_type=float]
    For further information visit https://errors.pydantic.dev/2.13/v/value_error

## Neural Settings

- supported_models: `['deeplob_style', 'transformer']`
- lookback: `1`
- transformer_window_length: `4`
- batch_size: `4`
- max_epochs: `1`
- learning_rate: `0.001`
- weight_decay: `0.0`
- gradient_clip_norm: `1.0`
- device: `cpu`
- deterministic: `True`
- dropout: `0.0`
- deeplob_conv_channels: `4`
- deeplob_lstm_hidden_size: `8`
- deeplob_use_batch_norm: `False`
- transformer_field_embedding_dim: `4`
- transformer_model_dim: `16`
- transformer_num_heads: `2`
- transformer_num_layers: `1`
- transformer_feedforward_dim: `32`
- transformer_max_levels_per_side: `2`

## Metric Groups

- predictive metrics emitted:
  - accuracy
  - macro_f1
  - weighted_f1
  - matthews_corrcoef
  - balanced_accuracy
  - n_samples
  - class_count_train
  - class_count_test
- calibration metrics emitted:
  - log_loss
  - brier_score
  - mean_confidence
  - expected_calibration_error
  - calibration_bins
- execution-aware sensitivity metrics emitted:
  - gross_signal_return_proxy
  - net_signal_return_proxy
  - turnover_proxy
  - hit_rate_proxy
- all emitted metrics:
  - accuracy
  - macro_f1
  - weighted_f1
  - matthews_corrcoef
  - balanced_accuracy
  - log_loss
  - n_samples
  - class_count_train
  - class_count_test
  - brier_score
  - mean_confidence
  - expected_calibration_error

## Artefacts

- `config.yaml` (config)
- `data_manifest.json` (data_manifest)
- `results.json` (results)
- `predictions.csv` (predictions)
- `model_card.md` (model_card)
- `confusion_matrix.json` (confusion_matrix)
- `runner_summary.json` (runner_summary)
- `calibration_bins.csv` (calibration_bins)
- `execution_sensitivity.csv` (execution_sensitivity)

## Calibration Evidence

- reliability bins computed from held-out test predictions and written to `calibration_bins.csv`.
- models with calibration bins: `majority`, `logistic`, `random_forest`, `gradient_boosting`, `deeplob_style`
- calibration artefacts are derived from stored prediction rows; no calibrator is fitted on test data.

## Execution-Aware Sensitivity

- cost-aware signal sensitivity rows are computed from stored prediction rows under explicit simple assumptions and are written to `execution_sensitivity.csv`.
- models with sensitivity rows: `majority`, `logistic`, `random_forest`, `gradient_boosting`, `deeplob_style`
- this is a simplified proxy analysis, not a production backtest, and does not claim tradable profitability or live execution.

## Leakage Controls

- Train, validation and test indices come from the deterministic temporal splitter; no random or stratified shuffling is used.
- Per-model train-only feature standardisation is applied for models that require it; standardisation statistics are never fit on validation or test rows.
- No model-selection choice, calibrator, bucket boundary or threshold is fitted on validation or test rows in this phase.
- Label, split and timestamp columns are excluded from the feature matrix.

## Limitations

- This phase supports `majority`, `logistic`, `ridge`, `elastic_net`, `random_forest`, `gradient_boosting`, `deeplob_style` and `transformer`. The DeepLOB-style path is not an exact external-paper reproduction.
- Plot generation remains tracked under a later phase; this runner emits calibration and execution-sensitivity evidence as CSV artefacts only.
- Reported numbers are run-specific and must not be interpreted as profitability, deployability or live-trading evidence.

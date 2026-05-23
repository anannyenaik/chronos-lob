# DeepLOB-style supervised baseline

This note documents the supervised CNN-LSTM neural baseline introduced
in Phase 7B. It uses UK English and is deliberately honest: the model
is a DeepLOB-*style* baseline rather than an exact reproduction of the
original DeepLOB paper, and no benchmark performance is claimed.

## Why this baseline is included

A serious deep learning project should clear two bars before adding
transformers or self-supervised learning. First, the classical
baselines from Phase 6 establish a reproducible floor under temporal
validation. Second, a canonical supervised neural baseline — the
DeepLOB-style CNN-LSTM — establishes whether a deep model can move
above that floor under the same leakage-safe protocol. Phase 7B
delivers the second bar.

The intent is auditability rather than state-of-the-art performance:
the architecture is compact, the training loop is short, the model
trains on CPU and the entire experiment is exercised by deterministic
smoke tests on the bundled synthetic fixture.

## Architecture summary

The model lives in `chronoslob/models/deeplob.py` and ships:

- `DeepLOBConfig` — frozen dataclass with `input_features`, `n_classes`,
  `conv_channels` (default 16), `conv_kernel_size` (default 3),
  `lstm_hidden_size` (default 32), `lstm_layers` (default 1),
  `dropout` (default 0.1) and `use_batch_norm` (default `True`).
- `DeepLOBModel` — `torch.nn.Module` that applies two padded 1D
  convolutions over the feature-as-channel layout, optionally wraps
  each convolution in `BatchNorm1d`, applies ReLU and dropout, then
  feeds the per-time-step representation to an LSTM. The final
  time-step output is projected to class logits.
- `create_deeplob_model(config)` — factory that returns a configured
  model.

The forward pass expects input tensors of shape
`[batch, lookback, n_features]` and returns logits of shape
`[batch, n_classes]`. Invalid input rank or mismatched feature count
raises `ValueError` with a clear message. The model exposes a
`predict_logits` helper that puts the model into eval mode and disables
gradients, and an `n_parameters` helper for reporting.

## Input shape

The data layer from Phase 7A already enforces past-only sequence
windows. `SequenceDataset` emits feature tensors of shape
`[lookback, n_features]` and integer scalar targets, and
`collate_fixed_length_batch` stacks them into
`[batch, lookback, n_features]` and `[batch]` tensors. The DeepLOB-style
model therefore plugs straight into the existing DataLoader path.

## Training protocol

Training is implemented in `chronoslob/training/torch_training.py`:

- `set_torch_deterministic(seed)` seeds Python, NumPy and PyTorch and
  sets deterministic flags where practical, without assuming CUDA.
- `train_one_epoch` trains for one epoch, returns the sample-weighted
  mean loss and refuses an empty dataloader.
- `evaluate_torch_classifier` runs `eval` mode under `no_grad`,
  collects logits/probabilities/predictions/targets and reuses the
  existing `compute_classification_metrics` and
  `confusion_matrix_as_dict` helpers so neural and classical baselines
  share an evaluation surface.
- `fit_torch_classifier` runs a short Adam-based loop with
  `CrossEntropyLoss`, evaluates the validation loader if supplied and
  returns a list of `TorchEpochResult` entries.

CPU is the only device exercised by tests. Non-CPU device strings are
parsed at runtime and raise a clear error if CUDA is requested but
unavailable.

## Train-only standardisation

`chronoslob/training/torch_experiment.py` fits feature-wise mean and
standard deviation on *training rows only*, replaces zero standard
deviations with `1.0` for numerical stability and applies the fitted
statistics to the entire feature frame before sequence windows are
constructed. Validation and test rows never alter scaler statistics.
The fitted mean and std are included in the result payload as
serialisable Python floats. The boundary is identical in spirit to
`TrainOnlyStandardScaler` and `TorchSequenceStandardiser`, but the
experiment runner performs the transform on the dataframe before
window construction so the sequence dataset and DataLoaders consume
already-scaled features.

## Leakage controls

The neural baseline reuses every leakage control already exercised by
the classical baseline runner:

- `validate_no_lookahead(feature_frame, label_frame)` runs first and
  raises if any error-severity issue is reported.
- `select_feature_columns` rejects obvious label-like names so they
  cannot be passed in as features by accident.
- `temporal_train_validation_test_split` is the only splitter used by
  this phase; random splitting is not supported.
- `SequenceWindowConfig` and `build_dataloaders_for_split` constrain
  each partition's sequence windows to that partition's row indices.
- The train-only class mapping inferred from training rows is shared
  with the validation and test loaders, so unseen classes raise rather
  than being silently re-mapped.

## Limitations

The DeepLOB-style model:

- is **not** an exact reproduction of the published DeepLOB
  architecture (which uses an inception block over the 40-feature
  FI-2010 layout with a 100-row lookback);
- targets CPU smoke testing, not state-of-the-art results;
- does not save model checkpoints in this phase;
- does not produce backtest results, PnL or trading metrics;
- does not implement transformers, self-supervised pretraining,
  event tokenisation, multi-task heads, abstention or calibration.

Results from the synthetic-fixture smoke command must never be reported
as FI-2010 benchmark performance. The smoke command emits an explicit
notice to that effect, and the experiment result dictionary carries a
`notes` field with the same warning.

## No benchmark performance is claimed

No FI-2010 benchmark numbers are reported by this repository. A future
phase may match the original DeepLOB architecture exactly, run it
against user-supplied benchmark data and publish the resulting metrics
under the same leakage-safe protocol — until then, no such numbers
exist in the code, configs, reports or tests.

## Future work

- Exact DeepLOB architecture replication, including the inception
  block and the 40-feature, 100-row input layout.
- A real FI-2010 run against user-supplied benchmark data with full
  metric reporting under temporal splits and purged/embargoed
  validation.
- Calibration diagnostics and abstention rules layered on top of the
  classifier outputs in a later phase.
- Transformer baselines and self-supervised pretraining objectives as
  separate later phases.

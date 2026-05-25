# Classical Baselines

ChronosLOB now includes the first supervised modelling layer for classical
forecasting baselines. The purpose is to make later deep learning work auditable:
a larger model is only interesting if it improves on simple baselines under the
same leakage-safe temporal validation protocol.

## Implemented Scope

Implemented baseline types:

- majority class;
- logistic regression;
- ridge classifier;
- elastic-net logistic regression;
- random forest;
- gradient boosting.

These models live behind a small common interface in `chronoslob.models`. The
experiment runner in `chronoslob.training.baseline_experiment` aligns feature and
label frames, builds temporal splits, fits preprocessing on train rows only,
trains each configured baseline and returns validation/test metrics in memory.

## Train-Only Preprocessing

Feature matrices are built from numeric feature columns only. Metadata columns
such as `timestamp`, `symbol`, `split`, `horizon_start` and `horizon_end` are
excluded from model inputs. Obvious label-like names, including `label`,
`future_`, `target`, `direction_`, `return_quantile_`, fill-proxy labels and
adverse-selection labels, are rejected as features.

The standard scaler wrapper must be fitted before transform and stores statistics
from the training partition only. Validation and test rows are transformed with
those fixed training statistics.

## Temporal Validation

Baseline experiments use Phase 5 temporal train/validation/test splitters by
default. Random train/test splitting is not used. Where label horizon metadata is
available, purge and embargo settings can remove training rows whose label
horizons overlap validation or test periods.

## Metrics

The metrics layer reports accuracy, macro F1, weighted F1, Matthews correlation
coefficient and balanced accuracy. Binary Brier score and log loss are reported
only when compatible probabilities are available.

These are forecasting diagnostics. They are not evidence of tradable signal
quality, execution performance or cost-aware signal quality. Classification quality and
tradability remain separate research questions.

## Smoke Tests

The `run-baseline-smoke` CLI command uses the tiny synthetic FI-2010-style fixture
under `tests/fixtures`. It is a pipeline smoke test only. Its output must not be
reported as FI-2010 benchmark performance.

No benchmark tables, model checkpoints or run artefacts are committed by this
module. Metrics are only written when a command is explicitly run with output
writing enabled, and those outputs belong under the gitignored `runs/` tree.

## Later Work

Future phases can add PyTorch dataset and batching infrastructure, followed by a
DeepLOB-style CNN-LSTM supervised baseline. Transformer and self-supervised
learning work should remain later phases and should continue to compare against
these classical baselines under the same temporal validation discipline.

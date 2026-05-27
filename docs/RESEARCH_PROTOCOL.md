# ChronosLOB Research Protocol

This document defines the empirical study protocol that ChronosLOB commits to
for limit order book mid-price direction forecasting. It separates evidence
that already exists from evidence the protocol still requires, and it bounds
the claims that any report built on top of those artefacts is allowed to make.

The protocol is deliberately conservative. It exists to make the final research
conclusion testable, to keep evaluation leakage-safe and to keep public claims
inside what stored artefacts can support.

## 1. Research Question

Given a leakage-safe temporal pipeline over the FI-2010 limit order book
benchmark, how do well-specified classical baselines, a DeepLOB-style
convolutional-recurrent baseline and a normalised-matrix transformer compare on
short-horizon mid-price direction forecasting under official split-aware
evaluation, when each model is judged jointly on predictive quality, calibration
quality and execution-aware proxy sensitivity?

The question targets representation and forecasting quality on a public
benchmark. It does not concern trading profitability and does not concern
broker-integrated execution.

## 2. Scope

In scope:

- FI-2010 NoAuction ZScore normalised matrices, official train/test split
  semantics, with validation carved only from official train rows.
- Mid-price direction targets at the canonical FI-2010 horizons.
- Classical baselines, a DeepLOB-style baseline and a normalised-matrix
  transformer baseline.
- Calibration diagnostics and execution-aware proxy diagnostics on held-out
  test rows.
- Local systems measurements for loader, feature, runner and inference paths.

Out of scope:

- Profitability claims, broker integration and any form of live tradability.
- Self-supervised pretraining result claims until a traceable train-only
  pretraining and supervised fine-tuning path exists in the runner.
- Cross-asset transfer claims until adapters for additional public limit order
  book datasets are added with leakage-safe evaluation.

## 3. Dataset Protocol

The repository does not download FI-2010, does not commit raw FI-2010 archives,
does not commit processed FI-2010 CSV files and does not commit prediction or
intermediate matrix files. The acquisition, verification and conversion path is
documented in [FI2010_DATA_ACQUISITION.md](FI2010_DATA_ACQUISITION.md).

The canonical local layout for a single fold is a combined CSV file with the
FI-2010 normalised columns, the official horizon label columns and a `split`
column whose values mark each row as `train` or `test`. The order of rows
within each partition is preserved as the temporal order, because the canonical
FI-2010 file does not provide explicit timestamps.

## 4. FI-2010 Fold Plan

The study evaluates the official FI-2010 NoAuction ZScore folds. For the
purposes of this protocol the planned fold identifiers are `1`, `2`, `3`, `4`
and `5`, corresponding to the official NoAuction ZScore train/test file pairs.

Each fold is evaluated under its own combined CSV file. The combined file is
built locally from the official `.txt` matrices using the documented conversion
path, and is not checked into the repository.

A fold is considered evaluated only when all required artefacts in section 16
are present for that fold.

## 5. Official Split Semantics

The combined CSV `split` column is the source of truth for evaluation:

- Rows where `split == "train"` form the official training partition. They are
  the only rows allowed for preprocessing decisions, model fitting,
  hyperparameter selection, calibrator fitting and early stopping.
- Rows where `split == "test"` form the official test partition. They are not
  used for preprocessing, fitting, validation or model selection.

The protocol does not permit any leakage from `test` rows into any preprocessing
statistic, label statistic, calibrator parameter or model selection decision.

## 6. Train, Validation and Test Rules

Validation rows are carved from the official train partition using a contiguous
temporal tail, controlled by `validation_fraction_within_train`. The protocol
default is `0.15`. Validation rows are used for hyperparameter selection,
calibrator fitting and early stopping. The official test partition is held out
end-to-end.

Standardisation, label encoding and any other fittable preprocessing must be
fit on the train partition only and then applied to validation and test rows.
The runner enforces this contract through the existing train-only fitting
tests.

## 7. Allowed Model Families

The study reports four model families:

- A majority-class baseline.
- Well-specified classical baselines (logistic regression, optional ridge or
  elastic-net classifiers, random forest, gradient boosting).
- A DeepLOB-style convolutional-recurrent baseline trained on the normalised
  FI-2010 matrix.
- A normalised-matrix transformer baseline.

Self-supervised pretraining is gated under section 10 and is not part of the
reportable model set until the gate is satisfied.

## 8. Baseline Requirements

For each evaluated fold, the protocol requires:

- The majority-class baseline as a sanity lower bound.
- At least logistic regression and gradient boosting as classical references.
- The DeepLOB-style baseline as the standard sequence-model reference.

Random forest and other classical baselines are recorded when they run cleanly.
Each baseline must be evaluated on the same official train and test partitions
as the neural baselines, with the same label horizon and the same metrics.

## 9. Neural Benchmark Requirements

The neural baselines must:

- Read normalised FI-2010 rows through the matrix adapter. Raw order-book
  snapshot schemas remain strict and z-score-normalised rows are not coerced
  into raw snapshots.
- Be fit only on the official train partition with validation carved from the
  train tail.
- Use a fixed seed list shared with the classical baselines, so seed-induced
  variance can be measured.
- Produce stored predictions and confidences sufficient to reconstruct the
  predictive metrics, calibration bins and execution-aware sensitivity rows.

## 10. Self-Supervised Claim Gate

The protocol does not allow any self-supervised result claim until all of the
following are true:

- The paper runner supports a traceable self-supervised pretraining stage that
  uses only rows from the official train partition (and a validation tail
  carved from that partition).
- The runner supports a supervised fine-tuning stage on the same train
  partition with the same label horizon.
- Stored artefacts include the pretraining config, the fine-tuning config, a
  pretraining manifest and the same predictive, calibration and execution-aware
  evidence streams used for the supervised baselines.
- The ablation suite distinguishes the pretrained-then-fine-tuned variant from
  the supervised-only transformer baseline within the same fold.

Until those conditions hold, self-supervised pretraining is recorded as a
skipped ablation with an explicit reason, never as a result.

## 11. Metrics

Each evaluated model on each evaluated fold reports the same metric set on the
official test partition:

- Predictive metrics: macro F1, accuracy, log loss, Brier score.
- Confidence summary: mean predicted confidence.
- Calibration metrics: expected calibration error and a reliability-bin table.
- Execution-aware proxy metrics: turnover and a cost-aware signal proxy under
  the configured cost, threshold and latency grids.

Predictive metrics, calibration metrics and execution-aware proxy metrics are
reported as separate evidence streams. They are not collapsed into a single
ranking number.

## 12. Calibration Diagnostics

Calibration evidence is computed from held-out test predictions only. The
runner never refits calibrators on test rows and never alters model selection
based on test results. The reliability table uses the configured number of
bins (default ten) and is stored as `calibration_bins.csv` alongside the
expected calibration error and the mean predicted confidence in `results.json`.

A model is considered calibration-evaluated for a fold only when both the
reliability table and the calibration scalars exist for that fold.

## 13. Execution-Aware Proxy Diagnostics

Execution evidence is a simplified proxy under explicit cost, threshold and
latency assumptions. It is not a backtest, not a tradable strategy and carries
no claim of live tradability.

For each evaluated fold the runner stores `execution_sensitivity.csv` with
rows for the configured confidence thresholds, cost levels (in basis points)
and latency steps. The proxy uses a mid-forward-return target computed from
the bid and ask price columns at the configured price level. The proxy is
reported as cost-aware signal quality, not as profit.

## 14. Statistical Uncertainty

Each evaluated configuration must be run under a seed list of at least three
seeds. For each metric the protocol reports the mean and standard deviation
across seeds within the same fold. Across folds, the protocol reports per-fold
results and a per-metric mean and standard deviation across folds.

The protocol does not claim significance from a single seed on a single fold.
Single-seed, single-fold runs may be recorded as preparatory artefacts but
must not be presented as the main empirical result.

## 15. Ablation Requirements

Each fold's main run is accompanied by a controlled ablation suite under the
existing paper-ablation contract. Required ablation axes are:

- Lookback window length for the sequence models.
- Feature group toggles supported by the runner.
- A no-calibrator versus calibrated variant for at least one neural baseline.
- A self-supervised pretraining ablation, recorded as `skipped` with an
  explicit reason until section 10 is satisfied.

Each ablation entry must record either a complete child experiment directory
or an explicit skip status with a documented reason. Hidden skips are not
permitted.

## 16. External Benchmark Comparison

The study reports its results against published FI-2010 numbers under the
same official split semantics and the same label horizons. For each fold the
report cites the comparison source, the metric used and any preprocessing
differences. The protocol does not claim parity with, or superiority over,
external published numbers unless the comparison is made on the same fold,
the same horizon and the same metric definition.

The protocol explicitly does not claim state-of-the-art results. The
comparison is positioned as a transparency check against the published record.

## 17. Artefact Traceability

For each evaluated fold the protocol requires the following artefacts to be
present:

- The exact config used for preparation and for the paper experiment run.
- The combined CSV path and a content hash recorded in the data manifest.
- The split summary recording official split semantics and the validation
  carve.
- `results.json` with the metric set in section 11.
- `calibration_bins.csv` as required in section 12.
- `execution_sensitivity.csv` as required in section 13.
- Plots produced from stored artefacts (reliability curve, cost sensitivity,
  confusion matrix).
- An ablation directory with the ablation manifest required in section 15.
- A systems benchmark directory with the loader, feature, runner and
  inference measurements for the same environment.

Any reported metric must trace to a versioned config, data source identifier,
seed, split definition, code commit where available and stored output files.

## 18. Public Claim Boundaries

Reports built on top of these artefacts may describe:

- Reproducible empirical protocol design and execution.
- Leakage-safe temporal validation and official split-aware evaluation.
- Predictive, calibration and execution-aware proxy results on held-out
  official test rows.
- Cost and latency sensitivity of model-derived signals under the configured
  proxy.
- Local systems measurements in the recorded environment.

Reports may not describe ChronosLOB as a profitable trading system, may not
claim guaranteed returns, may not claim market-beating performance, may not
claim state-of-the-art performance, may not claim self-supervised result
parity until section 10 is satisfied and may not describe the platform as a
foundation-model release. The platform is research software for offline
market microstructure experiments with no claim of live tradability.

## 19. What This Study Can Prove

Once the protocol is satisfied across multiple folds, the study can support:

- A leakage-safe, official split-aware comparison of classical, DeepLOB-style
  and normalised-matrix transformer baselines on FI-2010 mid-price direction.
- Calibrated forecasting evidence with reliability tables and expected
  calibration error per model per fold.
- Execution-aware proxy diagnostics that describe how predictive quality maps
  to cost-aware signal quality under the configured assumptions.
- Cross-seed variance estimates within each fold and cross-fold variance
  estimates per metric.
- Local systems measurements that describe loader, feature, runner and
  inference behaviour in the recorded environment.

## 20. What This Study Cannot Prove

The study cannot support:

- Any claim about trading profitability, alpha generation or risk-adjusted
  performance of a deployable strategy.
- Any claim about behaviour under broker, exchange or market-microstructure
  conditions outside the FI-2010 file boundary.
- Any claim about self-supervised pretraining benefits until the gate in
  section 10 is satisfied.
- Any claim about cross-asset transfer until additional public limit order
  book datasets are integrated under the same protocol.
- Any claim about market impact, slippage realism or fill realism beyond what
  the configured proxy explicitly models.

These boundaries are intentional. They preserve the integrity of the
empirical study and the trustworthiness of every artefact it produces.

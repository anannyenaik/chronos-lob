# Calibration and Uncertainty

Phase 15 adds calibration and uncertainty utilities for supervised
classification outputs. The focus is probabilistic forecast quality: whether a
model's stated confidence is consistent with observed label frequencies. This
is useful for noisy market microstructure modelling because short-horizon
market-state labels can be unstable across regimes, venues and liquidity
conditions.

## Accuracy Versus Confidence

Accuracy measures how often the top predicted class is correct. Calibration
asks a different question: when a model assigns roughly 80 percent confidence,
is it correct roughly 80 percent of the time? A classifier can have useful
accuracy while being overconfident, underconfident or poorly calibrated in
specific confidence ranges.

The forecasting-versus-tradability gap remains important. Better calibrated
probabilities do not imply cost-adjusted signal quality or execution viability.

## Metrics

Negative log-likelihood measures the probability assigned to the realised
class. It penalises confident wrong predictions strongly and is the objective
used by temperature scaling in this phase.

Brier score measures the squared distance between the predicted probability
vector and the one-hot target. Lower values indicate probabilities closer to the
realised class labels.

Expected calibration error groups predictions into confidence bins. For each
bin, the average confidence is compared with empirical accuracy. The ECE is the
count-weighted average gap across bins.

Reliability bins expose the data needed for later plotting. Each bin records
the bin edges, count, accuracy, average confidence, calibration gap and
contribution to ECE. Phase 15 creates this data only; it does not create a
dashboard or notebook output.

## Temperature Scaling

Temperature scaling learns one positive scalar applied to logits before
softmax. A temperature above one softens probabilities; a temperature below one
sharpens them. The scalar is fitted by minimising negative log-likelihood on a
calibration split.

The fitted temperature must be learned only from calibration or validation
logits and labels. Test data may be transformed and evaluated with the fitted
temperature, but it must not influence the parameter. Fitting on evaluation or
test logits would leak future evaluation information into the probabilistic
post-processing stage.

For multi-task heads, the natural extension is one temperature per task, fitted
from that task's calibration labels. Tasks remain separate because direction,
volatility regime, spread widening and proxy labels can have different
calibration behaviour.

## Confidence Filtering and Abstention

Confidence filtering reports coverage and accuracy after retaining examples
whose maximum softmax probability exceeds a threshold. Abstention curves sort
examples by confidence and evaluate retained accuracy at specified coverage
levels.

These diagnostics are not trading rules. They do not model spreads, fees,
latency, queue position, market impact, partial fills or venue rules. A
high-confidence prediction is not a tradable signal by itself, and abstention
analysis is not execution-aware validation.

## Split Discipline

Calibration should respect the same leakage-safe discipline as model training:

- features use only information available at or before timestamp `t`;
- labels may use future information only as documented targets;
- transforms and calibration parameters are fitted on the training or explicit
  calibration partition only;
- evaluation and test partitions are transformed with already fitted
  parameters;
- no random time-series split is introduced by this phase.

## Scope and Limitations

The synthetic smoke command is a reproducible code-path check. It creates local
synthetic logits and labels, fits temperature scaling on the first synthetic
subset and evaluates on a separate synthetic subset. The outputs are not
benchmark results, market evidence, Sharpe claims or tradability
claims.

Phase 15 does not implement execution simulation, transaction-cost modelling,
queue-position modelling, latency sensitivity, turnover, PnL, backtesting,
dashboards, attention analysis, real FI-2010 benchmark results or live data
ingestion.

The next step is Phase 16, execution-aware validation, where any simplified
research simulation assumptions must be explicit and auditable.

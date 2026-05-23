# CLI Smoke Outputs

These outputs were captured locally for report-writing reference. Synthetic fixture commands are labelled synthetic and are not market evidence, benchmark evidence or execution evidence.

- Optional smoke-training commands included: `False`
- Commands captured: `13`

## 1. Environment and package smoke check.

- Synthetic fixture or synthetic smoke: `False`
- Optional command: `False`
- Exit code: `0`
- Timed out: `False`

Command:

```bash
python -m chronoslob.cli doctor
```

Stdout:

```text
ChronosLOB Doctor                             
+-------------------------------------------------------------------------+
| Check              | Value                                              |
|--------------------+----------------------------------------------------|
| Python             | 3.11.9                                             |
| Package import     | chronoslob 0.1.0                                   |
| Project root       | C:\Users\Lenovo\Programming\ChronosLOB\chronos-lob |
| Folder: configs    | present                                            |
| Folder: chronoslob | present                                            |
| Folder: tests      | present                                            |
| Folder: notebooks  | present                                            |
| Folder: reports    | present                                            |
+-------------------------------------------------------------------------+
```

Stderr:

```text
<empty>
```

## 2. Local repository audit summary.

- Synthetic fixture or synthetic smoke: `False`
- Optional command: `False`
- Exit code: `0`
- Timed out: `False`

Command:

```bash
python -m chronoslob.cli run-project-audit
```

Stdout:

```text
ChronosLOB project audit
  root:                         C:\Users\Lenovo\Programming\ChronosLOB\chronos-lob
  configs:                      25
  reports:                      35
  tests:                        74
  CLI commands:                 32
  required paths status:        pass
  forbidden-claim issue count:  0
  synthetic-labelling issues:   0
  large-file issue count:       0
  public README status:         pass
  public docs status:           pass
  public wording issue count:   0
  issues:                       none
  network calls:                none performed
  outputs:                      not written
  final status:                 pass
```

Stderr:

```text
<empty>
```

## 3. FI-2010-style loader inspection on a synthetic fixture.

- Synthetic fixture or synthetic smoke: `True`
- Optional command: `True`
- Exit code: `0`
- Timed out: `False`

Command:

```bash
python -m chronoslob.cli inspect-fi2010 --path tests/fixtures/fi2010/tiny_fi2010_like.csv
```

Stdout:

```text
ChronosLOB FI-2010 inspection
  path:         tests\fixtures\fi2010\tiny_fi2010_like.csv
  rows:         6
  features:     11
  labels:       0
  has labels:   False
  has split:    True
  has ts col:   True
  ok:           True
  errors:       0
  warnings:     0
```

Stderr:

```text
<empty>
```

## 4. Feature-pipeline inspection on a synthetic FI-2010-style fixture.

- Synthetic fixture or synthetic smoke: `True`
- Optional command: `True`
- Exit code: `0`
- Timed out: `False`

Command:

```bash
python -m chronoslob.cli inspect-features-fi2010 --path tests/fixtures/fi2010/tiny_fi2010_like.csv
```

Stdout:

```text
ChronosLOB FI-2010 feature inspection
  path:                tests\fixtures\fi2010\tiny_fi2010_like.csv
  rows:                6
  feature columns:     22
  synthetic_time:      False
  skipped time feats:  False
  validation ok:       True
  validation errors:   0
  validation warnings: 0
  sample columns:      ['best_bid_price', 'best_ask_price', 'best_bid_quantity', 'best_ask_quantity', 'mid_price', 'spread', 'relative_spread', 'microprice', 'bid_depth_1', 'ask_depth_1']
```

Stderr:

```text
<empty>
```

## 5. Label-pipeline inspection on a synthetic FI-2010-style fixture.

- Synthetic fixture or synthetic smoke: `True`
- Optional command: `True`
- Exit code: `0`
- Timed out: `False`

Command:

```bash
python -m chronoslob.cli inspect-labels-fi2010 --path tests/fixtures/fi2010/tiny_fi2010_like.csv
```

Stdout:

```text
ChronosLOB FI-2010 label inspection
  path:                tests\fixtures\fi2010\tiny_fi2010_like.csv
  rows:                6
  label columns:       3
  validation ok:       True
  validation errors:   0
  validation warnings: 0
  sample columns:      ['label_10', 'label_50', 'label_100']
```

Stderr:

```text
<empty>
```

## 6. Canonical event-log inspection on a synthetic fixture.

- Synthetic fixture or synthetic smoke: `True`
- Optional command: `True`
- Exit code: `0`
- Timed out: `False`

Command:

```bash
python -m chronoslob.cli inspect-event-log --path tests/fixtures/event_logs/synthetic_snapshots.jsonl
```

Stdout:

```text
WARNING: event log path is a synthetic fixture; outputs are not real market data.
ChronosLOB event log inspection
  path:             tests\fixtures\event_logs\synthetic_snapshots.jsonl
  records:          6
  book events:      0
  snapshots:        6
  symbols:          TESTUSDT
  timestamp range:  2024-01-01T00:00:00+00:00 to 2024-01-01T00:00:05+00:00
  sequence range:   1..6
  sha256 prefix:    4838e65f460d
  outputs:          not written
  network calls:    none performed
```

Stderr:

```text
<empty>
```

## 7. Event-token inspection on a synthetic event-log fixture.

- Synthetic fixture or synthetic smoke: `True`
- Optional command: `True`
- Exit code: `0`
- Timed out: `False`

Command:

```bash
python -m chronoslob.cli inspect-event-tokens --path tests/fixtures/event_logs/synthetic_snapshots.jsonl
```

Stdout:

```text
WARNING: event log path is a synthetic fixture; outputs are not real market data.
ChronosLOB event-token inspection
  path:                    tests\fixtures\event_logs\synthetic_snapshots.jsonl
  symbol filter:           none
  input records:           6
  tokenised records:       30
  token windows:           30
  window length:           8
  snapshot-derived tokens: yes
  vocabulary sizes:
    event_type: 13
    side: 9
    price_bucket: 14
    quantity_bucket: 12
    time_delta_bucket: 13
    context_bucket: 14
    source: 7
  first token ids:
    pos=0 ids={'event_type': 5, 'side': 7, 'price_bucket': 5, 'quantity_bucket': 5, 'time_delta_bucket': 5, 'context_bucket': 11, 'source': 6}
    pos=1 ids={'event_type': 6, 'side': 5, 'price_bucket': 8, 'quantity_bucket': 9, 'time_delta_bucket': 5, 'context_bucket': 11, 'source': 6}
    pos=2 ids={'event_type': 6, 'side': 5, 'price_bucket': 7, 'quantity_bucket': 10, 'time_delta_bucket': 5, 'context_bucket': 11, 'source': 6}
    pos=3 ids={'event_type': 6, 'side': 6, 'price_bucket': 11, 'quantity_bucket': 10, 'time_delta_bucket': 5, 'context_bucket': 11, 'source': 6}
    pos=4 ids={'event_type': 6, 'side': 6, 'price_bucket': 12, 'quantity_bucket': 10, 'time_delta_bucket': 5, 'context_bucket': 11, 'source': 6}
  outputs:                 not written
  network calls:           none performed
```

Stderr:

```text
<empty>
```

## 8. Transformer architecture support summary.

- Synthetic fixture or synthetic smoke: `False`
- Optional command: `True`
- Exit code: `0`
- Timed out: `False`

Command:

```bash
python -m chronoslob.cli inspect-transformer
```

Stdout:

```text
ChronosLOB Market Transformer encoder
  Supervised encoder over field-wise tokenised market microstructure.
  No self-supervised objective, calibration or execution claim.
  token fields expected:    7
  token field names:        ['event_type', 'side', 'price_bucket', 'quantity_bucket', 'time_delta_bucket', 'context_bucket', 'source']
  vocab sizes (default):
    event_type: 5
    side: 5
    price_bucket: 5
    quantity_bucket: 5
    time_delta_bucket: 5
    context_bucket: 5
    source: 5
  defaults:
    field_embedding_dim:    16
    model_dim:              64
    num_heads:              4
    num_layers:             2
    feedforward_dim:        128
    dropout:                0.1
    max_sequence_length:    128
    num_classes:            3
    pooling:                mean
    activation:             gelu
    use_layer_norm:         True
    pad_token_id:           0
  model parameter count:    83379
  No training was run.
```

Stderr:

```text
<empty>
```

## 9. Self-supervised objective support summary.

- Synthetic fixture or synthetic smoke: `False`
- Optional command: `True`
- Exit code: `0`
- Timed out: `False`

Command:

```bash
python -m chronoslob.cli inspect-ssl
```

Stdout:

```text
ChronosLOB SSL Transformer wrapper
  Self-supervised pretraining over field-wise tokenised market microstructure.
  No supervised market labels, calibration, execution simulation or benchmark claim.
  enabled objectives:       ['masked_field', 'next_field']
  masked fields:            ['event_type', 'side', 'price_bucket', 'quantity_bucket']
  next-predicted fields:    ['event_type', 'side', 'price_bucket', 'quantity_bucket']
  ignore_index:             -100
  contrastive enabled:      False
  masking config:
    mask_probability:        0.15
    mask_token_probability:  0.8
    random_token_probability: 0.1
    keep_token_probability:  0.1
    force_at_least_one_mask: True
  loss weights:
    masked_field: 1.0
    next_field: 1.0
    contrastive: 0.0
  transformer backbone:
    model_dim:              64
    num_heads:              4
    num_layers:             2
    max_sequence_length:    128
  model parameter count:    85979
  No training was run.
```

Stderr:

```text
<empty>
```

## 10. Multi-task fine-tuning support summary.

- Synthetic fixture or synthetic smoke: `False`
- Optional command: `True`
- Exit code: `0`
- Timed out: `False`

Command:

```bash
python -m chronoslob.cli inspect-multitask
```

Stdout:

```text
ChronosLOB Multi-Task Transformer
  Supervised fine-tuning heads over a shared field-wise token transformer backbone.
  No calibration, confidence filtering, execution simulation, backtesting or performance claim.
  supervised tasks:
    direction: type=classification, classes=3, loss_weight=1.0
    return_quantile: type=classification, classes=5, loss_weight=1.0
    volatility_regime: type=classification, classes=3, loss_weight=1.0
    spread_widening: type=classification, classes=2, loss_weight=1.0
    fill_probability: type=classification, classes=2, loss_weight=1.0
    adverse_selection: type=classification, classes=2, loss_weight=1.0
  transformer backbone:
    token fields:          ['event_type', 'side', 'price_bucket', 'quantity_bucket', 'time_delta_bucket', 'context_bucket', 'source']
    model_dim:             64
    num_heads:             4
    num_layers:            2
    feedforward_dim:       128
    dropout:               0.1
    max_sequence_length:   128
    pooling:               mean
  head dropout:            0.1
  freeze backbone:         False
  model parameter count:   84484
  No training was run.
```

Stderr:

```text
<empty>
```

## 11. Calibration and uncertainty support summary.

- Synthetic fixture or synthetic smoke: `False`
- Optional command: `True`
- Exit code: `0`
- Timed out: `False`

Command:

```bash
python -m chronoslob.cli inspect-calibration
```

Stdout:

```text
ChronosLOB calibration and uncertainty
  supported metrics:
    negative_log_likelihood
    brier_score
    expected_calibration_error
    reliability_bins
    confidence_filtering
    abstention_curve
  default ECE bins:          10
  default confidence range:  0.0..1.0
  default thresholds:        [0.5, 0.6, 0.7, 0.8, 0.9]
  temperature scaling:       one positive scalar fitted on a calibration split by minimising NLL
  training run:              none
  outputs:                   not written
  performance claims:        none
```

Stderr:

```text
<empty>
```

## 12. Execution-aware validation support summary.

- Synthetic fixture or synthetic smoke: `False`
- Optional command: `True`
- Exit code: `0`
- Timed out: `False`

Command:

```bash
python -m chronoslob.cli inspect-execution-validation
```

Stdout:

```text
ChronosLOB execution-aware validation
  supported execution modes:
    aggressive
    passive
    hybrid
  supported cost components:
    fixed_fee_per_trade
    proportional_fee_bps
    aggressive half-spread or full-spread convention
    passive adverse-selection/slippage assumptions
  supported risk constraints:
    inventory_limit
    max_trades
    max_turnover
    optional max_drawdown
  summary metrics:
    coverage, fill_rate, hit_rate
    gross_pnl_simulated, total_cost_simulated, net_pnl_simulated
    turnover, adverse_selection_rate, latency sensitivity
    confidence-threshold sweep
  training run:       none
  live trading:       not implemented
  outputs:            not written
  statement:          simulation infrastructure only; no tradability claim
```

Stderr:

```text
<empty>
```

## 13. Transfer, regime, ablation and sensitivity support summary.

- Synthetic fixture or synthetic smoke: `False`
- Optional command: `True`
- Exit code: `0`
- Timed out: `False`

Command:

```bash
python -m chronoslob.cli inspect-analysis
```

Stdout:

```text
ChronosLOB analysis layer
  supported analysis types:
    regime
    transfer
    ablation
    sensitivity
    summary
  supported regime kinds:
    volatility
    spread
    liquidity
    confidence
    latency
  supported ablation categories:
    feature
    token_field
    model_component
    objective
    task_head
    execution_setting
  supported sensitivity parameters:
    confidence_threshold
    latency_steps
    fee_bps
    spread_multiplier
    turnover_cap
    inventory_cap
    mask_probability
    temperature
  supported metric names:
    accuracy
    macro_f1
    mcc
    nll
    brier_score
    ece
    coverage
    fill_rate
    simulated_net_pnl
    total_cost
    turnover
    adverse_selection_rate
    max_drawdown
    latency_steps
  predictive metric names:
    accuracy
    macro_f1
    mcc
    nll
    brier_score
    ece
  execution metric names:
    coverage
    fill_rate
    simulated_net_pnl
    total_cost
    turnover
    adverse_selection_rate
    max_drawdown
    latency_steps
  note: analysis summaries require real upstream experiment records and do not generate evidence by themselves.
  outputs:             not written (read-only command)
  network calls:       none performed
```

Stderr:

```text
<empty>
```

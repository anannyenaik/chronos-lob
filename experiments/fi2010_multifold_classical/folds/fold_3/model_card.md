# FI-2010 Multi-Fold Classical Fold 3

## Scope

- runner: classical baselines only
- dataset: `FI-2010`
- task: `midprice_direction`
- target: `label_10`
- fold CSV: `data\processed\fi2010\fold3_combined.csv`
- full predictions: not written

## Split

- train rows: 90477
- validation rows: 15967
- official test rows: 37023
- validation is carved only from official train rows
- preprocessing is fitted only on train rows

## Models

- `majority`
- `logistic`
- `ridge`
- `elastic_net`
- `random_forest`
- `gradient_boosting`

## Limitations

- No neural or self-supervised models are run by this runner.
- Execution-sensitivity rows are proxy diagnostics under configured assumptions.
- No profitability or deployment claim is made.

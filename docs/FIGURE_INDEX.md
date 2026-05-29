# FI-2010 Figure Index

`build-fi2010-figures` generates diagnostic figures from stored FI-2010
neural full-grid artefacts. It does not train models, load the raw FI-2010
dataset or invent missing diagnostics.

`build-fi2010-ablation-figures` separately generates figures from stored
microstructure feature-ablation artefacts.

The current figure set is real, but not every possible diagnostic is present:
17 neural/full-grid figures are completed and unsupported regime plots are
skipped because the stored prediction artefacts do not include regime labels.
Skipped plots are part of the evidence trail, not missing positive results.

## Command

```bash
python -m chronoslob.cli build-fi2010-figures \
  --neural-full-grid experiments/fi2010_neural_full_grid \
  --execution-v3 experiments/fi2010_execution_v3 \
  --out reports/figures/fi2010_neural_full_grid \
  --models all \
  --horizons all \
  --folds all \
  --seeds all \
  --overwrite \
  --strict
```

Smoke-test grids require `--allow-smoke-test`. Smoke figures are labelled as
diagnostics only and do not support empirical claims.

## Mapping Audit

Before plotting, the builder writes `label_mapping_audit.json`.

- Raw FI-2010 label `1` maps to `up`.
- Raw FI-2010 label `2` maps to `stationary`.
- Raw FI-2010 label `3` maps to `down`.
- Probability columns are consumed in canonical class order:
  `prob_up`, `prob_stationary`, `prob_down`.

Strict mode fails when the mapping is missing or ambiguous.

## Figures

| Figure | Source artefact | Supports | Does not support | Real grid required |
| --- | --- | --- | --- | --- |
| Confusion matrix | `runs/**/predictions.csv` selected by `best_model_selection.json` | Class-level error patterns for the best stored model per horizon | Claims about unavailable horizons or missing prediction files | Real or smoke/synthetic with predictions |
| Reliability curve | `runs/**/predictions.csv` | Calibration diagnostics using predicted confidence | Calibration evidence when probabilities are missing | Real or smoke/synthetic with predictions |
| Macro-F1 across folds | `results_summary.csv` | Fold-level variability by model/objective | Per-sample error analysis | Real or smoke/synthetic with result rows |
| Macro-F1 across horizons | `results_summary.csv` | Mean horizon trend with fold/seed error bars when available | Unrun horizons | Real or smoke/synthetic with result rows |
| ECE across horizons | `results_summary.csv` | Calibration-error trend by horizon | Reliability curve detail without prediction rows | Real or smoke/synthetic with result rows |
| SSL matched delta | `results_summary.csv` | Matched supervised-vs-SSL deltas for fold/horizon/seed/lookback pairs | Unmatched SSL comparisons | Real or smoke/synthetic with matched rows |
| Confidence threshold fraction | `runs/**/predictions.csv` | Fraction of samples retained by confidence threshold | Execution quality or trading capacity | Real or smoke/synthetic with predictions |
| Confidence threshold retained Macro-F1 | `runs/**/predictions.csv` | Predictive quality on retained high-confidence samples | Performance at thresholds with too few retained samples | Real or smoke/synthetic with predictions |
| Cost-adjusted proxy | Execution proxy CSVs, if present | Conservative proxy diagnostics only | Execution-aware evidence when v3 artefacts are absent | Requires valid proxy artefacts |
| Execution-v3 confidence active fraction | `confidence_threshold_aggregate.csv` | Offline execution-aware proxy diagnostic for retained active-trade fraction | Capacity or trading-volume claims | Requires execution-v3 artefacts |
| Execution-v3 confidence cost-adjusted proxy | `confidence_threshold_aggregate.csv` | Offline execution-aware proxy diagnostic for confidence-filtered signal quality | Profitability or live-trading claims | Requires execution-v3 artefacts |
| Execution-v3 cost sensitivity | `cost_sensitivity_summary.csv` | Cost sensitivity of the cost-adjusted proxy | PnL, venue costs or broker routing claims | Requires execution-v3 artefacts |
| Execution-v3 latency sensitivity | `latency_sensitivity_summary.csv` | Row-step latency degradation within fold/partition boundaries | Network or exchange latency claims | Requires execution-v3 artefacts |
| Execution-v3 fill assumption comparison | `fill_assumption_summary.csv` | Filled-count and proxy-quality differences by fill assumption | Real queue-position or fill-quality claims | Requires execution-v3 artefacts |
| Execution-v3 adverse selection | `adverse_selection_summary.csv` | Adverse-selection proxy by confidence bucket | Post-trade market-impact evidence | Requires execution-v3 artefacts |
| Execution-v3 regime execution | `regime_execution_summary.csv` | Execution proxy metrics by explicit regime label | Unobserved or inferred regime claims | Requires execution-v3 regime rows |
| Regime breakdown | Prediction rows with explicit `regime` labels | Regime-labelled predictive diagnostics | Regime claims inferred from timestamps or row numbers | Requires explicit regime labels |

## Outputs

Every completed figure has:

- a PNG file
- a source CSV under `source_data/`
- a metadata JSON under `metadata/`
- an entry in `figure_manifest.json`

Skipped plots are also recorded in `figure_manifest.json` with a reason. Missing
execution-v3 or regime artefacts are skipped explicitly.

## Feature-Ablation Figures

```bash
python -m chronoslob.cli build-fi2010-ablation-figures \
  --feature-ablations experiments/fi2010_feature_ablations \
  --out reports/figures/fi2010_feature_ablations \
  --allow-smoke-test
```

The ablation figure builder produces feature-group delta macro-F1, delta MCC,
only-one-group, remove-one-group degradation, proxy-versus-non-proxy and
horizon-specific feature-importance plots when the supporting CSV rows exist.
Every completed plot has a PNG, source CSV, metadata JSON and manifest entry;
unavailable plots are skipped with a reason.

## Evidence-Pack Use

The evidence pack reads each `figure_manifest.json`, records smoke status,
completed and skipped figures, and marks figure outputs stale if their source
artefacts changed after the manifest was generated. Figure rows support visual
traceability only; they do not upgrade smoke diagnostics into empirical
evidence.

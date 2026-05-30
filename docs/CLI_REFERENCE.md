# CLI Reference

The CLI is a local research-engineering interface. Commands are
read-only unless their help text explicitly says outputs are written,
and no command performs hidden network calls. Commands that take
fixture paths under `tests/fixtures/` operate on small synthetic files.

Run commands with:

```bash
python -m chronoslob.cli <command> [options]
```

## Diagnostics

| Command   | Description                                       |
| --------- | ------------------------------------------------- |
| `version` | Print the installed package version.              |
| `doctor`  | Print Python, package and key-folder checks.      |

## Data Inspection

| Command                                       | Description                                                                  |
| --------------------------------------------- | ---------------------------------------------------------------------------- |
| `inspect-fi2010 --path PATH`                  | Load a local FI-2010-style file and print a data-quality summary.            |
| `inspect-event-log --path PATH`               | Inspect a canonical local JSONL event log.                                   |
| `inspect-binance-replay --snapshot --updates` | Reconstruct a local Binance-style order book offline from supplied JSON.     |
| `replay-binance-l2-sample [--snapshot] [--updates] [--out]` | Replay a local Binance Spot L2 snapshot-plus-diff sample offline into a reconstructed book, replay-quality report, update-continuity audit and event-level feature summary; writes the Binance L2 extension artefacts. Aggregated L2 diff-depth replay only; crypto-market engineering evidence, not equity, not live trading and not profitability evidence. See [BINANCE_L2_EXTENSION.md](BINANCE_L2_EXTENSION.md). |
| `inspect-event-tokens --path PATH`            | Tokenise a canonical event log and print vocabulary and window counts.      |

## Features, Labels and Splits

| Command                                              | Description                                                                  |
| ---------------------------------------------------- | ---------------------------------------------------------------------------- |
| `inspect-features-fi2010 --path PATH`                | Build and validate leakage-safe microstructure features locally.             |
| `event-log-to-features --path PATH`                  | Replay a canonical event log into past-only feature rows.                    |
| `inspect-labels-fi2010 --path PATH`                  | Extract FI-2010 labels or build ChronosLOB labels and print counts.          |
| `inspect-split --rows N`                             | Print default temporal train, validation and test split counts.              |
| `init-run --name NAME --phase PHASE --seed --root`   | Create a metadata-only run directory.                                        |

## Models

| Command                                                                | Description                                              |
| ---------------------------------------------------------------------- | -------------------------------------------------------- |
| `inspect-baselines`                                                    | List supported classical baseline model types.           |
| `run-baseline-smoke --path PATH`                                       | Run a deterministic synthetic-fixture baseline check.    |
| `inspect-torch-dataset --path PATH --lookback N`                       | Build a tiny sequence `DataLoader` and print shapes.     |
| `inspect-deeplob`                                                      | Print DeepLOB-style supervised baseline defaults.        |
| `run-deeplob-smoke --path PATH --lookback N --epochs N`                | Run a deterministic synthetic DeepLOB-style check.       |
| `inspect-transformer`                                                  | Print supervised transformer encoder defaults.           |
| `run-transformer-smoke --path PATH`                                    | Run a deterministic synthetic transformer check.         |
| `inspect-ssl`                                                          | Print self-supervised transformer wrapper defaults.      |
| `run-ssl-smoke --path PATH`                                            | Run a tiny synthetic self-supervised objective check.    |
| `inspect-multitask`                                                    | Print multi-task transformer defaults.                   |
| `run-multitask-smoke --path PATH`                                      | Run a tiny synthetic supervised multi-task check.        |

## Calibration, Execution and Analysis

| Command                              | Description                                                       |
| ------------------------------------ | ----------------------------------------------------------------- |
| `inspect-calibration`                | Print calibration and uncertainty utility support.                |
| `run-calibration-smoke`              | Run deterministic synthetic calibration diagnostics.              |
| `inspect-execution-validation`       | Print execution-aware validation support.                         |
| `run-execution-validation-smoke`     | Run deterministic synthetic execution-validation diagnostics.     |
| `inspect-analysis`                   | Print transfer, regime, ablation and sensitivity support.         |
| `run-robustness-analysis-smoke`      | Run deterministic synthetic robustness-analysis diagnostics.      |

## Experiment Artefacts

| Command                                                              | Description                                                        |
| -------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `inspect-experiment-artifacts --experiment PATH`                     | Inspect an experiment directory against the artefact contract.     |
| `prepare-fi2010-benchmark --config PATH --data-path PATH --out PATH` | Prepare a local-only FI-2010 benchmark input (no model run).       |
| `verify-fi2010-local --data-path PATH`                               | Inspect a local FI-2010 ``.txt`` matrix safely without loading it into memory; prints byte size, SHA-256, row and column counts, label class distribution and layout issues. |
| `convert-fi2010-official --input PATH --output PATH [--split train\|test] [--overwrite]` | Convert one official FI-2010 ``.txt`` matrix into a header-bearing CSV file matching the existing loader convention. Operates on user-supplied local files only. |
| `inspect-fi2010-multifold --config PATH --extracted-root PATH [--processed-root PATH] [--folds all\|1,2,...]` | Report which configured FI-2010 fold train and test source files exist under a local extracted dataset root. Read-only. See [FI2010_MULTIFOLD_PROTOCOL.md](FI2010_MULTIFOLD_PROTOCOL.md). |
| `prepare-fi2010-multifold --config PATH --extracted-root PATH [--processed-root PATH] --out PATH [--folds all\|1,2,...] [--overwrite]` | Convert configured FI-2010 train and test source files into one split-aware combined CSV per fold under ``--processed-root``, plus per-fold manifests and a ``summary.json`` under ``--out``. Preparation only; no model is trained. |
| `run-fi2010-multifold-classical --config PATH [--processed-root PATH] --out PATH [--models majority[,logistic,ridge,elastic_net,random_forest,gradient_boosting]] [--folds all\|1,2,...] [--overwrite]` | Run classical baselines across prepared FI-2010 fold CSVs and write `summary.json`, fold metrics, aggregate metrics, calibration summary, execution proxy summary and per-fold lightweight artefacts. Full prediction rows are not written by default. |
| `inspect-fi2010-neural-plan --config PATH [--folds all\|1,2,...] [--models deeplob_style[,matrix_transformer]]` | Inspect the serious FI-2010 neural benchmark grid without training or writing outputs. See [NEURAL_BENCHMARK_PROTOCOL.md](NEURAL_BENCHMARK_PROTOCOL.md). |
| `run-fi2010-neural-benchmark --config PATH --processed-root PATH --out PATH [--folds fold_1[,fold_2]] [--models deeplob_style[,matrix_transformer]] [--seeds 11] [--lookbacks 20] [--max-epochs 1] [--overwrite]` | Run selected supervised neural configurations on prepared fold CSVs and write lightweight aggregate artefacts. The default options are smoke-level; the full configured grid requires `--allow-full-benchmark`. See [FI2010_NEURAL_BENCHMARKS.md](FI2010_NEURAL_BENCHMARKS.md). |
| `run-fi2010-ssl-neural-benchmark --config PATH --processed-root PATH --out PATH [--folds fold_1] [--seeds 0] [--lookbacks 10] [--objective masked_field\|next_field\|both] [--mask-probability 0.15] [--next-field-bucket-count 3] [--pretrain-epochs 1] [--max-epochs 1] [--batch-size 16] [--device cpu] [--overwrite] [--fail-fast] [--no-write-full-predictions]` | Pretrain a transformer encoder on FI-2010 training rows only with a masked-field and/or next-field self-supervised objective, save the encoder checkpoint, config snapshot, metrics JSON, git commit and SHA256 manifest, then fine-tune on mid-price direction and compare against a supervised baseline of identical architecture, folds, horizons, seeds and preprocessing. The validation pretraining loss is carved from training rows; official test rows are never consulted during pretraining. No SSL effectiveness is claimed. See [FI2010_SSL_BENCHMARKS.md](FI2010_SSL_BENCHMARKS.md). |
| `run-fi2010-ssl-v2-benchmark --processed-root PATH [--config PATH] [--out experiments/fi2010_ssl_v2_benchmark] [--baseline-source experiments/fi2010_neural_proper_training_subset_v2] [--folds 1] [--horizons 10,50] [--seeds 0] [--lookbacks 50] [--objectives supervised,masked_reconstruction,market_state_multitask] [--pretrain-epochs 5] [--max-epochs 25] [--patience 5] [--batch-size 1024] [--mask-probability 0.30] [--future-bucket-count 3] [--contrastive] [--device cpu]` | Run the second-generation market-state-aware SSL objective with structured group masking and future-state auxiliary heads. Baselines can be imported from matched proper-training artefacts; imported baselines are disclosed. Outputs include `ssl_v2_comparison.csv`, loss components, configs, manifests and a partial/complete evidence label. |
| `run-fi2010-neural-full-grid --config PATH --processed-root PATH --folds 1,2,3,4,5 --horizons 10,20,50 --seeds 0,1,2 --lookbacks 20 --objectives supervised,masked_reconstruction,next_field --pretrain-epochs N --max-epochs N --batch-size N --device cpu\|cuda --out PATH [--res<!-- -->ume\|--no-res<!-- -->ume] [--smoke-test]` | Run and aggregate the supervised matrix transformer against masked-reconstruction and next-field SSL matrix-transformer variants. The runner keeps horizon selection config-bound by writing per-horizon config snapshots, records failed and reused-existing runs explicitly, writes per-run predictions and SHA256 manifests, and produces aggregate and matched-delta tables. Smoke-test outputs are marked as smoke only and are not empirical evidence. |
| `build-fi2010-figures --neural-full-grid PATH [--execution-v3 PATH] [--out reports/figures/fi2010_neural_full_grid] [--models all] [--horizons all] [--folds all] [--seeds all] [--overwrite\|--no-overwrite] [--allow-smoke-test] [--strict\|--no-strict]` | Generate reproducible FI-2010 neural, SSL and optional execution-v3 diagnostic figures from stored full-grid artefacts. The command first validates the canonical FI-2010 mapping `1=up`, `2=stationary`, `3=down`; strict mode refuses ambiguous class/probability mappings. Every completed figure has a PNG, source CSV, metadata JSON and `figure_manifest.json` entry. Missing execution-v3 or regime artefacts are skipped explicitly. See [FIGURE_INDEX.md](FIGURE_INDEX.md). |
| `audit-fi2010-features [--path PATH] [--feature-groups all] [--label-columns label_10,...] [--split-column split] [--strict\|--no-strict]` | Audit the FI-2010 microstructure feature registry and builder for label exclusion, future-horizon exclusion, rolling past-only logic, snapshot-delta partition boundaries and row alignment. |
| `run-fi2010-feature-ablations [--config PATH] [--processed-root PATH] [--data-path PATH] --folds LIST --horizons LIST --seeds LIST --models LIST --feature-groups LIST --ablation-modes LIST --out PATH [--res<!-- -->ume\|--no-res<!-- -->ume] [--strict\|--no-strict] [--smoke-test] [--summary-only\|--no-summary-only] [--save-predictions\|--no-save-predictions] [--save-heavy-artefacts\|--no-save-heavy-artefacts]` | Run classical FI-2010 microstructure feature ablations, write per-run configs/metrics/status files and root `results_summary.csv`, `aggregate_summary.csv` and `feature_delta_summary.csv`. Summary-only mode is the storage-light default; prediction rows and cached feature matrices are optional. Snapshot-flow columns are labelled proxies, not true OFI. See [FEATURE_ABLATIONS.md](FEATURE_ABLATIONS.md). |
| `build-fi2010-ablation-figures --feature-ablations PATH --out PATH [--overwrite\|--no-overwrite] [--allow-smoke-test]` | Generate reproducible feature-ablation figures with PNG, source CSV, metadata JSON and skipped-plot manifest entries. |
| `analyse-fi2010-feature-ablations --feature-ablations PATH [--extra-feature-ablations PATHS] --out PATH [--figures\|--no-figures] [--overwrite\|--no-overwrite] [--allow-smoke-test]` | Build a scoped feature-stability analysis from retained lightweight feature-ablation tables, including horizon/model/fold/seed deltas, `snapshot_order_flow_proxy` scope, claim assessment and optional figures. Raw predictions are not required; execution-aware ablation diagnostics require retained prediction-level outputs or a targeted rerun. |
| `analyse-fi2010-uncertainty [--classical PATH] [--neural PATH] --out PATH [--baseline gradient_boosting] [--ci-level 0.95] [--bootstrap-iterations 1000] [--bootstrap-seed 0] [--overwrite]` | Compute fold-level confidence intervals, paired model comparisons against the baseline, rank stability and a combined ranking from stored multi-fold tables. Diagnostic only. See [STATISTICAL_UNCERTAINTY.md](STATISTICAL_UNCERTAINTY.md). |
| `analyse-fi2010-ssl-results [--full-grid PATH] [--proper-training PATH] [--out PATH] [--figures\|--no-figures] [--overwrite]` | Build the SSL failure-analysis report from retained lightweight comparison tables. Separates the completed one-epoch matched full grid from the longer-training proper-training subset, writes per-objective/horizon/fold/seed delta CSVs and a claim assessment, and never requires deleted raw prediction files or checkpoints. The completed grid does not support a broad SSL improvement claim; the only positive predictive-metric signal is narrow to fold 1, horizon 50 in the partial subset, while calibration worsened. See [reports/ssl_failure_analysis/ssl_failure_analysis.md](../reports/ssl_failure_analysis/ssl_failure_analysis.md). |
| `run-fi2010-brutal-ablations --config PATH [--neural-config PATH] [--processed-root PATH] [--classical PATH] [--neural PATH] --out PATH [--families all\|feature_groups,...] [--folds all\|fold_1,...] [--models NAMES] [--neural-lookbacks 20,50] [--max-epochs 5] [--overwrite] [--dry-run]` | Run the brutal ablation layer across feature groups, model class, lookback, horizon, calibration threshold and execution cost/latency. Cheap families refit a fast linear baseline; model-class, calibration and execution reuse stored evidence; the lookback sweep is skipped by default and recorded with a reason. Execution numbers are proxy diagnostics only. See [FI2010_BRUTAL_ABLATIONS.md](FI2010_BRUTAL_ABLATIONS.md). |
| `run-fi2010-execution-v2 [--classical PATH] [--neural PATH] [--ablations PATH] --out PATH [--models NAMES] [--cost-bps 0,1,5] [--latency-steps 0,1] [--confidence-thresholds 0,0.6] [--overwrite]` | Build execution-aware v2 proxy diagnostics (cost, latency, confidence, turnover, adverse-selection, fill and statistical-to-execution degradation) from stored multi-fold and ablation artefacts. Consumes no full predictions or checkpoints. Every metric is a proxy diagnostic; no profitability or tradability claim is made. See [FI2010_EXECUTION_V2.md](FI2010_EXECUTION_V2.md). |
| `build-fi2010-execution-v3 --neural-full-grid PATH [--feature-ablations PATH] [--out experiments/fi2010_execution_v3] [--models all] [--horizons all] [--folds all] [--seeds all] [--confidence-thresholds LIST] [--fee-bps LIST] [--spread-multipliers LIST] [--latency-steps LIST] [--fill-assumptions LIST] [--allow-smoke-test] [--strict\|--no-strict] [--overwrite\|--no-overwrite]` | Build an offline execution-aware proxy diagnostic from FI-2010 full-grid prediction artefacts, or explicitly supplied feature-ablation predictions. It evaluates confidence filtering, costs, row-step latency, fill assumptions, adverse-selection proxies and explicit regime breakdowns when context exists. Missing market context is recorded as skipped or as a documented fallback. This is not a live trading system or profitability claim. See [EXECUTION_VALIDATION_V3.md](EXECUTION_VALIDATION_V3.md). |
| `analyse-fi2010-execution-v3 [--execution-v3 experiments/fi2010_execution_v3] [--out reports/execution_v3_analysis] [--figures\|--no-figures] [--overwrite]` | Build a richer reviewer-facing execution-aware proxy analysis from the retained execution-v3 output tables. It aggregates confidence filtering, active fraction, turnover proxy, cost sensitivity, latency sensitivity, fill-assumption sensitivity and the adverse-selection proxy, records regime diagnostics as an explicit skip, and writes a claim assessment plus optional figures. It reads only retained tables and never requires deleted raw predictions. All outputs are offline proxy diagnostics; not PnL, not live trading and not a production execution simulator. See [EXECUTION_VALIDATION_V3.md](EXECUTION_VALIDATION_V3.md). |
| `build-execution-centrepiece [--execution-analysis reports/execution_v3_analysis] [--execution-v3 experiments/fi2010_execution_v3] [--neural-full-grid experiments/fi2010_neural_full_grid] [--out reports/execution_centrepiece] [--figures\|--no-figures] [--overwrite]` | Build the forecasting-versus-signal-quality execution centrepiece from retained execution-v3 analysis tables and retained neural full-grid aggregate summaries. It writes the central figure, CSVs, report, summary and claim assessment without requiring deleted raw predictions. Offline diagnostic only. See [EXECUTION_VALIDATION_V3.md](EXECUTION_VALIDATION_V3.md). |
| `run-paper-experiment --config PATH --data-path PATH --out PATH [--models majority[,logistic,ridge,elastic_net,random_forest,gradient_boosting,deeplob_style,transformer]] [--overwrite] [--build-plots]` | Run the paper benchmark suite and write a validated artefact directory. Phase F additionally emits `calibration_bins.csv` and `execution_sensitivity.csv` when their inputs are available. Phase G adds `--build-plots`, which generates reproducible plots from stored artefacts. |
| `run-paper-ablations --config PATH --data-path PATH --out PATH [--models majority[,logistic,...]] [--ablation-set smoke|standard] [--overwrite] [--build-plots]` | Run controlled paper-experiment ablations and write `ablation_summary.json`, `ablation_results.csv`, `ablation_manifest.json`, child experiment directories for run ablations and explicit skip reports for unsupported ablations. |
| `run-system-benchmarks --config PATH --data-path PATH --out PATH [--benchmark-set smoke|standard] [--models majority[,logistic,...]] [--overwrite]` | Run local systems benchmarks and write `system_benchmark_summary.json`, `system_benchmark_results.csv`, `environment.json`, category reports and a validated child paper experiment for runner timing. |
| `inspect-system-benchmarks --benchmark PATH`                         | Print a concise, read-only summary of a systems benchmark directory. |
| `build-paper-plots --experiment PATH [--overwrite]`                  | Generate paper experiment plots (`plots/reliability_curve.png`, `plots/cost_sensitivity.png`, `plots/confusion_matrix.png`, and `plots/regime_breakdown.png` when genuine regime data is present) from the artefacts stored inside a completed paper experiment directory. |
| `inspect-paper-experiment --experiment PATH`                         | Print a concise, read-only summary of a paper experiment directory (validation status, evidence streams, prediction/calibration/execution row counts, plot inventory, fixture flag). |
| `build-paper-report --experiment PATH [--ablations PATH] [--systems PATH] --out PATH [--overwrite]` | Build a structured empirical report from stored paper experiment, ablation and systems benchmark artefacts. Missing optional artefacts are marked unavailable or skipped. |
| `build-final-empirical-report --classical PATH --neural PATH --uncertainty PATH [--ablations PATH] [--feature-ablations PATH] [--feature-ablation-analysis PATH] [--execution PATH] [--execution-v3 PATH] [--execution-centrepiece PATH] [--external PATH] [--synthetic-lob PATH] [--binance-l2 PATH] [--ssl PATH] [--neural-full-grid PATH] [--proper-training PATH] [--ssl-v2-analysis PATH] [--evidence-pack PATH] --out PATH [--overwrite]` | Build the final FI-2010 empirical report and summary JSON from stored multi-fold, uncertainty, ablation, execution-proxy and external-context artefacts. With SSL-v2 analysis, feature-ablation analysis and evidence-pack inputs, the report includes scoped SSL-v2 interpretation, stability and claim-audit status. Optional synthetic-lob, Binance L2 and execution-centrepiece inputs add clearly bounded extension sections. |
| `build-evidence-pack --out PATH --neural-full-grid PATH --figures PATH --execution-v3 PATH [--execution-centrepiece PATH] --feature-ablations PATH [--feature-ablation-analysis PATH] --ablation-figures PATH --final-report PATH [--binance-l2 PATH] [--strict\|--no-strict] [--allow-smoke-test] [--overwrite\|--no-overwrite]` | Build the release evidence pack: artefact inventory, claim audit, conservative README snapshot, public bullet files, reproduction commands and release checklist. Smoke artefacts and fixture-only Binance L2 samples remain labelled as diagnostics or partial engineering evidence, not empirical market evidence. See [EVIDENCE_PACK.md](EVIDENCE_PACK.md). |
| `inspect-paper-report --report PATH`                                 | Inspect a generated empirical report and its companion summary JSON. |

## Synthetic Event-Level Extension

| Command                                          | Description                                                            |
| ------------------------------------------------ | ---------------------------------------------------------------------- |
| `run-synthetic-lob-benchmark [--out PATH] [--events-per-regime N] [--seed N] [--horizon N] [--smoke] [--make-figures] [--overwrite]` | Run the synthetic event-level limit-order-book pipeline: deterministic event generation, replay into snapshots, event-level features, future-horizon labels, small baselines and known-regime stress-test diagnostics. Synthetic controlled stress test only; not real-market evidence, no tradability and no change to FI-2010 limitations. See [SYNTHETIC_LOB_EXTENSION.md](SYNTHETIC_LOB_EXTENSION.md). |

## Binance L2 Replay Extension

| Command                                          | Description                                                            |
| ------------------------------------------------ | ---------------------------------------------------------------------- |
| `replay-binance-l2-sample [--snapshot PATH] [--updates PATH] [--out PATH] [--max-depth N] [--window-events N] [--no-stop-on-gap] [--allow-crossed] [--make-figures] [--overwrite]` | Replay a local Binance Spot L2 depth snapshot plus aggregated diff-depth JSONL stream into a reconstructed book and storage-light summaries. Default inputs are Binance-shaped synthetic fixtures. Crypto-market L2 engineering evidence only; not equity, not live trading and not profitability evidence. See [BINANCE_L2_EXTENSION.md](BINANCE_L2_EXTENSION.md). |

## Audit and Evidence Archive

| Command                                          | Description                                                            |
| ------------------------------------------------ | ---------------------------------------------------------------------- |
| `inspect-release-readiness`                      | Inspect README, documentation structure and wording without writing.   |
| `run-project-audit [--strict] [--root PATH]`     | Run local repository audit checks and print inventory counts.          |
| `build-report-archive [--output PATH] [--strict] [--include-smoke-training]` | Build or update the local evidence archive.    |
| `inspect-report-archive [--output PATH]`         | List expected archive files and whether they are present.              |

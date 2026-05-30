# Reproduction Commands

Commands are ordered from feature checks through release audit. Real runs require local FI-2010 data prepared under `data/processed/fi2010`; no runtime duration is promised.

## Feature Audit

Smoke-test version:

```bash
python -m chronoslob.cli audit-fi2010-features --path tests/fixtures/fi2010/tiny_fi2010_like.csv --feature-groups all --no-strict
```

Real-run version:

```bash
python -m chronoslob.cli audit-fi2010-features --path data/processed/fi2010/fold1_combined.csv --feature-groups all --strict
```

Expected output: `read-only CLI output`

Compute and dependency caveat: Requires a local FI-2010-style CSV with label and split columns.

## Feature Ablations

Smoke-test version:

```bash
python -m chronoslob.cli run-fi2010-feature-ablations --data-path tests/fixtures/fi2010/tiny_fi2010_like.csv --out experiments/fi2010_feature_ablations --smoke-test --no-strict
```

Real-run version:

```bash
python -m chronoslob.cli run-fi2010-feature-ablations --config configs/experiments/fi2010_multifold.yaml --processed-root data/processed/fi2010 --folds all --horizons 10,20,50 --out experiments/fi2010_feature_ablations --strict
```

Expected output: `experiments/fi2010_feature_ablations`

Compute and dependency caveat: Real run depends on prepared folds and selected model/grid scope.

## Neural Full Grid

Smoke-test version:

```bash
python -m chronoslob.cli run-fi2010-neural-full-grid --processed-root data/processed/fi2010 --out experiments/fi2010_neural_full_grid --smoke-test
```

Real-run version:

```bash
python -m chronoslob.cli run-fi2010-neural-full-grid --processed-root data/processed/fi2010 --folds 1,2,3,4,5 --horizons 10,20,50 --seeds 0,1,2 --out experiments/fi2010_neural_full_grid
```

Expected output: `experiments/fi2010_neural_full_grid`

Compute and dependency caveat: Real run requires local compute suitable for neural training.

## Proper-Training Neural Subset

Smoke-test version:

```bash
python -m chronoslob.cli run-fi2010-neural-proper-training-subset --config configs/experiments/fi2010_neural_proper_training_smoke.yaml --processed-root data/processed/fi2010 --out experiments/fi2010_neural_proper_training_subset_v2 --folds 1 --horizons 10 --seeds 0 --lookbacks 10 --objectives supervised,masked_reconstruction,next_field --pretrain-epochs 1 --max-epochs 2 --patience 1 --batch-size 16 --smoke-test
```

Real-run version:

```bash
python -m chronoslob.cli run-fi2010-neural-proper-training-subset --config configs/experiments/fi2010_neural_proper_training.yaml --processed-root data/processed/fi2010 --out experiments/fi2010_neural_proper_training_subset_v2 --folds 1,2,3 --horizons 10,50 --seeds 0 --lookbacks 50 --objectives supervised,masked_reconstruction,next_field --pretrain-epochs 5 --max-epochs 25 --patience 5 --batch-size 1024 --device cpu
```

Expected output: `experiments/fi2010_neural_proper_training_subset_v2`

Compute and dependency caveat: Fallback real evidence is partial_real; complete_real requires folds 1-5 at horizons 10 and 50 with the same longer-training protocol.

## SSL Benchmark

Smoke-test version:

```bash
python -m chronoslob.cli run-fi2010-ssl-neural-benchmark --config configs/experiments/fi2010_ssl_smoke.yaml --processed-root data/processed/fi2010 --out experiments/fi2010_ssl --folds fold_1 --seeds 0 --lookbacks 10
```

Real-run version:

```bash
python -m chronoslob.cli run-fi2010-ssl-neural-benchmark --config configs/experiments/fi2010_ssl_smoke.yaml --processed-root data/processed/fi2010 --out experiments/fi2010_ssl --folds all --seeds 0,1,2 --objective both
```

Expected output: `experiments/fi2010_ssl`

Compute and dependency caveat: Real SSL evidence requires train-only pretraining and verified checkpoints.

## Execution-V3

Smoke-test version:

```bash
python -m chronoslob.cli build-fi2010-execution-v3 --neural-full-grid experiments/fi2010_neural_full_grid --out experiments/fi2010_execution_v3 --allow-smoke-test --no-strict
```

Real-run version:

```bash
python -m chronoslob.cli build-fi2010-execution-v3 --neural-full-grid experiments/fi2010_neural_full_grid --out experiments/fi2010_execution_v3 --strict
```

Expected output: `experiments/fi2010_execution_v3`

Compute and dependency caveat: Consumes stored predictions; outputs are offline proxy diagnostics.

## Figures

Smoke-test version:

```bash
python -m chronoslob.cli build-fi2010-figures --neural-full-grid experiments/fi2010_neural_full_grid --out reports/figures/fi2010_neural_full_grid --allow-smoke-test --no-strict
```

Real-run version:

```bash
python -m chronoslob.cli build-fi2010-figures --neural-full-grid experiments/fi2010_neural_full_grid --execution-v3 experiments/fi2010_execution_v3 --out reports/figures/fi2010_neural_full_grid --strict
```

Expected output: `reports/figures/fi2010_neural_full_grid`

Compute and dependency caveat: Requires aggregate tables and optional execution-v3 inputs.

## Ablation Figures

Smoke-test version:

```bash
python -m chronoslob.cli build-fi2010-ablation-figures --feature-ablations experiments/fi2010_feature_ablations --out reports/figures/fi2010_feature_ablations --allow-smoke-test
```

Real-run version:

```bash
python -m chronoslob.cli build-fi2010-ablation-figures --feature-ablations experiments/fi2010_feature_ablations --out reports/figures/fi2010_feature_ablations
```

Expected output: `reports/figures/fi2010_feature_ablations`

Compute and dependency caveat: Requires stored feature-ablation tables.

## Feature Ablation Analysis

Smoke-test version:

```bash
python -m chronoslob.cli analyse-fi2010-feature-ablations --feature-ablations experiments/fi2010_feature_ablations --out reports/feature_ablation_analysis --allow-smoke-test --overwrite
```

Real-run version:

```bash
python -m chronoslob.cli analyse-fi2010-feature-ablations --feature-ablations experiments/fi2010_feature_ablations --out reports/feature_ablation_analysis --overwrite
```

Expected output: `reports/feature_ablation_analysis`

Compute and dependency caveat: Consumes retained lightweight feature-ablation tables; prediction-level execution-aware diagnostics require a targeted prediction-retaining rerun.

## Final Empirical Report

Smoke-test version:

```bash
python -m chronoslob.cli build-final-empirical-report --classical experiments/fi2010_multifold_classical --neural experiments/fi2010_multifold_neural --uncertainty experiments/fi2010_uncertainty --neural-full-grid experiments/fi2010_neural_full_grid --proper-training experiments/fi2010_neural_proper_training_subset_v2 --feature-ablations experiments/fi2010_feature_ablations --feature-ablation-analysis reports/feature_ablation_analysis --execution-v3 experiments/fi2010_execution_v3 --out reports/chronoslob_final_empirical_report.md --overwrite
```

Real-run version:

```bash
python -m chronoslob.cli build-final-empirical-report --classical experiments/fi2010_multifold_classical --neural experiments/fi2010_multifold_neural --uncertainty experiments/fi2010_uncertainty --ablations experiments/fi2010_brutal_ablations --external experiments/fi2010_external_context --neural-full-grid experiments/fi2010_neural_full_grid --proper-training experiments/fi2010_neural_proper_training_subset_v2 --feature-ablations experiments/fi2010_feature_ablations --feature-ablation-analysis reports/feature_ablation_analysis --execution-v3 experiments/fi2010_execution_v3 --evidence-pack reports/evidence_pack --out reports/chronoslob_final_empirical_report.md --overwrite
```

Expected output: `reports/chronoslob_final_empirical_report.md`

Compute and dependency caveat: Consumes stored artefacts only.

## Evidence Pack

Smoke-test version:

```bash
python -m chronoslob.cli build-evidence-pack --out reports/evidence_pack --neural-full-grid experiments/fi2010_neural_full_grid --figures reports/figures/fi2010_neural_full_grid --execution-v3 experiments/fi2010_execution_v3 --feature-ablations experiments/fi2010_feature_ablations --feature-ablation-analysis reports/feature_ablation_analysis --ablation-figures reports/figures/fi2010_feature_ablations --final-report reports/chronoslob_final_empirical_report.md --allow-smoke-test --no-strict --overwrite
```

Real-run version:

```bash
python -m chronoslob.cli build-evidence-pack --out reports/evidence_pack --neural-full-grid experiments/fi2010_neural_full_grid --figures reports/figures/fi2010_neural_full_grid --execution-v3 experiments/fi2010_execution_v3 --feature-ablations experiments/fi2010_feature_ablations --feature-ablation-analysis reports/feature_ablation_analysis --ablation-figures reports/figures/fi2010_feature_ablations --final-report reports/chronoslob_final_empirical_report.md --strict --overwrite
```

Expected output: `reports/evidence_pack`

Compute and dependency caveat: Builds summaries and claim audits; it does not train models.

## Project Audit

Smoke-test version:

```bash
python -m chronoslob.cli run-project-audit
```

Real-run version:

```bash
python -m chronoslob.cli run-project-audit --strict
```

Expected output: `read-only CLI output`

Compute and dependency caveat: Review unrelated or untracked worktree files before release.

# DeepLOB-style under the proper-training neural protocol

Status: **implemented and completed on `main`.**

The broader proper-training workflow can execute the requested 180-cell
supervised grid:

- models: `matrix_transformer`, `deeplob_style`
- folds: 1-5
- seeds: 0-2
- lookbacks: 20, 50, 100
- horizons: 10, 50
- training: validation-only early stopping and best-checkpoint restore

The retained benchmark completed all 180 cells with zero failed runs. Results
remain mixed by model, lookback and horizon.

## Protocol parity

Both supervised model paths use the shared `fit_torch_classifier` training core,
which performs validation-only early stopping and restores the best validation
state before the single official-test evaluation. Both paths use train-only
standardisation and split-confined contiguous windows.

DeepLOB-style proper training supports the supervised objective only. Existing
SSL objectives remain matrix-transformer-only and retain their separate evidence
and claim boundaries.

## Implementation

- `run-fi2010-neural-proper-training-subset` accepts
  `--models matrix_transformer,deeplob_style`.
- Proper-training run specifications carry a model axis and model-aware reuse
  signatures.
- Existing matrix-transformer run paths remain backward-compatible.
- DeepLOB-style runs use the isolated leaf
  `runs/fold_F/horizon_H/seed_S/lookback_L/deeplob_style/supervised/`.
- The proper-training config enables both architectures.
- The Slurm CSV and array cover 180 independent cells.
- Consolidation validates every expected cell before producing aggregates.

## Integrity checks

Tests cover model-axis expansion, collision-free run paths, DeepLOB supervised
enforcement, reuse behaviour and retained artefact generation. The Hamilton
workflow passed a representative timing gate, a six-cell cross-model stage,
both 90-cell model-family arrays and final 180-cell consolidation.

## Scope guarantees

- The published `experiments/fi2010_neural_proper_training_subset_v2` tree and
  SSL-v2 evidence are not modified.
- The one-epoch matched comparison grid remains distinct from this
  performance-oriented proper-training benchmark.
- DeepLOB-style and matrix-transformer results remain separately identifiable.
- No broad model-family ranking follows beyond the exact stored scope.

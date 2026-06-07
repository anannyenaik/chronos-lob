#!/bin/bash
#SBATCH --job-name=chronos_proper_neural_timing
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Representative single-cell timing smoke test for the broader proper-training
# neural benchmark (matrix transformer). It uses the BENCHMARK config (not the
# fixture-scale smoke config) restricted to one cell so the measured walltime is
# representative of real grid cells. It is a feasibility probe only and is not,
# on its own, FI-2010 evidence. Do not commit its artefacts, predictions or
# checkpoints. Modelled on the working ssl_v2_timing.sbatch conventions.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
module purge
module load python/3.12.6
module load cuda/12.4.1
source .venv/bin/activate

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "job=$SLURM_JOB_ID"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

/usr/bin/time -v python -m chronoslob.cli run-fi2010-neural-proper-training-subset \
  --config configs/experiments/fi2010_neural_proper_training.yaml \
  --processed-root data/processed/fi2010 \
  --out experiments/fi2010_proper_neural_hamilton_timing/fold_1_seed_0_lookback_50_horizon_10 \
  --folds 1 \
  --horizons 10 \
  --seeds 0 \
  --lookbacks 50 \
  --models matrix_transformer \
  --objectives supervised \
  --device cuda \
  --no-resume

echo "done=$(date --iso-8601=seconds)"

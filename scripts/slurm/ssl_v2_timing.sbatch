#!/bin/bash
#SBATCH --job-name=chronos_sslv2_timing
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
module purge
module load python/3.12.6
module load cuda/12.4.1
source .venv/bin/activate

echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "job=$SLURM_JOB_ID"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

/usr/bin/time -v python -m chronoslob.cli run-fi2010-ssl-v2-benchmark \
  --processed-root data/processed/fi2010 \
  --out experiments/fi2010_ssl_v2_hamilton_timing/fold_1_horizon_10_seed_1 \
  --baseline-source experiments/fi2010_neural_proper_training_subset_v2 \
  --folds 1 \
  --horizons 10 \
  --seeds 1 \
  --lookbacks 50 \
  --objectives supervised,market_state_multitask \
  --max-epochs 25 \
  --patience 5 \
  --batch-size 1024 \
  --pretrain-epochs 5 \
  --device cuda \
  --resume

echo "done=$(date --iso-8601=seconds)"

#!/bin/bash
#SBATCH --job-name=chronos_sslv2
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=1-20%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
module purge
module load python/3.12.6
module load cuda/12.4.1
source .venv/bin/activate

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" scripts/slurm/ssl_v2_jobs.csv)
IFS=',' read -r fold horizon seed <<< "$line"
task_out="experiments/fi2010_ssl_v2_hamilton_tasks/task_${SLURM_ARRAY_TASK_ID}"

echo "fold=$fold horizon=$horizon seed=$seed"
echo "job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

/usr/bin/time -v python -m chronoslob.cli run-fi2010-ssl-v2-benchmark \
  --processed-root data/processed/fi2010 \
  --out "$task_out" \
  --baseline-source experiments/fi2010_neural_proper_training_subset_v2 \
  --folds "$fold" \
  --horizons "$horizon" \
  --seeds "$seed" \
  --lookbacks 50 \
  --objectives supervised,market_state_multitask \
  --max-epochs 25 \
  --patience 5 \
  --batch-size 1024 \
  --pretrain-epochs 5 \
  --device cuda \
  --resume

echo "done=$(date --iso-8601=seconds)"

#!/bin/bash
#SBATCH --job-name=chronos_proper_neural
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=1-90%1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Broader proper-training neural benchmark (matrix transformer) as a Slurm job
# array: one task per grid cell. 90 cells = 5 folds x 3 seeds x 3 lookbacks x 2
# horizons. Each task trains the matrix transformer with validation-only early
# stopping and best-checkpoint restore (supervised objective) under the
# benchmark config, writing storage-light per-run summaries into an isolated
# per-task tree. Modelled on the working ssl_v2_array.sbatch conventions.
#
# IMPORTANT (set before launching the full array):
#   * --array throttle is %1 by default (one concurrent task) because GPU count
#     and per-GPU memory headroom on the cuda partition have NOT been confirmed.
#     Only raise to %2/%4 after `sinfo -p cuda` + a successful smoke run show
#     multiple GPUs and that one cell fits comfortably in GPU memory.
#   * --time=02:00:00 is a placeholder. Reset it from the timing smoke walltime
#     (e.g. ~3x the measured single-cell walltime) and the cuda partition limit.
#   * Do NOT launch all 90 at once. Run a staged subset first (see runbook):
#       sbatch --array=1-4%1 scripts/slurm/proper_neural_array.sbatch
#     verify outputs, then widen the range.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
module purge
module load python/3.12.6
module load cuda/12.4.1
source .venv/bin/activate

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" scripts/slurm/proper_neural_jobs.csv)
if [ -z "$line" ]; then
  echo "no CSV row for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi
IFS=',' read -r fold seed lookback horizon <<< "$line"

task_out="experiments/fi2010_proper_neural_hamilton_tasks/task_${SLURM_ARRAY_TASK_ID}"

echo "fold=$fold seed=$seed lookback=$lookback horizon=$horizon"
echo "job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

/usr/bin/time -v python -m chronoslob.cli run-fi2010-neural-proper-training-subset \
  --config configs/experiments/fi2010_neural_proper_training.yaml \
  --processed-root data/processed/fi2010 \
  --out "$task_out" \
  --folds "$fold" \
  --horizons "$horizon" \
  --seeds "$seed" \
  --lookbacks "$lookback" \
  --objectives supervised \
  --device cuda \
  --resume

echo "done=$(date --iso-8601=seconds)"

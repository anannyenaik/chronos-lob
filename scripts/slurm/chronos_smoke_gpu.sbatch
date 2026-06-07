#!/bin/bash
#SBATCH --job-name=chronos_smoke_gpu
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "host=$(hostname)"
echo "user=$(whoami)"
echo "date=$(date --iso-8601=seconds)"
echo "pwd=$(pwd)"
echo "job=${SLURM_JOB_ID:-unset}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

module purge
module load python/3.12.6
module load cuda/12.4.1

echo "python=$(python --version 2>&1)"
echo "python_path=$(command -v python)"
echo "git=$(git --version)"
echo "nvcc=$(nvcc --version | tail -n 1)"
nvidia-smi

python - <<'PY'
import sys

print("python_full", sys.version)
try:
    import torch

    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch_check_failed", repr(exc))
PY

df -h .

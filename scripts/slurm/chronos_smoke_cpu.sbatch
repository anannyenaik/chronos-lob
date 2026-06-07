#!/bin/bash
#SBATCH --job-name=chronos_smoke_cpu
#SBATCH --partition=test
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "host=$(hostname)"
echo "user=$(whoami)"
echo "date=$(date --iso-8601=seconds)"
echo "pwd=$(pwd)"
echo "job=${SLURM_JOB_ID:-unset}"

module purge
module load python/3.12.6

echo "python=$(python --version 2>&1)"
echo "python_path=$(command -v python)"
echo "git=$(git --version)"

python - <<'PY'
import sys

print("python_full", sys.version)
try:
    import torch

    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
except Exception as exc:
    print("torch_check_failed", repr(exc))
PY

df -h .

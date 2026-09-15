#!/bin/bash
#SBATCH --job-name=chronos_sslv2_merge
#SBATCH --partition=test
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
module purge
module load python/3.12.6
source .venv/bin/activate

canonical="experiments/fi2010_ssl_v2_benchmark"
for task_dir in experiments/fi2010_ssl_v2_hamilton_tasks/task_*; do
  rsync -a "$task_dir/runs/" "$canonical/runs/"
done

python - <<'PY'
from pathlib import Path

root = Path("experiments/fi2010_ssl_v2_benchmark/runs")
required = ("metrics.json", "curves.csv", "config.json", "status.txt")
missing: list[str] = []
for seed in (0, 1, 2):
    for fold in (1, 2, 3, 4, 5):
        for horizon in (10, 50):
            for objective in ("supervised", "market_state_multitask"):
                run = (
                    root
                    / f"fold_{fold}"
                    / f"horizon_{horizon}"
                    / f"seed_{seed}"
                    / "lookback_50"
                    / objective
                )
                absent = [name for name in required if not (run / name).is_file()]
                if absent:
                    missing.append(f"{run}: {','.join(absent)}")
                    continue
                status = (run / "status.txt").read_text(encoding="utf-8").strip()
                if status not in {"completed", "skipped_existing", "imported_existing"}:
                    missing.append(f"{run}: unsuccessful status {status!r}")

if missing:
    raise SystemExit("Missing completed runs:\n" + "\n".join(missing))
print("validated_completed_runs=60")
PY

python -m chronoslob.cli run-fi2010-ssl-v2-benchmark \
  --processed-root data/processed/fi2010 \
  --out "$canonical" \
  --baseline-source experiments/fi2010_neural_proper_training_subset_v2 \
  --folds 1,2,3,4,5 \
  --horizons 10,50 \
  --seeds 0,1,2 \
  --lookbacks 50 \
  --objectives supervised,market_state_multitask \
  --max-epochs 25 \
  --patience 5 \
  --batch-size 1024 \
  --pretrain-epochs 5 \
  --device cpu \
  --resume

echo "done=$(date --iso-8601=seconds)"

#!/bin/bash
#SBATCH --job-name=chronos_proper_neural_merge
#SBATCH --partition=test
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Consolidate the per-task proper-training neural runs into one canonical tree,
# validate that all 180 expected two-model cells completed, then re-run
# the CLI in consolidation mode (--resume) to regenerate aggregate artefacts.
# CPU-only (no training happens here; completed runs are skipped). Modelled on
# the working ssl_v2_consolidate.sbatch conventions.
#
# Run this only after the array has finished and the staged checks passed.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
module purge
module load python/3.12.6
source .venv/bin/activate

# Separate directory from the published subset_v2 so the existing SSL-v2
# baseline evidence and its committed aggregates are never mutated.
canonical="experiments/fi2010_neural_proper_training_broader"

# Merge per-task run trees into the canonical runs/ directory. rsync -a is
# idempotent; per-cell paths do not collide because each cell has a unique
# fold/horizon/seed/lookback/objective leaf.
mkdir -p "$canonical/runs"
for task_dir in experiments/fi2010_proper_neural_hamilton_tasks/task_*; do
  [ -d "$task_dir/runs" ] || continue
  rsync -a "$task_dir/runs/" "$canonical/runs/"
done

# Validate every expected cell is present and successful before aggregating.
python - <<'PY'
from pathlib import Path

root = Path("experiments/fi2010_neural_proper_training_broader/runs")
required = ("status.txt", "metrics.json", "predictions.csv",
            "curves.csv", "curves.json", "config.json")
success = {"completed", "skipped_existing"}

models = ("matrix_transformer", "deeplob_style")
folds = (1, 2, 3, 4, 5)
seeds = (0, 1, 2)
lookbacks = (20, 50, 100)
horizons = (10, 50)
objective = "supervised"

missing: list[str] = []
expected = 0
for model in models:
    for seed in seeds:
        for fold in folds:
            for lookback in lookbacks:
                for horizon in horizons:
                    expected += 1
                    run = (
                        root
                        / f"fold_{fold}"
                        / f"horizon_{horizon}"
                        / f"seed_{seed}"
                        / f"lookback_{lookback}"
                    )
                    if model == "deeplob_style":
                        run = run / model
                    run = run / objective
                    absent = [name for name in required if not (run / name).is_file()]
                    if absent:
                        missing.append(f"{run}: missing {','.join(absent)}")
                        continue
                    status = (run / "status.txt").read_text(encoding="utf-8").strip()
                    if status not in success:
                        missing.append(f"{run}: unsuccessful status {status!r}")

if missing:
    raise SystemExit(
        f"Incomplete grid ({expected - len(missing)}/{expected} ok):\n"
        + "\n".join(missing)
    )
print(f"validated_completed_runs={expected}")
PY

# Regenerate aggregate artefacts over the full canonical grid. With --resume all
# completed cells are skipped, so no training is repeated; only summaries are
# rebuilt across the whole 180-cell grid.
python -m chronoslob.cli run-fi2010-neural-proper-training-subset \
  --config configs/experiments/fi2010_neural_proper_training.yaml \
  --processed-root data/processed/fi2010 \
  --out "$canonical" \
  --folds 1,2,3,4,5 \
  --horizons 10,50 \
  --seeds 0,1,2 \
  --lookbacks 20,50,100 \
  --models matrix_transformer,deeplob_style \
  --objectives supervised \
  --device cpu \
  --resume

python -m chronoslob.analysis.proper_neural_analysis --source "$canonical"

echo "done=$(date --iso-8601=seconds)"

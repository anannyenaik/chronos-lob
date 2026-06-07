#!/bin/bash
#SBATCH --job-name=chronos_sslv2_post
#SBATCH --partition=test
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
module purge
module load python/3.12.6
source .venv/bin/activate

python -m chronoslob.cli build-final-empirical-report \
  --classical experiments/fi2010_multifold_classical \
  --neural experiments/fi2010_multifold_neural \
  --uncertainty experiments/fi2010_uncertainty \
  --ablations experiments/fi2010_brutal_ablations \
  --feature-ablations experiments/fi2010_feature_ablations \
  --feature-ablation-analysis reports/feature_ablation_analysis \
  --execution experiments/fi2010_execution_v2 \
  --execution-v3 experiments/fi2010_execution_v3 \
  --execution-centrepiece reports/execution_centrepiece \
  --external experiments/fi2010_external_context \
  --neural-full-grid experiments/fi2010_neural_full_grid \
  --proper-training experiments/fi2010_neural_proper_training_subset_v2 \
  --ssl-v2-analysis reports/ssl_v2_analysis \
  --evidence-pack reports/evidence_pack \
  --synthetic-lob reports/synthetic_lob_extension \
  --binance-l2 reports/binance_l2_extension \
  --out reports/chronoslob_final_empirical_report.md \
  --overwrite

python -m chronoslob.cli build-evidence-pack \
  --out reports/evidence_pack \
  --neural-full-grid experiments/fi2010_neural_full_grid \
  --figures reports/figures/fi2010_neural_full_grid \
  --execution-v3 experiments/fi2010_execution_v3 \
  --execution-centrepiece reports/execution_centrepiece \
  --feature-ablations experiments/fi2010_feature_ablations \
  --feature-ablation-analysis reports/feature_ablation_analysis \
  --ablation-figures reports/figures/fi2010_feature_ablations \
  --final-report reports/chronoslob_final_empirical_report.md \
  --strict \
  --overwrite

python -m chronoslob.cli inspect-release-readiness
python -m chronoslob.cli run-project-audit --strict

echo "done=$(date --iso-8601=seconds)"

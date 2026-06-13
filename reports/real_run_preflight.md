# ChronosLOB Real-Run Preflight

Generated as Phase 0 of the real FI-2010 evidence run.

- Date: 2026-05-27
- Repository root: `C:\Users\Lenovo\Programming\ChronosLOB\chronos-lob`
- Branch: `main`
- Commit at preflight: `a72d46f0a1d7e0ccb62853eee6004375f7b5358c`

This run does not delete, stage or commit any worktree file automatically.

## Git status summary

Tracked files modified (not staged):

- `README.md`
- `chronoslob/cli.py`
- `chronoslob/experiments/final_report.py`
- `chronoslob/experiments/neural_adapters.py`
- `chronoslob/models/matrix_transformer.py`
- `docs/CLI_REFERENCE.md`
- `docs/EXPERIMENT_EVIDENCE_INDEX.md`
- `docs/FINAL_EMPIRICAL_REPORT.md`
- `docs/PROJECT_STATUS.md`
- `reports/chronoslob_final_empirical_report.md`
- `reports/chronoslob_final_empirical_report_summary.json`
- `tests/test_final_empirical_report.py`

Untracked infrastructure (new modules, tests, docs, configs) already present
for: execution-v3, figures, ablation figures, label mapping, evidence pack,
feature ablations, neural grid, SSL runner/datasets/experiment, microstructure
features, feature registry, matrix SSL model. Untracked artefact directories
already present: `experiments/fi2010_feature_ablations/` (README only),
`experiments/fi2010_neural_full_grid/` (README only), `reports/evidence_pack/`.

No files were staged, committed or deleted during preflight.

## Dataset availability

FI-2010 NoAuction Z-score, local-only (gitignored), under
`data/processed/fi2010/`:

| Fold | Combined CSV | Rows | Train rows | Test rows |
| --- | --- | --- | --- | --- |
| 1 | `fold1_combined.csv` | 77,909 | 39,512 | 38,397 |
| 2 | `fold2_combined.csv` | present | n/a | n/a |
| 3 | `fold3_combined.csv` | present | n/a | n/a |
| 4 | `fold4_combined.csv` | present | n/a | n/a |
| 5 | `fold5_combined.csv` | 217,404 | 178,252 | 39,152 |

- All five per-fold combined CSVs exist and load.
- Columns = 150, including the `split` column (train/test) and label columns
  `label_10`, `label_20`, `label_50` (plus `label_30`, `label_100`).
- Raw release `data/raw/fi2010/BenchmarkDatasets.zip` and `extracted/` present.
- Config `configs/experiments/fi2010_multifold.yaml` points at
  `data/processed/fi2010` with template `fold{fold}_combined.csv`.

Horizons 10/20/50 and folds 1-5 are therefore supported by the prepared data.

## Command pass/fail status

| Command | Status | Notes |
| --- | --- | --- |
| `python -m pytest` | PASS | 1254 passed, 1 skipped, 2 warnings in 225.31s. The single skip is intentional (`test_paper_report_builder.py:313`: real report already exists). |
| `python -m ruff check .` | PASS | All checks passed. |
| `python -m mypy chronoslob` | PASS | No issues in 116 source files. |
| `python -m chronoslob.cli doctor` | PASS | Python 3.11.9, package import ok, all scaffold folders present. |
| `python -m chronoslob.cli inspect-release-readiness` | PASS | README, docs, formatting, AI-artefact scan, safety/claims scan all pass. |
| `python -m chronoslob.cli run-project-audit --strict` | PASS | 0 forbidden-claim, 0 synthetic-labelling, 0 large-file, 0 public-wording issues. |

## Environment

- Python 3.11.9.
- PyTorch 2.12.0+cpu: **CPU only, CUDA not available.**
- numpy 1.26.4, pandas 3.0.3, scikit-learn 1.8.0.

### Compute note (material to scope)

The neural full grid and feature ablations must run on CPU. A prior real
multi-fold neural run (25 epochs, early stopping) recorded
`matrix_transformer` wall-clock of ~596s (fold 1) to ~3104s (fold 5) per run.
The full-grid command uses 1 pretrain + 1 fine-tune epoch per run, so per-run
cost is far lower, but the full grid is 5 folds x 3 horizons x 3 seeds x 3
objectives = 135 runs and the feature-ablation grid at full scope is large.
Runs skip already-completed run directories by default, write per-run artefacts
incrementally, and may complete as `partial_real` with exact remaining commands
recorded rather than be forced.

## Blockers

None blocking. All quality gates pass and the dataset is present and valid.
The only material constraint is CPU-only compute, which affects wall-clock for
the neural grid and feature ablations (handled via restartable runs that skip
completed work, with partial_real classification when not finished in one pass).

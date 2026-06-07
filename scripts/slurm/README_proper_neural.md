# Broader proper-training neural benchmark — Hamilton runbook

Handoff runbook for executing the broader proper-training neural benchmark on
Durham University Hamilton/NCC HPC. Everything here is Slurm-driven; nothing runs
on a laptop. This benchmark is **matrix-transformer only** (90 cells). See the
"Scope and honesty boundaries" section before changing that.

## What this benchmark is (and is not)

- **Is:** the matrix transformer trained with validation-only early stopping and
  best-checkpoint restore, across 5 folds x 3 seeds x 3 lookbacks (20/50/100) x
  2 horizons (10/50) = **90 cells**, supervised objective, under the benchmark
  config `configs/experiments/fi2010_neural_proper_training.yaml`.
- **Is not:** a two-model (DeepLOB + transformer) grid. DeepLOB-style training
  has full protocol parity (same `fit_torch_classifier` core with best-checkpoint
  restore) **but no proper-training entrypoint** — it is only reachable via the
  one-epoch full-grid runner, which clamps `max_epochs` to 1. A two-model proper
  grid needs new code (loop `_run_deeplob_style` through the proper-training
  harness + unit tests + leakage re-check) and is deliberately out of scope here.
- **Is not** the published `experiments/fi2010_neural_proper_training_subset_v2`
  directory. That tree is the SSL-v2 baseline source and stays untouched. This
  benchmark writes to a **separate** `experiments/fi2010_neural_proper_training_broader`.

## Files

| File | Purpose |
|---|---|
| `proper_neural_jobs.csv` | 90 grid cells, columns `fold,seed,lookback,horizon`. |
| `proper_neural_timing.sbatch` | Single-cell timing smoke test (feasibility gate). |
| `proper_neural_array.sbatch` | Job array, one task per cell, `%1` throttle. |
| `proper_neural_consolidate.sbatch` | Merge + validate 90 runs + regenerate aggregates (CPU). |

## Prerequisites (Phases 2-3)

```bash
cd chronos-lob
git checkout main && git pull --ff-only      # must include these scripts
[ -d .venv ] || python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
mkdir -p logs

python -m chronoslob.cli doctor
python -m pytest tests/test_ssl_v2.py -q
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"

du -sh data/processed/fi2010 || echo "MISSING: data/processed/fi2010"
find data/processed/fi2010 -maxdepth 2 -type f | head
```

`torch.cuda.is_available()` is expected to be `False` on a login node; the GPU
only appears inside a Slurm job. If processed FI-2010 data is missing, transfer
only the processed tree (it is gitignored; never commit it).

## Phase 4 — timing smoke test (the gate)

```bash
sbatch scripts/slurm/proper_neural_timing.sbatch
squeue -u "$USER"
# when done:
grep -E "Elapsed|Maximum resident|Percent of CPU|wall clock" \
  logs/chronos_proper_neural_timing_*.out logs/chronos_proper_neural_timing_*.err
cat experiments/fi2010_proper_neural_hamilton_timing/*/summary.json
```

Record from the timing run: wall-clock, max RSS, GPU name + memory used
(`nvidia-smi` lines in the log), and confirm one cell wrote
`status.txt=completed`, `metrics.json`, `curves.csv/json`, `config.json`.

**Decision rule (do not skip):**
- Full grid walltime estimate = single-cell walltime x 90 / (array concurrency).
- Proceed only if the full grid fits in a few days AND queue limits allow AND
  GPU memory comfortably holds one cell. Otherwise STOP and report that Hamilton
  timing makes the full benchmark impractical. Do not force it.

## Phase 5 — set array concurrency and walltime

Before the array, confirm GPU reality:

```bash
sinfo -p cuda -o "%P %a %l %D %t %G"
```

- Default array throttle is `%1` (one concurrent task) and `--time=02:00:00`,
  both placeholders. If `sinfo` shows multiple GPUs and the smoke run shows one
  cell fits with memory headroom, raise to `%2`/`%4`. Set `--time` to roughly
  3x the measured single-cell walltime, capped by the partition limit.
- Edit `#SBATCH --array` / `#SBATCH --time` in `proper_neural_array.sbatch`, or
  override per submission with `sbatch --array=... --time=...`.

## Phase 6 — staged run (do NOT launch all 90 at once)

```bash
# Stage 1: a small spread of cells.
sbatch --array=1-4%1 scripts/slurm/proper_neural_array.sbatch
squeue -u "$USER"
# Verify stage 1 before widening:
for t in 1 2 3 4; do
  echo "== task $t =="; cat experiments/fi2010_proper_neural_hamilton_tasks/task_$t/runs/*/*/*/*/supervised/status.txt 2>/dev/null
done
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS,ReqMem

# Stage 2 (if stable): one seed family, e.g. seed 0 = tasks 1-30.
sbatch --array=1-30%1 scripts/slurm/proper_neural_array.sbatch
# Stage 3: remaining cells.
sbatch --array=31-90%1 scripts/slurm/proper_neural_array.sbatch
```

`--resume` means re-submitting an already-completed task is a cheap skip, so
overlapping ranges are safe. After each stage check: failures (`sacct` State),
that summaries exist, disk usage (`du -sh experiments/fi2010_proper_neural_hamilton_tasks`),
and logs for any sign of data leakage, broken early stopping, or output
corruption. Stop and report if any appear.

## Phase 7 — consolidate

```bash
sbatch scripts/slurm/proper_neural_consolidate.sbatch
# on success the log prints validated_completed_runs=90 and writes aggregates to:
#   experiments/fi2010_neural_proper_training_broader/{aggregate_summary.json,
#   results_summary.csv, summary.json, ssl_comparison.csv, ...}
```

## What gets committed vs ignored

- **Committed (storage-light):** top-level aggregate summaries under
  `experiments/fi2010_neural_proper_training_broader/` (CSV/JSON), the Slurm
  scripts, and the CSV — exactly as for the existing subset_v2 directory.
- **Ignored (gitignored, never committed):** `…_broader/runs/`, the per-task
  tree `…_hamilton_tasks/`, the timing tree `…_hamilton_timing/`, all
  `predictions.csv`, checkpoints (`*.pt`), and cluster logs (`logs/`).

Bring the lightweight aggregates back to the laptop for analysis, doc updates,
and validation (Phases 7-10 of the master plan). Local validation before any
commit:

```bash
python -m pytest -q
python -m ruff check .
python -m mypy chronoslob
python -m chronoslob.cli doctor
python -m chronoslob.cli run-project-audit --strict
git diff --check
```

## Scope and honesty boundaries

- Do not claim broad neural superiority. Report mixed results as mixed.
- Keep the forecasting-vs-signal-quality story intact.
- Distinguish this performance-oriented proper-training grid from the one-epoch
  matched comparison grid.
- Add Hamilton provenance only if the benchmark actually completed on Hamilton.
- Do not retag v0.2.0. Any v0.3.0 notes are drafts pending explicit approval.

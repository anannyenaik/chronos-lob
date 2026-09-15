# Selected results

These committed summaries support the public tables and figures. Scientific CSV and JSON
files are preserved from source revision `beed52603a25d231c6576f7d6e0610c6c5cc6685`.
The path and SHA-256 digest of each selected file are recorded in
[`manifest.json`](manifest.json).

| Directory | Scientific purpose | Main tables |
| --- | --- | --- |
| `classical/` | Six classical models over five official FI-2010 folds at horizon 10 | `results_summary.csv`, `results_by_fold.csv` |
| `proper_training/` | 180 supervised neural cells over model, fold, seed, lookback and horizon | `results_summary.csv`, `lookback_summary.csv`, `model_summary.csv` |
| `ssl_v2/` | Thirty matched supervised and market-state SSL-v2 comparisons | `ssl_v2_comparison.csv`, `delta_by_horizon.csv`, `delta_by_seed.csv` |
| `feature_ablation/` | 2,520 leakage-safe snapshot-feature ablations | `feature_group_stability.csv`, `feature_delta_by_horizon.csv` |
| `execution/` | Confidence, coverage, cost and latency proxy diagnostics | `confidence_threshold_tradeoff.csv`, `latency_cost_gap.csv` |

`ssl_v2_loss_components.csv` is header-only because epoch-level component histories were
not available in the selected run outputs. The empty table records that absence and is
not used by the public figures.

The manifest is the integrity record for this directory. Historical paths and hashes
inside copied run-summary JSON files describe their original output directories and may
refer to larger files that are not distributed here.

Verify all 37 selected files from the repository root with:

```bash
python scripts/plot_results.py --verify-only
```

The plotting command performs the same check before regenerating the public figures:

```bash
python scripts/plot_results.py
```

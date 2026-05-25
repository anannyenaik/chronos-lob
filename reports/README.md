# Reports

This directory contains implementation notes for the major
ChronosLOB subsystems. Each note describes the technical scope of one
module or layer.

Empirical results, when added, must trace to a versioned config, data
source, seed, code commit and stored output artefact.

The [10/10 empirical upgrade plan](10_10_upgrade_plan.md) records the
next benchmark artefact contract and implementation phases.

`reports/report_archive/` is a generated technical evidence archive of
inventories, current command outputs and Mermaid diagrams. Rebuild it
with `python -m chronoslob.cli build-report-archive`.

`reports/chronoslob_empirical_report.md` is a builder-produced summary
of the real FI-2010 mid-price direction run on the official
NoAuction ZScore fold 1 train/test pair. It links to the paper
experiment directory at `experiments/fi2010_midprice_h10/` and the
ablation directory at `experiments/fi2010_midprice_h10_ablations/`,
records every model that ran, every skip with its reason and every
optional artefact that was omitted, and does not claim profitability,
deployability or live trading.

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

# Reports

This directory contains implementation notes for the major
ChronosLOB subsystems. Each note describes the technical scope of one
module or layer.

Empirical results must trace to a versioned config, data source, seed,
code commit where available and stored output artefacts.

The [FI-2010 empirical release note](10_10_upgrade_plan.md) records the
current evidence status and remaining research work.

The [10/10 research protocol note](10_10_research_protocol.md) is the
implementation-facing companion to the public protocol in
[docs/RESEARCH_PROTOCOL.md](../docs/RESEARCH_PROTOCOL.md) and records the
staged multi-fold upgrade contract.

The [statistical uncertainty layer](../docs/STATISTICAL_UNCERTAINTY.md)
quantifies fold-level variance for the stored multi-fold tables and
writes diagnostic artefacts under `experiments/fi2010_uncertainty/`.
It does not retrain any model and does not promote any model beyond
the reduced-scope evidence already stored.

`reports/report_archive/` is a generated technical evidence archive of
inventories, current command outputs and Mermaid diagrams. Rebuild it
with `python -m chronoslob.cli build-report-archive`.

`reports/chronoslob_empirical_report.md` is a builder-produced summary
of the real FI-2010 mid-price direction run on the official
NoAuction ZScore fold 1 train/test pair. It links to the paper
experiment directory at `experiments/fi2010_midprice_h10/` and the
ablation directory at `experiments/fi2010_midprice_h10_ablations/` and
the systems benchmark directory at
`experiments/fi2010_midprice_h10_systems/`. The report records the
official split-aware evaluation, the normalised FI-2010 matrix path for
the supervised transformer baseline, grouped warnings, every skip with
its reason and every optional artefact that was omitted. It does not
present trading or execution-system claims.

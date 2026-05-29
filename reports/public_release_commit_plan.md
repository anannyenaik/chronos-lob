# Public Release Commit Plan

Date: 2026-05-29

This plan groups the public-release changes for review before commit/push.

## 1. SSL And Neural-Grid Infrastructure

- Include: neural-grid code, tests, configs, `experiments/fi2010_neural_full_grid/README.md`, aggregate CSV/JSON summaries.
- Exclude: transient caches, local data, checkpoints and any incomplete scratch runs.
- Large artefact concerns: per-run predictions are large; include only if the repository policy accepts stored evidence artefacts.
- Generated files: commit generated summaries only when they are the evidence source for public claims.
- Review notes: keep the one-epoch full grid separate from the 25-epoch reduced-scope neural benchmark.

## 2. Figures And Label Mapping

- Include: figure builders, tests, figure manifests, label-mapping audit, source CSVs and public PNGs.
- Exclude: unsupported or manually edited figure outputs.
- Large artefact concerns: PNGs and source data are reviewable but add size.
- Generated files: commit if figures are part of the public evidence pack.
- Review notes: skipped regime plots are expected because regime labels are unavailable.

## 3. Execution-V3

- Include: execution-v3 builder, tests, `experiments/fi2010_execution_v3/` summaries and notes.
- Exclude: any feature-ablation execution-v3 attempt that mixes out-of-manifest prediction scopes.
- Large artefact concerns: CSV summaries are large but are the stored evidence for the diagnostic.
- Generated files: commit regenerated execution-v3 outputs only as a coherent set with the manifest.
- Review notes: use "offline execution-aware proxy diagnostic" and "cost-adjusted proxy"; do not use PnL claims.

## 4. Microstructure Features And Ablations

- Include: feature registry, FI-2010 feature builder, ablation runner, tests, docs and canonical root summaries.
- Exclude: scratch/out-of-manifest feature-ablation run directories unless intentionally reviewed.
- Large artefact concerns: per-run predictions can be large.
- Generated files: commit feature-ablation summaries/manifests if they are cited as evidence.
- Review notes: `snapshot_order_flow_proxy` is a proxy only, not true event-level OFI.

## 5. Evidence Pack And Claim Audit

- Include: evidence-pack generator, tests, `reports/evidence_pack/artefact_inventory.csv`, manifest, summaries, claim audit, supported/unsupported claims, reproduction commands and release checklist.
- Exclude: legacy evidence-pack bullet files with old naming; the generator now writes `public_bullets_*.md`.
- Large artefact concerns: minimal.
- Generated files: commit as a coherent generated bundle after strict rebuild.
- Review notes: SSL macro-F1/calibration improvement claims are unsupported; SSL execution improvement still needs real matched proxy evidence.

## 6. Real Empirical Artefacts

- Include: refreshed summaries and manifests that support `complete_real` / `partial_real` status.
- Exclude: local FI-2010 data, caches, checkpoints and interrupted scratch artefacts.
- Large artefact concerns: inspect CSV/PNG size before push.
- Generated files: commit only when hashes/manifests match the regenerated state.
- Review notes: final report and evidence pack should agree on status counts.

## 7. Public Writing Polish

- Include: `README.md`, docs polish, generated public reports, public audit, claim-safety scan, release notes and this commit plan.
- Exclude: personal-profile pages, employment-targeted wording, manual paper drafts and local legacy evidence-pack bullet files with old naming.
- Large artefact concerns: none beyond generated reports.
- Generated files: final report and evidence pack should be regenerated from artefacts, not hand-edited.
- Review notes: keep public language conservative, evidence-backed and explicit about non-claims.

# Public Release Checklist

Use this checklist before a public repository review.

## Repository Presentation

- [ ] README is concise, technical and research-oriented.
- [ ] CI badge is present if GitHub Actions is enabled.
- [ ] Repository description is updated manually on GitHub.
- [ ] Repository topics are reviewed manually on GitHub.
- [ ] Licence file is present.
- [ ] Roadmap and contribution guidance are present.
- [ ] No internal workflow artefacts or build-instruction files remain.
- [ ] No external-positioning language remains in public documentation.

## Validation

- [ ] `python -m chronoslob.cli inspect-release-readiness` passes.
- [ ] `python -m chronoslob.cli run-project-audit --strict` passes.
- [ ] `python -m pytest` passes.
- [ ] `python -m compileall -q chronoslob tests` passes.
- [ ] `python -m ruff check .` passes.
- [ ] `python -m mypy chronoslob` passes.

## Documentation

- [ ] `docs/CLI_REFERENCE.md` lists all CLI commands.
- [ ] `docs/REPRODUCIBILITY.md` gives canonical Python commands.
- [ ] `docs/PROJECT_STATUS.md` distinguishes implemented and planned work.
- [ ] `docs/SAFETY_AND_LIMITATIONS.md` states public caveats.
- [ ] `docs/REPORT_EVIDENCE_INDEX.md` maps report sections to evidence.
- [ ] Technical report support material is clearly separate from final report
  prose.

## Repository Hygiene

- [ ] No secrets, API keys, access tokens or private paths are committed.
- [ ] No real venue data is committed.
- [ ] No notebook outputs or dashboard artefacts are committed.
- [ ] Large generated files are absent.
- [ ] Synthetic fixtures remain clearly labelled.

## Claims Safety

- [ ] No fake results, invented metrics or manually fabricated tables are added.
- [ ] Synthetic smoke outputs are not presented as market evidence.
- [ ] Prediction quality, calibration quality and execution-aware validation are
  kept separate.
- [ ] No live execution, production readiness or investment-usefulness claim is
  made.
- [ ] Any future performance claim cites reproducible configs, seeds, data
  provenance, code version and output artefacts.
- [ ] The final technical report can be added later from reproducible outputs.

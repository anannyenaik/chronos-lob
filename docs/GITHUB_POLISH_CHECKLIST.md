# GitHub Polish Checklist

Use this checklist before a public-facing release or recruiter review.

## README

- [ ] Project identity is clear and research-oriented.
- [ ] The repository is described as a research platform, not a trading bot.
- [ ] Quickstart commands work on a clean environment.
- [ ] Core smoke commands are present and labelled appropriately.
- [ ] Links point to reproducibility, CLI, status, safety and report evidence
  documentation.
- [ ] Current outputs are described as synthetic plumbing unless real
  experiments are separately run.

## Repository Hygiene

- [ ] CI badge is present if GitHub Actions is enabled.
- [ ] `python -m chronoslob.cli run-project-audit --strict` passes.
- [ ] `python -m pytest`, `compileall`, `ruff` and `mypy` pass.
- [ ] No real venue data is committed.
- [ ] No secrets, API keys, access tokens or private paths are committed.
- [ ] No notebook outputs or dashboard artefacts are committed.
- [ ] Large generated files are absent.

## Documentation

- [ ] `docs/CLI_REFERENCE.md` lists all CLI commands.
- [ ] `docs/REPRODUCIBILITY.md` gives canonical Python commands.
- [ ] `docs/PROJECT_STATUS.md` distinguishes implemented and planned work.
- [ ] `docs/SAFETY_AND_LIMITATIONS.md` states public caveats.
- [ ] `docs/REPORT_EVIDENCE_INDEX.md` maps report sections to evidence.
- [ ] `docs/REPORT_WRITING_GUIDE.md` avoids writing the final report for the
  user.

## Claims Safety

- [ ] No fake results, invented metrics or manually fabricated tables are added.
- [ ] Synthetic smoke outputs are not presented as market evidence.
- [ ] Prediction quality, calibration quality and execution-aware validation are
  kept separate.
- [ ] Any future performance claim cites reproducible configs, seeds, data
  provenance, code version and output artefacts.
- [ ] Public-facing caveats mention synthetic fixtures, public data limits,
  crypto transfer limits and simplified execution assumptions.

## Report Archive

- [ ] `python -m chronoslob.cli build-report-archive` succeeds.
- [ ] `python -m chronoslob.cli inspect-report-archive` reports all expected
  files present.
- [ ] Mermaid diagrams remain text assets and do not require rendering
  dependencies.
- [ ] `reports/report_archive/report_claims_checklist.md` is reviewed before
  writing final report claims.

# Full Audit And CI Hardening

This hardening pass supports ChronosLOB as a reproducible research-engineering
artefact. It does not add model architectures, training objectives, benchmark
results, dashboards, notebook outputs or trading functionality.

## Purpose

The hardening pass checks that the repository is internally consistent before
later report-writing and public documentation work. The focus is quality, safety,
reproducibility and claim discipline across code, CLI commands, configs, reports,
tests and documentation.

## CI Design

The GitHub Actions workflow runs on push and pull request using Ubuntu and Python
3.11. It installs the package with `.[dev,torch]` because existing tests cover
torch-backed data and model plumbing. CI then runs package import, doctor,
pytest, compileall, ruff and mypy checks.

CI does not require real FI-2010 data, real exchange data, secrets, API keys,
remote services or live data downloads. Network access is only used for
dependency installation by GitHub Actions.

## Audit Utilities

`chronoslob.utils.audit` provides local-only helpers for:

- collecting CLI, config, report and test inventories;
- checking required repository paths;
- scanning for unsupported trading or performance claim phrases;
- checking synthetic fixture and smoke-config labelling;
- detecting unexpectedly large repository-facing files;
- returning structured audit results for CLI and tests.

The audit does not call the GitHub API, shell out, mutate files or contact
external services.

## Data And Claim Safety

The claim scanner is a conservative heuristic. It reports file path, line number
and matched phrase for suspicious unsupported claims, while allowing documented
"do not say this" and limitation contexts. Human review remains required before a
public release.

Synthetic fixture checks focus on smoke configs and fixture READMEs. They help
keep plumbing outputs clearly separated from benchmark evidence and real market
evidence.

## Config And Report Inventory

New tests parse YAML configs, check fixture path references, reject obvious secret
fields and ensure configs do not require network data by default. Report tests
check expected reports, limitation coverage, claim discipline and implementation
report discoverability.

## CLI Documentation

`docs/CLI_REFERENCE.md` groups commands by area, including version and doctor,
FI-2010 inspection, feature and label inspection, split and registry, baselines,
torch datasets, DeepLOB smoke, Binance replay, event logs, tokenisation,
transformer, SSL, multi-task, calibration, execution validation, robustness
analysis and audit.

The new `run-project-audit` command prints local inventory counts and issue
counts without writing output files. `--strict` exits non-zero if warnings or
failures are found.

## Reproducibility Documentation

`docs/REPRODUCIBILITY.md` records Python expectations, install commands, local
validation commands, the Windows `make` caveat, data policy, deterministic smoke
notes and interpretation rules for synthetic smoke outputs.

## Remaining Limitations

ChronosLOB remains a research infrastructure project. It has no live trading,
broker integration, production queue model, production partial-fill model,
market impact model, portfolio optimiser or real benchmark result claims.

Before final public release, contributors should:

- rerun the full validation suite in a clean environment;
- confirm GitHub Actions passes remotely;
- review docs for stale release-history wording;
- verify that no local data, generated artefacts, notebook outputs or secrets are
  staged;
- generate any future report metrics only from reproducible experiment artefacts.

This hardening work adds no new model results, benchmark claims, execution
evidence or production-use claims.

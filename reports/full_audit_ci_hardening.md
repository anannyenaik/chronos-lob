# Audit and CI

This note describes the local audit utilities and continuous
integration setup for ChronosLOB. It does not add model architectures,
training objectives or any evaluation evidence.

## CI

The GitHub Actions workflow runs on push and pull request using Ubuntu
and Python 3.11. It installs the package with `.[dev,torch]` because
existing tests cover torch-backed data and model code paths. CI runs
package import, the doctor command, pytest, compileall, ruff and mypy.

CI does not require real FI-2010 data, real exchange data, secrets or
API keys. Network access is used only for dependency installation by
GitHub Actions.

## Audit Utilities

`chronoslob.utils.audit` provides local-only helpers for:

- collecting CLI, config, report and test inventories;
- checking required repository paths;
- scanning for unsupported trading or performance claim phrases;
- checking synthetic fixture and smoke-config labelling;
- detecting unexpectedly large repository-facing files;
- inspecting the public README for required structure and links;
- returning structured audit results for the CLI and tests.

The audit does not call the GitHub API, shell out, mutate files or
contact external services.

## Claim Scanner

The claim scanner is a conservative heuristic. It reports the file
path, line number and matched phrase for suspicious unsupported
claims and allows documented limitation contexts. Human review
remains required before a public release.

## Config and Report Inventory

Tests parse YAML configs, check fixture path references, reject
obvious secret fields and ensure configs do not require network data
by default. Report tests check that expected reports exist and stay
within the claim discipline.

## CLI Surface

The CLI exposes the audit and evidence-archive commands described in
[docs/CLI_REFERENCE.md](../docs/CLI_REFERENCE.md). The
`run-project-audit --strict` command exits non-zero if any warning or
failure is detected.

## Validation Path

The canonical local validation path is documented in
[docs/REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md).

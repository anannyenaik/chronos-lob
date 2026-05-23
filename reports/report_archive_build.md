# Technical Evidence Archive Build

ChronosLOB includes a structured evidence archive for later manual technical
report writing. It does not write the final report, add model functionality,
create benchmark tables or introduce trading-use claims.

## What The Archive Contains

The archive under `reports/report_archive/` contains:

- project, config, module and test inventories;
- an implementation release history;
- curated local CLI smoke outputs;
- limitations and claim-safety indexes;
- reproducibility command bundles;
- Mermaid diagram sources for architecture, data flow, model stack, evaluation
  stack and report dependencies.

## How To Rebuild

```bash
python -m chronoslob.cli build-report-archive
python -m chronoslob.cli inspect-report-archive
```

The default build captures lightweight inspect and audit commands only. It does
not run smoke-training commands unless `--include-smoke-training` is passed.

## Mermaid Text Assets

Diagrams are stored as `.mmd` files so the repository does not need rendering
dependencies or generated images. The final report can render or redraw them
later if needed.

## CLI Output Capture

CLI outputs are captured locally with command, exit code, stdout and stderr.
Synthetic fixture commands are labelled synthetic. Optional command failures are
recorded as warnings by default and become failures only in strict mode.

## Synthetic Scope

Bundled fixtures and smoke outputs are synthetic plumbing checks. They are not
market evidence, benchmark evidence, execution evidence or proof of signal
quality.

## What Is Not Evidence Yet

The archive does not provide real FI-2010 benchmark results, real venue
experiments, production execution assumptions, market impact modelling or final
report conclusions. Those require separate reproducible experiment artefacts.

## Support For Manual Report Writing

The archive gives maintainers stable references for architecture, implemented
scope, validation commands, limitations and claim boundaries. It is intended to
make manual report writing faster without inventing results.

## Remaining Public-Release Work

Manual GitHub metadata review remains outside the local archive build. In
particular, the repository description and topics should be checked in the
GitHub UI before a public release.

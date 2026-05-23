# Report Evidence Archive

This directory is a reference archive for writing the ChronosLOB technical report manually. It is not the final report.

The archive contains repository inventories, phase history, current CLI smoke outputs, config and test cross-references, limitations, claim-safety checks and Mermaid diagram sources.

Synthetic fixture outputs are labelled synthetic. They are useful for checking local plumbing, but they are not market evidence, benchmark evidence, execution evidence or proof of signal quality.

Real benchmark results must be generated separately from documented data, configs, temporal splits, seeds, code versions and output artefacts before the final report makes result claims.

Rebuild with:

```bash
python -m chronoslob.cli build-report-archive
```

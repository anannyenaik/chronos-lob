# Reports

Reports should be generated from reproducible experiment outputs.

Do not add manually invented result tables, placeholder metrics or fake plots. Every
future reported metric should be traceable to a config, code version, data version,
seed and stored experiment artefact.

Current implementation reports document infrastructure and limitations. Synthetic
smoke outputs are plumbing checks only and should not be described as benchmark
results or market evidence.

`reports/report_archive/` is a technical evidence archive for manual report
writing. It is not the final report and should not be treated as performance
evidence.

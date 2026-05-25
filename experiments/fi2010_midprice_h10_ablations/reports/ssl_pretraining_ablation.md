# SSL Pretraining Ablation

Ablation type: `ssl_pretraining`

## Purpose

Document the SSL pretraining status explicitly so an unsupported model family is not hidden inside the ablation grid.

## What Changed

- No child experiment is run for SSL pretraining in this suite.
- The ablation is recorded as skipped in `ablation_summary.json`, `ablation_manifest.json` and `ablation_results.csv`.

## Held Fixed

- paper-runner model registry
- no SSL checkpoint, pretraining config or fine-tuning config
- held-out evaluation requirement before any SSL result is reported

## Artefacts Used

- `ablation_summary.json`
- `ablation_results.csv`
- `ablation_manifest.json`

## Status

- skipped

## Reason

- reason: no traceable runner support for SSL pretraining/fine-tuning yet
- no traceable runner support for SSL pretraining/fine-tuning yet; ssl_transformer is not registered in the paper-runner model registry
- `ssl_transformer` is intentionally not registered in the paper runner model registry, and the ablation runner does not report SSL results without a run.

## Requirements Before Enabling

- a train-only pretraining stage with a stored pretraining config
- a stored checkpoint or weight-transfer trace so the fine-tuning stage is reproducible
- a fine-tuning config that uses the pretrained representation and is fitted without test-row input
- a held-out evaluation that validates the artefact contract before any SSL claim is made

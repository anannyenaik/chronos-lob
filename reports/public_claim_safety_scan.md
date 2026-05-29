# Public Claim Safety Scan

Date: 2026-05-29

Scan command:

```bash
rg -n -i --glob "*.md" "profitable|profit|PnL|alpha|tradable|live trading|automated order-placement|SOTA|state-of-the-art|state of the art|foundation model|foundation-model|true OFI|order-flow imbalance|queue position|production simulator|production execution simulator|backtest" README.md docs reports experiments
```

## Actions

| Term | Main files | Action taken | Remaining justified occurrences |
| --- | --- | --- | --- |
| profitable / profit | README, docs, reports, experiment model cards | Kept only in explicit non-claim language; replaced older result-style wording where present. | Claim-audit forbidden rows and limitation statements. |
| PnL | README, execution-v3 docs, final report, older reports | Replaced visible report wording with `cost-adjusted proxy` where it described diagnostics. | Remaining occurrences explicitly say not PnL or appear in historical/generated archives. |
| alpha | README title, claim audit, real-run summary | Kept only in the requested project title and in blocked `tradable alpha` language. | No tradable-alpha claim remains. |
| tradable / live trading / automated order-placement | README, safety docs, execution docs, generated reports | Kept only as explicit non-claims. README now states the project is not automated order-placement software. | Historical model cards use non-claim wording. |
| SOTA / state-of-the-art | README, final report, research protocol, claim audit | Kept only as blocked/non-claim wording; final report now says no SOTA status. | Claim audit and historical protocol limitations. |
| foundation model / foundation-model | README, docs, claim audit | Kept only as blocked/non-claim wording. | Claim audit and limitation statements. |
| true OFI / order-flow imbalance | README, microstructure docs, feature docs, claim audit | Public wording now uses `snapshot_order_flow_proxy` for the proxy and reserves true event-level language for non-claims. | Unsupported-feature caveats and forbidden-claim rows. |
| queue position | microstructure docs, execution docs, reports | Kept only to state FI-2010 does not expose it or the simulator does not model it. | Limitation statements and generated manifests. |
| production simulator / production execution simulator | README, claim audit | Kept only as a blocked claim. | Forbidden-claim rows. |
| backtest | legacy reports, execution-v2 docs, archive files | Replaced with offline proxy language in the most visible public docs where technically appropriate. | Remaining uses are historical, module-path references or explicit "not a backtest" caveats. |

## Remaining Manual-Review Items

- `reports/report_archive/*` contains archived CLI/module output and still mentions backtest/PnL terms as historical module names or CLI fields.
- Older generated experiment model cards under `experiments/fi2010_midprice_h10_ablations/` retain caveated "not a production backtest" wording.
- Legacy local evidence-pack bullet files with old naming are no longer produced
  by the generator; exclude them from public commits.

Status: unsupported claims are absent from the current README, docs and regenerated public reports, or they appear only as explicit blocked claims / limitations.

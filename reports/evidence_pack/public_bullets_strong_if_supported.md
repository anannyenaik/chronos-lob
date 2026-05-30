# Stronger Public Bullets With Support Conditions

Bullet:
"Compared supervised and self-supervised transformer variants across FI-2010 folds, horizons and seeds under leakage-safe evaluation."

Status:
supported only if:
- neural full-grid status = complete_real or archived_valid
- supervised, masked SSL and next-field SSL objectives completed
- aggregate_summary.csv exists
- ssl_comparison.csv exists
- no broad SSL improvement is claimed unless aggregate deltas support it

Current status:
- neural full-grid status = archived_valid

Safe fallback:
"Built infrastructure to compare supervised and self-supervised transformer variants under leakage-safe FI-2010 evaluation."

Bullet:
"Generated artefact-backed execution-aware proxy diagnostics for stored FI-2010 predictions."

Status:
supported only if:
- execution-v3 status = complete_real or archived_valid
- summary.json and execution_v3_manifest.json exist
- inputs are non-smoke and not stale

Safe fallback:
"Built offline execution-aware proxy diagnostics with explicit release limitations."

Bullet:
"Identified the strongest stored classical FI-2010 baseline by macro-F1."

Status:
supported by classical result artefacts.

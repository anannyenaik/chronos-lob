# Limitations Index

Use this index to keep report claims aligned with implemented evidence and documented caveats.

## Primary References

- `../limitations.md`: current limitations statement.
- `../../docs/SAFETY_AND_LIMITATIONS.md`: public safety boundaries.
- `../../docs/REPRODUCIBILITY.md`: validation and smoke-command caveats.
- `../../docs/PROJECT_STATUS.md`: implemented versus not implemented scope.

## Implementation Reports With Limitation Context

- `../data_quality.md`
- `../feature_engine.md`
- `../label_engine.md`
- `../leakage_controls.md`
- `../validation_protocol.md`
- `../calibration_uncertainty.md`
- `../execution_aware_validation.md`
- `../transfer_regime_ablation_analysis.md`
- `../full_audit_ci_hardening.md`

## Caveats To Preserve

- Public data may have coverage, preprocessing and timestamp limitations.
- Synthetic fixtures are plumbing checks only.
- Crypto-style reconstruction examples should not be overclaimed as equity market evidence.
- Execution-aware validation is a simplified research simulation.
- Queue position, partial fills, latency realism and venue rules remain explicit assumptions.
- No production market impact model is implemented.
- Real result claims require reproducible experiment artefacts.

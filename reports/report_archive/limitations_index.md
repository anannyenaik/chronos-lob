# Limitations Index

Pointer index to the canonical scope and limitation documents.

## Primary References

- `../limitations.md`: technical caveats for extending the platform.
- `../../docs/SAFETY_AND_LIMITATIONS.md`: canonical scope statement.
- `../../docs/REPRODUCIBILITY.md`: validation and reproducibility path.
- `../../docs/PROJECT_STATUS.md`: implemented and current limitations.

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

## Core Caveats

- Public data may have coverage, preprocessing and timestamp limitations.
- Synthetic fixtures exercise code paths only.
- Crypto-style reconstruction examples should not be treated as equity-market evidence.
- Execution-aware validation is a simplified research simulation.
- Queue position, partial fills, latency realism and venue rules remain explicit assumptions.
- No production market impact model is implemented.
- Reported metrics must trace to versioned configs, data, seeds and stored outputs.

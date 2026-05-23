# ChronosLOB Build Plan

This plan keeps implementation staged, auditable and leakage-aware. Each phase
should leave the repository in a tested state, with limitations stated before
any result is reported.

## Completed Phases

0. Repository scaffold, project rules and tooling.
1. Core schemas, event types and utilities.
2. Local FI-2010-style loader and validation.
3. Leakage-safe microstructure feature engine.
4. Future-window label engine and no-look-ahead checks.
5. Temporal, walk-forward and purged or embargoed splitters plus experiment
   registry skeleton.
6. Classical baselines, train-only preprocessing and metrics.
7. PyTorch sequence-window data layer.
8. DeepLOB-style supervised CNN-LSTM baseline and smoke training utilities.
9. Offline Binance-style local order book reconstruction.
10. Canonical JSONL event-log storage and replay-to-feature or replay-to-label
    integration.
11. Deterministic event tokenisation and transformer input preparation.
12. Supervised transformer encoder architecture.
13. Self-supervised masked-field and next-field objectives.
14. Multi-task fine-tuning infrastructure.
15. Calibration, uncertainty and abstention diagnostics.
16. Execution-aware validation with explicit costs, latency, turnover, risk and
    adverse-selection handling.
17. Transfer, regime, ablation and sensitivity analysis utilities.
18. Full audit and CI hardening.

## Phase 18 Acceptance Criteria

- GitHub Actions CI exists and runs the local validation suite.
- Reproducibility, CLI reference, project status, safety and limitation docs are
  present.
- Local audit utilities and the `run-project-audit` CLI command are tested.
- Config and report inventory tests pass.
- README and limitations statements match the implemented project state.
- No fake results, benchmark claims, live trading functionality or deployable
  trading claims are added.

## Future Phases

19. Technical report and GitHub polish.
20. Recruiter-facing packaging, if requested, using only verified implemented
    artefacts.

Future phases must not invent metrics, plots, benchmarks or CV claims. Any
reported performance must be generated from reproducible experiment artefacts
with documented configs, seeds, data versions and code versions.

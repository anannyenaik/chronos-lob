# Phase Timeline

This timeline summarises implementation history for report writing. It does not imply that benchmark experiments or final report results have been produced.

| Phase | Implemented scope |
|---|---|
| Phase 0 | repository scaffold, tooling and documentation conventions. |
| Phase 1 | schemas for events, order books, features, labels and quality issues. |
| Phase 2 | local FI-2010-style loading and validation. |
| Phase 3 | past-only microstructure feature engine. |
| Phase 4 | future-window labels and no-look-ahead leakage checks. |
| Phase 5 | temporal, walk-forward and purged or embargoed splitters. |
| Phase 6 | classical baselines, metrics and train-only preprocessing. |
| Phase 7A | PyTorch sequence-window data layer. |
| Phase 7B | DeepLOB-style supervised CNN-LSTM baseline. |
| Phase 8 | offline Binance-style order book reconstruction. |
| Phase 9 | canonical JSONL event logs and replay-to-feature/label integration. |
| Phase 10 | event-log storage and replay planning absorbed into Phase 9 outputs. |
| Phase 11 | deterministic event tokenisation and transformer inputs. |
| Phase 12 | supervised transformer encoder architecture. |
| Phase 13 | self-supervised masked-field and next-field objectives. |
| Phase 14 | multi-task fine-tuning infrastructure. |
| Phase 15 | calibration, uncertainty and confidence filtering. |
| Phase 16 | execution-aware validation under explicit simplified assumptions. |
| Phase 17 | transfer, regime, ablation and sensitivity analysis. |
| Phase 18 | local audit utilities, CI hardening and reproducibility documentation. |

Phase 19 adds this evidence archive and GitHub polish material without adding new modelling functionality.

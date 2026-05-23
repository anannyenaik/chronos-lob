# Release History

Summary of implementation milestones in this repository.

| Milestone | Implemented scope |
|---|---|
| Foundation | repository scaffold, tooling and documentation conventions. |
| Data contracts | schemas for events, order books, features, labels and quality issues. |
| Local loading | FI-2010-style loading and validation. |
| Feature engine | past-only microstructure feature generation. |
| Label engine | future-window labels and no-look-ahead leakage checks. |
| Validation protocols | temporal, walk-forward and purged or embargoed splitters. |
| Classical baselines | baseline interfaces, metrics and train-only preprocessing. |
| Torch data layer | PyTorch sequence-window datasets and loaders. |
| DeepLOB-style path | supervised CNN-LSTM baseline plumbing. |
| Book reconstruction | offline Binance-style order book reconstruction. |
| Event logs | canonical JSONL storage and replay-to-feature/label integration. |
| Tokenisation | deterministic event tokenisation and transformer inputs. |
| Transformer modelling | supervised transformer encoder architecture. |
| Self-supervision | masked-field and next-field objectives. |
| Multi-task modelling | fine-tuning infrastructure. |
| Calibration | uncertainty and confidence-filtering diagnostics. |
| Execution-aware validation | explicit simplified assumptions for costs and latency. |
| Robustness analysis | transfer, regime, ablation and sensitivity summaries. |
| Audit and CI | local audit utilities, CI hardening and reproducibility documentation. |
| Evidence archive | technical evidence archive and public documentation polish. |

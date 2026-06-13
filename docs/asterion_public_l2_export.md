# Recorded-public-L2 ChronosLOB → Asterion ONNX export

`tools/export_asterion_public_l2_onnx.py` trains and exports a **tiny, real-data**
ChronosLOB DeepLOB model to ONNX for the [Asterion](https://github.com/krakxn)
order-book engine's optional ONNX Runtime inference backend.

It supersedes the synthetic-toy `tools/export_tiny_asterion_onnx.py`
(`chronoslob_tiny_real`, 4-feature single-timestep) **for model-contract
evidence**: it consumes a compact, committed **recorded public Binance crypto L2
depth** sample, builds a genuine multi-timestep window, applies deterministic
normalisation, trains a small DeepLOB-style CNN-LSTM and emits a richer windowed
model contract plus expected-fixture and checksum artefacts.

## What it produces

Into Asterion's `data/models/` (paths overridable):

* `chronoslob_public_l2_tiny.onnx`: windowed `[1, 16, 40] → [1, 3]` DeepLOB
  artefact (a few KB), trained on recorded public L2 depth.
* `chronoslob_public_l2_tiny.metadata.json`: model contract: shapes, per-timestep
  feature count, **window length**, feature schema, **normalisation metadata**
  (mid-relative + per-feature z-score mean/std), expected input/output, training
  summary (diagnostic only), source-data + model checksums, ChronosLOB source
  commit and explicit claim boundary.
* `chronoslob_public_l2_tiny.expected_input.json` / `.expected_output.json`:
  the recorded deterministic test window and its ONNX Runtime output.
* `chronoslob_public_l2_tiny.manifest.json`: SHA-256 + byte sizes of every
  emitted artefact and of the source dataset.

## Claim boundary

This artefact is **recorded-public-data model-contract evidence for moving a
research-style LOB model into Asterion's deterministic inference path.** It is:

* trained only on a compact **recorded public Binance crypto L2 depth** window
  sample (BTCUSDT, public REST `/api/v3/depth` snapshots), no API keys, no
  account/order endpoints, no authenticated connectivity, no private data;
* a deliberately **tiny** DeepLOB-style baseline on a small, heavily-overlapping
  window set: a systems/integration artefact, not a research result.

It is **not** evidence of predictive quality, profitability, alpha, live trading,
production model-serving, production HFT, portable latency, equities-market
realism or L3 exchange-feed realism. Any accuracy/loss in the metadata is
**diagnostic context only**, with no trading significance.

## Architecture / contract

| field | value |
|---|---|
| model class | `DeepLOBModel` (CNN-LSTM) |
| input | `features`, shape `[1, 16, 40]` (window length 16 × 40-dim LOB frame) |
| output | `logits`, shape `[1, 3]` (raw `[down, flat, up]` logits) |
| per-timestep features | 40 = 10 levels × {bid price, bid qty, ask price, ask qty}, mid-relative + z-scored |
| label | direction of mid price 5 steps ahead (median-abs-move threshold); diagnostic only |
| opset / IR | `17` / `8` |
| trained | yes (recorded public L2 smoke run) |

## Determinism

Training is seeded and single-threaded. The committed ONNX artefact (not
bit-exact regeneration across machines/library versions) is the source of truth:
`--verify` confirms the artefact reproduces its recorded `expected_test_output`
through ONNX Runtime within a small tolerance. The source dataset is checksummed
(`source_data_sha256`) so the recorded-data input is pinned.

## Reproduce

```bash
pip install -e '.[torch]' onnx onnxruntime    # from the chronos-lob repo root
python tools/export_asterion_public_l2_onnx.py \
  --dataset ../Asterion/data/samples/binance_public_l2_window_sample.jsonl \
  --output  ../Asterion/data/models/chronoslob_public_l2_tiny.onnx
# verify a committed artefact reproduces its recorded expected output via ORT,
# and optionally measure isolated ONNX inference latency (local, not portable):
python tools/export_asterion_public_l2_onnx.py --verify --benchmark 20000 \
  --output  ../Asterion/data/models/chronoslob_public_l2_tiny.onnx
```

The recorded dataset itself is produced (manually, opt-in, no CI) by Asterion's
`tools/capture_binance_depth.py` against the public REST depth endpoint; only a
compact top-10 subset is committed. None of `torch`/`onnx`/`onnxruntime` are
runtime dependencies of ChronosLOB or Asterion.

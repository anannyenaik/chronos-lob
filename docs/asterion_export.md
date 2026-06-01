# Tiny ChronosLOB → Asterion ONNX export

`tools/export_tiny_asterion_onnx.py` trains and exports a **tiny, real**
ChronosLOB DeepLOB model to ONNX for the [Asterion](https://github.com/krakxn)
order-book engine's optional ONNX Runtime inference backend.

## What it produces

* A deterministically-trained `chronoslob.models.DeepLOBModel`
  (`create_deeplob_model`) configured tiny enough to commit as a few-KB ONNX
  graph (~900 parameters).
* A metadata sidecar (`*.metadata.json`) describing the artefact: model class,
  artefact type, ChronosLOB source commit, input/output shapes, feature
  ordering and version, the deterministic test vector, the training summary and
  the explicit claim boundary.

## Claim boundary

This artefact is **a tiny ChronosLOB research-model artefact exported into
Asterion for systems-integration and inference-latency evaluation.** It is:

* trained only on **synthetic toy data** (seeded, standardised, O(1) features
  positionally aligned to Asterion's L2 feature ordering) — not FI-2010 and not
  any private/market dataset;
* a deliberately **reduced** 4-feature, single-timestep (`lookback == 1`)
  simplification of DeepLOB, chosen so the four caller-owned Asterion L2
  features map 1:1 to the model input.

It is **not** evidence of predictive quality, trading profitability, live
trading, production model-serving, production HFT or SOTA modelling. The smoke
run reduces a cross-entropy loss so the network is genuinely *trained* rather
than random, but the learned relationship is an artificial synthetic rule with
no market meaning. The exported logits are consumed by Asterion as a
deterministic **plumbing score**, not an alpha signal.

## Architecture / contract

| field | value |
|---|---|
| model class | `DeepLOBModel` (CNN-LSTM) |
| input | `features`, shape `[1, 1, 4]` |
| output | `logits`, shape `[1, 3]` (raw `[down, flat, up]` logits) |
| feature order | `spread_ticks, mid_price_ticks, top_level_imbalance, top_level_quantity` |
| feature version | `1` |
| opset / IR | `17` / `8` |
| trained | yes (synthetic toy smoke run) |

## Dependencies

The optional `[torch]` extra plus `onnx`. `onnxruntime`, when importable, is
used to compute the recorded expected output through the same runtime family
Asterion links, so the metadata vector matches Asterion's C++ ONNX path
byte-for-byte; otherwise the PyTorch eval output is recorded and
`expected_test_output_engine` is set to `pytorch`. None of these are runtime
dependencies of ChronosLOB or Asterion.

## Reproduce

```bash
pip install -e '.[torch]' onnx onnxruntime    # from the chronos-lob repo root
python tools/export_tiny_asterion_onnx.py \
  --output    ../Asterion/data/models/chronoslob_tiny_real.onnx \
  --metadata-output ../Asterion/data/models/chronoslob_tiny_real.metadata.json
# verify a committed artefact reproduces its recorded expected output via ORT:
python tools/export_tiny_asterion_onnx.py --verify \
  --output    ../Asterion/data/models/chronoslob_tiny_real.onnx \
  --metadata-output ../Asterion/data/models/chronoslob_tiny_real.metadata.json
```

Training is seeded and single-threaded, but the committed ONNX artefact (not
bit-exact regeneration across machines/library versions) is the source of
truth: `--verify` confirms the artefact reproduces its recorded
`expected_test_output` through ONNX Runtime within a small tolerance.

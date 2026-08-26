#!/usr/bin/env python3
"""Train and export a tiny *real* ChronosLOB DeepLOB model to ONNX for Asterion.

This script lives in ChronosLOB and uses ChronosLOB's own model code
(``chronoslob.models.DeepLOBModel`` via :func:`create_deeplob_model`). It builds
a deliberately tiny DeepLOB-style CNN-LSTM, runs a short *deterministic* training
smoke pass on *synthetic toy data*, exports the trained network to ONNX and
writes a metadata sidecar that Asterion's optional ONNX Runtime backend can load
and validate.

Claim boundary
--------------
The exported artefact is a *tiny ChronosLOB research-model artefact exported into
Asterion for systems-integration and inference-latency evaluation*. It is:

* trained only on **synthetic toy data**, not FI-2010 and not any private/market
  dataset;
* a **reduced-feature** (4-feature, single-timestep) simplification chosen to
  match Asterion's L2 feature buffer ordering, not ChronosLOB's full feature
  frame;
* **not** evidence of predictive quality, trading profitability, live trading,
  production model-serving, production HFT or SOTA modelling.

The toy training reduces a cross-entropy loss so the network is genuinely
*trained* rather than random, but the learned relationship is an artificial
synthetic rule with no market meaning.

Dependencies
------------
Requires the optional ``[torch]`` extra plus ``onnx``. ``onnxruntime`` is used,
when importable, to compute the deterministic expected output through the same
runtime family Asterion uses; otherwise the PyTorch eval output is recorded and
a note is set. None of these are runtime dependencies of Asterion.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import pathlib
import subprocess
import sys
from typing import Any

import numpy as np

# Make the chronoslob package importable without requiring an editable install.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

MODEL_NAME = "chronoslob_tiny_real"
MODEL_CLASS = "DeepLOBModel"
SEED = 7
OPSET_VERSION = 17

# Asterion's L2 feature buffer ordering. The tiny model is trained to consume
# these four features as a single-timestep (lookback == 1) DeepLOB input.
FEATURE_ORDER = [
    "spread_ticks",
    "mid_price_ticks",
    "top_level_imbalance",
    "top_level_quantity",
]
FEATURE_VERSION = 1
INPUT_FEATURES = len(FEATURE_ORDER)
LOOKBACK = 1
N_CLASSES = 3
INPUT_SHAPE = [1, LOOKBACK, INPUT_FEATURES]
OUTPUT_SHAPE = [1, N_CLASSES]

INPUT_NAME = "features"
OUTPUT_NAME = "logits"

# Tiny architecture: ~900 parameters. Small enough to commit as a few-KB ONNX.
CONV_CHANNELS = 8
CONV_KERNEL_SIZE = 3
LSTM_HIDDEN_SIZE = 8
LSTM_LAYERS = 1

# Deterministic toy-training schedule (full-batch Adam). Batch norm is left OFF:
# with lookback==1 its running-stat estimate is poor, opening a train/eval gap
# that would make the *exported* (eval-mode) model behave differently from the
# trained one. Instead the synthetic toy features are standardised (O(1)) so the
# smoke run genuinely fits the rule, and the exported eval-mode network is exactly
# what was trained (no dropout, no batch norm => deterministic, no train/eval gap).
N_SAMPLES = 512
TRAIN_STEPS = 400
LEARNING_RATE = 0.03
USE_BATCH_NORM = False

# Deterministic test vector: a representative in-distribution standardised toy
# sample (see _build_synthetic_toy_dataset). It is synthetic, not a real L2
# snapshot; it exists only to pin a reproducible input->output pair.
EXPECTED_TEST_INPUT = [0.5, -0.25, 0.8, 0.2]

DEFAULT_MODEL_OUTPUT = _REPO_ROOT / "runs" / "asterion_export" / f"{MODEL_NAME}.onnx"
DEFAULT_METADATA_OUTPUT = (
    _REPO_ROOT / "runs" / "asterion_export" / f"{MODEL_NAME}.metadata.json"
)

# Reproducible cross-repo command recorded in the metadata.
EXPORT_COMMAND = (
    "python ../ChronosLOB/chronos-lob/tools/export_tiny_asterion_onnx.py "
    "--output data/models/chronoslob_tiny_real.onnx "
    "--metadata-output data/models/chronoslob_tiny_real.metadata.json"
)


def _git_value(repo: pathlib.Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


def _source_status() -> dict[str, Any]:
    commit = _git_value(_REPO_ROOT, "rev-parse", "HEAD")
    status = _git_value(_REPO_ROOT, "status", "--short")
    return {
        "source_repo": "https://github.com/anannyenaik/chronos-lob",
        "source_repo_path": "ChronosLOB/chronos-lob",
        "source_commit": commit,
        "source_dirty": bool(status),
    }


def _build_synthetic_toy_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Build a small, fully synthetic, seeded toy dataset.

    The four columns are positionally aligned to Asterion's L2 feature ordering
    (spread_ticks, mid_price_ticks, top_level_imbalance, top_level_quantity) but
    hold *standardised O(1) synthetic values*, not real tick/quantity scales. The
    standardisation keeps the no-batch-norm smoke run well conditioned. Labels
    follow an artificial 3-class rule driven by the imbalance and spread channels
    plus seeded noise. Nothing here is market data or an alpha signal, and the
    toy distribution deliberately does not match Asterion's real feature scale.
    """
    rng = np.random.default_rng(SEED)
    spread = rng.normal(0.0, 1.0, N_SAMPLES).astype(np.float32)
    mid = rng.normal(0.0, 1.0, N_SAMPLES).astype(np.float32)
    imbalance = rng.uniform(-1.0, 1.0, N_SAMPLES).astype(np.float32)
    quantity = rng.normal(0.0, 1.0, N_SAMPLES).astype(np.float32)
    features = np.stack([spread, mid, imbalance, quantity], axis=1).astype(np.float32)

    # Artificial label rule: imbalance dominates, spread nudges, plus noise. mid
    # and quantity carry no label information (the net must learn to ignore them).
    signal = imbalance + 0.25 * spread + rng.normal(0.0, 0.15, N_SAMPLES)
    labels = np.ones(N_SAMPLES, dtype=np.int64)  # default: flat (class 1)
    labels[signal < -0.5] = 0  # down
    labels[signal > 0.5] = 2  # up

    # DeepLOB expects [N, lookback, features]; lookback is 1 for this export.
    return features.reshape(N_SAMPLES, LOOKBACK, INPUT_FEATURES), labels


def _train_tiny_deeplob() -> tuple[Any, dict[str, Any]]:
    import torch

    from chronoslob.models import DeepLOBConfig, create_deeplob_model

    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    # Best-effort determinism; harmless if unsupported for some ops.
    with contextlib.suppress(Exception):
        torch.use_deterministic_algorithms(True)

    config = DeepLOBConfig(
        input_features=INPUT_FEATURES,
        n_classes=N_CLASSES,
        conv_channels=CONV_CHANNELS,
        conv_kernel_size=CONV_KERNEL_SIZE,
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        lstm_layers=LSTM_LAYERS,
        dropout=0.0,
        use_batch_norm=USE_BATCH_NORM,
    )
    model = create_deeplob_model(config)

    x_np, y_np = _build_synthetic_toy_dataset()
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    initial_loss = float("nan")
    final_loss = float("nan")
    for step in range(TRAIN_STEPS):
        optimiser.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimiser.step()
        if step == 0:
            initial_loss = float(loss.detach())
        final_loss = float(loss.detach())

    # Report metrics in eval mode (the mode that is exported), so the numbers
    # describe the artefact that ships, not a different train-mode behaviour.
    model.eval()
    with torch.no_grad():
        eval_logits = model(x)
        eval_loss = float(loss_fn(eval_logits, y))
        eval_accuracy = float((eval_logits.argmax(dim=1) == y).float().mean())

    training = {
        "data": "synthetic_toy",
        "data_description": (
            "Seeded synthetic standardised (O(1)) 4-feature samples positionally "
            "aligned to Asterion's L2 feature ordering; artificial 3-class label "
            "rule driven by imbalance and spread. Not FI-2010, not market data, "
            "not an alpha signal; toy scale deliberately differs from Asterion's "
            "real feature scale."
        ),
        "seed": SEED,
        "samples": N_SAMPLES,
        "steps": TRAIN_STEPS,
        "optimiser": "adam",
        "learning_rate": LEARNING_RATE,
        "loss": "cross_entropy",
        "initial_train_loss": round(initial_loss, 6),
        "final_train_loss": round(final_loss, 6),
        "eval_loss": round(eval_loss, 6),
        "eval_accuracy": round(eval_accuracy, 6),
        "parameter_count": model.n_parameters(),
    }
    return model, training


def _export_onnx(model: Any, path: pathlib.Path) -> bytes:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.tensor(
        np.array(EXPECTED_TEST_INPUT, dtype=np.float32).reshape(INPUT_SHAPE)
    )
    model.eval()
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        opset_version=OPSET_VERSION,
        dynamo=False,
    )
    import onnx

    loaded = onnx.load(str(path))
    onnx.checker.check_model(loaded)
    return path.read_bytes()


def _expected_output(model: Any, path: pathlib.Path) -> tuple[list[float], str]:
    """Compute the deterministic expected output for EXPECTED_TEST_INPUT.

    Prefers ONNX Runtime (same family Asterion links) so the recorded vector
    matches the C++ ONNX path; falls back to the PyTorch eval output otherwise.
    """
    input_array = np.array(EXPECTED_TEST_INPUT, dtype=np.float32).reshape(INPUT_SHAPE)
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        result = session.run(None, {INPUT_NAME: input_array})[0]
        return [float(value) for value in np.asarray(result).reshape(-1)], "onnxruntime"
    except ImportError:
        import torch

        with torch.no_grad():
            result = model(torch.from_numpy(input_array)).numpy()
        return [float(value) for value in np.asarray(result).reshape(-1)], "pytorch"


def build_metadata(
    model_bytes: bytes, training: dict[str, Any], expected_output: list[float],
    expected_output_engine: str,
) -> dict[str, Any]:
    import onnx
    import torch

    metadata: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "model_class": MODEL_CLASS,
        "artefact_type": "trained_synthetic_smoke",
        "framework": "pytorch",
        "framework_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "opset_version": OPSET_VERSION,
        "ir_version": onnx.load_model_from_string(model_bytes).ir_version,
        "export_command": EXPORT_COMMAND,
        "input_name": INPUT_NAME,
        "input_shape": INPUT_SHAPE,
        "output_name": OUTPUT_NAME,
        "output_shape": OUTPUT_SHAPE,
        "feature_count": INPUT_FEATURES,
        "feature_version": FEATURE_VERSION,
        "feature_order": FEATURE_ORDER,
        "feature_mapping": {
            "description": (
                "Asterion's 4 caller-owned L2 features map 1:1, in order, to the "
                "model's single-timestep [1, 1, 4] input. lookback==1 is a "
                "deliberate simplification of DeepLOB's multi-row window. Because "
                "the model was trained on a standardised synthetic toy "
                "distribution (not Asterion's real feature scale and not market "
                "data), feeding Asterion's live L2 features yields a deterministic "
                "plumbing score only, with no predictive meaning."
            ),
            "asterion_to_model": {name: index for index, name in enumerate(FEATURE_ORDER)},
        },
        "expected_test_input": EXPECTED_TEST_INPUT,
        "expected_test_output": expected_output,
        "expected_test_output_engine": expected_output_engine,
        "trained_model": True,
        "deterministic_fixture": False,
        "training": training,
        "onnx_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "output_semantics": (
            "Raw 3-class logits [down, flat, up]. Asterion consumes output[0] as a "
            "scalar plumbing score; it is not a probability, alpha signal or "
            "trading decision."
        ),
        "claim_boundary": (
            "A tiny ChronosLOB research-model artefact (DeepLOB-style CNN-LSTM) "
            "exported into Asterion for systems-integration and inference-latency "
            "evaluation."
        ),
        "claim_limitations": [
            "No predictive quality claim.",
            "No trading profitability claim.",
            "Trained on synthetic toy data only; not FI-2010 or market data.",
            "Reduced 4-feature, single-timestep simplification of DeepLOB.",
            "Not live trading infrastructure.",
            "Not production model-serving infrastructure.",
            "Not production-HFT infrastructure.",
            "Not a portable latency guarantee.",
        ],
    }
    metadata.update(_source_status())
    return metadata


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument(
        "--metadata-output", type=pathlib.Path, default=DEFAULT_METADATA_OUTPUT
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the committed artefact reproduces its recorded expected "
        "output through ONNX Runtime (does not retrain or rewrite).",
    )
    args = parser.parse_args(argv)

    if args.verify:
        return _verify(args.output, args.metadata_output)

    model, training = _train_tiny_deeplob()
    model_bytes = _export_onnx(model, args.output)
    expected_output, engine = _expected_output(model, args.output)
    metadata = build_metadata(model_bytes, training, expected_output, engine)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output} ({len(model_bytes)} bytes)")
    print(f"wrote {args.metadata_output}")
    print(
        f"  trained_model=True initial_train_loss={training['initial_train_loss']} "
        f"eval_loss={training['eval_loss']} eval_accuracy={training['eval_accuracy']}"
    )
    print(f"  expected_test_output ({engine}) = {expected_output}")
    return 0


def _verify(model_path: pathlib.Path, metadata_path: pathlib.Path) -> int:
    if not model_path.exists():
        print(f"missing ONNX artefact: {model_path}", file=sys.stderr)
        return 1
    if not metadata_path.exists():
        print(f"missing metadata: {metadata_path}", file=sys.stderr)
        return 1
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_bytes = model_path.read_bytes()

    digest = hashlib.sha256(model_bytes).hexdigest()
    if "onnx_sha256" in metadata and metadata["onnx_sha256"] != digest:
        print(
            f"onnx_sha256 mismatch: metadata={metadata['onnx_sha256']} actual={digest}",
            file=sys.stderr,
        )
        return 1

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; cannot verify expected output", file=sys.stderr)
        return 2

    input_array = np.array(
        metadata["expected_test_input"], dtype=np.float32
    ).reshape(metadata["input_shape"])
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    result = np.asarray(
        session.run(None, {metadata["input_name"]: input_array})[0]
    ).reshape(-1)
    expected = np.asarray(metadata["expected_test_output"], dtype=np.float64)
    max_diff = float(np.max(np.abs(result.astype(np.float64) - expected)))
    if max_diff > 1e-4:
        print(f"expected output mismatch: max_abs_diff={max_diff}", file=sys.stderr)
        return 1
    print(
        f"OK: {model_path.name} ({len(model_bytes)} bytes) reproduces expected "
        f"output via ONNX Runtime (max_abs_diff={max_diff:.2e})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""Train and export a tiny *real-data* ChronosLOB DeepLOB model to ONNX for Asterion.

This script lives in ChronosLOB and uses ChronosLOB's own model code
(``chronoslob.models.DeepLOBModel`` via :func:`create_deeplob_model`). Unlike the
older ``export_tiny_asterion_onnx.py`` (which trained on *synthetic toy* data with
a 4-feature single-timestep input), this exporter consumes a **recorded public
Binance crypto L2 depth** dataset, builds a genuine multi-timestep window, applies
deterministic mid-relative + z-score normalisation, trains a small DeepLOB-style
CNN-LSTM and exports a windowed ``[1, window, 40]`` -> ``[1, 3]`` ONNX artefact
plus a metadata sidecar, expected input/output fixtures and a checksum manifest.

Claim boundary
--------------
The exported artefact is a *recorded-public-fixture-trained systems/integration
artefact* used to move a research-style LOB model into Asterion's deterministic
inference path and to validate a richer model contract. It is:

* trained only on a small, compact, **recorded public Binance L2 depth** window
  sample (BTCUSDT, public REST ``/api/v3/depth`` snapshots) — no authenticated
  connectivity, no API keys, no account/order endpoints, no private data;
* a deliberately **tiny** DeepLOB-style baseline, not a SOTA model;
* **not** evidence of predictive quality, trading profitability, alpha, live
  trading, production model-serving, production HFT, portable latency, equities
  realism or L3 exchange-feed realism.

Any accuracy/loss reported by this script is *diagnostic context only* (it shows
the network genuinely fits a recorded-data direction-of-mid label on a tiny,
heavily-overlapping window set); it carries no trading significance.

Dependencies
------------
Requires the optional ``[torch]`` extra plus ``onnx``. ``onnxruntime`` is used,
when importable, to compute the deterministic expected output through the same
runtime family Asterion links; otherwise the PyTorch eval output is recorded and
a note is set. None of these are runtime dependencies of Asterion or ChronosLOB.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import pathlib
import subprocess
import sys
import time
from typing import Any

import numpy as np

# Make the chronoslob package importable without requiring an editable install.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

MODEL_NAME = "chronoslob_public_l2_tiny"
MODEL_CLASS = "DeepLOBModel"
ARTEFACT_TYPE = "trained_recorded_public_l2"
SEED = 7
OPSET_VERSION = 17

# Recorded-public-L2 feature contract: a classic DeepLOB-style 40-dim LOB frame
# (10 price levels per side x {bid price, bid qty, ask price, ask qty}) over a
# multi-timestep window. The per-timestep feature ordering groups the four
# channels (task's suggested schema: top-N bid prices, bid quantities, ask
# prices, ask quantities). Prices are stored mid-relative (price - mid) before
# the per-feature z-score normalisation recorded in the metadata.
N_LEVELS = 10
WINDOW_LENGTH = 16
HORIZON = 5
N_CLASSES = 3
FEATURE_VERSION = 1

_FEATURE_ORDER: list[str] = (
    [f"bid_price_rel_{i}" for i in range(N_LEVELS)]
    + [f"bid_qty_{i}" for i in range(N_LEVELS)]
    + [f"ask_price_rel_{i}" for i in range(N_LEVELS)]
    + [f"ask_qty_{i}" for i in range(N_LEVELS)]
)
FEATURE_COUNT = len(_FEATURE_ORDER)  # 40 (per timestep)
INPUT_SHAPE = [1, WINDOW_LENGTH, FEATURE_COUNT]
OUTPUT_SHAPE = [1, N_CLASSES]

INPUT_NAME = "features"
OUTPUT_NAME = "logits"

# Tiny architecture. Small enough to commit as a few-KB ONNX graph.
CONV_CHANNELS = 8
CONV_KERNEL_SIZE = 3
LSTM_HIDDEN_SIZE = 8
LSTM_LAYERS = 1
USE_BATCH_NORM = False  # see export_tiny_asterion_onnx.py rationale (no train/eval gap)

# Deterministic training schedule.
TRAIN_STEPS = 300
LEARNING_RATE = 0.02

_STD_FLOOR = 1e-8

# ChronosLOB lives at .../Programming/ChronosLOB/chronos-lob; Asterion is a sibling
# of the ChronosLOB directory at .../Programming/Asterion.
_ASTERION_ROOT = _REPO_ROOT.parent.parent / "Asterion"
DEFAULT_DATASET = (
    _ASTERION_ROOT / "data" / "samples" / "binance_public_l2_window_sample.jsonl"
)
_DEFAULT_ASTERION_MODELS = _ASTERION_ROOT / "data" / "models"
DEFAULT_MODEL_OUTPUT = _DEFAULT_ASTERION_MODELS / f"{MODEL_NAME}.onnx"
DEFAULT_METADATA_OUTPUT = _DEFAULT_ASTERION_MODELS / f"{MODEL_NAME}.metadata.json"
DEFAULT_EXPECTED_INPUT_OUTPUT = _DEFAULT_ASTERION_MODELS / f"{MODEL_NAME}.expected_input.json"
DEFAULT_EXPECTED_OUTPUT_OUTPUT = _DEFAULT_ASTERION_MODELS / f"{MODEL_NAME}.expected_output.json"
DEFAULT_MANIFEST_OUTPUT = _DEFAULT_ASTERION_MODELS / f"{MODEL_NAME}.manifest.json"

EXPORT_COMMAND = (
    "python ../ChronosLOB/chronos-lob/tools/export_asterion_public_l2_onnx.py "
    "--dataset data/samples/binance_public_l2_window_sample.jsonl "
    "--output data/models/chronoslob_public_l2_tiny.onnx"
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


# ----------------------------------------------------------------------------
# Recorded public L2 dataset -> windowed, normalised features + diagnostic label
# ----------------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_snapshots(dataset_path: pathlib.Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the compact recorded-public-L2 JSONL dataset.

    Returns ``(snapshots, banner_meta)``. Each snapshot has compact keys
    ``t`` (capture ns), ``u`` (lastUpdateId), ``b`` (bids) and ``a`` (asks),
    each side a list of ``[price_str, qty_str]`` levels best-first.
    """
    banner: dict[str, Any] = {}
    snapshots: list[dict[str, Any]] = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "_meta" in obj:
            banner = obj
            continue
        snapshots.append(obj)
    if len(snapshots) < WINDOW_LENGTH + HORIZON + 1:
        raise SystemExit(
            f"recorded dataset too small: {len(snapshots)} snapshots, need at least "
            f"{WINDOW_LENGTH + HORIZON + 1}"
        )
    return snapshots, banner


def _raw_frame(snapshot: dict[str, Any]) -> tuple[np.ndarray, float]:
    """Build the raw per-timestep 40-dim frame and the mid price for a snapshot.

    Returns ``(frame_raw[40], mid)`` where price columns are *absolute* prices
    (mid subtraction happens after, in :func:`_build_dataset`, so the recorded
    mid is preserved in metadata-free form here).
    """
    bids = snapshot["b"][:N_LEVELS]
    asks = snapshot["a"][:N_LEVELS]
    if len(bids) < N_LEVELS or len(asks) < N_LEVELS:
        raise SystemExit("snapshot has fewer levels than N_LEVELS; dataset is malformed")
    bid_p = np.array([float(p) for p, _ in bids], dtype=np.float64)
    bid_q = np.array([float(q) for _, q in bids], dtype=np.float64)
    ask_p = np.array([float(p) for p, _ in asks], dtype=np.float64)
    ask_q = np.array([float(q) for _, q in asks], dtype=np.float64)
    mid = 0.5 * (bid_p[0] + ask_p[0])
    frame = np.concatenate([bid_p, bid_q, ask_p, ask_q]).astype(np.float64)
    return frame, mid


def _build_dataset(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert recorded snapshots into normalised windows + diagnostic labels.

    Deterministic. Returns a dict with the normalised windows ``X``
    (``[n_windows, WINDOW_LENGTH, FEATURE_COUNT]``), integer labels ``y``,
    the per-feature normalisation ``mean``/``std``, the label threshold and a
    small class-balance summary.
    """
    frames_raw = np.zeros((len(snapshots), FEATURE_COUNT), dtype=np.float64)
    mids = np.zeros(len(snapshots), dtype=np.float64)
    for i, snap in enumerate(snapshots):
        frame, mid = _raw_frame(snap)
        frames_raw[i] = frame
        mids[i] = mid

    # Mid-relative price columns (bid prices: 0..N, ask prices: 2N..3N). Quantity
    # columns are left in raw recorded units; the per-feature z-score below
    # handles their different scale.
    frames = frames_raw.copy()
    price_cols = list(range(0, N_LEVELS)) + list(range(2 * N_LEVELS, 3 * N_LEVELS))
    for col in price_cols:
        frames[:, col] = frames[:, col] - mids

    # Per-feature standardisation computed over every timestep (no leakage of the
    # forward label into the feature scale). std floored to avoid divide-by-zero.
    mean = frames.mean(axis=0)
    std = frames.std(axis=0)
    std = np.where(std < _STD_FLOOR, 1.0, std)
    frames_norm = (frames - mean) / std

    # Diagnostic 3-class label: direction of the mid price HORIZON steps ahead,
    # relative to the window's last timestep. Threshold = median absolute forward
    # move over the dataset (a deterministic function of the recorded data), so
    # the classes are not degenerate. This is a diagnostic target, NOT alpha.
    last_index = len(snapshots) - HORIZON
    deltas = []
    for end in range(WINDOW_LENGTH - 1, last_index):
        deltas.append(mids[end + HORIZON] - mids[end])
    deltas_arr = np.array(deltas, dtype=np.float64)
    threshold = float(np.median(np.abs(deltas_arr))) if deltas_arr.size else 0.0

    windows = []
    labels = []
    for w_i, end in enumerate(range(WINDOW_LENGTH - 1, last_index)):
        start = end - WINDOW_LENGTH + 1
        windows.append(frames_norm[start : end + 1])
        delta = deltas_arr[w_i]
        if delta > threshold:
            labels.append(2)  # up
        elif delta < -threshold:
            labels.append(0)  # down
        else:
            labels.append(1)  # flat
    X = np.stack(windows).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    counts = {
        int(k): int(v)
        for k, v in zip(*np.unique(y, return_counts=True), strict=True)
    }
    return {
        "X": X,
        "y": y,
        "mean": mean.astype(np.float64),
        "std": std.astype(np.float64),
        "threshold": threshold,
        "class_counts": {
            "down": counts.get(0, 0),
            "flat": counts.get(1, 0),
            "up": counts.get(2, 0),
        },
        "n_windows": int(X.shape[0]),
        "n_snapshots": len(snapshots),
    }


def _train_tiny_deeplob(dataset: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    import torch

    from chronoslob.models import DeepLOBConfig, create_deeplob_model

    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    with contextlib.suppress(Exception):
        torch.use_deterministic_algorithms(True)

    config = DeepLOBConfig(
        input_features=FEATURE_COUNT,
        n_classes=N_CLASSES,
        conv_channels=CONV_CHANNELS,
        conv_kernel_size=CONV_KERNEL_SIZE,
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        lstm_layers=LSTM_LAYERS,
        dropout=0.0,
        use_batch_norm=USE_BATCH_NORM,
    )
    model = create_deeplob_model(config)

    x = torch.from_numpy(dataset["X"])
    y = torch.from_numpy(dataset["y"])

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

    model.eval()
    with torch.no_grad():
        eval_logits = model(x)
        eval_loss = float(loss_fn(eval_logits, y))
        eval_accuracy = float((eval_logits.argmax(dim=1) == y).float().mean())

    training = {
        "data": "recorded_public_binance_l2",
        "data_description": (
            "Recorded public Binance crypto L2 depth snapshots (BTCUSDT, public REST "
            "/api/v3/depth), windowed into multi-timestep DeepLOB-style 40-dim frames "
            "with mid-relative prices and per-feature z-score normalisation. The label "
            "is the direction of the mid price HORIZON steps ahead (down/flat/up) with "
            "a median-absolute-move threshold. Diagnostic target only: no alpha, no "
            "predictive-quality and no trading significance is claimed."
        ),
        "seed": SEED,
        "snapshots": dataset["n_snapshots"],
        "windows": dataset["n_windows"],
        "window_length": WINDOW_LENGTH,
        "horizon": HORIZON,
        "label_threshold_mid_units": round(dataset["threshold"], 8),
        "class_counts": dataset["class_counts"],
        "steps": TRAIN_STEPS,
        "optimiser": "adam",
        "learning_rate": LEARNING_RATE,
        "loss": "cross_entropy",
        "initial_train_loss": round(initial_loss, 6),
        "final_train_loss": round(final_loss, 6),
        "eval_loss": round(eval_loss, 6),
        "eval_accuracy_diagnostic": round(eval_accuracy, 6),
        "parameter_count": model.n_parameters(),
    }
    return model, training


def _export_onnx(model: Any, path: pathlib.Path, sample_window: np.ndarray) -> bytes:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.from_numpy(sample_window.reshape(INPUT_SHAPE).astype(np.float32))
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


def _run_ort(path: pathlib.Path, model: Any, flat_input: np.ndarray) -> tuple[list[float], str]:
    """Compute the deterministic expected output for ``flat_input``.

    Prefers ONNX Runtime (the same family Asterion links) so the recorded vector
    matches the C++ ONNX path; falls back to the trained PyTorch model in eval
    mode when ONNX Runtime is not importable.
    """
    array = flat_input.astype(np.float32).reshape(INPUT_SHAPE)
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        result = session.run(None, {INPUT_NAME: array})[0]
        return [float(v) for v in np.asarray(result).reshape(-1)], "onnxruntime"
    except ImportError:
        import torch

        model.eval()
        with torch.no_grad():
            result = model(torch.from_numpy(array)).numpy()
        return [float(v) for v in np.asarray(result).reshape(-1)], "pytorch"


def build_metadata(
    *,
    model_bytes: bytes,
    training: dict[str, Any],
    dataset: dict[str, Any],
    dataset_path: pathlib.Path,
    dataset_sha256: str,
    banner: dict[str, Any],
    expected_input: list[float],
    expected_output: list[float],
    expected_output_engine: str,
) -> dict[str, Any]:
    import onnx
    import torch

    metadata: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "model_class": MODEL_CLASS,
        "artefact_type": ARTEFACT_TYPE,
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
        "feature_count": FEATURE_COUNT,
        "window_length": WINDOW_LENGTH,
        "feature_version": FEATURE_VERSION,
        "feature_order": _FEATURE_ORDER,
        "feature_levels_per_side": N_LEVELS,
        "normalisation": {
            "scheme": "mid_relative_then_per_feature_zscore",
            "price_reference": "mid_relative_units (price - mid)",
            "quantity_reference": "raw_recorded_units",
            "mean": [round(float(v), 10) for v in dataset["mean"]],
            "std": [round(float(v), 10) for v in dataset["std"]],
        },
        "expected_test_input": expected_input,
        "expected_test_output": expected_output,
        "expected_test_output_engine": expected_output_engine,
        "trained_model": True,
        "deterministic_fixture": False,
        "training": training,
        "source_data": {
            "kind": "recorded_public_binance_l2_depth",
            "symbol": banner.get("symbol", "BTCUSDT"),
            "source": banner.get("source", "https://api.binance.com"),
            "endpoint_path": banner.get("endpoint_path", "/api/v3/depth"),
            "stream_type": banner.get("stream_type", "rest_depth_snapshot_poll"),
            "file": "data/samples/" + dataset_path.name,
            "snapshots": dataset["n_snapshots"],
            "levels_per_side": N_LEVELS,
            "sha256": dataset_sha256,
        },
        "source_data_sha256": dataset_sha256,
        "onnx_sha256": _sha256(model_bytes),
        "output_semantics": (
            "Raw 3-class logits [down, flat, up] over a recorded-public-L2 window. "
            "Consumed by Asterion as a scalar plumbing/contract score (output[0]); it "
            "is not a probability, alpha signal or trading decision."
        ),
        "claim_boundary": (
            "Recorded-public-data model-contract evidence for moving a research-style "
            "LOB model into Asterion's deterministic inference path. A tiny DeepLOB-style "
            "CNN-LSTM trained on recorded public Binance crypto L2 depth, exported for "
            "systems-integration and model-contract validation."
        ),
        "claim_limitations": [
            "No predictive quality claim.",
            "No trading profitability or alpha claim.",
            "Accuracy/loss are diagnostic context only, with no trading significance.",
            "Recorded public Binance crypto L2 depth only; not L3, equities or "
            "market realism.",
            "No live trading, no authenticated connectivity, no order placement.",
            "Not production model-serving infrastructure.",
            "Not production-HFT infrastructure.",
            "Not a portable latency guarantee.",
            "Tiny, heavily-overlapping window set; a small smoke-scale model.",
        ],
    }
    metadata.update(_source_status())
    return metadata


def _write_json(path: pathlib.Path, payload: Any) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    data = text.encode("utf-8")
    path.write_bytes(data)
    return data


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=pathlib.Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--metadata-output", type=pathlib.Path, default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument(
        "--expected-input-output", type=pathlib.Path, default=DEFAULT_EXPECTED_INPUT_OUTPUT
    )
    parser.add_argument(
        "--expected-output-output", type=pathlib.Path, default=DEFAULT_EXPECTED_OUTPUT_OUTPUT
    )
    parser.add_argument("--manifest-output", type=pathlib.Path, default=DEFAULT_MANIFEST_OUTPUT)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the committed artefact reproduces its recorded expected output "
        "through ONNX Runtime (does not retrain or rewrite).",
    )
    parser.add_argument(
        "--benchmark",
        type=int,
        default=0,
        metavar="N",
        help="After export/verify, measure isolated ONNX Runtime inference latency over "
        "N steady-state iterations (local diagnostic only, not portable).",
    )
    args = parser.parse_args(argv)

    if args.verify:
        rc = _verify(args.output, args.metadata_output)
        if rc == 0 and args.benchmark > 0:
            _benchmark(args.output, args.metadata_output, args.benchmark)
        return rc

    dataset_bytes = args.dataset.read_bytes()
    dataset_sha256 = _sha256(dataset_bytes)
    snapshots, banner = _load_snapshots(args.dataset)
    dataset = _build_dataset(snapshots)

    model, training = _train_tiny_deeplob(dataset)
    sample_window = dataset["X"][0]
    model_bytes = _export_onnx(model, args.output, sample_window)
    expected_input = [float(v) for v in sample_window.reshape(-1)]
    expected_output, engine = _run_ort(
        args.output, model, np.array(expected_input, dtype=np.float32)
    )

    metadata = build_metadata(
        model_bytes=model_bytes,
        training=training,
        dataset=dataset,
        dataset_path=args.dataset,
        dataset_sha256=dataset_sha256,
        banner=banner,
        expected_input=expected_input,
        expected_output=expected_output,
        expected_output_engine=engine,
    )
    metadata_bytes = _write_json(args.metadata_output, metadata)

    expected_input_payload = {
        "model_name": MODEL_NAME,
        "input_name": INPUT_NAME,
        "shape": INPUT_SHAPE,
        "data": expected_input,
    }
    expected_output_payload = {
        "model_name": MODEL_NAME,
        "output_name": OUTPUT_NAME,
        "shape": OUTPUT_SHAPE,
        "engine": engine,
        "data": expected_output,
    }
    expected_input_bytes = _write_json(args.expected_input_output, expected_input_payload)
    expected_output_bytes = _write_json(args.expected_output_output, expected_output_payload)

    manifest = {
        "model_name": MODEL_NAME,
        "generated_by": "tools/export_asterion_public_l2_onnx.py",
        "export_command": EXPORT_COMMAND,
        "source_commit": metadata["source_commit"],
        "source_dirty": metadata["source_dirty"],
        "seed": SEED,
        "artefacts": {
            args.output.name: {"sha256": _sha256(model_bytes), "bytes": len(model_bytes)},
            args.metadata_output.name: {
                "sha256": _sha256(metadata_bytes),
                "bytes": len(metadata_bytes),
            },
            args.expected_input_output.name: {
                "sha256": _sha256(expected_input_bytes),
                "bytes": len(expected_input_bytes),
            },
            args.expected_output_output.name: {
                "sha256": _sha256(expected_output_bytes),
                "bytes": len(expected_output_bytes),
            },
        },
        "source_data": {
            "file": "data/samples/" + args.dataset.name,
            "sha256": dataset_sha256,
            "bytes": len(dataset_bytes),
            "snapshots": dataset["n_snapshots"],
        },
    }
    _write_json(args.manifest_output, manifest)

    print(f"wrote {args.output} ({len(model_bytes)} bytes)")
    print(f"wrote {args.metadata_output}")
    print(f"wrote {args.expected_input_output}")
    print(f"wrote {args.expected_output_output}")
    print(f"wrote {args.manifest_output}")
    print(
        f"  trained_model=True windows={dataset['n_windows']} "
        f"class_counts={dataset['class_counts']} "
        f"initial_train_loss={training['initial_train_loss']} "
        f"eval_loss={training['eval_loss']} "
        f"eval_accuracy_diagnostic={training['eval_accuracy_diagnostic']}"
    )
    print(f"  source_data_sha256={dataset_sha256}")
    print(f"  onnx_sha256={metadata['onnx_sha256']}")
    print(f"  expected_test_output ({engine}) = {expected_output}")
    if args.benchmark > 0:
        _benchmark(args.output, args.metadata_output, args.benchmark)
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

    digest = _sha256(model_bytes)
    if metadata.get("onnx_sha256") and metadata["onnx_sha256"] != digest:
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

    input_array = np.array(metadata["expected_test_input"], dtype=np.float32).reshape(
        metadata["input_shape"]
    )
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    result = np.asarray(session.run(None, {metadata["input_name"]: input_array})[0]).reshape(-1)
    expected = np.asarray(metadata["expected_test_output"], dtype=np.float64)
    max_diff = float(np.max(np.abs(result.astype(np.float64) - expected)))
    if max_diff > 1e-4:
        print(f"expected output mismatch: max_abs_diff={max_diff}", file=sys.stderr)
        return 1
    print(
        f"OK: {model_path.name} ({len(model_bytes)} bytes) reproduces expected output "
        f"via ONNX Runtime (max_abs_diff={max_diff:.2e})"
    )
    return 0


def _benchmark(model_path: pathlib.Path, metadata_path: pathlib.Path, iterations: int) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; skipping benchmark", file=sys.stderr)
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    input_array = np.array(metadata["expected_test_input"], dtype=np.float32).reshape(
        metadata["input_shape"]
    )
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    name = metadata["input_name"]
    for _ in range(min(200, iterations)):  # warm-up
        session.run(None, {name: input_array})
    samples = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        start = time.perf_counter_ns()
        session.run(None, {name: input_array})
        samples[i] = time.perf_counter_ns() - start
    samples.sort()

    def pct(p: float) -> float:
        idx = min(len(samples) - 1, int(p * len(samples)))
        return float(samples[idx])

    total_s = float(samples.sum()) / 1e9
    print(
        "isolated ONNX Runtime inference latency (local Python onnxruntime, not portable):"
    )
    print(
        f"  iterations={iterations} p50={pct(0.50):.0f}ns p95={pct(0.95):.0f}ns "
        f"p99={pct(0.99):.0f}ns p99.9={pct(0.999):.0f}ns max={samples[-1]:.0f}ns "
        f"throughput={iterations / total_s:,.0f}/s"
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

"""Tests for the public selected-results plotting entry point."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest


def _load_plot_module() -> ModuleType:
    script = Path(__file__).parents[1] / "scripts" / "plot_results.py"
    spec = importlib.util.spec_from_file_location("plot_results", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load plotting script: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pooled_feature_deltas_reconstruct_selected_means() -> None:
    module = _load_plot_module()
    root = Path(__file__).parents[1] / "experiments" / "selected_results"
    grouped = pd.read_csv(root / "feature_ablation" / "feature_delta_by_horizon.csv")
    selected = pd.read_csv(root / "feature_ablation" / "feature_group_stability.csv")

    pooled = module.pooled_feature_deltas(grouped)
    comparison = pooled.merge(
        selected[["feature_group", "run_count", "mean_delta_macro_f1"]],
        on="feature_group",
        suffixes=("_pooled", "_selected"),
        validate="one_to_one",
    )

    assert comparison["run_count_pooled"].tolist() == comparison["run_count_selected"].tolist()
    assert comparison["mean_delta_macro_f1_pooled"].to_numpy() == pytest.approx(
        comparison["mean_delta_macro_f1_selected"].to_numpy(), abs=1e-14
    )
    assert (comparison["std_delta_macro_f1"] >= 0.0).all()


def test_verify_selected_results_accepts_committed_inventory() -> None:
    module = _load_plot_module()
    results_dir = Path(__file__).parents[1] / "experiments" / "selected_results"

    verified = module.verify_selected_results(results_dir)

    assert len(verified) == 37
    assert all(path.is_file() for path in verified)


def test_verify_selected_results_rejects_hash_mismatch(tmp_path: Path) -> None:
    module = _load_plot_module()
    results_dir = tmp_path / "selected_results"
    results_dir.mkdir()
    result_path = results_dir / "summary.csv"
    result_path.write_text("metric,value\nmacro_f1,0.5\n", encoding="utf-8")
    manifest = {
        "manifest_version": 1,
        "file_count": 1,
        "files": [
            {
                "path": "experiments\\selected_results\\summary.csv",
                "sha256": hashlib.sha256(b"different contents").hexdigest(),
            }
        ],
    }
    (results_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module.verify_selected_results(results_dir)


def test_generate_figures_writes_png_and_pdf(tmp_path: Path) -> None:
    module = _load_plot_module()
    results_dir = Path(__file__).parents[1] / "experiments" / "selected_results"

    written = module.generate_figures(results_dir, tmp_path)

    assert len(written) == 10
    assert {path.suffix for path in written} == {".png", ".pdf"}
    assert {path.stem for path in written} == set(module.FIGURE_STEMS)
    assert all(path.stat().st_size > 1_000 for path in written)

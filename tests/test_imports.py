"""Import smoke tests for the initial package scaffold."""

from __future__ import annotations

import importlib


def test_chronoslob_imports() -> None:
    package = importlib.import_module("chronoslob")

    assert package.__version__


def test_key_subpackages_import() -> None:
    modules = [
        "chronoslob.data",
        "chronoslob.book",
        "chronoslob.features",
        "chronoslob.labels",
        "chronoslob.models",
        "chronoslob.training",
        "chronoslob.backtest",
        "chronoslob.analysis",
        "chronoslob.utils",
    ]

    for module in modules:
        assert importlib.import_module(module)


def test_cli_module_imports() -> None:
    assert importlib.import_module("chronoslob.cli")

"""Tests for deterministic seeding helpers."""

from __future__ import annotations

import random

import numpy as np

from chronoslob.utils.seeding import seed_everything


def test_seed_everything_reproduces_numpy_sequence() -> None:
    seed_everything(123)
    first = np.random.random(5)

    seed_everything(123)
    second = np.random.random(5)

    np.testing.assert_array_equal(first, second)


def test_repeated_seeding_reproduces_python_random_sequence() -> None:
    seed_everything(456)
    first = [random.random() for _ in range(5)]

    seed_everything(456)
    second = [random.random() for _ in range(5)]

    assert first == second

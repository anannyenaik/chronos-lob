"""Reproducibility helpers."""

from __future__ import annotations

import importlib
import importlib.util
import os
import random
from typing import Any, cast

import numpy as np

MAX_NUMPY_SEED = 2**32 - 1


def seed_everything(seed: int) -> int:
    """Seed Python and NumPy random number generators.

    This helper also seeds PyTorch if it is installed, but PyTorch is intentionally
    not a scaffold dependency. Setting `PYTHONHASHSEED` here records the desired
    value for child processes; Python hash randomisation is only fully controlled
    when the variable is set before interpreter startup.

    Determinism is limited by hardware, library versions, parallelism and any
    future data-loading or modelling code that introduces non-deterministic
    operations. Experiment configs should still record seeds explicitly.
    """
    if not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if seed < 0 or seed > MAX_NUMPY_SEED:
        raise ValueError(f"seed must be between 0 and {MAX_NUMPY_SEED}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if importlib.util.find_spec("torch") is not None:
        torch = cast(Any, importlib.import_module("torch"))

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    return seed

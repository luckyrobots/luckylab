"""Random number generator seeding utilities."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_rng(seed: int, torch_deterministic: bool = False) -> None:
    """Seed all random number generators for reproducibility.

    Seeds Python's random module, NumPy, and PyTorch for consistent
    results across runs.

    Args:
        seed: The seed value to use.
        torch_deterministic: If True, use deterministic algorithms in PyTorch
            (slower but reproducible).

    Example:
        >>> from luckylab.utils import seed_rng
        >>> seed_rng(42)  # All RNGs now seeded with 42
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    # Ref: https://docs.pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)  # Seed RNG for all devices.

    # Use deterministic algorithms when possible.
    if torch_deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

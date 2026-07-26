"""Deterministic seeding: identical data order across arms.

Arms must differ only in geometry. The data generator is seeded independently
of the model initializer so that changing the arm cannot shift the data
stream, and both are derived from one run seed.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Guarded deliberately: an unconditional torch.cuda.manual_seed_all creates
    # a CUDA context even for a CPU-only run, and a 12-worker CPU sweep then
    # tries to allocate twelve contexts on a GPU another study is holding. That
    # is how this line first announced itself (Phase 00a, OOM at launch).
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)


def data_generator(seed: int, device=None) -> torch.Generator:
    """A data stream keyed only on the run seed, never on the arm."""
    gen = torch.Generator(device=device or "cpu")
    gen.manual_seed(seed * 1_000_003 + 17)
    return gen

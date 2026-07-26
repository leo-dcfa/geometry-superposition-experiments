"""Test-suite defaults: CPU-pinned and deterministic.

The RTX 5090 in this workstation is routinely held by another study; the test
suite must never contend for it, and CI has no GPU at all (rule R5: a fresh
clone runs the suite).
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _cpu_and_deterministic():
    torch.manual_seed(0)
    prev = torch.get_default_device()
    torch.set_default_device("cpu")
    yield
    torch.set_default_device(prev)

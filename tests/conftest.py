"""Test-suite defaults: CPU-pinned and deterministic.

The suite computes nothing on a GPU, and must never touch one — not "prefer
CPU", *never touch it*. A GPU may be absent (CI has none, and rule R5 requires
a fresh clone to run green), or present but fully allocated by an unrelated
process.

The distinction is not pedantic. An earlier version pinned CPU with
``torch.set_default_device("cpu")`` but read ``torch.get_default_device()``
first, which queries the accelerator and initializes a CUDA context. That is
harmless on an idle GPU and fails outright on a busy one: with the device
nearly full, tests died with ``AcceleratorError: CUDA error: out of memory`` in
a suite that runs entirely on CPU.

Hiding the device before torch is imported is the only version of "CPU-pinned"
that holds regardless of what else is running.
"""

from __future__ import annotations

import os

# Must precede the torch import: CUDA visibility is read at initialization.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)
    yield


def test_gpu_is_not_visible_to_the_suite():
    """Guards the pin itself. If this fails, every other test's isolation is a
    matter of luck rather than configuration."""
    assert not torch.cuda.is_available(), (
        "the test suite must not see a GPU; CUDA_VISIBLE_DEVICES was not "
        "cleared before torch was imported"
    )

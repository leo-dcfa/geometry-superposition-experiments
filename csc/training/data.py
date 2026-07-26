"""Sparse feature data for the toy models.

Standard TMS generative model: each of N features is independently active with
probability ``1 − sparsity``; an active feature takes a value uniform on
(0, 1]. Feature importance is geometric, I_i = decay^i, and enters the loss as
a per-feature weight.

``sparsity`` is the *probability a feature is off*, so sparsity → 1 is the
sparse regime and sparsity → 0 the dense one. P2's phase boundary S* is stated
in these units.
"""

from __future__ import annotations

import torch
from torch import Tensor


def importance_spectrum(n_features: int, decay: float = 0.9, device=None) -> Tensor:
    """Geometric importance I_i = decay^i, normalized to mean 1."""
    imp = decay ** torch.arange(n_features, dtype=torch.float32, device=device)
    return imp / imp.mean()


def sample_batch(
    batch_size: int,
    n_features: int,
    sparsity: float,
    generator: torch.Generator | None = None,
    device=None,
) -> Tensor:
    """``(batch_size, n_features)`` sparse non-negative features."""
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity is P(feature off) and must lie in [0, 1)")
    kw = {"generator": generator, "device": device}
    active = torch.rand(batch_size, n_features, **kw) >= sparsity
    values = torch.rand(batch_size, n_features, **kw)
    return values * active


def weighted_mse(pred: Tensor, target: Tensor, importance: Tensor) -> Tensor:
    return ((pred - target).square() * importance).sum(-1).mean()

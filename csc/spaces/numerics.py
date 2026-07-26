"""Numerically safe primitives shared by every space.

Found by the Phase-00b gradient audit: ``d(0, expmap0(v))`` and every pairwise
distance end in a square root, and the gradient of √· at 0 is infinite. Any
pair of coincident points therefore produced NaN gradients that propagate
through the whole batch — at *every* radius, not just near the boundary, so
this was not a saturation problem and R2 would never have caught it.

Coincident points are not exotic here. Prototypes collapse onto each other
exactly when a model runs out of room, which is the regime the capacity
metrics are designed to probe, so the failure would have arrived precisely in
the cells that matter most and looked like a training divergence.

Two fixes, applied uniformly across all arms so that no geometry gets a
numerically better deal than another (the 00c parity concern, in the
arithmetic rather than the readout):

- ``safe_sqrt`` adds an ε² floor inside the root. At coincidence the outer
  factor is exactly 0, so the product is 0 rather than 0·∞ = NaN.
- ``pairwise_sq_dist`` computes squared distances by direct broadcasting.
  ``torch.cdist`` is exact enough in the forward pass but has the same NaN
  gradient at coincident points and cannot be given an ε.
"""

from __future__ import annotations

import torch
from torch import Tensor

_SQRT_EPS = {
    torch.float64: 1e-12,
    torch.float32: 1e-6,
    torch.bfloat16: 1e-2,
    torch.float16: 1e-3,
}


def sqrt_eps(dtype: torch.dtype) -> float:
    return _SQRT_EPS.get(dtype, 1e-6)


def safe_sqrt(x: Tensor) -> Tensor:
    """√(x + ε²) — finite gradient at x = 0, negligible bias for x >> ε²."""
    eps = sqrt_eps(x.dtype)
    return (x.clamp_min(0.0) + eps * eps).sqrt()


def pairwise_sq_dist(x: Tensor, y: Tensor) -> Tensor:
    """``(..., n, d) x (m, d) -> (..., n, m)`` squared L2 distances.

    Direct broadcasting rather than the expanded ‖x‖²+‖y‖²−2⟨x,y⟩ form: the
    expansion loses catastrophic-cancellation precision for nearby points, and
    nearby points are the entire subject of the interference metrics (P3).
    Memory is (n·m·d), which is comfortable at the widths this study uses
    (d ≤ 8, n·m ≤ ~10⁶); a chunked path is the Phase-2 problem, where the
    bottleneck is narrow but the prototype count is vocabulary-sized.
    """
    return (x.unsqueeze(-2) - y).square().sum(-1)


def pairwise_dist(x: Tensor, y: Tensor) -> Tensor:
    """``(..., n, d) x (m, d) -> (..., n, m)`` L2 distances, gradient-safe at 0."""
    return safe_sqrt(pairwise_sq_dist(x, y))

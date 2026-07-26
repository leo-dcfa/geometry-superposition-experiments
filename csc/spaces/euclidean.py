"""Flat baseline: expmap0/logmap0 are the identity, dist is the L2 norm.

This is the K = 0 arm and the reference against which every capacity claim in
H-MAIN is stated. It is also the exact limit the curved spaces are tested
against (``test_spaces.py::test_curvature_limit_is_euclidean``).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from csc.spaces.numerics import pairwise_dist, safe_sqrt


def euclidean_dist_matrix(x: Tensor, y: Tensor) -> Tensor:
    """``(..., n, d) x (m, d) -> (..., n, m)`` pairwise L2 distances.

    Not ``torch.cdist``, for two measured reasons. Its default silently
    switches to the expanded ‖x‖²+‖y‖²−2⟨x,y⟩ matmul above 25 rows, losing
    catastrophic-cancellation precision exactly for *nearby* points (4e-8 on a
    float64 self-distance); and every ``cdist`` mode produces NaN gradients at
    coincident points, with no way to pass an epsilon. Nearby and coincident
    points are the entire subject of the interference metrics (P3), so neither
    is acceptable here. See ``spaces/numerics.py``.
    """
    return pairwise_dist(x, y)


class EuclideanSpace(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    @property
    def kappa(self) -> float:
        return 0.0

    def expmap0(self, v: Tensor) -> Tensor:
        return v

    def logmap0(self, x: Tensor) -> Tensor:
        return x

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        return safe_sqrt((x - y).square().sum(-1))

    def project(self, x: Tensor) -> Tensor:
        return x

    def dist_matrix(self, x: Tensor, y: Tensor) -> Tensor:
        return euclidean_dist_matrix(x, y)

    def radius(self, x: Tensor) -> Tensor:
        return safe_sqrt(x.square().sum(-1))

    def saturation_fraction(self, x: Tensor) -> Tensor:
        return torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, kappa=0.0"

"""Flat baseline: expmap0/logmap0 are the identity, dist is the L2 norm.

This is the K = 0 arm and the reference against which every capacity claim in
H-MAIN is stated. It is also the exact limit the curved spaces are tested
against (``test_spaces.py::test_curvature_limit_is_euclidean``).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def euclidean_dist_matrix(x: Tensor, y: Tensor) -> Tensor:
    """``(..., n, d) x (m, d) -> (..., n, m)`` pairwise L2 distances.

    ``compute_mode`` is pinned to the direct form. ``cdist``'s default silently
    switches to the expanded ‖x‖²+‖y‖²−2⟨x,y⟩ matmul above 25 rows, which loses
    catastrophic-cancellation precision exactly for *nearby* points — measured
    here at 4e-8 on a self-distance in float64. Nearby points are the entire
    subject of the interference metrics (P3) and of the minimum-pairwise-
    distance metric, so the mm path is not an acceptable default for this
    study. The latent widths here are d ≤ 8, where the matmul trick buys
    almost nothing anyway.
    """
    return torch.cdist(x, y, p=2, compute_mode="donot_use_mm_for_euclid_dist")


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
        return (x - y).norm(dim=-1)

    def project(self, x: Tensor) -> Tensor:
        return x

    def dist_matrix(self, x: Tensor, y: Tensor) -> Tensor:
        return euclidean_dist_matrix(x, y)

    def radius(self, x: Tensor) -> Tensor:
        return x.norm(dim=-1)

    def saturation_fraction(self, x: Tensor) -> Tensor:
        return torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, kappa=0.0"

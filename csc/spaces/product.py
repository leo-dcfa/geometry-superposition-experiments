"""Product manifold: the Phase-2 bottleneck shape (k independent curved factors).

Coordinates are the concatenation of factor coordinates; maps apply
factor-wise and the product distance is the L2 combination of factor
distances, which preserves the shared normalization d(0, expmap0(v)) = ‖v‖.

A product of k copies of H² at curvature K is *not* H^{2k} at curvature K —
its volume growth is the product of the factors' — which is exactly why
Phase 2 pre-registers a product of narrow factors rather than one wide curved
space: the wide version is the saturated full-width readout the audit
condemned.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from csc.spaces.numerics import safe_sqrt
from csc.spaces.stereographic import StereographicSpace


class ProductSpace(nn.Module):
    def __init__(self, factors: list[nn.Module]) -> None:
        super().__init__()
        if not factors:
            raise ValueError("a product space needs at least one factor")
        self.factors = nn.ModuleList(factors)
        self.dim = sum(f.dim for f in factors)

    @classmethod
    def hyperbolic(cls, n_factors: int, dim_per_factor: int, kappa: float) -> ProductSpace:
        """``n_factors`` copies of a curvature-K space of equal width."""
        return cls([StereographicSpace(dim_per_factor, kappa) for _ in range(n_factors)])

    @property
    def kappa(self) -> float:
        """Common curvature of the factors; ``nan`` if they differ (mixed product)."""
        values = {f.kappa for f in self.factors}
        return next(iter(values)) if len(values) == 1 else float("nan")

    def _slices(self):
        start = 0
        for factor in self.factors:
            yield factor, slice(start, start + factor.dim)
            start += factor.dim

    def _map(self, x: Tensor, op: str) -> Tensor:
        return torch.cat([getattr(f, op)(x[..., s]) for f, s in self._slices()], dim=-1)

    def expmap0(self, v: Tensor) -> Tensor:
        return self._map(v, "expmap0")

    def logmap0(self, x: Tensor) -> Tensor:
        return self._map(x, "logmap0")

    def project(self, x: Tensor) -> Tensor:
        return self._map(x, "project")

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        per_factor = torch.stack([f.dist(x[..., s], y[..., s]) for f, s in self._slices()], dim=-1)
        return safe_sqrt(per_factor.square().sum(-1))

    def dist_matrix(self, x: Tensor, y: Tensor) -> Tensor:
        per_factor = torch.stack(
            [f.dist_matrix(x[..., s], y[..., s]) for f, s in self._slices()], dim=-1
        )
        return safe_sqrt(per_factor.square().sum(-1))

    def radius(self, x: Tensor) -> Tensor:
        per_factor = torch.stack([f.radius(x[..., s]) for f, s in self._slices()], dim=-1)
        return safe_sqrt(per_factor.square().sum(-1))

    def saturation_fraction(self, x: Tensor) -> Tensor:
        """Worst factor's fraction — a single saturated factor invalidates the point."""
        per_factor = torch.stack(
            [f.saturation_fraction(x[..., s]) for f, s in self._slices()], dim=-1
        )
        return per_factor.max(dim=-1).values

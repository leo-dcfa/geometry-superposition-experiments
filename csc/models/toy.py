"""Toy-Models-of-Superposition autoencoder, parametric over Space.

N sparse features are encoded linearly into the tangent space at the origin,
pushed onto the manifold, and reconstructed through a distance-to-prototype
readout:

    p    = expmap0(W x + b)
    x̂_i = head_i(−d(p, P_i))          per-prototype bias + scale (R1)

The manifold is the only thing that differs between arms. Encoder shape,
prototype count, head, initialization and parameter count are identical
across κ, which is what makes a capacity comparison a comparison.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from csc.layers.readout import DistanceReadout, ResponseHead


class ToySuperposition(nn.Module):
    def __init__(
        self,
        space,
        n_features: int,
        head: str = "rbf",
        proto_init_scale: float = 0.2,
    ) -> None:
        super().__init__()
        self.space = space
        self.n_features = n_features
        self.encoder = nn.Linear(n_features, space.dim)
        self.readout = DistanceReadout(space, n_features, init_scale=proto_init_scale)
        self.head = ResponseHead(n_features, kind=head)

    @property
    def latent_dim(self) -> int:
        return self.space.dim

    def encode(self, x: Tensor) -> Tensor:
        """``(batch, n_features) -> (batch, dim)`` points on the manifold."""
        return self.space.expmap0(self.encoder(x))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.readout(self.encode(x)))

    @torch.no_grad()
    def geometry_report(self, x: Tensor) -> dict:
        """R2 quantities for one batch: geodesic radii and saturation fractions."""
        points = self.encode(x)
        radius = self.space.radius(points)
        kappa = float(self.space.kappa)
        scaled = radius * abs(kappa) ** 0.5
        frac = self.space.saturation_fraction(points)
        quantiles = torch.tensor(
            [0.05, 0.25, 0.5, 0.75, 0.95], dtype=radius.dtype, device=radius.device
        )
        return {
            "kappa": kappa,
            "radius_median": radius.median().item(),
            "scaled_radius_median": scaled.median().item(),
            "scaled_radius_quantiles": torch.quantile(scaled, quantiles).tolist(),
            "saturation_median": frac.median().item(),
            "saturation_max": frac.max().item(),
            "saturation_p95": torch.quantile(frac, 0.95).item(),
        }


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

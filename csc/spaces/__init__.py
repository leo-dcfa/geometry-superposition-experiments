"""Geometry conditions: every CSC model is parametric over a Space.

``kappa`` is the true sectional curvature everywhere in this package. See
``base.py`` for the three conventions the test suite enforces.
"""

from __future__ import annotations

from csc.spaces.base import Space
from csc.spaces.controls import ClampedEuclideanSpace, NormalizedEuclideanSpace
from csc.spaces.euclidean import EuclideanSpace
from csc.spaces.product import ProductSpace
from csc.spaces.stereographic import StereographicSpace

__all__ = [
    "Space",
    "EuclideanSpace",
    "StereographicSpace",
    "ClampedEuclideanSpace",
    "NormalizedEuclideanSpace",
    "ProductSpace",
    "make_space",
]


def make_space(arm: str, dim: int, kappa: float = 0.0, **kwargs):
    """Construct the arm named in a run config.

    Arms: ``euclidean``, ``curved`` (kappa != 0), ``clamped`` (needs
    ``max_dist``), ``normalized``, ``product`` (needs ``n_factors``).
    """
    if arm == "euclidean":
        return EuclideanSpace(dim)
    if arm == "curved":
        return StereographicSpace(dim, kappa, **kwargs)
    if arm == "clamped":
        return ClampedEuclideanSpace(dim, **kwargs)
    if arm == "normalized":
        return NormalizedEuclideanSpace(dim, **kwargs)
    if arm == "product":
        n_factors = kwargs.pop("n_factors")
        return ProductSpace.hyperbolic(n_factors, dim // n_factors, kappa)
    raise ValueError(f"unknown arm {arm!r}")

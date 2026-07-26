"""Space protocol: the geometry axis every CSC model is parametric over.

Conventions, enforced by ``tests/test_spaces.py``:

**1. ``kappa`` is the true constant sectional curvature K.**
The parent program's κ was the κ-stereographic *parameter*, which equals
¼ of the sectional curvature under its halved-distance normalization. CSC's
central pre-registered form (P1: ``log N* = α·√|κ|·R + β``) is fitted against
√|κ|, so a factor-of-2 convention slip would land directly in α. Here
``space.kappa`` is K, and ``curvature_calibration`` in the test suite pins it
against the closed-form law of cosines rather than against another
implementation.

**2. Distance is normalized so ``d(0, expmap0(v)) == ‖v‖``.**
A tangent vector's norm *is* its geodesic radius from the origin. This makes
the R2 operating band √|K|·r readable straight off the tangent
parametrization with no extra map, and it is why the saturation monitor can
report a geodesic radius and a coordinate ball fraction from the same tensor.

**3. K → 0 is exactly Euclidean**, not approximately: the K = 0 branch is the
identity/L2 path, and the curved branches converge to it continuously.

**4. Tangent-space parametrization.** Trainable parameters are ordinary
Euclidean tensors; geometry enters only through ``expmap0``/``logmap0``/
``dist`` in the forward pass. Spaces may own geometry scalars (a learnable
curvature magnitude), which is why concrete spaces are ``nn.Module``s.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class Space(Protocol):
    dim: int

    @property
    def kappa(self) -> float:
        """True sectional curvature K of the manifold (0.0 for flat spaces)."""
        ...

    def expmap0(self, v: Tensor) -> Tensor:
        """Tangent at origin -> manifold. ``(..., dim) -> (..., dim)``."""
        ...

    def logmap0(self, x: Tensor) -> Tensor:
        """Manifold -> tangent at origin. Inverse of ``expmap0`` on its domain."""
        ...

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        """Geodesic distance, broadcast over leading axes. ``(...,)``."""
        ...

    def project(self, x: Tensor) -> Tensor:
        """Retraction onto the numerically valid region. Idempotent."""
        ...

    def dist_matrix(self, x: Tensor, y: Tensor) -> Tensor:
        """Pairwise distances ``(..., n, dim) x (m, dim) -> (..., n, m)``."""
        ...

    def radius(self, x: Tensor) -> Tensor:
        """Geodesic radius of each point from the origin. ``(..., dim) -> (...,)``."""
        ...

    def saturation_fraction(self, x: Tensor) -> Tensor:
        """R2 monitor: fraction of the maximal representable radius, in [0, 1).

        Geometry-specific by necessity and reported as such:

        - K < 0: coordinate ball fraction ‖x‖ / R_ball with R_ball = 2/√|K|.
          This is the quantity that saturates ``artanh`` and produces the
          boundary-pinned regime the audit condemned.
        - K > 0: geodesic fraction r / (π/√K) toward the antipode, since the
          spherical stereographic chart has no finite coordinate ball but does
          have a finite diameter.
        - K = 0: identically zero — a flat space has no boundary to pin to.
        """
        ...

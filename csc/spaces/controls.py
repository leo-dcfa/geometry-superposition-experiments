"""Rule R3 controls: flat spaces that mimic a curved readout's *conditioning*.

The audit's standing worry (H1b) is that a bounded-diameter distance readout
behaves like an implicitly normalized one, and that any capacity advantage
attributed to curvature is really the conditioning of a bounded readout. Both
controls here are flat — sectional curvature is exactly 0 — and both ship in
the same sweep as the curved arm they control, never afterwards.

If ``ClampedEuclideanSpace`` reproduces ≥ 70% of the hyperbolic capacity gain,
falsifier F1.3 fires and H-MAIN is dead at toy scale. That is the intended
outcome of a working control, not a failure of it.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from csc.spaces.euclidean import euclidean_dist_matrix


class ClampedEuclideanSpace(nn.Module):
    """Flat space whose distances are hard-clipped at ``max_dist``.

    Reproduces the bounded-diameter conditioning of a spherical readout (and
    the matched operating diameter of a hyperbolic one) with no curvature.

    Known asymmetry, stated because it bears on how F1.3 is read: the clip is
    a *hard* one, so pairs beyond ``max_dist`` receive exactly zero gradient,
    whereas a sphere's metric saturates smoothly and keeps a small gradient
    everywhere below the antipode. The control therefore reproduces the
    bounded *range* of the readout but not its smooth compression. It is
    implemented as the spec's R3 defines it; the caveat is logged here and in
    VALIDATION.md so that a clean F1.3 is not over-read as ruling out every
    conditioning account.
    """

    def __init__(self, dim: int, max_dist: float) -> None:
        super().__init__()
        if max_dist <= 0:
            raise ValueError("max_dist must be positive")
        self.dim = dim
        self.max_dist = float(max_dist)

    @property
    def kappa(self) -> float:
        return 0.0

    def expmap0(self, v: Tensor) -> Tensor:
        return v

    def logmap0(self, x: Tensor) -> Tensor:
        return x

    def project(self, x: Tensor) -> Tensor:
        return x

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        return (x - y).norm(dim=-1).clamp(max=self.max_dist)

    def dist_matrix(self, x: Tensor, y: Tensor) -> Tensor:
        return euclidean_dist_matrix(x, y).clamp(max=self.max_dist)

    def radius(self, x: Tensor) -> Tensor:
        return x.norm(dim=-1)

    def saturation_fraction(self, x: Tensor) -> Tensor:
        """Radius as a fraction of the clip radius (``max_dist``/2 from centre)."""
        return (2 * x.norm(dim=-1) / self.max_dist).clamp(max=1.0)

    def clipped_fraction(self, x: Tensor, y: Tensor) -> Tensor:
        """Diagnostic: fraction of pairs sitting on the clip (zero-gradient)."""
        return ((x - y).norm(dim=-1) >= self.max_dist).to(x.dtype).mean()

    def extra_repr(self) -> str:
        return f"dim={self.dim}, kappa=0.0, max_dist={self.max_dist:.4g}"


class NormalizedEuclideanSpace(nn.Module):
    """Flat space with a unit-norm latent: ``expmap0`` projects onto S^{dim-1}.

    Distances are chordal (straight-line in the ambient space), not geodesic
    on the sphere — this is the "normalized-Euclidean" control of R3, i.e. the
    ordinary L2-normalize-then-compare readout, not a spherical manifold. The
    spherical *manifold* arm is ``StereographicSpace(kappa > 0)``; keeping the
    two distinct is what lets a positive-curvature result be separated from a
    normalization result.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    @property
    def kappa(self) -> float:
        return 0.0

    def expmap0(self, v: Tensor) -> Tensor:
        return v / v.norm(dim=-1, keepdim=True).clamp_min(self.eps)

    def logmap0(self, x: Tensor) -> Tensor:
        return x

    def project(self, x: Tensor) -> Tensor:
        return self.expmap0(x)

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        return (self.project(x) - self.project(y)).norm(dim=-1)

    def dist_matrix(self, x: Tensor, y: Tensor) -> Tensor:
        return euclidean_dist_matrix(self.project(x), self.project(y))

    def radius(self, x: Tensor) -> Tensor:
        """Identically 1 on the unit sphere — the control has no radial channel."""
        return self.project(x).norm(dim=-1)

    def saturation_fraction(self, x: Tensor) -> Tensor:
        return torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, kappa=0.0, unit_norm=True"

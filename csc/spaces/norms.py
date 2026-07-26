"""Non-Euclidean *norms* — the axis curvature could not reach.

Study 1 established that no Riemannian curvature can change angular capacity,
and the reason is structural rather than empirical: at any point of a
Riemannian manifold the set of directions is S^(d−1), and in normal coordinates
the metric there is a scalar multiple of the identity. **Angles at a point are
Euclidean whatever the curvature is.** Curvature governs how angles evolve
along geodesics, never how many directions exist. Sweeping κ harder was never
going to work.

Superposition capacity is angular — it is set by how many near-orthogonal
directions fit, and interference is directional overlap. So the way to move it
is to leave Riemannian geometry and change the **norm**.

The spaces here are flat (zero curvature everywhere) and differ only in how
distance is computed:

- ``LpSpace(p)`` — the ℓ^p family. p=2 is Euclidean and is the control; the
  interesting cases are p=1 and p→∞, whose unit balls have qualitatively
  different extremal structure.
- ``FinslerSpace`` — a direction-dependent norm with learnable per-axis
  weights, the smallest departure from Riemannian geometry that genuinely
  distorts angles rather than radii.

**Why ℓ^∞ is the pre-registered candidate.** The ℓ^∞ unit ball is a hypercube
with **2^d vertices** — exponentially many maximally separated extreme points
in d dimensions, with no exponential radial blow-up. That is the shape
superposition wants, and it is why hyperdimensional computing and
vector-symbolic architectures get large capacity from high-dimensional ±1
codes. Hamming space is the geometry whose capacity is genuinely angular.

**The honest counter-hypothesis, registered with it.** Euclidean space already
admits exp(Ω(ε²d)) unit vectors with pairwise |⟨u,v⟩| ≤ ε (Johnson-
Lindenstrauss), which is far more than Study 1's models ever used — they
recovered ~18 features in d=8. If flat ℓ² capacity is already unreached, the
binding constraint is readout resolution and optimization, not the geometry,
and **no norm will help either**. That outcome would be more informative than a
win: it would say stop looking at spaces and start looking at readouts.

All spaces keep the Space protocol so every Study-1 control, monitor and metric
applies unchanged.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from csc.spaces.numerics import safe_sqrt, sqrt_eps


def _lp_norm(diff: Tensor, p: float, dim: int = -1) -> Tensor:
    """‖·‖_p with a gradient that survives the origin.

    ``torch.norm`` is NaN-at-zero for p ≤ 1 and for the p=∞ subgradient, which
    is the same defect Study 1's 00b audit found in the Euclidean path — and
    for the same reason it matters: coincident points are what a model out of
    room actually produces.
    """
    eps = sqrt_eps(diff.dtype)
    if p == 2.0:
        return safe_sqrt(diff.square().sum(dim))
    if p == float("inf"):
        # smooth-max would blur the very extremal structure being tested, so
        # the true max is used and the eps only guards the all-zero case
        return diff.abs().amax(dim).clamp_min(eps)
    if p == 1.0:
        return diff.abs().sum(dim).clamp_min(eps)
    return (diff.abs().clamp_min(eps) ** p).sum(dim) ** (1.0 / p)


class LpSpace(nn.Module):
    """Flat space under the ℓ^p norm. ``p=2`` is exactly ``EuclideanSpace``.

    Curvature is zero by construction, so any capacity difference against the
    p=2 control is attributable to the norm and to nothing else — which is the
    comparison Study 1 could not make.
    """

    def __init__(self, dim: int, p: float = 2.0) -> None:
        super().__init__()
        if p < 1.0 and p != float("inf"):
            raise ValueError("p < 1 is not a norm (triangle inequality fails)")
        self.dim = dim
        self.p = float(p)

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
        return _lp_norm(x - y, self.p)

    def dist_matrix(self, x: Tensor, y: Tensor) -> Tensor:
        return _lp_norm(x.unsqueeze(-2) - y, self.p)

    def radius(self, x: Tensor) -> Tensor:
        return _lp_norm(x, self.p)

    def saturation_fraction(self, x: Tensor) -> Tensor:
        """Flat and unbounded — nothing to pin against, so R2 never fires."""
        return torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, p={self.p}, kappa=0.0"


class FinslerSpace(nn.Module):
    """Direction-dependent norm: ‖v‖ = ‖w ⊙ v‖_p with learnable positive w.

    The minimal departure from Riemannian geometry that distorts *angles*.
    A Riemannian metric rescales space uniformly at a point; this rescales each
    axis differently, so the unit ball is an ℓ^p ball stretched per-axis and
    directions are no longer interchangeable.

    Rule R1 applies as ever: the weights are extra free parameters, so any arm
    using this must be compared against a control with matched parameter count,
    or the comparison measures capacity-per-parameter rather than geometry.
    ``n_extra_parameters`` reports the count so a sweep can match it.
    """

    def __init__(self, dim: int, p: float = 2.0, learnable: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.p = float(p)
        raw = torch.zeros(dim)  # softplus(0) ≈ 0.693, uniform => starts isotropic
        if learnable:
            self.log_weights = nn.Parameter(raw)
        else:
            self.register_buffer("log_weights", raw)

    @property
    def kappa(self) -> float:
        return 0.0

    @property
    def n_extra_parameters(self) -> int:
        return self.dim if isinstance(self.log_weights, nn.Parameter) else 0

    def weights(self, dtype: torch.dtype | None = None) -> Tensor:
        w = nn.functional.softplus(self.log_weights)
        return w.to(dtype) if dtype is not None else w

    def expmap0(self, v: Tensor) -> Tensor:
        return v

    def logmap0(self, x: Tensor) -> Tensor:
        return x

    def project(self, x: Tensor) -> Tensor:
        return x

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        return _lp_norm(self.weights(x.dtype) * (x - y), self.p)

    def dist_matrix(self, x: Tensor, y: Tensor) -> Tensor:
        return _lp_norm(self.weights(x.dtype) * (x.unsqueeze(-2) - y), self.p)

    def radius(self, x: Tensor) -> Tensor:
        return _lp_norm(self.weights(x.dtype) * x, self.p)

    def saturation_fraction(self, x: Tensor) -> Tensor:
        return torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)

    def anisotropy(self) -> float:
        """max/min axis weight — 1.0 means it stayed isotropic (Riemannian)."""
        w = self.weights()
        with torch.no_grad():
            return float(w.max() / w.min().clamp_min(1e-9))

    def extra_repr(self) -> str:
        return f"dim={self.dim}, p={self.p}, kappa=0.0, anisotropy={self.anisotropy():.3f}"

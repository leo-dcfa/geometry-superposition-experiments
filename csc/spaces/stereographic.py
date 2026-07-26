"""Constant-curvature space of true sectional curvature K, hyperbolic or spherical.

The κ-stereographic model is implemented directly here rather than delegated,
so that the study owns its curvature convention end to end (see
``spaces/base.py`` for why a factor of 2 matters to P1). With
``s = √|K| / 2`` the model is the κ-stereographic chart at parameter κ = K/4,
and every returned distance is half the standard one — which simultaneously
buys ``d(0, expmap0(v)) = ‖v‖`` and true sectional curvature K.

Closed forms used (r = ‖v‖, n = ‖x‖):

    expmap0(v)  = tan_K(s·r) / (s·r) · v
    logmap0(x)  = tan_K⁻¹(s·n) / (s·n) · x
    dist(x, y)  = tan_K⁻¹(s·w) / s,
                  w² = ‖x−y‖² / ((1+κ‖x‖²)(1+κ‖y‖²) − κ‖x−y‖²)

with tan_K = tanh for K < 0 and tan for K > 0. The ``w`` identity is the
gyro-distance ‖(−x) ⊕_κ y‖ in closed form; it avoids materialising Möbius
addition, which is the usual source of sign errors and of catastrophic
cancellation for nearby points.

Domain and saturation:

- K < 0: coordinates live in the open ball of radius R_ball = 2/√|K|. The
  boundary is where ``artanh`` diverges; rule R2 exists because a readout
  operating there is measuring the clamp, not the geometry.
- K > 0: the chart covers the sphere minus a point; there is no coordinate
  bound, but the diameter is finite (π/√K, the antipode) and ``expmap0``
  wraps. ``logmap0 ∘ expmap0`` is the identity only for r < π/√K, which is
  the domain the tests pin.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from csc.spaces.numerics import pairwise_sq_dist, safe_sqrt

# Boundary standoff for the artanh/ball domain, per dtype. fp32's value is the
# one Phase 00b audits; bf16 gets a far looser standoff because its ~3 decimal
# digits of mantissa cannot resolve a 1e-6 approach to the boundary at all.
_EPS = {torch.float32: 1e-6, torch.float64: 1e-12, torch.bfloat16: 1e-2, torch.float16: 1e-3}

# Ceiling on the gyro-distance w. Only reachable near antipodal pairs on a
# sphere, where the distance has already saturated at π/√K; the clamp keeps the
# gradient of the intervening division finite instead of NaN.
_W_MAX = 1e6


def _eps(dtype: torch.dtype) -> float:
    return _EPS.get(dtype, 1e-6)


class StereographicSpace(nn.Module):
    """Constant-curvature space. ``kappa`` is the true sectional curvature.

    ``learnable`` makes the curvature *magnitude* trainable while the sign is
    fixed at construction. The sign is deliberately not free: a sign-crossing
    κ needs the series-continued branch at |K| ≈ 0 to avoid NaN gradients from
    the unselected branch, which is only built if P4 runs. The parent
    program's F6 finding (a ``κ = −softplus`` parametrization cannot express
    the κ > 0 regime and so its null was parametrization-limited, not a null)
    is recorded here as the reason this limitation is stated up front rather
    than discovered later.
    """

    def __init__(self, dim: int, kappa: float, learnable: bool = False) -> None:
        super().__init__()
        if kappa == 0.0:
            raise ValueError("kappa=0 is EuclideanSpace; the flat branch is not a special case here")
        self.dim = dim
        self.learnable = learnable
        self.curvature_sign = 1 if kappa > 0 else -1
        # A fixed curvature is held as a Python float, not an fp32 buffer: it is
        # a *constant of the experiment*, and an fp32 round-trip puts ~1e-8
        # relative error into √|K|, which the law-of-cosines calibration test
        # resolves. Learnable arms take the fp32 parameter path, where that
        # precision is irrelevant because κ is being optimized anyway.
        self._kappa_const = float(kappa)
        if learnable:
            # log-magnitude keeps |K| > 0 without a softplus's flat region
            self.log_magnitude = nn.Parameter(
                torch.tensor(math.log(abs(kappa)), dtype=torch.float32)
            )

    # ---- geometry scalars -------------------------------------------------

    @property
    def kappa(self) -> float:
        if self.learnable:
            return float(self.curvature_sign * self.log_magnitude.exp())
        return self._kappa_const

    def kappa_tensor(self, dtype: torch.dtype | None = None) -> Tensor | float:
        if not self.learnable:
            return self._kappa_const
        k = self.curvature_sign * self.log_magnitude.exp()
        return k.to(dtype) if dtype is not None else k

    def _s(self, dtype: torch.dtype) -> Tensor | float:
        """s = √|K| / 2 — the κ-stereographic scale at parameter κ = K/4."""
        if not self.learnable:
            return math.sqrt(abs(self._kappa_const)) / 2
        return (self.log_magnitude.exp().sqrt() / 2).to(dtype)

    @property
    def ball_radius(self) -> float:
        """Coordinate radius of the model. ``inf`` for K > 0 (unbounded chart)."""
        if self.curvature_sign > 0:
            return math.inf
        return 2.0 / math.sqrt(abs(self.kappa))

    @property
    def diameter(self) -> float:
        """Maximal geodesic distance. ``inf`` for K < 0."""
        if self.curvature_sign < 0:
            return math.inf
        return math.pi / math.sqrt(abs(self.kappa))

    # ---- maps -------------------------------------------------------------

    def expmap0(self, v: Tensor) -> Tensor:
        eps = _eps(v.dtype)
        s = self._s(v.dtype)
        r = safe_sqrt(v.square().sum(-1, keepdim=True)).clamp_min(eps)
        arg = s * r
        if self.curvature_sign < 0:
            coeff = torch.tanh(arg) / arg
        else:
            # tan diverges at the antipode (arg = π/2); stand off from it so a
            # runaway tangent norm produces a large-but-finite coordinate
            # rather than an inf that poisons the whole batch
            coeff = torch.tan(arg.clamp(max=math.pi / 2 - eps)) / arg
        return coeff * v

    def logmap0(self, x: Tensor) -> Tensor:
        eps = _eps(x.dtype)
        s = self._s(x.dtype)
        x = self.project(x)
        n = safe_sqrt(x.square().sum(-1, keepdim=True)).clamp_min(eps)
        arg = s * n
        if self.curvature_sign < 0:
            coeff = torch.atanh(arg.clamp(max=1 - eps)) / arg
        else:
            coeff = torch.atan(arg) / arg
        return coeff * x

    def project(self, x: Tensor) -> Tensor:
        if self.curvature_sign > 0:
            return x  # the spherical chart is all of R^d
        eps = _eps(x.dtype)
        s = self._s(x.dtype)
        max_norm = (1 - eps) / s
        n = safe_sqrt(x.square().sum(-1, keepdim=True)).clamp_min(_eps(x.dtype))
        return torch.where(n > max_norm, x * (max_norm / n), x)

    # ---- distances --------------------------------------------------------

    def _from_gyro(self, w: Tensor) -> Tensor:
        eps = _eps(w.dtype)
        s = self._s(w.dtype)
        arg = s * w
        if self.curvature_sign < 0:
            return torch.atanh(arg.clamp(max=1 - eps)) / s
        return torch.atan(arg) / s

    def _gyro_norm(self, sq_diff: Tensor, nx2: Tensor, ny2: Tensor) -> Tensor:
        """‖(−x) ⊕_κ y‖ from squared quantities only (κ = K/4)."""
        kappa_param = self.kappa_tensor(sq_diff.dtype) / 4
        den = (1 + kappa_param * nx2) * (1 + kappa_param * ny2) - kappa_param * sq_diff
        # den vanishes exactly at antipodal spherical pairs, where the distance
        # has already saturated at π/√K; clamping keeps the gradient finite
        den = den.clamp_min(_eps(sq_diff.dtype))
        return safe_sqrt((sq_diff / den).clamp(0.0, _W_MAX**2))

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.project(x)
        y = self.project(y)
        sq_diff = (x - y).square().sum(-1)
        w = self._gyro_norm(sq_diff, x.square().sum(-1), y.square().sum(-1))
        return self._from_gyro(w)

    def dist_matrix(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.project(x)
        y = self.project(y)
        sq_diff = pairwise_sq_dist(x, y)
        nx2 = x.square().sum(-1).unsqueeze(-1)
        ny2 = y.square().sum(-1).unsqueeze(-2)
        return self._from_gyro(self._gyro_norm(sq_diff, nx2, ny2))

    # ---- R2 monitor quantities -------------------------------------------

    def radius(self, x: Tensor) -> Tensor:
        """Geodesic radius from the origin, ``= ‖logmap0(x)‖`` by construction."""
        return safe_sqrt(self.logmap0(x).square().sum(-1))

    def saturation_fraction(self, x: Tensor) -> Tensor:
        if self.curvature_sign < 0:
            return safe_sqrt(x.square().sum(-1)) / self.ball_radius
        return self.radius(x) / self.diameter

    def extra_repr(self) -> str:
        return f"dim={self.dim}, kappa={self.kappa:+.4g}"

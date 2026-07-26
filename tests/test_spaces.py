"""Geometry contract tests.

The load-bearing one is ``test_kappa_is_true_sectional_curvature``: it pins
``space.kappa`` against the closed-form law of cosines for a space of constant
curvature K, so the number P1 fits ``log N*`` against is verified against
geometry rather than against another implementation's convention. Everything
else here is hygiene that the audit's findings made non-optional.
"""

from __future__ import annotations

import math

import pytest
import torch

from csc.spaces import (
    ClampedEuclideanSpace,
    EuclideanSpace,
    NormalizedEuclideanSpace,
    ProductSpace,
    StereographicSpace,
)

CURVATURES = [-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
DTYPE = torch.float64


def _safe_radius(space: StereographicSpace, frac: float = 0.6) -> float:
    """A geodesic radius comfortably inside the space's usable domain."""
    if space.kappa > 0:
        return frac * space.diameter
    return frac * 6.0 / math.sqrt(abs(space.kappa))


# --------------------------------------------------------------------------
# 1. curvature convention
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kappa", CURVATURES)
@pytest.mark.parametrize("theta", [0.3, 1.0, 2.0, 3.0])
def test_kappa_is_true_sectional_curvature(kappa, theta):
    """Two geodesics leaving the origin at angle θ obey the law of cosines for K.

    For a space of constant sectional curvature K with s = √|K|, two points at
    geodesic radius r separated by angle θ at the origin lie at distance c:

        K < 0:  cosh(s·c) = cosh²(s·r) − sinh²(s·r)·cos θ
        K > 0:  cos(s·c)  = cos²(s·r)  + sin²(s·r)·cos θ

    ``expmap0`` is a radial isometry, so the angle between the tangent vectors
    *is* θ. If ``kappa`` were the κ-stereographic parameter rather than the
    sectional curvature (the parent program's convention), this test fails by
    a factor of 2 inside s — which is precisely the failure mode it exists to
    catch.
    """
    space = StereographicSpace(2, kappa)
    s = math.sqrt(abs(kappa))
    r = _safe_radius(space, 0.4)

    v1 = torch.tensor([r, 0.0], dtype=DTYPE)
    v2 = torch.tensor([r * math.cos(theta), r * math.sin(theta)], dtype=DTYPE)
    measured = space.dist(space.expmap0(v1), space.expmap0(v2)).item()

    if kappa < 0:
        cosh_sc = math.cosh(s * r) ** 2 - math.sinh(s * r) ** 2 * math.cos(theta)
        expected = math.acosh(cosh_sc) / s
    else:
        cos_sc = math.cos(s * r) ** 2 + math.sin(s * r) ** 2 * math.cos(theta)
        expected = math.acos(min(1.0, max(-1.0, cos_sc))) / s

    assert measured == pytest.approx(expected, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("kappa", CURVATURES)
def test_distance_normalization(kappa):
    """d(0, expmap0(v)) == ‖v‖ — a tangent norm is a geodesic radius."""
    space = StereographicSpace(3, kappa)
    origin = torch.zeros(3, dtype=DTYPE)
    for frac in (0.05, 0.2, 0.5, 0.9):
        r = _safe_radius(space, frac)
        v = torch.tensor([r, 0.0, 0.0], dtype=DTYPE)
        assert space.dist(origin, space.expmap0(v)).item() == pytest.approx(r, rel=1e-9)
        assert space.radius(space.expmap0(v)).item() == pytest.approx(r, rel=1e-9)


@pytest.mark.parametrize("kappa", [-1e-4, -1e-6, 1e-4, 1e-6])
def test_curvature_limit_is_euclidean(kappa):
    """K → 0 recovers EuclideanSpace on maps and distances."""
    curved = StereographicSpace(4, kappa)
    flat = EuclideanSpace(4)
    x = torch.randn(64, 4, dtype=DTYPE) * 0.5
    y = torch.randn(64, 4, dtype=DTYPE) * 0.5

    tol = 10 * abs(kappa)
    assert torch.allclose(curved.expmap0(x), flat.expmap0(x), atol=tol)
    assert torch.allclose(curved.logmap0(x), flat.logmap0(x), atol=tol)
    assert torch.allclose(curved.dist(x, y), flat.dist(x, y), atol=tol)
    assert torch.allclose(curved.dist_matrix(x, y), flat.dist_matrix(x, y), atol=tol)


@pytest.mark.parametrize("kappa", CURVATURES)
def test_ball_radius_and_diameter(kappa):
    space = StereographicSpace(2, kappa)
    if kappa < 0:
        assert space.ball_radius == pytest.approx(2.0 / math.sqrt(abs(kappa)))
        assert space.diameter == math.inf
    else:
        assert space.ball_radius == math.inf
        assert space.diameter == pytest.approx(math.pi / math.sqrt(kappa))


# --------------------------------------------------------------------------
# 2. map and metric axioms
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kappa", CURVATURES)
def test_logmap_inverts_expmap(kappa):
    space = StereographicSpace(5, kappa)
    r_max = _safe_radius(space, 0.8)
    v = torch.randn(128, 5, dtype=DTYPE)
    v = v / v.norm(dim=-1, keepdim=True) * torch.rand(128, 1, dtype=DTYPE) * r_max
    back = space.logmap0(space.expmap0(v))
    assert torch.allclose(back, v, atol=1e-9, rtol=1e-7)


@pytest.mark.parametrize("kappa", CURVATURES)
def test_metric_axioms(kappa):
    space = StereographicSpace(3, kappa)
    r_max = _safe_radius(space, 0.7)
    v = torch.randn(48, 3, dtype=DTYPE)
    pts = space.expmap0(v / v.norm(dim=-1, keepdim=True) * r_max * torch.rand(48, 1, dtype=DTYPE))

    d = space.dist_matrix(pts, pts)
    assert torch.allclose(d, d.T, atol=1e-10)
    assert torch.allclose(d.diagonal(), torch.zeros(48, dtype=DTYPE), atol=1e-7)
    assert (d >= -1e-12).all()
    # triangle inequality on every triple
    assert (d.unsqueeze(2) <= d.unsqueeze(1) + d.unsqueeze(0).transpose(1, 2) + 1e-9).all()


@pytest.mark.parametrize("kappa", CURVATURES)
def test_dist_matrix_matches_dist(kappa):
    space = StereographicSpace(3, kappa)
    r_max = _safe_radius(space, 0.7)
    x = space.expmap0(torch.randn(16, 3, dtype=DTYPE) * r_max / 3)
    y = space.expmap0(torch.randn(11, 3, dtype=DTYPE) * r_max / 3)
    pairwise = space.dist(x.unsqueeze(1), y.unsqueeze(0))
    assert torch.allclose(space.dist_matrix(x, y), pairwise, atol=1e-9)


@pytest.mark.parametrize("kappa", CURVATURES)
def test_project_is_idempotent_and_in_domain(kappa):
    space = StereographicSpace(2, kappa)
    x = torch.randn(256, 2, dtype=DTYPE) * 50  # deliberately far outside
    p = space.project(x)
    assert torch.allclose(space.project(p), p, atol=1e-12)
    if kappa < 0:
        assert (p.norm(dim=-1) < space.ball_radius).all()
    assert torch.isfinite(space.dist_matrix(p, p)).all()


# --------------------------------------------------------------------------
# 3. R2 monitor quantities
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kappa", CURVATURES)
def test_saturation_fraction_is_bounded_and_monotone(kappa):
    space = StereographicSpace(2, kappa)
    radii = torch.linspace(0.01, _safe_radius(space, 0.95), 40, dtype=DTYPE)
    v = torch.stack([radii, torch.zeros_like(radii)], dim=-1)
    frac = space.saturation_fraction(space.expmap0(v))
    assert (frac >= 0).all() and (frac <= 1.0).all()
    assert (frac.diff() > 0).all(), "saturation must increase with radius"


def test_hyperbolic_saturation_matches_analytic_form():
    """‖x‖/R_ball = tanh(√|K|·r/2), so the R2 band maps onto a known fraction."""
    space = StereographicSpace(2, -1.0)
    r = 3.0  # top of the target operating band √|K|·r ∈ [0.5, 3.0]
    v = torch.tensor([r, 0.0], dtype=DTYPE)
    frac = space.saturation_fraction(space.expmap0(v)).item()
    assert frac == pytest.approx(math.tanh(math.sqrt(1.0) * r / 2), rel=1e-9)
    assert frac < 0.99, "the top of the operating band must sit clear of the R2 gate"


# --------------------------------------------------------------------------
# 4. controls (R3)
# --------------------------------------------------------------------------


def test_clamped_euclidean_clips_at_max_dist():
    space = ClampedEuclideanSpace(3, max_dist=2.0)
    x = torch.zeros(5, 3, dtype=DTYPE)
    y = torch.zeros(5, 3, dtype=DTYPE)
    y[:, 0] = torch.tensor([0.5, 1.0, 2.0, 5.0, 50.0], dtype=DTYPE)
    d = space.dist(x, y)
    assert torch.allclose(d, torch.tensor([0.5, 1.0, 2.0, 2.0, 2.0], dtype=DTYPE))
    assert space.kappa == 0.0
    assert space.clipped_fraction(x, y).item() == pytest.approx(3 / 5)


def test_normalized_euclidean_is_unit_norm():
    space = NormalizedEuclideanSpace(4)
    v = torch.randn(32, 4, dtype=DTYPE) * 7
    x = space.expmap0(v)
    assert torch.allclose(x.norm(dim=-1), torch.ones(32, dtype=DTYPE), atol=1e-12)
    assert (space.dist_matrix(x, x) <= 2.0 + 1e-12).all()
    assert space.kappa == 0.0


# --------------------------------------------------------------------------
# 5. product manifold
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kappa", [-2.0, -1.0, 1.0])
def test_product_preserves_distance_normalization(kappa):
    space = ProductSpace.hyperbolic(n_factors=3, dim_per_factor=2, kappa=kappa)
    v = torch.randn(64, 6, dtype=DTYPE) * 0.2
    origin = torch.zeros(6, dtype=DTYPE)
    assert torch.allclose(space.dist(origin, space.expmap0(v)), v.norm(dim=-1), atol=1e-9)
    assert space.kappa == pytest.approx(kappa)


def test_product_dist_matrix_matches_dist():
    space = ProductSpace.hyperbolic(n_factors=2, dim_per_factor=2, kappa=-1.0)
    x = space.expmap0(torch.randn(9, 4, dtype=DTYPE) * 0.3)
    y = space.expmap0(torch.randn(7, 4, dtype=DTYPE) * 0.3)
    assert torch.allclose(
        space.dist_matrix(x, y), space.dist(x.unsqueeze(1), y.unsqueeze(0)), atol=1e-9
    )


# --------------------------------------------------------------------------
# 6. gradient safety (regression tests for the Phase-00b finding)
# --------------------------------------------------------------------------


ALL_SPACES = {
    "euclidean": lambda: EuclideanSpace(3),
    "hyperbolic": lambda: StereographicSpace(3, -1.0),
    "spherical": lambda: StereographicSpace(3, +1.0),
    "clamped": lambda: ClampedEuclideanSpace(3, max_dist=4.0),
    "normalized": lambda: NormalizedEuclideanSpace(3),
    "product": lambda: ProductSpace.hyperbolic(3, 1, -1.0),
}


@pytest.mark.parametrize("name", list(ALL_SPACES))
def test_gradients_are_finite_at_coincident_points(name):
    """The 00b finding: √· has an infinite derivative at 0, so a self-distance
    or any exactly-coincident pair produced NaN gradients through the whole
    batch, at every radius. Coincident prototypes are what a model out of room
    actually does, so this had to be fixed everywhere, not guarded against."""
    space = ALL_SPACES[name]()
    x = torch.randn(16, 3, dtype=torch.float32) * 0.3
    x[3] = x[2]  # an exactly coincident pair, in addition to the diagonal
    x = x.requires_grad_(True)
    space.dist_matrix(space.expmap0(x), space.expmap0(x)).sum().backward()
    assert torch.isfinite(x.grad).all(), f"{name}: non-finite gradient at coincidence"


@pytest.mark.parametrize("name", list(ALL_SPACES))
def test_gradients_are_finite_at_the_origin(name):
    """A zero tangent vector is the other place a norm's gradient blows up."""
    space = ALL_SPACES[name]()
    v = torch.zeros(4, 3, dtype=torch.float32, requires_grad=True)
    space.dist_matrix(space.expmap0(v), space.expmap0(v) + 0.5).sum().backward()
    assert torch.isfinite(v.grad).all(), f"{name}: non-finite gradient at the origin"


@pytest.mark.parametrize("kappa", CURVATURES)
def test_gradients_are_finite_across_and_beyond_the_operating_band(kappa):
    space = StereographicSpace(4, kappa)
    s = math.sqrt(abs(kappa))
    for scaled in (0.1, 0.5, 1.5, 3.0, 6.0):
        if kappa > 0 and scaled >= math.pi:
            continue
        v = torch.randn(32, 4, dtype=torch.float32)
        v = (v / v.norm(dim=-1, keepdim=True) * (scaled / s)).requires_grad_(True)
        pts = space.expmap0(v)
        space.dist_matrix(pts, pts).sum().backward()
        assert torch.isfinite(v.grad).all(), f"kappa={kappa} scaled_radius={scaled}"


def test_safe_sqrt_bias_is_negligible_in_band():
    """The ε² floor must not move a distance anyone will read."""
    space = StereographicSpace(2, -1.0)
    v = torch.tensor([[1.5, 0.0], [0.0, 1.5]], dtype=DTYPE)
    pts = space.expmap0(v)
    d = space.dist(pts[0], pts[1]).item()
    # closed form: law of cosines at r=1.5, theta=pi/2, K=-1
    expected = math.acosh(math.cosh(1.5) ** 2)
    assert d == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------------------
# 7. independent cross-check against geoopt
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kappa", CURVATURES)
def test_agrees_with_geoopt_under_the_stated_conversion(kappa):
    """geoopt at parameter K/4, distance halved, must reproduce our distance.

    This is the conversion documented in CLAUDE.md. Asserting it here means a
    future geoopt upgrade that changes conventions breaks a test rather than
    silently rescaling every curvature in the study.
    """
    geoopt_math = pytest.importorskip("geoopt.manifolds.stereographic.math")
    space = StereographicSpace(3, kappa)
    k = torch.tensor(kappa / 4, dtype=DTYPE)

    r_max = _safe_radius(space, 0.6)
    v = torch.randn(32, 3, dtype=DTYPE)
    v = v / v.norm(dim=-1, keepdim=True) * r_max * torch.rand(32, 1, dtype=DTYPE)
    w = torch.randn(32, 3, dtype=DTYPE)
    w = w / w.norm(dim=-1, keepdim=True) * r_max * torch.rand(32, 1, dtype=DTYPE)

    ours = space.dist(space.expmap0(v), space.expmap0(w))
    theirs = (
        geoopt_math.dist(
            geoopt_math.expmap0(v, k=k),
            geoopt_math.expmap0(w, k=k),
            k=k,
        )
        / 2
    )
    assert torch.allclose(ours, theirs, atol=1e-8, rtol=1e-7)

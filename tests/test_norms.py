"""Norm-space contracts — Study 2's geometry axis.

Study 1 established that no Riemannian curvature changes angular capacity,
because the tangent-space direction sphere is metric-independent. These spaces
change the *norm* instead, which is the only way to move the angular structure
while keeping curvature at zero. The tests pin the properties that make the
comparison meaningful: p=2 must be exactly the Euclidean control, curvature
must be exactly zero everywhere, the metric axioms must hold, and gradients
must survive coincident points (the 00b defect, which recurs for every new
distance function and is not inherited automatically).
"""

from __future__ import annotations

import math

import pytest
import torch

from csc.models.toy import ToySuperposition, parameter_count
from csc.spaces import EuclideanSpace, FinslerSpace, LpSpace

PS = [1.0, 1.5, 2.0, 3.0, float("inf")]
DTYPE = torch.float64


@pytest.mark.parametrize("p", PS)
def test_norm_spaces_are_exactly_flat(p):
    """Any capacity difference must be attributable to the norm, not curvature."""
    assert LpSpace(4, p).kappa == 0.0
    assert FinslerSpace(4, p).kappa == 0.0


def test_p2_is_exactly_euclidean():
    """p=2 is the control arm; if it drifts from Euclidean the comparison is void."""
    x = torch.randn(9, 5, dtype=DTYPE)
    y = torch.randn(7, 5, dtype=DTYPE)
    assert torch.allclose(LpSpace(5, 2.0).dist_matrix(x, y), EuclideanSpace(5).dist_matrix(x, y))
    assert torch.allclose(LpSpace(5, 2.0).dist(x[0], y[0]), EuclideanSpace(5).dist(x[0], y[0]))


@pytest.mark.parametrize("p", PS)
def test_metric_axioms(p):
    space = LpSpace(4, p)
    x = torch.randn(24, 4, dtype=DTYPE)
    d = space.dist_matrix(x, x)
    assert torch.allclose(d, d.T, atol=1e-9)
    assert (d.diagonal() < 1e-6).all()
    assert (d >= 0).all()
    # triangle inequality holds for every p >= 1
    assert (d.unsqueeze(2) <= d.unsqueeze(1) + d.unsqueeze(0).transpose(1, 2) + 1e-9).all()


@pytest.mark.parametrize("p", PS)
def test_dist_matrix_matches_dist(p):
    space = LpSpace(3, p)
    x = torch.randn(6, 3, dtype=DTYPE)
    y = torch.randn(4, 3, dtype=DTYPE)
    assert torch.allclose(space.dist_matrix(x, y), space.dist(x.unsqueeze(1), y.unsqueeze(0)))


@pytest.mark.parametrize("p", PS)
def test_gradients_finite_at_coincident_points(p):
    """The 00b defect does not inherit — every new distance function reintroduces it."""
    x = torch.randn(8, 4, dtype=torch.float32)
    x[2] = x[1]
    x = x.requires_grad_(True)
    LpSpace(4, p).dist_matrix(x, x).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_norm_ordering_is_correct():
    """‖v‖_inf <= ‖v‖_2 <= ‖v‖_1 for any vector — a sanity check on the implementations."""
    v = torch.randn(200, 6, dtype=DTYPE)
    zero = torch.zeros(6, dtype=DTYPE)
    inf_, two, one = (LpSpace(6, p).dist(v, zero) for p in (float("inf"), 2.0, 1.0))
    assert (inf_ <= two + 1e-9).all()
    assert (two <= one + 1e-9).all()


def test_linf_unit_ball_has_exponentially_many_extreme_points():
    """The pre-registered reason to expect a capacity gain: the l-inf ball is a
    hypercube with 2^d vertices, all mutually at l-inf distance 2 — exponentially
    many maximally separated directions with no radial blow-up."""
    for d in (2, 3, 4):
        verts = torch.tensor(
            [[1.0 if (i >> k) & 1 else -1.0 for k in range(d)] for i in range(2**d)],
            dtype=DTYPE,
        )
        assert verts.shape[0] == 2**d
        space = LpSpace(d, float("inf"))
        off = space.dist_matrix(verts, verts)[~torch.eye(2**d, dtype=torch.bool)]
        assert torch.allclose(off, torch.full_like(off, 2.0))
        # all vertices sit on the unit sphere of the norm
        assert torch.allclose(space.radius(verts), torch.ones(2**d, dtype=DTYPE))


def test_finsler_starts_isotropic_and_can_become_anisotropic():
    space = FinslerSpace(5, 2.0)
    assert space.anisotropy() == pytest.approx(1.0)
    with torch.no_grad():
        space.log_weights[0] += 2.0
    assert space.anisotropy() > 1.5


def test_finsler_declares_its_extra_parameters():
    """R1: a learnable geometry must not smuggle in free parameters unmatched."""
    learn, fixed = FinslerSpace(6, 2.0), FinslerSpace(6, 2.0, learnable=False)
    assert learn.n_extra_parameters == 6
    assert fixed.n_extra_parameters == 0
    lp = parameter_count(ToySuperposition(LpSpace(6, 2.0), 8))
    assert parameter_count(ToySuperposition(learn, 8)) == lp + 6
    assert parameter_count(ToySuperposition(fixed, 8)) == lp


def test_lp_spaces_are_matched_in_parameter_count():
    counts = {p: parameter_count(ToySuperposition(LpSpace(4, p), 10)) for p in PS}
    assert len(set(counts.values())) == 1, counts


def test_rejects_non_norms():
    with pytest.raises(ValueError):
        LpSpace(4, 0.5)


def test_infinite_norm_is_the_max_not_a_smooth_approximation():
    """A smooth-max would blur the extremal structure the hypothesis is about."""
    v = torch.tensor([[3.0, -1.0, 2.0]], dtype=DTYPE)
    zero = torch.zeros(3, dtype=DTYPE)
    assert LpSpace(3, float("inf")).dist(v, zero).item() == pytest.approx(3.0)
    assert LpSpace(3, 1.0).dist(v, zero).item() == pytest.approx(6.0)
    assert LpSpace(3, 2.0).dist(v, zero).item() == pytest.approx(math.sqrt(14))

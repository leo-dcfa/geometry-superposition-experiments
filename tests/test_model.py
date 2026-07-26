"""Model, readout and monitor contract tests.

The two that matter most are ``test_r1_bias_and_scale_are_structural`` (rule
R1 must not be defeatable by configuration) and
``test_parameter_count_is_matched_across_arms`` (a capacity comparison between
arms with different parameter counts is not a comparison).
"""

from __future__ import annotations

import pytest
import torch

from csc.interp.capacity import capacity_from_sweep, dead_unit_fraction, probe_metrics
from csc.layers.readout import HEADS, ResponseHead
from csc.models.toy import ToySuperposition, parameter_count
from csc.spaces import (
    ClampedEuclideanSpace,
    EuclideanSpace,
    NormalizedEuclideanSpace,
    StereographicSpace,
)
from csc.training.data import sample_batch
from csc.training.monitor import SaturationMonitor
from csc.training.toy_loop import ToyConfig, train_toy

N_FEATURES = 12
DIM = 2


def _arms():
    return {
        "euclidean": EuclideanSpace(DIM),
        "hyperbolic": StereographicSpace(DIM, -1.0),
        "spherical": StereographicSpace(DIM, +1.0),
        "clamped": ClampedEuclideanSpace(DIM, max_dist=6.0),
        "normalized": NormalizedEuclideanSpace(DIM),
    }


# --------------------------------------------------------------------------
# R1 — curvature is never the only free scalar
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", HEADS)
def test_r1_bias_and_scale_are_structural(kind):
    head = ResponseHead(N_FEATURES, kind=kind)
    names = dict(head.named_parameters())
    assert "bias" in names and names["bias"].shape == (N_FEATURES,)
    assert "scale" in names and names["scale"].shape == (N_FEATURES,)
    assert names["bias"].requires_grad and names["scale"].requires_grad


@pytest.mark.parametrize("kind", HEADS)
@pytest.mark.parametrize("arm", list(_arms()))
def test_r1_holds_in_every_arm_including_controls(arm, kind):
    model = ToySuperposition(_arms()[arm], N_FEATURES, head=kind)
    names = dict(model.named_parameters())
    assert "head.bias" in names and "head.scale" in names


def test_r1_bias_and_scale_receive_gradient():
    """A parameter that exists but never gets gradient would satisfy R1 only on paper."""
    model = ToySuperposition(StereographicSpace(DIM, -1.0), N_FEATURES, head="rbf")
    batch = sample_batch(64, N_FEATURES, sparsity=0.8)
    model(batch).square().mean().backward()
    assert model.head.bias.grad is not None and model.head.bias.grad.abs().sum() > 0
    assert model.head.scale.grad is not None and model.head.scale.grad.abs().sum() > 0


# --------------------------------------------------------------------------
# matched conditions across arms
# --------------------------------------------------------------------------


def test_parameter_count_is_matched_across_arms():
    counts = {
        name: parameter_count(ToySuperposition(space, N_FEATURES, head="rbf"))
        for name, space in _arms().items()
    }
    assert len(set(counts.values())) == 1, f"arms differ in parameter count: {counts}"


def test_fixed_curvature_adds_no_trainable_parameter():
    """A fixed-κ arm must not smuggle in a free scalar the flat arm lacks (R1)."""
    curved = ToySuperposition(StereographicSpace(DIM, -1.0), N_FEATURES)
    flat = ToySuperposition(EuclideanSpace(DIM), N_FEATURES)
    assert parameter_count(curved) == parameter_count(flat)


def test_learnable_curvature_adds_exactly_one_parameter():
    curved = ToySuperposition(StereographicSpace(DIM, -1.0, learnable=True), N_FEATURES)
    flat = ToySuperposition(EuclideanSpace(DIM), N_FEATURES)
    assert parameter_count(curved) == parameter_count(flat) + 1


def test_learnable_curvature_receives_gradient():
    space = StereographicSpace(DIM, -1.0, learnable=True)
    model = ToySuperposition(space, N_FEATURES, head="rbf")
    batch = sample_batch(64, N_FEATURES, sparsity=0.8)
    model(batch).square().mean().backward()
    assert space.log_magnitude.grad is not None
    assert space.log_magnitude.grad.abs().item() > 0, "curvature must be load-bearing"


# --------------------------------------------------------------------------
# geometry is load-bearing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", HEADS)
def test_curvature_changes_the_output(kind):
    """Same weights, different κ, different reconstruction — or κ is decorative."""
    torch.manual_seed(1)
    hyp = ToySuperposition(StereographicSpace(DIM, -2.0), N_FEATURES, head=kind)
    torch.manual_seed(1)
    sph = ToySuperposition(StereographicSpace(DIM, +2.0), N_FEATURES, head=kind)
    batch = sample_batch(32, N_FEATURES, sparsity=0.8)
    assert not torch.allclose(hyp(batch), sph(batch), atol=1e-4)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def test_probe_metrics_shapes_and_bounds():
    model = ToySuperposition(EuclideanSpace(DIM), N_FEATURES)
    m = probe_metrics(model, value=0.8, tol=0.2)
    assert m["n_features"] == N_FEATURES
    assert 0 <= m["features_recovered"] <= N_FEATURES
    assert m["recovered_mask"].shape == (N_FEATURES,)
    assert 0.0 <= m["recovery_rate"] <= 1.0


def test_dead_unit_fraction_detects_a_fully_dead_head():
    model = ToySuperposition(EuclideanSpace(DIM), N_FEATURES, head="affine")
    with torch.no_grad():
        model.head.bias.fill_(-1e3)  # force every ReLU shut
    batch = sample_batch(64, N_FEATURES, sparsity=0.8)
    assert dead_unit_fraction(model, batch) == 1.0


def test_capacity_from_sweep_stops_at_first_failure():
    # a non-monotone curve must not let a later lucky cell inflate N*
    assert capacity_from_sweep({4: 1.0, 8: 0.95, 16: 0.5, 32: 0.99}, threshold=0.9) == 8
    assert capacity_from_sweep({4: 0.1, 8: 0.99}, threshold=0.9) is None


# --------------------------------------------------------------------------
# R2 monitor
# --------------------------------------------------------------------------


def _record(monitor, sat, scaled=1.0, step=0):
    monitor.record(
        step,
        {
            "kappa": -1.0,
            "radius_median": scaled,
            "scaled_radius_median": scaled,
            "scaled_radius_quantiles": [scaled] * 5,
            "saturation_median": sat,
            "saturation_max": sat,
            "saturation_p95": sat,
        },
    )


def test_monitor_flags_uninterpretable_above_five_percent():
    monitor = SaturationMonitor(eval_every=10)
    for i in range(100):
        _record(monitor, sat=0.995 if i < 6 else 0.5, step=i)
    assert monitor.saturated_fraction == pytest.approx(0.06)
    assert monitor.uninterpretable
    assert monitor.summary()["verdict"] == "UNINTERPRETABLE"


def test_monitor_passes_at_exactly_five_percent():
    """R2 says '> 5%', so 5% exactly is clean. The boundary is pre-registered."""
    monitor = SaturationMonitor(eval_every=10)
    for i in range(100):
        _record(monitor, sat=0.995 if i < 5 else 0.5, step=i)
    assert monitor.saturated_fraction == pytest.approx(0.05)
    assert not monitor.uninterpretable
    assert monitor.summary()["verdict"] == "OK"


def test_band_occupancy_is_separate_from_the_verdict():
    """Being outside the operating band is a calibration miss, not an exclusion."""
    monitor = SaturationMonitor(eval_every=10)
    for i in range(20):
        _record(monitor, sat=0.2, scaled=0.01, step=i)  # far below the band
    assert monitor.band_occupancy() == 0.0
    assert not monitor.uninterpretable


# --------------------------------------------------------------------------
# end-to-end smoke
# --------------------------------------------------------------------------


@pytest.mark.parametrize("arm,kappa,kwargs", [
    ("euclidean", 0.0, {}),
    ("curved", -1.0, {}),
    ("curved", 1.0, {}),
    ("clamped", 0.0, {"max_dist": 6.0}),
    ("normalized", 0.0, {}),
])
def test_train_toy_runs_and_serializes(arm, kappa, kwargs, tmp_path):
    import json

    cfg = ToyConfig(
        arm=arm, kappa=kappa, space_kwargs=kwargs, dim=2,
        n_features=8, steps=60, eval_every=30, batch_size=64,
    )
    run = train_toy(cfg)
    assert run.summary["final_loss"] >= 0
    assert run.summary["saturation"]["verdict"] in ("OK", "UNINTERPRETABLE")
    # R5: the summary must survive json.dump with no special-casing
    (tmp_path / "run.json").write_text(json.dumps(run.summary))

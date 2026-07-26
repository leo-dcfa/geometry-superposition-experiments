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


def test_capacity_from_sweep_takes_the_first_contiguous_run():
    # a non-monotone curve must not let a later lucky cell inflate N*
    assert capacity_from_sweep({4: 1.0, 8: 0.95, 16: 0.5, 32: 0.99}, threshold=0.9) == 8
    # ...but a degenerate cell at the BOTTOM of the grid must not erase N*
    # above it. This returned None before the fix, for four of six sparsity
    # levels in the 00d control.
    assert capacity_from_sweep({2: 0.0, 4: 1.0, 8: 0.95, 16: 0.5}, threshold=0.9) == 8
    # nothing passes anywhere -> a miss, not a zero
    assert capacity_from_sweep({4: 0.1, 8: 0.2}, threshold=0.9) is None


def test_probe_metrics_are_stable_under_probe_batch_size():
    """Regression: the primary head normalizes by batch mean distance, so
    evaluating probes as their own tiny batch measures a different readout than
    the one that was trained. Measured at up to 25% error, with the direction
    depending on N — disqualifying, since N-dependence is what H-MAIN measures.
    """
    torch.manual_seed(0)
    model = ToySuperposition(EuclideanSpace(DIM), N_FEATURES, head="norm_affine")
    small = sample_batch(64, N_FEATURES, sparsity=0.9)
    large = sample_batch(4096, N_FEATURES, sparsity=0.9)
    a = probe_metrics(model, context=small)["probe_rel_error_mean"]
    b = probe_metrics(model, context=large)["probe_rel_error_mean"]
    assert abs(a - b) < 0.02, "probe metric must not depend on the context batch size"


def test_probe_context_matters_for_a_batch_normalized_head():
    """The bug itself, pinned: without context the reading genuinely differs."""
    torch.manual_seed(0)
    model = ToySuperposition(EuclideanSpace(DIM), N_FEATURES, head="norm_affine")
    ctx = sample_batch(2048, N_FEATURES, sparsity=0.9)
    assert probe_metrics(model, context=None)["probe_rel_error_mean"] != pytest.approx(
        probe_metrics(model, context=ctx)["probe_rel_error_mean"], abs=1e-6
    )


def test_softmax_head_is_batch_independent():
    """softmax normalizes across prototypes, not across the batch — so it is
    immune to this failure. Recorded because it is a genuine point in its
    favour as the second instrument (D9)."""
    torch.manual_seed(0)
    model = ToySuperposition(EuclideanSpace(DIM), N_FEATURES, head="softmax")
    ctx = sample_batch(2048, N_FEATURES, sparsity=0.9)
    assert probe_metrics(model, context=None)["probe_rel_error_mean"] == pytest.approx(
        probe_metrics(model, context=ctx)["probe_rel_error_mean"], abs=1e-6
    )


# --------------------------------------------------------------------------
# R2 monitor
# --------------------------------------------------------------------------


def _record(monitor, sat, scaled=1.0, step=0, kappa=-1.0):
    monitor.record(
        step,
        {
            "kappa": kappa,
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


def test_spherical_runs_are_never_excluded_by_saturation():
    """Decision D2: the R2 exclusion applies to K < 0 only.

    For K > 0 there is no clamp — arctan is well-conditioned to the antipode
    (00b) — and what the rule would exclude is a cloud filling the diameter,
    which is the mechanism H-MAIN predicts for positive curvature. Applied to
    both signs, R2 could only ever discard evidence against the hypothesis.
    """
    monitor = SaturationMonitor(eval_every=10)
    for i in range(100):
        _record(monitor, sat=0.999, step=i, kappa=+1.0)
    assert monitor.saturated_fraction == pytest.approx(1.0)
    assert not monitor.uninterpretable
    summary = monitor.summary()
    assert summary["verdict"] == "OK"
    assert summary["exclusion_applies"] is False
    # the same measurement is still surfaced, under its own name
    assert summary["diameter_filling_fraction"] == pytest.approx(1.0)


def test_hyperbolic_runs_are_still_excluded_by_saturation():
    """D2 narrows the gate; it must not disable it where it is justified."""
    monitor = SaturationMonitor(eval_every=10)
    for i in range(100):
        _record(monitor, sat=0.999 if i < 20 else 0.3, step=i, kappa=-1.0)
    assert monitor.uninterpretable
    summary = monitor.summary()
    assert summary["verdict"] == "UNINTERPRETABLE"
    assert summary["exclusion_applies"] is True
    assert summary["diameter_filling_fraction"] is None


def test_flat_arms_are_never_excluded():
    monitor = SaturationMonitor(eval_every=10)
    for i in range(100):
        _record(monitor, sat=1.0, step=i, kappa=0.0)
    assert not monitor.uninterpretable


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


# --------------------------------------------------------------------------
# D12-D15: the power-analysis amendments
# --------------------------------------------------------------------------


def test_fixed_radius_calibration_varies_x_with_kappa():
    """D12: the Phase-1 rule must let the independent variable actually vary.

    Targeting a constant sqrt|K|*r gives every arm the same predicted capacity,
    which would falsify P1 by construction. This is the regression test for
    that defect.
    """
    from csc.calibration.scale_rule import (
        init_gain_fixed_radius,
        phase1_radius,
        predicted_band_position,
    )

    dim, n = 8, 64
    radius = phase1_radius(dim, n)
    assert radius is not None
    xs = []
    for kappa in (-0.5, -1.0, -2.0, -4.0):  # ascending |kappa|
        gain = init_gain_fixed_radius(kappa, dim, n, radius)
        xs.append(predicted_band_position(kappa, dim, n, gain))
    # x must increase with |kappa| -- this is what makes P1 testable at all
    assert xs == sorted(xs), f"x must increase with |kappa|, got {xs}"
    assert xs[-1] / xs[0] > 2.0, f"x must span a usable range, got {xs}"
    # and every arm must sit inside the operating band
    assert all(0.5 <= x <= 3.0 for x in xs), xs


def test_unreachable_cells_are_excluded_rather_than_run_out_of_band():
    """A below-band run measures nothing and R2 cannot detect it, so such
    cells must be refused rather than silently included."""
    from csc.calibration.scale_rule import phase1_radius

    assert phase1_radius(4, 16) is None  # gain cap cannot reach the band floor
    assert phase1_radius(8, 64) is not None


def test_capacity_max_recovered_is_smooth_where_nstar_is_brittle():
    """D15: the primary metric must not vanish when the >=90% bar is missed."""
    from csc.interp.capacity import capacity_max_recovered

    recovered = {4: 4, 8: 7, 16: 11, 32: 10}
    assert capacity_max_recovered(recovered) == 11
    # the secondary metric legitimately reports a miss on the same data
    rates = {4: 1.0, 8: 0.875, 16: 0.69, 32: 0.31}
    assert capacity_from_sweep(rates, threshold=0.9) == 4


def test_jonckheere_detects_a_real_decreasing_trend():
    from csc.interp.trend import jonckheere_terpstra

    groups = [[10.0, 11, 12, 10.5, 11.5], [8.0, 9, 7.5, 8.5, 9.5], [5.0, 6, 4.5, 5.5, 6.5]]
    out = jonckheere_terpstra(groups, n_permutations=2000, seed=0)
    assert out["p_value"] < 0.01
    assert out["normalized"] > 0.9


def test_jonckheere_does_not_fire_on_noise():
    """The criterion F1.1 replaced fired 53% of the time on a true hypothesis."""
    import numpy as np

    from csc.interp.trend import jonckheere_terpstra

    rng = np.random.default_rng(0)
    fired = 0
    trials = 30
    for t in range(trials):
        groups = [list(rng.normal(0, 1, 5)) for _ in range(4)]
        if jonckheere_terpstra(groups, n_permutations=500, seed=t)["p_value"] < 0.05:
            fired += 1
    assert fired <= 4, f"false-positive rate too high: {fired}/{trials}"


def test_spearman_uses_midranks_for_ties():
    from csc.interp.trend import spearman_midrank

    out = spearman_midrank([1, 2, 2, 3], [1, 2, 2, 3])
    assert out["rho"] == pytest.approx(1.0)
    assert out["tie_policy"] == "midranks"
    assert out["tied_mass_x"] == pytest.approx(0.25)

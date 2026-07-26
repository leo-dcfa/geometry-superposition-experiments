"""Phase 00a deliverable: the encoder init-gain selection rule.

SPEC §4 asks 00a for "a scale-selection rule (function of κ, d, N), committed
before Phase 1". This is it. Given a cell, it returns the encoder init gain
that puts the trained point cloud's median √|K|·r on the operating band, so
that curvature is felt: neither huddled at the origin, where every geometry is
locally flat and the experiment measures nothing, nor pinned to the rim, where
the readout measures its own clamp (rule R2).

Fitted from 1680 runs across two sweeps
(`CSC_RESULTS/phase00/00a_scale_sweep*.json`) on

    log(median √|K|·r) = a + b·log(gain) + c·log|K| + d·log(dim) + e·log(N)

restricted to gains ≤ 16, above which the model does not merely overshoot the
band — it lands exactly on the saturation ceiling (14.5 for K<0, which is the
clamp horizon 00b measured independently; π for K>0, the antipode) and the R2
gate fires on 68–100% of runs. There is no smooth regime beyond 16 to fit.

**Two facts about this rule that matter more than its coefficients.**

1. It is head-specific, and not just quantitatively. For `norm_affine` the
   gain exponent is 0.367 — a usable knob. For `softmax` it is 0.096, i.e.
   nearly inert: pushing softmax's gain by 100× moves its radius by ~1.6×.
   The two readouts differ in *whether the calibration knob works at all*,
   which is a two-instrument finding in its own right and is why softmax's
   band occupancy peaks at 0.60 while norm_affine's reaches 0.90.

2. An earlier version of this rule, fitted through the `rbf` head, reported
   the gain as completely inert (exponent 7e-4). `rbf` was then disqualified
   by the 00c parity fixture for collapsing to the all-zero solution. A
   calibration measured through an unvalidated readout measures the readout —
   which is the argument for running 00c *before* 00a, not after as SPEC
   orders them.

Residual scatter is large (σ ≈ 0.56 in log space, R² ≈ 0.50): the settled
operating radius varies materially between seeds. The rule therefore targets
the band's geometric centre rather than an edge, so that a typical residual
excursion still lands inside.
"""

from __future__ import annotations

import math

from csc.training.monitor import OPERATING_BAND

# log(median √|K|·r) = intercept + log_gain·log(g) + log_abs_kappa·log|K|
#                      + log_dim·log(d) + log_n_features·log(N)
COEFFICIENTS = {
    "norm_affine": {
        "intercept": -2.3069,
        "log_gain": 0.3672,
        "log_abs_kappa": 0.1989,
        "log_dim": 0.4152,
        "log_n_features": 0.3320,
        "r_squared": 0.502,
        "residual_std_log": 0.557,
        "n_runs": 531,
    },
    "softmax": {
        "intercept": -2.4068,
        "log_gain": 0.0959,
        "log_abs_kappa": 0.5302,
        "log_dim": 0.5931,
        "log_n_features": 0.2558,
        "r_squared": 0.542,
        "residual_std_log": 0.524,
        "n_runs": 576,
    },
}

PRIMARY_HEAD = "norm_affine"

# Geometric centre of [0.5, 3.0]. Targeting the centre in log space leaves the
# most room on both sides for the (large) seed-to-seed residual.
BAND_TARGET = math.sqrt(OPERATING_BAND[0] * OPERATING_BAND[1])

# The knob's ceiling is set by the TRANSIENT, not by the settled state.
#
# Measured in the confirmation run: every run at prescribed gain ≤ 8.8 is
# in-band and R2-clean (n=63), while every R2 flag came from gain ≥ 10.4 — and
# 15 of those 18 flagged runs still *ended* in-band. A high init gain launches
# the cloud onto the boundary and lets it contract back, so the endpoint looks
# healthy while the early gradients were clamp-dominated.
#
# That transient is not a technicality to be waived. It is the parent
# program's path-dependence argument in a new place: a readout that saturates
# at birth can shape what gets learned before the model routes around it, and
# the endpoint cannot show you that it happened. So the cap is set below where
# the transient appears rather than R2 being given a burn-in exemption.
#
# (Past gain ~16 the model does not overshoot smoothly at all — it lands
# exactly on the saturation ceiling, 14.5 for K<0 and π for K>0, and the fit
# has no support there either.)
MAX_CALIBRATED_GAIN = 9.0
MIN_GAIN = 0.25


def predicted_band_position(kappa: float, dim: int, n_features: int, gain: float,
                            head: str = PRIMARY_HEAD) -> float:
    """Predicted trained median √|K|·r for a cell at a given init gain."""
    if kappa == 0.0:
        raise ValueError("band position is undefined for a flat arm (√|K|·r ≡ 0)")
    c = COEFFICIENTS[head]
    return math.exp(
        c["intercept"]
        + c["log_gain"] * math.log(gain)
        + c["log_abs_kappa"] * math.log(abs(kappa))
        + c["log_dim"] * math.log(dim)
        + c["log_n_features"] * math.log(n_features)
    )


def init_gain(kappa: float, dim: int, n_features: int, head: str = PRIMARY_HEAD) -> float:
    """The committed rule: encoder init gain for a curved cell.

    Inverts the fit for the gain that lands the predicted median √|K|·r on
    ``BAND_TARGET``, clipped to the calibrated range.
    """
    c = COEFFICIENTS[head]
    log_gain = (
        math.log(BAND_TARGET)
        - c["intercept"]
        - c["log_abs_kappa"] * math.log(abs(kappa))
        - c["log_dim"] * math.log(dim)
        - c["log_n_features"] * math.log(n_features)
    ) / c["log_gain"]
    return min(MAX_CALIBRATED_GAIN, max(MIN_GAIN, math.exp(log_gain)))


def matched_flat_gain(kappa: float, dim: int, n_features: int,
                      head: str = PRIMARY_HEAD) -> float:
    """Init gain for a flat arm (Euclidean and both R3 controls).

    A flat arm has no √|K|·r to place, so there is nothing to calibrate — but
    it must not be handed a *different* init from the curved arm it controls,
    or the comparison acquires a second difference. The rule is therefore to
    give a flat control the gain of the curved cell it is matched to, which
    starts both arms at the same raw geodesic radius.
    """
    return init_gain(kappa, dim, n_features, head)

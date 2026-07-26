"""Rule R2: the saturation gate.

A distance readout whose points are pinned against the boundary of the model
is measuring its own clamp, not geometry. The parent flagship's LM ran there,
which is the single finding that most damaged its κ-allocation result. R2
makes that condition an automatic, pre-registered exclusion rather than a
post-hoc judgement call.

Two different quantities are logged at every eval step, and neither
substitutes for the other:

- **saturation fraction** — fraction of the maximal representable radius
  (coordinate ball for K < 0, geodesic diameter for K > 0). This is the
  *numerical* condition: near 1 the atanh clamp is doing the work.
- **scaled radius √|K|·r** — the *interpretability* condition. Near 0 every
  geometry is locally flat and the experiment measures nothing; the target
  operating band is [0.5, 3.0], set in Phase 00a.

Verdict, verbatim from R2: a run spending > 5% of steps with median fraction
> 0.99 is UNINTERPRETABLE and is excluded from hypothesis tests while still
appearing in the ledger. Eval steps are the sampling of training steps here;
the sampling interval is recorded so the proportion is auditable.

**Decision D2 (sealed, VALIDATION.md): the exclusion applies only to K < 0.**

R2's justification is numerical — a readout pinned to the boundary reports its
own clamp. That is real for hyperbolic space, where ``atanh`` diverges at the
ball boundary and 00b located the horizon. It does not transfer to K > 0:
``arctan`` is well-conditioned all the way to the antipode and 00b measured
finite gradients there, so there is no arithmetic reason to distrust the run.

What the unmodified rule *would* have excluded for K > 0 is a cloud spread to
fill the diameter — which is exactly the mechanism H-MAIN predicts for
positive curvature when the space runs out of room. Applied to both signs, R2
would therefore discard positive-curvature evidence for displaying the effect
under test, while being unable to fire against the hyperbolic arm for the
analogous reason: a gate that can only ever exclude evidence against the
hypothesis. In the first 00a sweep it had already fired on 4 cells, all
spherical.

So for K > 0 the same quantity is recorded as ``diameter_filling`` — a
diagnostic that never excludes. Capacity results on spherical cells are
reported with and without the diameter-filling cells as a robustness panel,
which keeps the check without the bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SATURATION_THRESHOLD = 0.99  # median fraction above this counts a step as saturated
MAX_SATURATED_FRACTION = 0.05  # > 5% of steps saturated => UNINTERPRETABLE
OPERATING_BAND = (0.5, 3.0)  # target √|K|·r for the bulk of points (Phase 00a)


@dataclass
class SaturationMonitor:
    """Accumulates R2 records across a run and renders the automatic verdict."""

    eval_every: int
    records: list[dict] = field(default_factory=list)

    def record(self, step: int, report: dict) -> None:
        self.records.append({"step": step, **report})

    # ---- verdict ----------------------------------------------------------

    @property
    def n_steps(self) -> int:
        return len(self.records)

    @property
    def curvature_sign(self) -> int:
        """Sign of K for this run; 0 for a flat arm."""
        if not self.records:
            return 0
        kappa = self.records[-1].get("kappa", 0.0)
        return (kappa > 0) - (kappa < 0)

    @property
    def saturated_fraction(self) -> float:
        """Fraction of eval steps whose median saturation exceeds the threshold.

        Reported for every arm. For K > 0 this is the ``diameter_filling``
        diagnostic and carries no exclusion (D2).
        """
        if not self.records:
            return 0.0
        hits = sum(r["saturation_median"] > SATURATION_THRESHOLD for r in self.records)
        return hits / len(self.records)

    @property
    def exclusion_applies(self) -> bool:
        """D2: only a hyperbolic arm can be excluded by boundary saturation."""
        return self.curvature_sign < 0

    @property
    def uninterpretable(self) -> bool:
        return self.exclusion_applies and self.saturated_fraction > MAX_SATURATED_FRACTION

    def band_occupancy(self) -> float:
        """Fraction of eval steps whose median √|K|·r sits inside the band.

        Reported separately from the verdict: being outside the band is a
        *calibration* miss (Phase 00a's job to fix), not an automatic
        exclusion. Only boundary pinning triggers UNINTERPRETABLE.
        """
        if not self.records:
            return 0.0
        lo, hi = OPERATING_BAND
        inside = sum(lo <= r["scaled_radius_median"] <= hi for r in self.records)
        return inside / len(self.records)

    def summary(self) -> dict:
        if not self.records:
            return {"verdict": "NO_DATA", "n_eval_steps": 0}
        final = self.records[-1]
        return {
            "verdict": "UNINTERPRETABLE" if self.uninterpretable else "OK",
            "curvature_sign": self.curvature_sign,
            "exclusion_applies": self.exclusion_applies,
            # D2: for K > 0 the same measurement is a diagnostic, not a gate.
            # A spherical cloud filling the diameter is the predicted mechanism,
            # not an instrument failure, so it is surfaced under its own name.
            "diameter_filling_fraction": (
                self.saturated_fraction if self.curvature_sign > 0 else None
            ),
            "n_eval_steps": self.n_steps,
            "eval_every": self.eval_every,
            "saturated_step_fraction": self.saturated_fraction,
            "saturation_threshold": SATURATION_THRESHOLD,
            "max_saturated_fraction": MAX_SATURATED_FRACTION,
            "band_occupancy": self.band_occupancy(),
            "operating_band": list(OPERATING_BAND),
            "final_saturation_median": final["saturation_median"],
            "final_saturation_p95": final["saturation_p95"],
            "final_scaled_radius_median": final["scaled_radius_median"],
            "final_scaled_radius_quantiles": final["scaled_radius_quantiles"],
            "max_saturation_median_over_run": max(r["saturation_median"] for r in self.records),
        }

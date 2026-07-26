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
    def saturated_fraction(self) -> float:
        if not self.records:
            return 0.0
        hits = sum(r["saturation_median"] > SATURATION_THRESHOLD for r in self.records)
        return hits / len(self.records)

    @property
    def uninterpretable(self) -> bool:
        return self.saturated_fraction > MAX_SATURATED_FRACTION

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

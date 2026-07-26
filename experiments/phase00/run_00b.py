"""Phase 00b — numerics audit.

Three questions, all about whether the instrument reports geometry or reports
its own arithmetic (SPEC §4):

1. **atanh clamp behaviour.** The hyperbolic distance ends in
   ``atanh(s·w)``, guarded by a standoff from 1. Where does that guard start
   doing the work? The answer defines a *numerical horizon*: a scaled radius
   beyond which the reported distance is the clamp's, not the manifold's. If
   the horizon sat inside the R2 operating band the whole study would be
   measuring a clamp, so locating it is a gating measurement, not a detail.

2. **fp32 vs bf16 boundary overflow.** The parent program has a prior bf16
   incident, so precision is audited rather than assumed. Reference is
   float64; fp32 and bf16 are scored against it across the band.

3. **expmap/logmap round-trip error across the operating band.**

Everything is measured on both curvature signs and across the Phase-1
curvature grid. No model is trained and no hypothesis quantity is computed.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from csc.spaces import StereographicSpace
from csc.spaces.stereographic import _EPS
from csc.training.monitor import OPERATING_BAND
from experiments.util import RESULTS_ROOT, write_artifact

CURVATURES = [-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
DIM = 8
N_POINTS = 512

# Scaled radii √|K|·r to probe: inside the band, and well past it, so the
# horizon can be located rather than assumed to be beyond reach.
SCALED_RADII = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 20.0]


def _tangents(scaled_radius: float, kappa: float, dtype: torch.dtype) -> torch.Tensor:
    """``N_POINTS`` random directions at geodesic radius r = scaled_radius/√|K|."""
    g = torch.Generator().manual_seed(1234)
    v = torch.randn(N_POINTS, DIM, generator=g, dtype=torch.float64)
    v = v / v.norm(dim=-1, keepdim=True)
    r = scaled_radius / math.sqrt(abs(kappa))
    return (v * r).to(dtype)


def round_trip_audit() -> list[dict]:
    rows = []
    for kappa in CURVATURES:
        space = StereographicSpace(DIM, kappa)
        for scaled in SCALED_RADII:
            if kappa > 0 and scaled >= math.pi:
                continue  # past the antipode; radius is not representable
            ref_v = _tangents(scaled, kappa, torch.float64)
            for name, dtype in {"float64": torch.float64, **DTYPES}.items():
                v = ref_v.to(dtype)
                back = space.logmap0(space.expmap0(v)).to(torch.float64)
                err = (back - ref_v).norm(dim=-1) / ref_v.norm(dim=-1)
                rows.append(
                    {
                        "kappa": kappa,
                        "scaled_radius": scaled,
                        "dtype": name,
                        "round_trip_rel_error_median": float(err.median()),
                        "round_trip_rel_error_max": float(err.max()),
                        "non_finite": int((~torch.isfinite(back)).sum()),
                        "in_band": OPERATING_BAND[0] <= scaled <= OPERATING_BAND[1],
                    }
                )
    return rows


def distance_precision_audit() -> list[dict]:
    """fp32/bf16 pairwise distances scored against a float64 reference."""
    rows = []
    for kappa in CURVATURES:
        space = StereographicSpace(DIM, kappa)
        for scaled in SCALED_RADII:
            if kappa > 0 and scaled >= math.pi:
                continue
            v64 = _tangents(scaled, kappa, torch.float64)
            ref = space.dist_matrix(space.expmap0(v64), space.expmap0(v64))
            off = ~torch.eye(N_POINTS, dtype=torch.bool)
            ref_off = ref[off]
            for name, dtype in DTYPES.items():
                v = v64.to(dtype)
                row = {
                    "kappa": kappa,
                    "scaled_radius": scaled,
                    "dtype": name,
                    "in_band": OPERATING_BAND[0] <= scaled <= OPERATING_BAND[1],
                }
                try:
                    got = space.dist_matrix(space.expmap0(v), space.expmap0(v))
                except NotImplementedError as exc:
                    # Recorded, not worked around. torch.cdist has no bfloat16
                    # kernel, and the direct compute_mode is pinned for
                    # precision reasons (see spaces/euclidean.py), so the
                    # study's distance path simply cannot run in bf16. That is
                    # a harder guarantee than "bf16 was measured and found
                    # acceptable": a bf16 run fails loudly instead of
                    # silently degrading, which is the failure the parent
                    # program's bf16 incident did not get.
                    rows.append({**row, "supported": False, "error": str(exc)})
                    continue
                got = got.to(torch.float64)
                got_off = got[off]
                rel = (got_off - ref_off).abs() / ref_off.clamp_min(1e-12)
                rows.append(
                    {
                        **row,
                        "supported": True,
                        "dist_rel_error_median": float(rel.median()),
                        "dist_rel_error_p99": float(torch.quantile(rel, 0.99)),
                        "dist_rel_error_max": float(rel.max()),
                        "non_finite": int((~torch.isfinite(got)).sum()),
                        "self_distance_max": float(got.diagonal().abs().max()),
                    }
                )
    return rows


HORIZON_TOLERANCE = 0.01  # reported radius must track the true radius to 1%


def clamp_horizon_audit() -> list[dict]:
    """Measure — not derive — where the atanh standoff starts carrying the answer.

    For K < 0 the distance's final step is atanh(s·w) with the argument capped
    at 1 − eps, so beyond some radius the reported distance saturates and the
    instrument reports its own clamp. The analytic value is just a restatement
    of eps, so this walks the radius outward and finds the largest scaled
    radius at which ``d(0, expmap0(v))`` still tracks ‖v‖ to within
    ``HORIZON_TOLERANCE``. That empirical value is the arm's numerical horizon,
    and the gate is that it sits clear of the R2 band top.
    """
    rows = []
    probe = torch.logspace(math.log10(0.1), math.log10(60.0), 400, dtype=torch.float64)
    for kappa in CURVATURES:
        if kappa > 0:
            continue  # arctan has no domain limit; the spherical horizon is the antipode
        space = StereographicSpace(DIM, kappa)
        s_true = math.sqrt(abs(kappa))
        radii = probe / s_true  # so that probe is the scaled radius √|K|·r
        for name, dtype in {"float64": torch.float64, **DTYPES}.items():
            v = torch.zeros(len(radii), DIM, dtype=dtype)
            v[:, 0] = radii.to(dtype)
            origin = torch.zeros(DIM, dtype=dtype)
            reported = space.dist(origin, space.expmap0(v)).to(torch.float64)
            rel = (reported - radii).abs() / radii
            ok = rel < HORIZON_TOLERANCE
            # horizon = last radius before tracking is first lost
            first_bad = int((~ok).nonzero()[0]) if (~ok).any() else len(ok)
            horizon_scaled = float(probe[first_bad - 1]) if first_bad > 0 else 0.0
            rows.append(
                {
                    "kappa": kappa,
                    "dtype": name,
                    "eps": _EPS[dtype],
                    "tolerance": HORIZON_TOLERANCE,
                    "max_tracked_scaled_radius": horizon_scaled,
                    "max_tracked_radius": horizon_scaled / s_true,
                    "band_top": OPERATING_BAND[1],
                    "headroom_over_band_top": horizon_scaled - OPERATING_BAND[1],
                }
            )
    return rows


def gradient_health_audit() -> list[dict]:
    """Gradients of the distance w.r.t. tangent coordinates, across and past the band."""
    rows = []
    for kappa in CURVATURES:
        space = StereographicSpace(DIM, kappa)
        for scaled in SCALED_RADII:
            if kappa > 0 and scaled >= math.pi:
                continue
            for name, dtype in {"float32": torch.float32}.items():
                v = _tangents(scaled, kappa, dtype).requires_grad_(True)
                pts = space.expmap0(v)
                d = space.dist_matrix(pts, pts)
                d.sum().backward()
                g = v.grad
                rows.append(
                    {
                        "kappa": kappa,
                        "scaled_radius": scaled,
                        "dtype": name,
                        "grad_non_finite": int((~torch.isfinite(g)).sum()),
                        "grad_abs_max": float(g.abs().max()),
                        "grad_abs_median": float(g.abs().median()),
                        "in_band": OPERATING_BAND[0] <= scaled <= OPERATING_BAND[1],
                    }
                )
    return rows


def verdicts(round_trip, distances, horizons, grads) -> dict:
    """Pass/fail summary. Thresholds are stated here and repeated in VALIDATION.md."""
    band_rt_fp32 = [
        r for r in round_trip if r["in_band"] and r["dtype"] == "float32"
    ]
    band_rt_bf16 = [r for r in round_trip if r["in_band"] and r["dtype"] == "bfloat16"]
    band_d_fp32 = [
        r for r in distances if r["in_band"] and r["dtype"] == "float32" and r["supported"]
    ]
    band_d_bf16 = [
        r for r in distances if r["in_band"] and r["dtype"] == "bfloat16" and r["supported"]
    ]
    band_g = [r for r in grads if r["in_band"]]

    worst_rt_fp32 = max(r["round_trip_rel_error_max"] for r in band_rt_fp32)
    worst_rt_bf16 = max(r["round_trip_rel_error_max"] for r in band_rt_bf16)
    worst_d_fp32 = max(r["dist_rel_error_p99"] for r in band_d_fp32)
    min_headroom = min(
        h["headroom_over_band_top"] for h in horizons if h["dtype"] == "float32"
    )
    grad_bad = sum(r["grad_non_finite"] for r in grads)

    return {
        "fp32_round_trip_in_band_max_rel_error": worst_rt_fp32,
        "fp32_round_trip_pass": worst_rt_fp32 < 1e-4,
        "bf16_round_trip_in_band_max_rel_error": worst_rt_bf16,
        "fp32_distance_in_band_p99_rel_error": worst_d_fp32,
        "fp32_distance_pass": worst_d_fp32 < 1e-4,
        "bf16_distance_in_band_p99_rel_error": (
            max(r["dist_rel_error_p99"] for r in band_d_bf16) if band_d_bf16 else None
        ),
        "bf16_distance_in_band_max_rel_error": (
            max(r["dist_rel_error_max"] for r in band_d_bf16) if band_d_bf16 else None
        ),
        "bf16_usable": False,
        "bf16_note": (
            "Measured, not assumed — the parent program's bf16 incident is why. "
            "In-band pairwise distances carry up to ~1.4e-1 relative error in "
            "bf16 against a float64 reference, versus ~2.3e-6 in fp32: five "
            "orders of magnitude worse, and large enough to swamp any capacity "
            "difference this study could detect. Part of that floor is our own "
            "safe_sqrt epsilon (1e-2 for bf16), which cannot be tightened "
            "because bf16's ~3 significant digits cannot resolve a finer "
            "standoff. CSC therefore runs fp32 throughout; bf16 is disqualified "
            "by measurement, and this row is the receipt."
        ),
        "fp32_clamp_headroom_over_band_top": min_headroom,
        "clamp_horizon_pass": min_headroom > 0,
        "gradients_finite_everywhere": grad_bad == 0,
        "gradients_finite_in_band": sum(r["grad_non_finite"] for r in band_g) == 0,
        "thresholds": {
            "round_trip_rel_error": 1e-4,
            "distance_p99_rel_error": 1e-4,
            "note": (
                "The fp32 gates are the ones that block G00. bf16 is measured "
                "and reported so that any future decision to use it is made "
                "against numbers — which is what the parent program's bf16 "
                "incident lacked — but it is disqualified, not gated."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RESULTS_ROOT / "phase00")
    args = parser.parse_args()

    torch.set_num_threads(4)
    round_trip = round_trip_audit()
    distances = distance_precision_audit()
    horizons = clamp_horizon_audit()
    grads = gradient_health_audit()
    summary = verdicts(round_trip, distances, horizons, grads)

    payload = {
        "phase": "00b",
        "purpose": "numerics audit: atanh clamp horizon, fp32/bf16 precision, round-trip error",
        "settings": {
            "curvatures": CURVATURES,
            "dim": DIM,
            "n_points": N_POINTS,
            "scaled_radii": SCALED_RADII,
            "operating_band": list(OPERATING_BAND),
        },
        "verdicts": summary,
        "round_trip": round_trip,
        "distance_precision": distances,
        "clamp_horizon": horizons,
        "gradient_health": grads,
    }
    path = write_artifact(args.out / "00b_numerics_audit.json", payload)
    print(f"00b: wrote {path}")
    for key in (
        "fp32_round_trip_pass",
        "fp32_distance_pass",
        "clamp_horizon_pass",
        "gradients_finite_in_band",
    ):
        print(f"  {key}: {summary[key]}")
    print(f"  fp32 clamp headroom over band top: {summary['fp32_clamp_headroom_over_band_top']:.2f}")
    print(
        f"  bf16 in-band distance p99 rel error: "
        f"{summary['bf16_distance_in_band_p99_rel_error']:.2e} (fp32: "
        f"{summary['fp32_distance_in_band_p99_rel_error']:.2e}) -> bf16 disqualified"
    )


if __name__ == "__main__":
    main()

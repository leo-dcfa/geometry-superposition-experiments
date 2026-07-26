"""CSC-2 E8 — the mechanism, measured directly: angular resolution vs radial capacity.

Every result in CSC-1 and CSC-2 follows from one property of curved space, and
this measures it in isolation — no model, no readout, no training, no data.
Just the geometry.

**The question.** Hyperbolic space demonstrably has exponentially more volume
(E7 confirms the classic result). Why does that never become capacity?

**The answer.** Curvature does not distort angles — the Poincaré model is
conformal, so local angles are exactly Euclidean. What it changes is the
*mapping from angle to distance*. For two points at geodesic radius r
separated by angle θ at the origin, the law of cosines gives:

    euclidean    c = 2r·sin(θ/2)                        — LINEAR in the angle
    hyperbolic   c ≈ 2r + (2/√|K|)·log sin(θ/2)         — LOGARITHMIC in it

Differentiating the hyperbolic form: ∂c/∂θ = (1/√|K|)·cot(θ/2), which is
**independent of r**. In flat space ∂c/∂θ = r·cos(θ/2) — moving outward buys
angular resolution. In hyperbolic space it buys none. The exponentially many
new points that appear at large radius are all at nearly the same distance from
each other, so a distance-based readout cannot rank them.

**Why this explains the whole project.** Superposition capacity is angular: it
is set by how many near-orthogonal directions fit, and interference is
directional overlap. Hyperbolic space spends its budget on a radial coordinate
that superposition cannot address, while degrading the angular resolution that
it can. Trees are the opposite case — a hierarchy's structure IS radial
(generality near the origin, specificity at the rim) and its exponential node
growth matches the exponential volume growth exactly.

So curvature is a dial trading **angular resolution** against **radial
capacity**, and the sign should be chosen to match where the data's structure
lives. CSC-1 asked an angular question and reached for the negative end.

Reported: the normalized distance-vs-angle profile at several radii, the
angular sensitivity ∂c/∂θ, and the circumference of a geodesic circle — the
"how much room is there to spread in" quantity, which peaks and then *shrinks*
for K > 0.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from csc.spaces import EuclideanSpace, StereographicSpace
from experiments.util import RESULTS_ROOT, write_artifact

RADII = [0.5, 1.0, 2.0, 3.0]
ANGLES_DEG = [1, 5, 15, 30, 45, 60, 90, 120, 150, 180]
CURVATURES = [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
DTYPE = torch.float64


def _space(kappa: float, dim: int = 2):
    return EuclideanSpace(dim) if kappa == 0.0 else StereographicSpace(dim, kappa)


def distance_profile(kappa: float, r: float) -> dict:
    """Distance vs angle at fixed radius, raw and normalized by the θ=180° value."""
    space = _space(kappa)
    out = {}
    for deg in ANGLES_DEG:
        th = math.radians(deg)
        v1 = torch.tensor([r, 0.0], dtype=DTYPE)
        v2 = torch.tensor([r * math.cos(th), r * math.sin(th)], dtype=DTYPE)
        out[deg] = float(space.dist(space.expmap0(v1), space.expmap0(v2)))
    span = out[180] or 1.0
    return {
        "raw": out,
        "normalized": {k: v / span for k, v in out.items()},
        "max_distance": span,
    }


def angular_sensitivity(kappa: float, r: float, deg: float = 60.0, eps: float = 1e-4) -> float:
    """dc/dtheta — how much distance a unit of angle buys at this radius."""
    space = _space(kappa)

    def dist_at(theta: float) -> float:
        v1 = torch.tensor([r, 0.0], dtype=DTYPE)
        v2 = torch.tensor([r * math.cos(theta), r * math.sin(theta)], dtype=DTYPE)
        return float(space.dist(space.expmap0(v1), space.expmap0(v2)))

    th = math.radians(deg)
    return (dist_at(th + eps) - dist_at(th - eps)) / (2 * eps)


def circle_circumference(kappa: float, r: float) -> float:
    """Room available at radius r: 2πr flat, 2π sinh(√|K|r)/√|K| curved, sin for K>0."""
    if kappa == 0.0:
        return 2 * math.pi * r
    s = math.sqrt(abs(kappa))
    return 2 * math.pi * (math.sinh(s * r) if kappa < 0 else math.sin(s * r)) / s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RESULTS_ROOT / "csc2" / "e8")
    args = parser.parse_args()

    profiles, sensitivity, room = {}, {}, {}
    for kappa in CURVATURES:
        for r in RADII:
            if kappa > 0 and math.sqrt(kappa) * r >= math.pi:
                continue  # past the antipode; not representable
            profiles[f"K={kappa:+g}|r={r}"] = distance_profile(kappa, r)
            sensitivity[f"K={kappa:+g}|r={r}"] = angular_sensitivity(kappa, r)
            room[f"K={kappa:+g}|r={r}"] = circle_circumference(kappa, r)

    # Does moving outward buy angular resolution? Flat: yes, linearly in r.
    # Hyperbolic: no — dc/dtheta is asymptotically r-independent.
    scaling = {}
    for kappa in CURVATURES:
        vals = [
            (r, sensitivity[f"K={kappa:+g}|r={r}"])
            for r in RADII
            if f"K={kappa:+g}|r={r}" in sensitivity
        ]
        if len(vals) >= 2:
            scaling[f"K={kappa:+g}"] = {
                "sensitivity_by_radius": dict(vals),
                "ratio_far_over_near": vals[-1][1] / vals[0][1],
            }

    payload = {
        "phase": "csc2-e8",
        "purpose": "the mechanism in isolation: curvature trades angular resolution for radial capacity",
        "no_model": "pure geometry — no training, readout, or data involved",
        "closed_forms": {
            "euclidean": "c = 2r sin(theta/2); dc/dtheta = r cos(theta/2)  -> grows with r",
            "hyperbolic_large_r": (
                "c ~ 2r + (2/sqrt|K|) log sin(theta/2); "
                "dc/dtheta = (1/sqrt|K|) cot(theta/2)  -> INDEPENDENT of r"
            ),
        },
        "settings": {"radii": RADII, "angles_deg": ANGLES_DEG, "curvatures": CURVATURES},
        "angular_sensitivity_scaling": scaling,
        "circle_circumference": room,
        "angular_sensitivity_at_60deg": sensitivity,
        "distance_vs_angle_profiles": profiles,
    }
    path = write_artifact(args.out / "e8_angular_resolution.json", payload)
    print(f"E8: wrote {path}")
    print("\nnormalized distance vs angle (fraction of the theta=180 distance), r=2:")
    hdr = "  " + "".join(f"{d:>8}" for d in ANGLES_DEG)
    print(f"  {'K':>16}" + hdr)
    for kappa in CURVATURES:
        key = f"K={kappa:+g}|r=2.0"
        if key in profiles:
            row = "".join(f"{v:>8.3f}" for v in profiles[key]["normalized"].values())
            print(f"  {kappa:>16.1f}  " + row)
    print("\ndoes moving outward buy angular resolution? (dc/dtheta at r=3 / at r=0.5)")
    for k, v in scaling.items():
        print(f"  {k:>10}: {v['ratio_far_over_near']:.2f}x")
    print("\ncircumference of a geodesic circle (room to spread in):")
    for r in RADII:
        vals = [(k, room.get(f"K={k:+g}|r={r}")) for k in (-1.0, 0.0, 1.0)]
        s = "  ".join(f"K={k:+g}: {v:8.3f}" for k, v in vals if v is not None)
        print(f"  r={r}:  {s}")


if __name__ == "__main__":
    main()

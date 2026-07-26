"""Phase 00a — input/init scale calibration.

Purpose (SPEC §4): guarantee curvature is *felt*. If trained point clouds
huddle at the origin every geometry is locally flat and the experiment
measures nothing; if they pin to the rim the readout is measuring its own
clamp (rule R2). The target operating band is √|K|·r ∈ [0.5, 3.0] for the
bulk of points.

Deliverable: a scale-selection rule as a function of (K, d, N), committed
before Phase 1 opens.

**Measured result that reshaped this runner: the encoder init gain is an inert
knob.** Sweeping it over a 16x range moves the trained median √|K|·r by a
fitted exponent of ~7e-4, and dropping it from the model changes R² in the
fourth decimal. The optimizer selects its own operating radius and an init
rescale is absorbed within the first few hundred steps. A calibration rule
phrased as "set the init gain to s(K, d, N)" would therefore have been a rule
that does nothing — committed, obeyed, and inert.

What the operating radius *is* controlled by is (|K|, d, N), so the rule this
runner commits is the honest one: predict the trained band position from the
cell, and invert it into the region of (K, d, N) whose predicted position
lands inside the band. Phase-1 primary cells are then chosen from inside that
region, which needs no architectural change and, importantly, no
regularization — the parent program's dose-response work established that
taxing geometry distorts exactly what these studies try to measure, so
weight decay is not an acceptable calibration knob here.

Three parts, all committed to one artifact:
  1. the grid sweep (init gain x K x shape x seed);
  2. a convergence probe — long runs recording the radius trajectory, so that
     "the trained cloud sits in the band" is a claim about the model and not
     about the step count at which we stopped looking;
  3. the fit and the admissible region.

No hypothesis-relevant quantity is read here: recovery, capacity and
interference are not computed, and the N-dependence of the radius is recorded
but deliberately not interpreted — its shape is Phase-1 evidence and reading
it now would breach the G00 blind.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from csc.training.monitor import OPERATING_BAND
from csc.training.toy_loop import ToyConfig, train_toy
from experiments.util import RESULTS_ROOT, parallel_map, write_artifact

CURVATURES = [-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
INIT_SCALES = [0.25, 0.5, 1.0, 2.0, 4.0]
SHAPES = [(2, 16), (2, 64), (4, 32), (8, 64)]  # (dim, n_features)
# Only the heads that survived the 00c parity fixture. The first 00a run used
# `rbf`, which 00c then disqualified as basin-fragile, so its calibration
# described a readout the study will not use and had to be redone.
HEADS_CALIBRATED = ["norm_affine", "softmax"]
SEEDS = [0, 1, 2]
# 10k rather than 2k: the convergence probe showed a spherical N=16 cell whose
# radius was still doubling after step 2000. Fitting the calibration rule on a
# not-yet-settled operating point would bake the step count into the rule.
STEPS = 10_000

# Geometric centre of the operating band — the target the rule solves for.
BAND_TARGET = math.sqrt(OPERATING_BAND[0] * OPERATING_BAND[1])


def _cell(spec: tuple) -> dict:
    kappa, scale, dim, n_features, seed, head = spec
    cfg = ToyConfig(
        arm="curved",
        kappa=kappa,
        dim=dim,
        n_features=n_features,
        encoder_init_scale=scale,
        steps=STEPS,
        eval_every=200,
        batch_size=256,
        seed=seed,
        head=head,
    )
    run = train_toy(cfg)
    sat = run.summary["saturation"]
    return {
        "kappa": kappa,
        "head": head,
        "init_scale": scale,
        "dim": dim,
        "n_features": n_features,
        "seed": seed,
        "scaled_radius_median": sat["final_scaled_radius_median"],
        "scaled_radius_quantiles": sat["final_scaled_radius_quantiles"],
        "saturation_median": sat["final_saturation_median"],
        "saturation_p95": sat["final_saturation_p95"],
        "max_saturation_median_over_run": sat["max_saturation_median_over_run"],
        "band_occupancy": sat["band_occupancy"],
        "verdict": sat["verdict"],
        "final_loss": run.summary["final_loss"],
        "dead_unit_fraction": run.summary["dead_unit_fraction"],
    }


def _ols(rows: list[dict], terms: list[str]) -> dict:
    """Least squares of log(scaled radius) on the named log-terms, plus R²."""
    key = {
        "log_init_scale": lambda c: math.log(c["init_scale"]),
        "log_abs_kappa": lambda c: math.log(abs(c["kappa"])),
        "log_dim": lambda c: math.log(c["dim"]),
        "log_n_features": lambda c: math.log(c["n_features"]),
    }
    design = np.array([[1.0] + [key[t](c) for t in terms] for c in rows])
    target = np.array([math.log(c["scaled_radius_median"]) for c in rows])
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ coef
    ss_tot = float(((target - target.mean()) ** 2).sum())
    return {
        "coefficients": dict(zip(["intercept"] + terms, coef.tolist())),
        "r_squared": 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else float("nan"),
        "residual_std_log": float(resid.std()),
        "n": len(rows),
    }


TERMS = ["log_abs_kappa", "log_dim", "log_n_features"]


def fit_scale_rule(cells: list[dict]) -> dict:
    """Fit the trained band position, and measure whether the init gain matters.

    Cells flagged UNINTERPRETABLE are excluded from the fit: their radius is a
    clamp artefact, not a measurement of where the optimizer put the cloud.
    They are counted in the artifact so the exclusion is visible.

    Hyperbolic and spherical arms are fitted separately. Pooling them is not a
    modelling nicety — a sphere's finite diameter caps its radius, so the two
    signs genuinely obey different laws, and the pooled fit is worse than
    either (reported).
    """
    usable = [c for c in cells if c["verdict"] == "OK" and c["scaled_radius_median"] > 0]
    hyperbolic = [c for c in usable if c["kappa"] < 0]
    spherical = [c for c in usable if c["kappa"] > 0]

    with_init = _ols(usable, ["log_init_scale"] + TERMS)
    without_init = _ols(usable, TERMS)
    init_coef = with_init["coefficients"]["log_init_scale"]

    return {
        "init_gain_inertness": {
            "log_init_scale_coefficient": init_coef,
            "r_squared_with_init_gain": with_init["r_squared"],
            "r_squared_without_init_gain": without_init["r_squared"],
            "delta_r_squared": with_init["r_squared"] - without_init["r_squared"],
            "init_gain_range_swept": [min(INIT_SCALES), max(INIT_SCALES)],
            "predicted_radius_ratio_across_swept_range": float(
                (max(INIT_SCALES) / min(INIT_SCALES)) ** init_coef
            ),
            "verdict": (
                "INERT — the init gain does not select the operating scale; the "
                "committed rule selects admissible (K, d, N) cells instead"
                if abs(init_coef) < 0.05
                else "ACTIVE — the init gain is a usable calibration knob"
            ),
        },
        "pooled": without_init,
        "hyperbolic": _ols(hyperbolic, TERMS) if hyperbolic else None,
        "spherical": _ols(spherical, TERMS) if spherical else None,
        "band_target": BAND_TARGET,
        "operating_band": list(OPERATING_BAND),
        "formula": (
            "log(median sqrt|K|*r) = intercept + c*log|K| + d*log(dim) + e*log(N), "
            "fitted separately per curvature sign. A cell is admissible when its "
            "predicted value lies inside the operating band."
        ),
        "n_cells_fitted": len(usable),
        "n_cells_excluded_uninterpretable": len(cells) - len(usable),
    }


def predict_band_position(rule: dict, kappa: float, dim: int, n_features: int) -> float:
    """Predicted trained median √|K|·r for a cell, from the committed rule."""
    fit = rule["hyperbolic" if kappa < 0 else "spherical"]
    c = fit["coefficients"]
    return math.exp(
        c["intercept"]
        + c["log_abs_kappa"] * math.log(abs(kappa))
        + c["log_dim"] * math.log(dim)
        + c["log_n_features"] * math.log(n_features)
    )


def admissible_region(rule: dict, curvatures, dims, feature_counts) -> list[dict]:
    """Which (K, d, N) cells the rule predicts will land inside the band."""
    lo, hi = OPERATING_BAND
    out = []
    for kappa in curvatures:
        for dim in dims:
            for n in feature_counts:
                pred = predict_band_position(rule, kappa, dim, n)
                out.append(
                    {
                        "kappa": kappa,
                        "dim": dim,
                        "n_features": n,
                        "predicted_scaled_radius": pred,
                        "in_band": bool(lo <= pred <= hi),
                    }
                )
    return out


def _trajectory(spec: tuple) -> dict:
    """Long run recording the radius trajectory (module-level so it pickles).

    Eval points are spaced at exactly ``STEPS``, so the probe answers the
    question that matters directly: is the value the grid sweep would have
    recorded the same as the value the run eventually settles on?
    """
    kappa, dim, n_features, seed, steps = spec
    cfg = ToyConfig(
        arm="curved",
        kappa=kappa,
        dim=dim,
        n_features=n_features,
        steps=steps,
        eval_every=STEPS,
        batch_size=256,
        seed=seed,
        head=HEADS_CALIBRATED[0],
    )
    run = train_toy(cfg)
    traj = [
        {"step": r["step"], "scaled_radius_median": r["scaled_radius_median"]}
        for r in run.summary["saturation_records"]
    ]
    at_sweep_length = next(t["scaled_radius_median"] for t in traj if t["step"] == STEPS)
    tail = [t["scaled_radius_median"] for t in traj[len(traj) // 2 :]]
    settled_value = float(np.mean(tail))
    return {
        "kappa": kappa,
        "dim": dim,
        "n_features": n_features,
        "seed": seed,
        "steps": steps,
        "sweep_length": STEPS,
        "trajectory": traj,
        "value_at_sweep_length": at_sweep_length,
        "settled_value": settled_value,
        "relative_gap": float(abs(at_sweep_length - settled_value) / max(settled_value, 1e-9)),
        "final_half_drift_ratio": float(
            (max(tail) - min(tail)) / settled_value if settled_value > 0 else float("nan")
        ),
        "converged_by_sweep_length": bool(
            abs(at_sweep_length - settled_value) / max(settled_value, 1e-9) < 0.15
        ),
    }


# Convergence probe: the corners of the sweep, where drift would show first.
# Spherical small-N is included because that is the cell that failed the probe
# at the original 2k sweep length.
PROBE_STEPS = 50_000
PROBE_CELLS = [
    (kappa, dim, n_features, 0, PROBE_STEPS)
    for kappa in (-4.0, -1.0, 0.5, 1.0, 2.0)
    for dim, n_features in ((2, 16), (2, 64))
]

# The Phase-1 grid the admissible region is evaluated over (P1's curvatures,
# the d* sweep's dimensions, and a capacity-style feature ladder).
PHASE1_CURVATURES = [-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
PHASE1_DIMS = [2, 3, 4, 6, 8]
PHASE1_FEATURES = [8, 16, 32, 64, 128, 256]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RESULTS_ROOT / "csc1" / "phase00")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--init-scales",
        type=float,
        nargs="+",
        default=None,
        help="override the init-gain grid (used to extend the sweep upward)",
    )
    parser.add_argument("--name", default="00a_scale_sweep", help="artifact basename")
    args = parser.parse_args()

    global INIT_SCALES
    if args.init_scales:
        INIT_SCALES = list(args.init_scales)

    specs = [
        (kappa, scale, dim, n_features, seed, head)
        for kappa in CURVATURES
        for scale in INIT_SCALES
        for dim, n_features in SHAPES
        for seed in SEEDS
        for head in HEADS_CALIBRATED
    ]
    print(f"00a: grid sweep, {len(specs)} runs x {STEPS} steps")
    cells = parallel_map(_cell, specs, max_workers=args.workers)

    print(f"00a: convergence probe, {len(PROBE_CELLS)} runs x {PROBE_STEPS} steps")
    probes = parallel_map(_trajectory, PROBE_CELLS, max_workers=args.workers)

    rules = {h: fit_scale_rule([c for c in cells if c["head"] == h]) for h in HEADS_CALIBRATED}
    primary = HEADS_CALIBRATED[0]
    rule = rules[primary]
    region = admissible_region(rule, PHASE1_CURVATURES, PHASE1_DIMS, PHASE1_FEATURES)
    lo, hi = OPERATING_BAND
    measured_in_band = {
        h: sum(
            lo <= c["scaled_radius_median"] <= hi for c in cells if c["head"] == h
        )
        / max(1, sum(c["head"] == h for c in cells))
        for h in HEADS_CALIBRATED
    }

    payload = {
        "phase": "00a",
        "purpose": "calibrate the operating scale so trained clouds occupy the R2 band",
        "grid": {
            "curvatures": CURVATURES,
            "init_scales": INIT_SCALES,
            "shapes": [list(s) for s in SHAPES],
            "seeds": SEEDS,
            "steps": STEPS,
            "heads": HEADS_CALIBRATED,
        },
        "headline": {
            "primary_head": primary,
            "measured_in_band_fraction_by_head": measured_in_band,
            "n_uninterpretable_cells": sum(c["verdict"] != "OK" for c in cells),
            "n_cells": len(cells),
            "convergence_all_cells_settled": all(p["converged_by_sweep_length"] for p in probes),
        },
        "rule": rule,
        "rule_by_head": rules,
        "convergence_probe": probes,
        "admissible_region": region,
        "cells": cells,
    }
    path = write_artifact(args.out / f"{args.name}.json", payload)
    print(f"00a: wrote {path}")
    print(f"00a: primary head {primary}; init gain -> {rule['init_gain_inertness']['verdict']}")
    for h, r in rules.items():
        print(
            f"00a: [{h}] fit R^2 hyperbolic {r['hyperbolic']['r_squared']:.3f} / "
            f"spherical {r['spherical']['r_squared']:.3f}; "
            f"in-band {measured_in_band[h]:.3f}"
        )
    print(f"00a: admissible Phase-1 cells {sum(r['in_band'] for r in region)}/{len(region)}")


if __name__ == "__main__":
    main()

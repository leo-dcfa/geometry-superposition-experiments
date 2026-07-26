"""Phase 00e — power analysis. Can Phase 1 detect the effect it is looking for?

Not in SPEC. Added because a null result is only informative if the design
could have found a positive one, and nothing so far establishes that.

Two inputs, kept deliberately separate because their reliability differs:

- **Noise: measured.** σ(log N*) across seeds, taken from the 00d control.
  That arm is Euclidean, so using it does not read anything about curvature.
- **Effect: theoretical.** The ratio of hyperbolic to Euclidean ball volume at
  matched geodesic radius R and matched absolute separation, which is the
  first-order proxy for "how many more features fit". In d dimensions with
  x = √|K|·R:

      V_H/V_E  =  d · ∫₀ˣ sinh(t)^{d-1} dt / x^d

  This is a heuristic, not a prediction: the model is not doing ideal packing
  and its minimum separation is learned rather than imposed. It is used only
  to locate the regime where an effect is large enough to see.

Reporting the minimum detectable effect separately from the predicted effect
means the noise result stands even if the effect model is wrong.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy import integrate, stats

from csc.interp.capacity import capacity_from_sweep
from csc.training.monitor import OPERATING_BAND
from experiments.util import RESULTS_ROOT, write_artifact

# P1's curvature grid and the d* sweep's dimensions.
P1_CURVATURES = [-4.0, -2.0, -1.0, -0.5]
DIMS = [2, 3, 4, 6, 8]
SEED_COUNTS = [3, 5, 10, 20]
ALPHA = 0.05


def volume_ratio(dim: int, x: float) -> float:
    """V_hyperbolic / V_euclidean for balls of matched geodesic radius.

    x = √|K|·R. Returns 1.0 at x → 0 (flat limit), grows with x and with dim.
    """
    if x <= 0:
        return 1.0
    num, _ = integrate.quad(lambda t: math.sinh(t) ** (dim - 1), 0.0, x)
    return dim * num / x**dim


def spherical_volume_ratio(dim: int, x: float) -> float:
    """Same, for K > 0. Defined for x < π; < 1, i.e. positive curvature costs room."""
    if x <= 0:
        return 1.0
    if x >= math.pi:
        return float("nan")
    num, _ = integrate.quad(lambda t: math.sin(t) ** (dim - 1), 0.0, x)
    return dim * num / x**dim


def seed_sigma_from_control(path: Path) -> dict:
    """σ(log N*) across seeds, measured on the Euclidean control (blind-safe)."""
    data = json.loads(path.read_text())
    cells = data["cells"]
    ns = sorted({c["n_features"] for c in cells})
    sparsities = sorted({c["sparsity"] for c in cells})
    seeds = sorted({c["seed"] for c in cells})

    per_sparsity = {}
    for s in sparsities:
        stars = []
        for sd in seeds:
            rec = {
                n: next(
                    (
                        c["recovery_rate"]
                        for c in cells
                        if c["n_features"] == n and c["sparsity"] == s and c["seed"] == sd
                    ),
                    0.0,
                )
                for n in ns
            }
            star = capacity_from_sweep(rec, 0.9)
            if star:
                stars.append(math.log(star))
        per_sparsity[str(s)] = {
            "n_seeds_defined": len(stars),
            "sigma_log_nstar": st.stdev(stars) if len(stars) >= 3 else None,
        }
    sigmas = [v["sigma_log_nstar"] for v in per_sparsity.values() if v["sigma_log_nstar"]]
    return {
        "by_sparsity": per_sparsity,
        "pooled_mean": st.mean(sigmas),
        "pooled_median": st.median(sigmas),
        "worst": max(sigmas),
        "note": (
            "Measured on the Euclidean arm of the 00d control. Includes the "
            "discretization noise of N* itself, since N* is read off a discrete "
            "feature-count grid — which is part of the real measurement error."
        ),
    }


def mde(sigma: float, n_seeds: int, power: float = 0.8) -> float:
    """Minimum detectable effect in log N* units, two-arm two-sided test."""
    se = sigma * math.sqrt(2.0 / n_seeds)
    return (stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(power)) * se


def power_of(effect_log: float, sigma: float, n_seeds: int) -> float:
    se = sigma * math.sqrt(2.0 / n_seeds)
    z = effect_log / se
    crit = stats.norm.ppf(1 - ALPHA / 2)
    return float(stats.norm.sf(crit - z) + stats.norm.cdf(-crit - z))


def monotonicity_risk(effects: list[float], sigma: float, n_seeds: int) -> dict:
    """F1.1 says ANY adjacent-pair inversion kills P1. How likely is that by chance?"""
    se = sigma * math.sqrt(2.0 / n_seeds)
    gaps = [b - a for a, b in pairwise(effects)]
    p_each = [float(stats.norm.cdf(-abs(g) / (se * math.sqrt(2)))) for g in gaps]
    p_none = float(np.prod([1 - p for p in p_each]))
    return {
        "adjacent_true_gaps_log": gaps,
        "p_inversion_per_pair": p_each,
        "p_no_inversion_anywhere": p_none,
        "p_at_least_one_inversion": 1 - p_none,
    }


def required_seeds(effects: list[float], sigma: float, max_n: int = 400) -> dict:
    """How many seeds does each test actually need?

    Two targets: F1.1's all-pairs-monotone criterion at a 10% false-rejection
    rate, and the two-arm contrast between the extreme curvatures — which is
    the same scientific claim asked in a form that does not require every
    adjacent pair to order correctly.
    """
    out = {}
    for n in range(3, max_n + 1):
        if (
            "f1_1_all_pairs_monotone" not in out
            and monotonicity_risk(effects, sigma, n)["p_at_least_one_inversion"] <= 0.10
        ):
            out["f1_1_all_pairs_monotone"] = n
        if (
            "extreme_pair_contrast" not in out
            and power_of(effects[-1] - effects[0], sigma, n) >= 0.80
        ):
            out["extreme_pair_contrast"] = n
        if len(out) == 2:
            break
    out.setdefault("f1_1_all_pairs_monotone", f">{max_n}")
    out.setdefault("extreme_pair_contrast", f">{max_n}")
    out["note"] = (
        "F1.1 as written ('any adjacent-pair inversion kills P1') is the "
        "expensive criterion: it asks every adjacent pair to order correctly, "
        "including pairs whose true separation is a fraction of the noise. A "
        "trend test over the whole grid answers the same scientific question "
        "for a fraction of the seeds."
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RESULTS_ROOT / "csc1" / "phase00")
    args = parser.parse_args()

    sigma_report = seed_sigma_from_control(args.out / "00d_positive_control.json")
    sigma = sigma_report["pooled_mean"]
    sigma_worst = sigma_report["worst"]

    band_lo, band_hi = OPERATING_BAND
    band_mid = math.sqrt(band_lo * band_hi)

    # --- effect size across dimension and band position -------------------
    effect_table = {
        f"d={d}": {
            f"x={x}": {
                "volume_ratio_H_over_E": volume_ratio(d, x),
                "log_effect": math.log(volume_ratio(d, x)),
            }
            for x in (band_lo, band_mid, 2.0, band_hi)
        }
        for d in DIMS
    }

    mde_table = {
        f"n_seeds={n}": {
            "mde_log": mde(sigma, n),
            "mde_ratio": math.exp(mde(sigma, n)),
            "mde_log_worst_sigma": mde(sigma_worst, n),
            "mde_ratio_worst_sigma": math.exp(mde(sigma_worst, n)),
        }
        for n in SEED_COUNTS
    }

    power_table = {
        f"d={d}|x={x}": {
            "effect_ratio": volume_ratio(d, x),
            **{
                f"power_n={n}": power_of(math.log(volume_ratio(d, x)), sigma, n)
                for n in SEED_COUNTS
            },
        }
        for d in DIMS
        for x in (band_mid, 2.0, band_hi)
    }

    # --- the design question: does the calibration rule erase P1's effect? --
    #
    # scale_rule.init_gain targets a CONSTANT √|K|·r for every cell. If every
    # curved arm sits at the same x, every arm has the same volume ratio, so
    # N* is predicted to be identical across κ — and P1's monotone prediction
    # becomes structurally untestable by the calibration procedure itself.
    scenario_matched_x = {
        f"kappa={k}": {"x": band_mid, "volume_ratio": volume_ratio(2, band_mid)}
        for k in P1_CURVATURES
    }
    # The alternative: hold the geodesic radius R fixed across arms, so that
    # x = √|K|·R varies with κ exactly as P1 intends.
    r_fixed = band_hi / math.sqrt(max(abs(k) for k in P1_CURVATURES))
    scenario_fixed_radius = {
        f"kappa={k}": {
            "x": math.sqrt(abs(k)) * r_fixed,
            "in_band": bool(band_lo <= math.sqrt(abs(k)) * r_fixed <= band_hi),
            "volume_ratio": volume_ratio(2, math.sqrt(abs(k)) * r_fixed),
        }
        for k in P1_CURVATURES
    }
    effects_fixed_r = [
        math.log(v["volume_ratio"]) for v in scenario_fixed_radius.values()
    ][::-1]  # ascending in κ (least to most negative reversed)

    payload = {
        "phase": "00e",
        "purpose": "power analysis: is Phase 1 able to detect the effect it tests for?",
        "blind": "noise measured on the Euclidean control only; effect sizes are theoretical",
        "noise": sigma_report,
        "minimum_detectable_effect": mde_table,
        "effect_size_theory": effect_table,
        "power": power_table,
        "design_check_calibration_vs_P1": {
            "problem": (
                "scale_rule.init_gain targets a CONSTANT sqrt|K|*r for every "
                "cell. Under it, every hyperbolic arm sits at the same x and "
                "therefore has the same predicted volume ratio, so N* is "
                "constant in kappa and P1's monotone prediction cannot be "
                "tested — the calibration procedure would erase the effect."
            ),
            "scenario_matched_x_current_rule": scenario_matched_x,
            "scenario_fixed_geodesic_radius": scenario_fixed_radius,
            "fixed_radius_value": r_fixed,
            "monotonicity_under_fixed_radius": monotonicity_risk(
                effects_fixed_r, sigma, 5
            ),
        },
        "required_seeds": required_seeds(effects_fixed_r, sigma),
        "f1_1_risk_note": (
            "F1.1 kills P1 on ANY adjacent-pair inversion. With a discrete N "
            "grid and the measured sigma, that criterion has a substantial "
            "false-rejection rate whenever adjacent true effects are close; "
            "see monotonicity_under_fixed_radius."
        ),
    }
    path = write_artifact(args.out / "00e_power_analysis.json", payload)
    print(f"00e: wrote {path}")
    print(f"00e: sigma(log N*) pooled {sigma:.3f}, worst {sigma_worst:.3f}")
    for n in SEED_COUNTS:
        m = mde_table[f"n_seeds={n}"]
        print(f"  n={n:>2} seeds -> MDE {m['mde_ratio']:.2f}x  (worst-case sigma {m['mde_ratio_worst_sigma']:.2f}x)")
    print("  effect (volume ratio H/E) by dim at band positions:")
    for d in DIMS:
        row = "  ".join(
            f"x={x}: {volume_ratio(d, x):.2f}x" for x in (band_mid, 2.0, band_hi)
        )
        print(f"    d={d}: {row}")
    mono = payload["design_check_calibration_vs_P1"]["monotonicity_under_fixed_radius"]
    print(f"  P(spurious inversion, fixed-R design, 5 seeds) = {mono['p_at_least_one_inversion']:.3f}")


if __name__ == "__main__":
    main()

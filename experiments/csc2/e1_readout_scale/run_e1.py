"""CSC-2 E1 — is CSC's null about geometry, or about a relative-scale readout?

CSC measured no capacity advantage for negative curvature, on **flat,
exchangeable features**, under two readouts that both normalize by a batch
statistic. The leading explanation was that hyperbolic geometry inflates the
batch-mean pairwise distance by the same exponential property that creates its
volume, so extra room arrives with a matching loss of relative resolution and
the two cancel.

That explanation makes a sharp prediction: a readout whose length scale is
**absolute** should not suffer the cancellation. This runs CSC's own factorial
unchanged except for the readout axis.

Design follows CSC's R4 — the nuisance variable is crossed with the independent
variable, never derived from it. Init gain is set explicitly, so gain and
curvature vary independently and the 0.971 collinearity that made CSC's Phase 1
unreadable cannot recur.

Pre-registered readings, committed before the run:

- **If absolute-scale heads show a hyperbolic advantage where relative-scale
  heads do not**, CSC's null is readout-specific and must be restated as such;
  the cancellation account is supported, and CSC-2's hierarchy framing needs
  rewriting because the capacity claim would be partly rescued.
- **If neither head family shows an advantage**, CSC's null is robust to the
  readout, the cancellation account is wrong or incomplete, and the angular-vs-
  radial account stands as the better explanation.
- Either way the trend must be sign-consistent across gain levels to count.
"""

from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

from csc.interp.trend import spearman_midrank
from csc.training.toy_loop import ToyConfig, train_toy
from experiments.util import RESULTS_ROOT, parallel_map, write_artifact

GAINS = [1.5, 3.0, 6.0, 9.0]
CURVATURES = [-4.0, -2.0, -1.0, -0.5, 1.0, 2.0]
DIMS = [4, 8]
N_FEATURES = 64
SEEDS = list(range(10))
STEPS = 10_000
# heads chosen by E0; these are the candidates, filtered at analysis time
RELATIVE = ["norm_affine", "softmax"]
ABSOLUTE = ["abs_rbf", "affine"]


def _cell(spec: tuple) -> dict:
    dim, gain, head, arm, kappa, seed = spec
    cfg = ToyConfig(
        arm=arm, kappa=kappa, dim=dim, n_features=N_FEATURES, sparsity=0.9,
        head=head, encoder_init_scale=gain, steps=STEPS, eval_every=1_000,
        batch_size=256, seed=seed,
    )
    run = train_toy(cfg)
    return {
        "dim": dim, "gain": gain, "head": head,
        "head_family": "absolute" if head in ABSOLUTE else "relative",
        "arm": arm, "kappa": kappa, "seed": seed,
        "arm_label": "euclidean" if arm == "euclidean" else f"curved(K={kappa:+g})",
        "features_recovered": run.summary["probes"]["features_recovered"],
        "interference_mean": run.summary["probes"]["interference_mean"],
        "min_pairwise_distance": run.summary["prototype_geometry"]["min_pairwise_distance"],
        "scaled_radius_median": run.summary["saturation"]["final_scaled_radius_median"],
        "final_loss": run.summary["final_loss"],
        "dead_unit_fraction": run.summary["dead_unit_fraction"],
        "saturation_verdict": run.summary["saturation"]["verdict"],
    }


def analyse(cells: list[dict]) -> dict:
    ok = [c for c in cells if c["saturation_verdict"] == "OK"]
    out = {}
    for head in RELATIVE + ABSOLUTE:
        for dim in DIMS:
            sub = [c for c in ok if c["head"] == head and c["dim"] == dim]
            if not sub:
                continue
            per_gain = {}
            for g in GAINS:
                curved = [c for c in sub if c["gain"] == g and c["arm"] == "curved"]
                euc = [c["features_recovered"] for c in sub
                       if c["gain"] == g and c["arm"] == "euclidean"]
                present = [
                    (k, [c["features_recovered"] for c in curved if c["kappa"] == k])
                    for k in sorted(CURVATURES)
                ]
                present = [(k, v) for k, v in present if len(v) >= 3]
                if len(present) < 3:
                    continue
                fk, fv = [], []
                for k, v in present:
                    fk.extend([k] * len(v)); fv.extend(v)
                hyp = [v for k, v in present if k < 0]
                best_hyp = max((st.mean(v) for v in hyp), default=None)
                per_gain[str(g)] = {
                    "rho_kappa_vs_capacity": spearman_midrank(fk, fv)["rho"],
                    "euclidean_capacity": st.mean(euc) if euc else None,
                    "best_hyperbolic_capacity": best_hyp,
                    "hyperbolic_advantage": (
                        best_hyp / st.mean(euc) if euc and best_hyp else None
                    ),
                    "capacity_by_kappa": {str(k): st.mean(v) for k, v in present},
                }
            rhos = [v["rho_kappa_vs_capacity"] for v in per_gain.values()]
            advs = [v["hyperbolic_advantage"] for v in per_gain.values()
                    if v["hyperbolic_advantage"]]
            out[f"{head}|d{dim}"] = {
                "head_family": "absolute" if head in ABSOLUTE else "relative",
                "by_gain": per_gain,
                "rho_by_gain": rhos,
                "sign_consistent": bool(
                    rhos and (all(r > 0 for r in rhos) or all(r < 0 for r in rhos))
                ),
                "max_hyperbolic_advantage": max(advs) if advs else None,
                "mean_hyperbolic_advantage": st.mean(advs) if advs else None,
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RESULTS_ROOT / "csc2" / "e1")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--heads", nargs="+", default=None)
    args = parser.parse_args()

    heads = args.heads or (RELATIVE + ABSOLUTE)
    arms = [("euclidean", 0.0)] + [("curved", k) for k in CURVATURES]
    specs = [
        (dim, gain, head, arm, kappa, seed)
        for dim in DIMS for gain in GAINS for head in heads
        for arm, kappa in arms for seed in SEEDS
    ]
    print(f"E1: {len(specs)} runs x {STEPS} steps")
    cells = parallel_map(_cell, specs, max_workers=args.workers)
    analysis = analyse(cells)

    rescued = [
        k for k, v in analysis.items()
        if v["head_family"] == "absolute" and v["sign_consistent"]
        and (v["max_hyperbolic_advantage"] or 0) > 1.2
    ]
    payload = {
        "phase": "csc2-e1",
        "question": "is CSC's null readout-specific?",
        "settings": {
            "gains": GAINS, "curvatures": CURVATURES, "dims": DIMS,
            "n_features": N_FEATURES, "seeds": len(SEEDS), "steps": STEPS,
            "relative_heads": RELATIVE, "absolute_heads": ABSOLUTE,
        },
        "verdict": (
            "NULL IS READOUT-SPECIFIC" if rescued else "NULL IS ROBUST TO READOUT"
        ),
        "cells_showing_advantage": rescued,
        "analysis": analysis,
        "cells": cells,
    }
    path = write_artifact(args.out / "e1_readout_scale.json", payload)
    print(f"E1: wrote {path}")
    print(f"E1: verdict {payload['verdict']}")
    for key, v in analysis.items():
        rhos = ", ".join(f"{r:+.2f}" for r in v["rho_by_gain"])
        adv = v["max_hyperbolic_advantage"]
        print(f"  {key:22} [{v['head_family']:8}] rho by gain [{rhos}] "
              f"consistent={v['sign_consistent']} max_adv="
              f"{adv:.3f}" if adv else f"  {key} [{v['head_family']}] no advantage computed")


if __name__ == "__main__":
    main()

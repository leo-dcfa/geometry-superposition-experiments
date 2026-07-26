"""CSC-2 E3 — the main event: does the advantage grow with hierarchy depth?

H2-MAIN: negative curvature buys representational room in proportion to the
hierarchical structure of the features, and the advantage vanishes for
exchangeable ones.

The design's most useful property is that **depth 0 is CSC's setting**. The
same generator produces both, so the anchor H2-a (no advantage without
hierarchy) is an internal control measured in the same sweep by the same code,
not a comparison across studies. If depth 0 shows an advantage here, something
is wrong with this run rather than with CSC.

Gated on E4 (the external positive control) and informed by E0 (which readouts
are fair) and E1 (whether the readout family matters). Gain is crossed with
curvature, never derived from it (R4).

Primary: A(κ, D), capacity of a curvature-κ arm relative to the matched
Euclidean arm at depth D. Prediction: ∂A/∂D > 0 for κ < 0, with A(κ, 0) = 1.

Secondary (P2-2, the mechanism): in hyperbolic arms at D > 0, prototype
geodesic radius should correlate with hierarchy level — general concepts near
the origin, specific ones near the rim. This is the coordinate CSC's features
gave no meaning to, and which its models left unused. If capacity improves
while radius stays uncorrelated with level, the stated mechanism is wrong even
if the headline holds.
"""

from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

import torch

from csc.interp.trend import spearman_midrank
from csc.training.hierarchy import FeatureTree
from csc.training.toy_loop import ToyConfig, train_toy
from experiments.util import RESULTS_ROOT, parallel_map, write_artifact

DEPTHS = [0, 2, 3, 4, 5]
BRANCHING = 2
CURVATURES = [-4.0, -2.0, -1.0, -0.5, 1.0]
DIMS = [4, 8]
GAINS = [1.5, 4.0, 9.0]
SEEDS = list(range(8))
STEPS = 10_000
HEADS = ["norm_affine", "abs_rbf"]
# depth 0 uses this many flat features so the anchor is comparable in size to
# the shallow trees (depth 2 binary = 7 nodes, depth 5 = 63)
FLAT_N_FEATURES = 31


def _radius_level_correlation(model, tree: FeatureTree | None) -> float | None:
    """P2-2: does prototype radius track hierarchy level?"""
    if tree is None:
        return None
    with torch.no_grad():
        protos = model.readout.prototypes()
        radii = model.space.radius(protos).cpu().tolist()
    return spearman_midrank(tree.level, radii)["rho"]


def _cell(spec: tuple) -> dict:
    dim, depth, gain, head, arm, kappa, seed = spec
    cfg = ToyConfig(
        arm=arm, kappa=kappa, dim=dim,
        n_features=FLAT_N_FEATURES, hierarchy_depth=depth, hierarchy_branching=BRANCHING,
        sparsity=0.9, head=head, encoder_init_scale=gain,
        steps=STEPS, eval_every=1_000, batch_size=256, seed=seed,
    )
    run = train_toy(cfg)
    tree = FeatureTree(depth, BRANCHING) if depth > 0 else None
    return {
        "dim": dim, "depth": depth, "gain": gain, "head": head,
        "arm": arm, "kappa": kappa, "seed": seed,
        "arm_label": "euclidean" if arm == "euclidean" else f"curved(K={kappa:+g})",
        "n_features": run.summary["n_features_effective"],
        "features_recovered": run.summary["probes"]["features_recovered"],
        "recovery_rate": run.summary["probes"]["recovery_rate"],
        "interference_mean": run.summary["probes"]["interference_mean"],
        "radius_level_rho": _radius_level_correlation(run.model, tree),
        "min_pairwise_distance": run.summary["prototype_geometry"]["min_pairwise_distance"],
        "final_loss": run.summary["final_loss"],
        "saturation_verdict": run.summary["saturation"]["verdict"],
    }


def analyse(cells: list[dict]) -> dict:
    ok = [c for c in cells if c["saturation_verdict"] == "OK"]
    out = {}
    for head in HEADS:
        for dim in DIMS:
            by_depth = {}
            for depth in DEPTHS:
                sub = [c for c in ok if c["head"] == head and c["dim"] == dim
                       and c["depth"] == depth]
                if not sub:
                    continue
                euc = [c["recovery_rate"] for c in sub if c["arm"] == "euclidean"]
                if not euc:
                    continue
                base = st.mean(euc)
                per_kappa = {}
                for k in CURVATURES:
                    v = [c["recovery_rate"] for c in sub if c["kappa"] == k]
                    if v:
                        per_kappa[str(k)] = {
                            "recovery": st.mean(v),
                            "advantage_vs_euclidean": st.mean(v) / base if base else None,
                        }
                hyp_adv = [
                    v["advantage_vs_euclidean"] for k, v in per_kappa.items()
                    if float(k) < 0 and v["advantage_vs_euclidean"]
                ]
                rhos = [c["radius_level_rho"] for c in sub
                        if c["kappa"] < 0 and c["radius_level_rho"] is not None]
                by_depth[str(depth)] = {
                    "euclidean_recovery": base,
                    "by_kappa": per_kappa,
                    "best_hyperbolic_advantage": max(hyp_adv) if hyp_adv else None,
                    "radius_level_rho_hyperbolic": st.mean(rhos) if rhos else None,
                }
            advs = [
                (int(d), v["best_hyperbolic_advantage"])
                for d, v in by_depth.items() if v["best_hyperbolic_advantage"]
            ]
            anchor = by_depth.get("0", {}).get("best_hyperbolic_advantage")
            out[f"{head}|d{dim}"] = {
                "by_depth": by_depth,
                "advantage_by_depth": advs,
                "anchor_depth0_advantage": anchor,
                "anchor_holds": (abs(anchor - 1.0) < 0.15) if anchor else None,
                "advantage_grows_with_depth": bool(
                    len(advs) >= 3
                    and spearman_midrank([a for a, _ in advs], [b for _, b in advs])["rho"] > 0.5
                ),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RESULTS_ROOT / "csc2" / "e3")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    arms = [("euclidean", 0.0)] + [("curved", k) for k in CURVATURES]
    specs = [
        (dim, depth, gain, head, arm, kappa, seed)
        for dim in DIMS for depth in DEPTHS for gain in GAINS
        for head in HEADS for arm, kappa in arms for seed in SEEDS
    ]
    print(f"E3: {len(specs)} runs x {STEPS} steps")
    cells = parallel_map(_cell, specs, max_workers=args.workers)
    analysis = analyse(cells)

    payload = {
        "phase": "csc2-e3",
        "hypothesis": "H2-MAIN — the curvature advantage grows with hierarchy depth",
        "settings": {
            "depths": DEPTHS, "branching": BRANCHING, "curvatures": CURVATURES,
            "dims": DIMS, "gains": GAINS, "heads": HEADS, "seeds": len(SEEDS),
            "steps": STEPS, "flat_n_features": FLAT_N_FEATURES,
        },
        "analysis": analysis,
        "cells": cells,
    }
    path = write_artifact(args.out / "e3_hierarchy.json", payload)
    print(f"E3: wrote {path}")
    for key, v in analysis.items():
        advs = ", ".join(f"D{d}:{a:.2f}" for d, a in v["advantage_by_depth"])
        print(f"  {key:20} anchor={v['anchor_depth0_advantage']} "
              f"holds={v['anchor_holds']} grows={v['advantage_grows_with_depth']} [{advs}]")


if __name__ == "__main__":
    main()

"""Phase 00d — positive control: does the flat arm reproduce known TMS behaviour?

Not in SPEC §4; added after Phase 00 found instrument-level problems in all
three of its own sub-gates. R3 gives the study controls for *geometry* — flat
arms that isolate curvature from conditioning — but nothing establishes that
the toy setup reproduces known superposition phenomenology at all. Every
capacity claim in H-MAIN is a comparison *against the Euclidean arm*, so if
that arm does not behave like a toy model of superposition, no curved
comparison against it means anything. This is the one check that catches a
whole-pipeline error rather than a component one.

**Euclidean only, deliberately.** Reading capacity numbers off a curved arm is
reading H-MAIN. This control measures the flat arm against *external*
published behaviour, not the curved arm against the flat one, so it stays
outside the hypothesis even though it is the first thing here to compute a
capacity number at all.

The architecture here is not standard TMS — reconstruction goes through a
distance-to-prototype readout rather than a tied linear W‌ᵀW — so the exact
polytope structure Elhage et al. report (digons, triangles, pentagons) is not
expected to transfer. What must transfer is the phenomenon the study depends
on: **sparsity drives superposition**. With d dimensions available:

- dense inputs → the model keeps roughly d features and drops the rest, because
  simultaneous features interfere and there is no room to overlap;
- sparse inputs → features rarely co-occur, interference is cheap, and the
  model represents many more than d.

Pre-registered expectations, committed before the run:

- **E1** dense (sparsity 0.0): recovered ≤ d + 1 for every N ≥ 4·d.
- **E2** sparse (sparsity 0.99): recovered ≥ 2·d for some N, and strictly more
  than the dense count at the same N.
- **E3** monotone: median recovered is non-decreasing in sparsity, allowing
  one adjacent-pair inversion for seed noise.

A failure of E1 or E2 means the setup does not exhibit superposition and
Phase 1 is measuring something else. E3 failing alone weakens P2 (whose whole
subject is where that boundary sits) without invalidating H-MAIN.
"""

from __future__ import annotations

import argparse
import statistics as st
from itertools import pairwise
from pathlib import Path

from csc.calibration.scale_rule import PRIMARY_HEAD
from csc.interp.capacity import capacity_from_sweep
from csc.training.toy_loop import ToyConfig, train_toy
from experiments.util import RESULTS_ROOT, parallel_map, write_artifact

DIM = 2
FEATURE_COUNTS = [2, 3, 4, 5, 6, 8, 12, 16, 24, 32]
SPARSITIES = [0.0, 0.5, 0.7, 0.9, 0.95, 0.99]
SEEDS = [0, 1, 2, 3, 4]
STEPS = 10_000


def _cell(spec: tuple) -> dict:
    n_features, sparsity, seed = spec
    cfg = ToyConfig(
        arm="euclidean",
        dim=DIM,
        n_features=n_features,
        sparsity=sparsity,
        head=PRIMARY_HEAD,
        steps=STEPS,
        eval_every=2_000,
        batch_size=256,
        seed=seed,
    )
    run = train_toy(cfg)
    return {
        "n_features": n_features,
        "sparsity": sparsity,
        "seed": seed,
        "features_recovered": run.summary["probes"]["features_recovered"],
        "recovery_rate": run.summary["probes"]["recovery_rate"],
        "interference_mean": run.summary["probes"]["interference_mean"],
        "min_pairwise_distance": run.summary["prototype_geometry"]["min_pairwise_distance"],
        "final_loss": run.summary["final_loss"],
        "dead_unit_fraction": run.summary["dead_unit_fraction"],
    }


def evaluate(cells: list[dict]) -> dict:
    """Score the three pre-registered expectations."""
    def med(n, s):
        vals = [c["features_recovered"] for c in cells if c["n_features"] == n and c["sparsity"] == s]
        return st.median(vals) if vals else None

    dense, sparse = min(SPARSITIES), max(SPARSITIES)
    large_n = [n for n in FEATURE_COUNTS if n >= 4 * DIM]

    e1_values = {n: med(n, dense) for n in large_n}
    e1 = all(v is not None and v <= DIM + 1 for v in e1_values.values())

    sparse_vs_dense = {n: (med(n, dense), med(n, sparse)) for n in FEATURE_COUNTS}
    e2_max = max(v for _, v in sparse_vs_dense.values() if v is not None)
    e2 = e2_max >= 2 * DIM and any(
        s > d for d, s in sparse_vs_dense.values() if d is not None and s is not None
    )

    # E3 on the largest N, where the ceiling is not the feature count itself
    curve = [med(max(FEATURE_COUNTS), s) for s in SPARSITIES]
    inversions = sum(1 for a, b in pairwise(curve) if b < a)
    e3 = inversions <= 1

    return {
        "E1_dense_keeps_about_d_features": {
            "pass": bool(e1),
            "criterion": f"median recovered <= d+1 = {DIM + 1} at sparsity {dense}, for N >= {4 * DIM}",
            "measured": e1_values,
        },
        "E2_sparsity_buys_superposition": {
            "pass": bool(e2),
            "criterion": f"max median recovered at sparsity {sparse} >= 2d = {2 * DIM}, and > dense at some N",
            "max_recovered_when_sparse": e2_max,
            "dense_vs_sparse_by_n": {str(k): v for k, v in sparse_vs_dense.items()},
        },
        "E3_monotone_in_sparsity": {
            "pass": bool(e3),
            "criterion": "median recovered non-decreasing in sparsity, <=1 adjacent inversion",
            "curve_at_max_n": curve,
            "sparsities": SPARSITIES,
            "inversions": inversions,
        },
        "verdict": "PASS" if (e1 and e2) else "FAIL",
        "note": (
            "E1 and E2 gate: they are the superposition phenomenon itself. E3 "
            "failing alone weakens P2 without invalidating H-MAIN."
        ),
        "capacity_by_sparsity": {
            str(s): capacity_from_sweep(
                {
                    n: st.median(
                        [
                            c["recovery_rate"]
                            for c in cells
                            if c["n_features"] == n and c["sparsity"] == s
                        ]
                    )
                    for n in FEATURE_COUNTS
                },
                threshold=0.9,
            )
            for s in SPARSITIES
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RESULTS_ROOT / "phase00")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    specs = [(n, s, seed) for n in FEATURE_COUNTS for s in SPARSITIES for seed in SEEDS]
    print(f"00d: {len(specs)} runs x {STEPS} steps (Euclidean only)")
    cells = parallel_map(_cell, specs, max_workers=args.workers)
    scored = evaluate(cells)

    payload = {
        "phase": "00d",
        "purpose": "positive control: the flat arm must reproduce known TMS superposition behaviour",
        "blind": "Euclidean only — no curved arm is trained, so G00 is not breached",
        "settings": {
            "dim": DIM,
            "feature_counts": FEATURE_COUNTS,
            "sparsities": SPARSITIES,
            "seeds": SEEDS,
            "steps": STEPS,
            "head": PRIMARY_HEAD,
        },
        "expectations": scored,
        "cells": cells,
    }
    path = write_artifact(args.out / "00d_positive_control.json", payload)
    print(f"00d: wrote {path}")
    print(f"00d: verdict {scored['verdict']}")
    for key in ("E1_dense_keeps_about_d_features", "E2_sparsity_buys_superposition",
                "E3_monotone_in_sparsity"):
        print(f"  {key}: {scored[key]['pass']}")
    print(f"  capacity N* by sparsity: {scored['capacity_by_sparsity']}")


if __name__ == "__main__":
    main()

"""Phase 1b — gain x curvature factorial: is the kappa-trend real or is it the gain?

Phase 1 was uninterpretable because Spearman(|kappa|, init gain) = 0.971: the
fixed-radius calibration (D12) tied the optimization knob to the variable under
test, so "capacity falls with |kappa|" and "capacity falls with gain" were the
same statement. This design breaks that tie by the only clean means available —
**setting the gain explicitly and crossing it with curvature**, rather than
deriving it from a calibration rule.

Consequences of that choice, all deliberate:

- No calibration rule is used here. Both curvature signs are therefore treated
  identically by construction, which also removes the Phase-1 coding error in
  which hyperbolic arms were calibrated by fixed radius and spherical arms by
  band centre.
- Arms at the same gain sit at *different* √|K|·r. That is not a confound to
  remove; it is the geometry doing its work. What must be matched across arms
  is the optimization condition, not the geometric outcome.
- Both matched-gain and matched-x comparisons are reported, so the two possible
  readings can be told apart rather than assumed apart.

Pre-registered readings, stated before the run:

- **If flat-arm capacity varies materially with gain** (|rho| > 0.3 in the
  Euclidean arm), Phase 1's kappa-trend is confounded and stands retracted
  regardless of what the curved arms do.
- **The clean test of P1** is the kappa-trend computed *within each gain
  level*, where gain is constant by construction. If that trend is consistent
  in sign across gain levels and heads, it is evidence; if it flips with gain
  or between heads, it is not.
"""

from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

from csc.calibration.scale_rule import PRIMARY_HEAD
from csc.interp.trend import jonckheere_terpstra, spearman_midrank
from csc.training.toy_loop import ToyConfig, train_toy
from experiments.util import RESULTS_ROOT, parallel_map, write_artifact

# Spans the range the Phase-1 arms actually used (1.50 -> 8.24 for norm_affine).
GAINS = [1.5, 2.5, 4.0, 6.5, 9.0]
CURVATURES = [-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
DIMS = [4, 6, 8]
N_FEATURES = 64
SPARSITY = 0.9
SEEDS = list(range(10))
STEPS = 10_000
HEADS = [PRIMARY_HEAD, "softmax"]


def _arms() -> list[dict]:
    arms = [
        {"arm": "euclidean", "kappa": 0.0, "kwargs": {}, "label": "euclidean"},
        {"arm": "normalized", "kappa": 0.0, "kwargs": {}, "label": "normalized"},
        {"arm": "clamped", "kappa": 0.0, "kwargs": {"max_dist": 3.0}, "label": "clamped"},
    ]
    arms += [
        {"arm": "curved", "kappa": k, "kwargs": {}, "label": f"curved(K={k:+g})"}
        for k in CURVATURES
    ]
    return arms


def _cell(spec: tuple) -> dict:
    dim, gain, head, arm, seed = spec
    cfg = ToyConfig(
        arm=arm["arm"],
        kappa=arm["kappa"],
        space_kwargs=dict(arm["kwargs"]),
        dim=dim,
        n_features=N_FEATURES,
        sparsity=SPARSITY,
        head=head,
        encoder_init_scale=gain,
        steps=STEPS,
        eval_every=1_000,
        batch_size=256,
        seed=seed,
    )
    run = train_toy(cfg)
    sat = run.summary["saturation"]
    return {
        "dim": dim,
        "gain": gain,
        "head": head,
        "arm": arm["arm"],
        "arm_label": arm["label"],
        "kappa": arm["kappa"],
        "seed": seed,
        "features_recovered": run.summary["probes"]["features_recovered"],
        "interference_mean": run.summary["probes"]["interference_mean"],
        "min_pairwise_distance": run.summary["prototype_geometry"]["min_pairwise_distance"],
        "final_loss": run.summary["final_loss"],
        "dead_unit_fraction": run.summary["dead_unit_fraction"],
        "scaled_radius_median": sat["final_scaled_radius_median"],
        "saturation_verdict": sat["verdict"],
    }


def analyse(cells: list[dict]) -> dict:
    ok = [c for c in cells if c["saturation_verdict"] == "OK"]
    out = {}

    for head in HEADS:
        for dim in DIMS:
            sub = [c for c in ok if c["head"] == head and c["dim"] == dim]
            if not sub:
                continue

            # 1. Does gain drive capacity in the FLAT arm? This is the question
            #    Phase 1 could not answer, and it decides whether that run's
            #    kappa-trend was an artifact.
            flat = [c for c in sub if c["arm"] == "euclidean"]
            gain_effect = None
            if flat:
                gain_effect = {
                    "spearman_gain_vs_capacity": spearman_midrank(
                        [c["gain"] for c in flat], [c["features_recovered"] for c in flat]
                    ),
                    "capacity_by_gain": {
                        str(g): st.mean(
                            c["features_recovered"] for c in flat if c["gain"] == g
                        )
                        for g in GAINS
                        if any(c["gain"] == g for c in flat)
                    },
                }

            # 2. The clean test: kappa-trend WITHIN each gain level.
            per_gain = {}
            for g in GAINS:
                at_g = [c for c in sub if c["gain"] == g and c["arm"] == "curved"]
                present = [
                    (k, [c["features_recovered"] for c in at_g if c["kappa"] == k])
                    for k in sorted(CURVATURES)
                ]
                present = [(k, v) for k, v in present if len(v) >= 3]
                if len(present) < 3:
                    continue
                flat_k, flat_v = [], []
                for k, v in present:
                    flat_k.extend([k] * len(v))
                    flat_v.extend(v)
                per_gain[str(g)] = {
                    "spearman_kappa_vs_capacity": spearman_midrank(flat_k, flat_v),
                    "jonckheere": jonckheere_terpstra(
                        [v for _, v in present], n_permutations=3000
                    ),
                    "capacity_by_kappa": {str(k): st.mean(v) for k, v in present},
                    "n_arms": len(present),
                }

            signs = [
                v["spearman_kappa_vs_capacity"]["rho"] for v in per_gain.values()
            ]
            out[f"{head}|d{dim}"] = {
                "gain_effect_in_flat_arm": gain_effect,
                "kappa_trend_within_gain_level": per_gain,
                "rho_by_gain": signs,
                "sign_consistent_across_gain": bool(
                    signs and (all(s > 0 for s in signs) or all(s < 0 for s in signs))
                ),
                "capacity_interference_frontier": {
                    lab: {
                        "capacity": st.mean(
                            c["features_recovered"] for c in sub if c["arm_label"] == lab
                        ),
                        "interference": st.mean(
                            c["interference_mean"] for c in sub if c["arm_label"] == lab
                        ),
                    }
                    for lab in sorted({c["arm_label"] for c in sub})
                },
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RESULTS_ROOT / "phase1")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    specs = [
        (dim, gain, head, arm, seed)
        for dim in DIMS
        for gain in GAINS
        for head in HEADS
        for arm in _arms()
        for seed in SEEDS
    ]
    print(f"P1b: {len(specs)} runs x {STEPS} steps")
    cells = parallel_map(_cell, specs, max_workers=args.workers)
    analysis = analyse(cells)

    payload = {
        "phase": "1b",
        "question": "is the Phase-1 kappa-trend real, or is it the init gain?",
        "design": {
            "gains": GAINS,
            "curvatures": CURVATURES,
            "dims": DIMS,
            "n_features": N_FEATURES,
            "sparsity": SPARSITY,
            "seeds": len(SEEDS),
            "steps": STEPS,
            "heads": HEADS,
            "note": "gain is set explicitly, not derived from a calibration rule",
        },
        "analysis": analysis,
        "cells": cells,
    }
    path = write_artifact(args.out / "p1b_factorial.json", payload)
    print(f"P1b: wrote {path}")
    for key, res in analysis.items():
        ge = res["gain_effect_in_flat_arm"]
        rho_g = ge["spearman_gain_vs_capacity"]["rho"] if ge else float("nan")
        rhos = ", ".join(f"{r:+.2f}" for r in res["rho_by_gain"])
        print(
            f"  {key}: flat-arm rho(gain,cap)={rho_g:+.3f} | "
            f"rho(kappa,cap) by gain [{rhos}] | "
            f"sign consistent={res['sign_consistent_across_gain']}"
        )


if __name__ == "__main__":
    main()

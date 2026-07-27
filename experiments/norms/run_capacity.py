"""Study 3 — does changing the NORM buy angular capacity where curvature could not?

Study 1 established that no Riemannian curvature changes angular capacity: at
any point the direction sphere is S^(d-1) and the metric there is a scalar
multiple of the identity, so angles at a point are Euclidean whatever the
curvature. Capacity is angular, so curvature was structurally the wrong dial.

Norms are the right dial. Every arm here is FLAT — curvature exactly zero — and
differs only in how distance is computed, so any capacity difference is
attributable to the norm alone.

**Registered prediction (before the run).** l-inf beats l2. Its unit ball is a
hypercube with 2^d vertices: exponentially many maximally separated directions
with no radial blow-up, which is the shape superposition wants.

**Registered counter-prediction, which matters more.** Euclidean space already
admits exp(O(eps^2 d)) near-orthogonal directions — far more than Study 1's
models ever used (~18 recovered in d=8). If flat l2 capacity is already
unreached, the binding constraint is readout resolution and optimization, not
geometry, and NO norm will help. That is the more useful outcome: it says stop
studying spaces and start studying readouts.

**The measurement that discriminates them** is `coherence_vs_welch`. The Welch
bound gives the smallest achievable max-|cos| for N unit vectors in R^d:

    mu_welch = sqrt((N - d) / (d (N - 1)))

If the learned prototypes sit far above it in every arm, the models are nowhere
near the packing limit and the geometry was never the constraint. That single
ratio tests the counter-prediction directly, and it is reported whatever the
capacity numbers do.

Design discipline carried from Study 1: gain crossed factorially with the arm
rather than derived from it (R4 — a calibration rule tied to the independent
variable produced rho=0.971 and cost 2200 runs); both validated readouts (R6);
10 seeds against a measured MDE of 1.39x (R7); flat controls in the same sweep
(R3); primary metric max-features-recovered (D15).

Note on R1: the Finsler arm carries `dim` extra parameters. Parameter counts
are reported per arm so the comparison can be read as capacity-per-parameter
where that matters, rather than the mismatch being silent.
"""

from __future__ import annotations

import argparse
import math
import statistics as st
from pathlib import Path

import torch

from csc.interp.capacity import capacity_max_recovered
from csc.interp.trend import extreme_pair_contrast
from csc.models.toy import parameter_count
from csc.training.toy_loop import ToyConfig, train_toy
from experiments.util import RESULTS_ROOT, parallel_map, write_artifact

# (label, arm, kwargs). p=2 is the control and must reproduce Euclidean.
ARMS = [
    ("lp_p1", "lp", {"p": 1.0}),
    ("lp_p1.5", "lp", {"p": 1.5}),
    ("lp_p2_CONTROL", "lp", {"p": 2.0}),
    ("lp_p3", "lp", {"p": 3.0}),
    ("lp_pinf", "lp", {"p": float("inf")}),
    ("finsler_p2", "finsler", {"p": 2.0}),
    ("normalized", "normalized", {}),
]
DIMS = [4, 8]
FEATURE_COUNTS = [16, 32, 64, 128]
GAINS = [1.5, 6.0]
HEADS = ["norm_affine", "softmax"]
SPARSITY = 0.9
SEEDS = list(range(10))
STEPS = 10_000


def welch_bound(n: int, d: int) -> float:
    """Smallest achievable max-|cos| among n unit vectors in R^d."""
    if n <= d:
        return 0.0
    return math.sqrt((n - d) / (d * (n - 1)))


@torch.no_grad()
def prototype_coherence(model) -> dict:
    """How close is the learned configuration to the packing limit?

    Coherence is computed on normalized prototype *directions*, deliberately:
    it is the angular quantity, and the whole programme's finding is that
    capacity is angular. A model far above the Welch bound is optimization- or
    readout-limited, not geometry-limited.
    """
    p = model.readout.prototypes()
    n, d = p.shape
    u = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    cos = (u @ u.T).abs()
    cos.fill_diagonal_(0.0)
    achieved = float(cos.max())
    bound = welch_bound(n, d)
    return {
        "max_abs_cosine": achieved,
        "mean_abs_cosine": float(cos.sum() / (n * (n - 1))),
        "welch_bound": bound,
        # 1.0 = at the packing limit; >>1 = nowhere near it
        "coherence_vs_welch": achieved / bound if bound > 0 else float("nan"),
    }


def _cell(spec: tuple) -> dict:
    label, arm, kwargs, dim, n_features, gain, head, seed = spec
    cfg = ToyConfig(
        arm=arm, space_kwargs=dict(kwargs), dim=dim, n_features=n_features,
        sparsity=SPARSITY, head=head, encoder_init_scale=gain,
        steps=STEPS, eval_every=2_000, batch_size=256, seed=seed,
    )
    run = train_toy(cfg)
    return {
        "label": label, "arm": arm, "dim": dim, "n_features": n_features,
        "gain": gain, "head": head, "seed": seed,
        "n_parameters": parameter_count(run.model),
        "features_recovered": run.summary["probes"]["features_recovered"],
        "recovery_rate": run.summary["probes"]["recovery_rate"],
        "interference_mean": run.summary["probes"]["interference_mean"],
        "min_pairwise_distance": run.summary["prototype_geometry"]["min_pairwise_distance"],
        "final_loss": run.summary["final_loss"],
        "dead_unit_fraction": run.summary["dead_unit_fraction"],
        **prototype_coherence(run.model),
    }


def analyse(cells: list[dict]) -> dict:
    out = {}
    control = "lp_p2_CONTROL"
    for head in HEADS:
        for dim in DIMS:
            for gain in GAINS:
                sub = [c for c in cells
                       if c["head"] == head and c["dim"] == dim and c["gain"] == gain]
                if not sub:
                    continue
                cap = {}
                for label, _, _ in ARMS:
                    per_seed = []
                    for s in SEEDS:
                        rows = {c["n_features"]: c["features_recovered"]
                                for c in sub if c["label"] == label and c["seed"] == s}
                        if rows:
                            per_seed.append(capacity_max_recovered(rows))
                    if per_seed:
                        cap[label] = per_seed
                if control not in cap:
                    continue
                base = st.mean(cap[control])
                per_arm = {
                    label: {
                        "capacity_mean": st.mean(v),
                        "capacity_per_seed": v,
                        "advantage_vs_l2": st.mean(v) / base if base else None,
                        "n_parameters": st.mean(
                            c["n_parameters"] for c in sub if c["label"] == label
                        ),
                        "coherence_vs_welch": st.mean(
                            c["coherence_vs_welch"] for c in sub
                            if c["label"] == label and not math.isnan(c["coherence_vs_welch"])
                        ),
                    }
                    for label, v in cap.items()
                }
                best = max(
                    (l for l in per_arm if l != control),
                    key=lambda l: per_arm[l]["advantage_vs_l2"] or 0,
                )
                out[f"{head}|d{dim}|gain{gain}"] = {
                    "by_arm": per_arm,
                    "best_non_control": best,
                    "best_advantage": per_arm[best]["advantage_vs_l2"],
                    "linf_advantage": per_arm.get("lp_pinf", {}).get("advantage_vs_l2"),
                    "linf_beats_l2_significantly": (
                        extreme_pair_contrast(cap["lp_pinf"], cap[control], n_permutations=5000)
                        if "lp_pinf" in cap else None
                    ),
                }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RESULTS_ROOT / "norms")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    specs = [
        (label, arm, kwargs, dim, n, gain, head, seed)
        for label, arm, kwargs in ARMS
        for dim in DIMS for n in FEATURE_COUNTS for gain in GAINS
        for head in HEADS for seed in SEEDS
    ]
    print(f"norms: {len(specs)} runs x {STEPS} steps")
    cells = parallel_map(_cell, specs, max_workers=args.workers)
    analysis = analyse(cells)

    # The counter-prediction, tested directly and independently of capacity.
    coh = [c["coherence_vs_welch"] for c in cells if not math.isnan(c["coherence_vs_welch"])]
    by_arm_coh = {
        label: st.mean(c["coherence_vs_welch"] for c in cells
                       if c["label"] == label and not math.isnan(c["coherence_vs_welch"]))
        for label, _, _ in ARMS
    }
    near_limit = st.mean(coh) < 1.5

    payload = {
        "phase": "study3-norms",
        "question": "does changing the norm buy angular capacity where curvature could not?",
        "predictions_registered_before_run": {
            "primary": "l-inf beats l2 (2^d hypercube vertices = exponentially many separated directions)",
            "counter": (
                "flat l2 capacity is already unreached, so the constraint is readout "
                "resolution rather than geometry and no norm helps"
            ),
            "discriminator": "coherence_vs_welch — far above 1.0 in every arm means never geometry-limited",
        },
        "settings": {
            "arms": [a[0] for a in ARMS], "dims": DIMS, "feature_counts": FEATURE_COUNTS,
            "gains": GAINS, "heads": HEADS, "sparsity": SPARSITY,
            "seeds": len(SEEDS), "steps": STEPS,
            "mde_at_10_seeds": 1.39,
        },
        "packing_limit_check": {
            "mean_coherence_vs_welch_all_arms": st.mean(coh),
            "by_arm": by_arm_coh,
            "models_near_packing_limit": near_limit,
            "reading": (
                "at the packing limit — geometry could plausibly be the constraint"
                if near_limit else
                "FAR from the packing limit in every arm — the models never approach "
                "what flat space already allows, so the binding constraint is readout "
                "resolution and optimization, not the geometry"
            ),
        },
        "analysis": analysis,
        "cells": cells,
    }
    path = write_artifact(args.out / "study3_norm_capacity.json", payload)
    print(f"norms: wrote {path}")
    print(f"\npacking limit: mean coherence/Welch = {st.mean(coh):.2f} "
          f"-> {payload['packing_limit_check']['reading']}")
    print("\ncapacity vs the l2 control:")
    for key, v in analysis.items():
        adv = "  ".join(
            f"{l.replace('lp_','')}:{d['advantage_vs_l2']:.2f}x"
            for l, d in v["by_arm"].items() if l != "lp_p2_CONTROL"
        )
        p = v["linf_beats_l2_significantly"]
        sig = f" | linf p={p['p_value']:.3f}" if p else ""
        print(f"  {key:26} {adv}{sig}")


if __name__ == "__main__":
    main()

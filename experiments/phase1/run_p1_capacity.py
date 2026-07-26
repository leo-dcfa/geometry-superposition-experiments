"""Phase 1 — P1: does capacity decrease as curvature increases?

The first hypothesis-relevant experiment in the study. Everything before this
was instrument work.

**Design, as sealed in VALIDATION.md §2.** Each cell is one (d, N) shape. Within
a cell every arm is calibrated to the *same geodesic radius* R, so that
x = √|K|·R varies with κ exactly as P1 predicts (D12) — targeting a constant
√|K|·r instead would have given every arm the same predicted capacity and
falsified P1 by construction.

Arms per cell: the four P1 curvatures, the Euclidean baseline, and both R3
controls. Controls are built in the same sweep, never afterwards.

Primary metric is **max features recovered** (D15); SPEC's ≥90% N* is computed
alongside as secondary. Primary dimensions are d ∈ {4, 6, 8} (D13) — at d = 2
the predicted effect sits below the measured detection floor, so d = 2 runs but
is reported as underpowered rather than as evidence.

Primary test is a **monotone-trend test** (D14), not F1.1's all-adjacent-pairs
criterion, which was measured to fire on 53% of true hypotheses. Reported:
Jonckheere–Terpstra with a permutation null, Spearman with midranks, and the
extreme-pair contrast κ=−4 vs κ=−0.5.

Falsifiers still live:
- F1.3 — if the clamped-Euclidean control reproduces ≥ 70% of the hyperbolic
  gain, the effect is conditioning rather than geometry and H-MAIN dies at toy
  scale. This is the one that matters most, and it is why the control is in
  every cell.
- Directional — a *spherical* capacity advantage surviving the controls
  falsifies H-MAIN outright (SPEC §1). It is not reinterpreted.
"""

from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

from csc.calibration.scale_rule import (
    PRIMARY_HEAD,
    init_gain,
    init_gain_fixed_radius,
    phase1_radius,
)
from csc.interp.capacity import capacity_from_sweep, capacity_max_recovered
from csc.interp.trend import extreme_pair_contrast, jonckheere_terpstra, spearman_midrank
from csc.training.toy_loop import ToyConfig, train_toy
from experiments.util import RESULTS_ROOT, parallel_map, write_artifact

# D13: primary dimensions; d=2 runs but is reported as underpowered.
PRIMARY_DIMS = [4, 6, 8]
ILLUSTRATIVE_DIMS = [2]
# Feature ladder per shape — capacity is read as the maximum over this sweep.
FEATURE_COUNTS = [8, 16, 32, 64, 128]
CURVATURES = [-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
SPARSITY = 0.9  # 00d: where the capacity metric is most sensitive
SEEDS = list(range(10))  # D14: above SPEC's minimum of 5, below F1.1's 58
STEPS = 10_000
HEADS = [PRIMARY_HEAD, "softmax"]


def _arms_for(dim: int, n_features: int, radius: float, head: str) -> list[dict]:
    """Every arm of one cell, with its calibrated init gain.

    R3 controls take the init gain of the curved arm they control, so the two
    start at the same raw geodesic radius and differ only in geometry.
    """
    reference_gain = init_gain_fixed_radius(-1.0, dim, n_features, radius, head)
    arms = [
        {"arm": "euclidean", "kappa": 0.0, "kwargs": {}, "gain": reference_gain},
        {
            "arm": "clamped",
            "kappa": 0.0,
            "kwargs": {"max_dist": 2.0 * radius},
            "gain": reference_gain,
        },
        {"arm": "normalized", "kappa": 0.0, "kwargs": {}, "gain": reference_gain},
    ]
    for kappa in CURVATURES:
        gain = (
            init_gain_fixed_radius(kappa, dim, n_features, radius, head)
            if kappa < 0
            else init_gain(kappa, dim, n_features, head)
        )
        arms.append({"arm": "curved", "kappa": kappa, "kwargs": {}, "gain": gain})
    return arms


def _cell(spec: tuple) -> dict:
    dim, n_features, radius, head, arm, seed = spec
    cfg = ToyConfig(
        arm=arm["arm"],
        kappa=arm["kappa"],
        space_kwargs=dict(arm["kwargs"]),
        dim=dim,
        n_features=n_features,
        sparsity=SPARSITY,
        head=head,
        encoder_init_scale=arm["gain"],
        steps=STEPS,
        eval_every=1_000,
        batch_size=256,
        seed=seed,
    )
    run = train_toy(cfg)
    sat = run.summary["saturation"]
    label = "euclidean" if arm["arm"] == "euclidean" else (
        arm["arm"] if arm["kappa"] == 0.0 else f"curved(K={arm['kappa']:+g})"
    )
    return {
        "dim": dim,
        "n_features": n_features,
        "radius": radius,
        "head": head,
        "arm": arm["arm"],
        "arm_label": label,
        "kappa": arm["kappa"],
        "gain": arm["gain"],
        "seed": seed,
        "features_recovered": run.summary["probes"]["features_recovered"],
        "recovery_rate": run.summary["probes"]["recovery_rate"],
        "interference_mean": run.summary["probes"]["interference_mean"],
        "interference_max": run.summary["probes"]["interference_max"],
        "min_pairwise_distance": run.summary["prototype_geometry"]["min_pairwise_distance"],
        "final_loss": run.summary["final_loss"],
        "dead_unit_fraction": run.summary["dead_unit_fraction"],
        "scaled_radius_median": sat["final_scaled_radius_median"],
        "saturation_verdict": sat["verdict"],
    }


def analyse(cells: list[dict]) -> dict:
    """Per (head, dim): capacity by arm, the trend test, and F1.3."""
    out = {}
    for head in HEADS:
        for dim in PRIMARY_DIMS + ILLUSTRATIVE_DIMS:
            sub = [
                c
                for c in cells
                if c["head"] == head and c["dim"] == dim and c["saturation_verdict"] == "OK"
            ]
            if not sub:
                continue
            labels = sorted({c["arm_label"] for c in sub})

            # capacity per (arm, seed): max recovered over the feature ladder
            cap = {}
            for label in labels:
                per_seed = []
                for seed in SEEDS:
                    rows = {
                        c["n_features"]: c["features_recovered"]
                        for c in sub
                        if c["arm_label"] == label and c["seed"] == seed
                    }
                    rates = {
                        c["n_features"]: c["recovery_rate"]
                        for c in sub
                        if c["arm_label"] == label and c["seed"] == seed
                    }
                    if rows:
                        per_seed.append(
                            {
                                "primary_max_recovered": capacity_max_recovered(rows),
                                "secondary_nstar": capacity_from_sweep(rates, 0.9),
                            }
                        )
                if per_seed:
                    cap[label] = per_seed

            # trend test across the curvature grid, ascending in kappa
            # keep kappa and its group paired: a missing arm must not shift the
            # curvature axis under the values, which a bare zip would allow
            present = [
                (k, [row["primary_max_recovered"] for row in cap[f"curved(K={k:+g})"]])
                for k in sorted(CURVATURES)
                if f"curved(K={k:+g})" in cap
            ]
            groups = [g for _, g in present]
            trend = None
            if len(groups) >= 3 and all(len(g) >= 3 for g in groups):
                flat_k, flat_v = [], []
                for k, g in present:
                    flat_k.extend([k] * len(g))
                    flat_v.extend(g)
                trend = {
                    "jonckheere_terpstra": jonckheere_terpstra(groups, n_permutations=5000),
                    "spearman": spearman_midrank(flat_k, flat_v),
                    "extreme_pair": extreme_pair_contrast(
                        groups[0], groups[-1], n_permutations=5000
                    ),
                    "group_means": [st.mean(g) for g in groups],
                    "curvature_order": [k for k, _ in present],
                }

            # F1.3: does the clamped control reproduce the hyperbolic gain?
            f13 = None
            euc = cap.get("euclidean")
            hyp = cap.get("curved(K=-4)")
            clamped = cap.get("clamped")
            if euc and hyp and clamped:
                m = lambda rows: st.mean(r["primary_max_recovered"] for r in rows)
                hyp_gain = m(hyp) - m(euc)
                clamp_gain = m(clamped) - m(euc)
                share = clamp_gain / hyp_gain if hyp_gain != 0 else float("nan")
                f13 = {
                    "euclidean_capacity": m(euc),
                    "hyperbolic_capacity": m(hyp),
                    "clamped_capacity": m(clamped),
                    "hyperbolic_gain": hyp_gain,
                    "clamped_gain": clamp_gain,
                    "clamped_share_of_gain": share,
                    "F1_3_fires": bool(share >= 0.70) if hyp_gain > 0 else None,
                    "criterion": "clamped control reproducing >=70% of the gain kills H-MAIN at toy scale",
                }

            out[f"{head}|d{dim}"] = {
                "capacity_by_arm": {
                    lab: {
                        "primary_mean": st.mean(r["primary_max_recovered"] for r in rows),
                        "primary_per_seed": [r["primary_max_recovered"] for r in rows],
                        "secondary_nstar_per_seed": [r["secondary_nstar"] for r in rows],
                    }
                    for lab, rows in cap.items()
                },
                "trend_test": trend,
                "F1_3_conditioning_control": f13,
                "is_primary_dimension": dim in PRIMARY_DIMS,
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RESULTS_ROOT / "phase1")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--dims", type=int, nargs="+", default=None)
    args = parser.parse_args()

    dims = args.dims or (PRIMARY_DIMS + ILLUSTRATIVE_DIMS)
    specs, skipped = [], []
    for dim in dims:
        for n in FEATURE_COUNTS:
            for head in HEADS:
                radius = phase1_radius(dim, n, head)
                if radius is None:
                    skipped.append({"dim": dim, "n_features": n, "head": head})
                    continue
                for arm in _arms_for(dim, n, radius, head):
                    for seed in SEEDS:
                        specs.append((dim, n, radius, head, arm, seed))

    print(f"P1: {len(specs)} runs x {STEPS} steps ({len(skipped)} cells unreachable, skipped)")
    cells = parallel_map(_cell, specs, max_workers=args.workers)
    analysis = analyse(cells)

    payload = {
        "phase": "1",
        "hypothesis": "P1 — capacity N*(kappa) decreases monotonically with kappa",
        "design": {
            "primary_dims": PRIMARY_DIMS,
            "illustrative_dims": ILLUSTRATIVE_DIMS,
            "feature_counts": FEATURE_COUNTS,
            "curvatures": CURVATURES,
            "sparsity": SPARSITY,
            "seeds": len(SEEDS),
            "steps": STEPS,
            "heads": HEADS,
            "calibration": "fixed geodesic radius per (d, N) cell (D12)",
            "primary_metric": "max features recovered (D15)",
            "primary_test": "Jonckheere-Terpstra permutation trend test (D14)",
        },
        "skipped_unreachable_cells": skipped,
        "analysis": analysis,
        "cells": cells,
    }
    path = write_artifact(args.out / "p1_capacity.json", payload)
    print(f"P1: wrote {path}")
    for key, res in analysis.items():
        t = res["trend_test"]
        f13 = res["F1_3_conditioning_control"]
        tag = "PRIMARY" if res["is_primary_dimension"] else "underpowered"
        line = f"  {key} [{tag}]:"
        if t:
            line += (
                f" JT p={t['jonckheere_terpstra']['p_value']:.4f}"
                f" rho={t['spearman']['rho']:+.3f}"
                f" extreme p={t['extreme_pair']['p_value']:.4f}"
            )
        if f13 and f13["F1_3_fires"] is not None:
            line += f" | F1.3 clamped share={f13['clamped_share_of_gain']:+.2f} fires={f13['F1_3_fires']}"
        print(line)


if __name__ == "__main__":
    main()

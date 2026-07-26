"""Phase 00a confirmation: does the committed rule actually land in the band?

A calibration rule fitted from a sweep and then reported from that same sweep
is only a description of the sweep. This applies `csc.calibration.scale_rule`
to held-out cells — including feature counts and dimensions the fit never saw
— trains them, and measures the resulting band occupancy. That number, not the
fit's R², is what decides sub-gate 00a.

Pre-registered pass criterion, stated before the run: **≥ 80% of runs on the
primary head land inside √|K|·r ∈ [0.5, 3.0], with no run flagged
UNINTERPRETABLE by R2.** 80 rather than 90 because the 00a fit's residual is
large (σ ≈ 0.56 in log space) and the band is only a factor of 6 wide, so
roughly one run in eight is expected outside on seed noise alone even with a
perfectly centred rule.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from csc.calibration.scale_rule import PRIMARY_HEAD, init_gain
from csc.training.monitor import OPERATING_BAND
from csc.training.toy_loop import ToyConfig, train_toy
from experiments.util import RESULTS_ROOT, parallel_map, write_artifact

PASS_THRESHOLD = 0.80
CURVATURES = [-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
# Deliberately includes shapes the rule was NOT fitted on (the fit saw
# (2,16), (2,64), (4,32), (8,64)); (3,24), (6,48) and N=128 are held out.
SHAPES = [(2, 16), (2, 48), (3, 24), (4, 32), (6, 48), (8, 128)]
SEEDS = [0, 1, 2]
HEADS = [PRIMARY_HEAD, "softmax"]
STEPS = 10_000


def _cell(spec: tuple) -> dict:
    kappa, dim, n_features, seed, head = spec
    gain = init_gain(kappa, dim, n_features, head)
    cfg = ToyConfig(
        arm="curved",
        kappa=kappa,
        dim=dim,
        n_features=n_features,
        encoder_init_scale=gain,
        head=head,
        steps=STEPS,
        eval_every=1_000,
        batch_size=256,
        seed=seed,
    )
    run = train_toy(cfg)
    sat = run.summary["saturation"]
    scaled = sat["final_scaled_radius_median"]
    lo, hi = OPERATING_BAND
    return {
        "kappa": kappa,
        "dim": dim,
        "n_features": n_features,
        "seed": seed,
        "head": head,
        "prescribed_gain": gain,
        "held_out_shape": (dim, n_features) not in [(2, 16), (2, 64), (4, 32), (8, 64)],
        "scaled_radius_median": scaled,
        "in_band": bool(lo <= scaled <= hi),
        "saturation_median": sat["final_saturation_median"],
        "verdict": sat["verdict"],
        "dead_unit_fraction": run.summary["dead_unit_fraction"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RESULTS_ROOT / "phase00")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    specs = [
        (kappa, dim, n_features, seed, head)
        for kappa in CURVATURES
        for dim, n_features in SHAPES
        for seed in SEEDS
        for head in HEADS
    ]
    print(f"00a-confirm: {len(specs)} runs x {STEPS} steps")
    cells = parallel_map(_cell, specs, max_workers=args.workers)

    def summarize(rows):
        if not rows:
            return None
        return {
            "n": len(rows),
            "in_band_fraction": sum(r["in_band"] for r in rows) / len(rows),
            "uninterpretable_fraction": sum(r["verdict"] != "OK" for r in rows) / len(rows),
            "below_band_fraction": sum(
                r["scaled_radius_median"] < OPERATING_BAND[0] for r in rows
            )
            / len(rows),
            "above_band_fraction": sum(
                r["scaled_radius_median"] > OPERATING_BAND[1] for r in rows
            )
            / len(rows),
        }

    by_head = {h: summarize([c for c in cells if c["head"] == h]) for h in HEADS}
    held_out = {
        h: summarize([c for c in cells if c["head"] == h and c["held_out_shape"]]) for h in HEADS
    }
    primary = by_head[PRIMARY_HEAD]
    passed = (
        primary["in_band_fraction"] >= PASS_THRESHOLD
        and primary["uninterpretable_fraction"] == 0.0
    )

    payload = {
        "phase": "00a-confirm",
        "purpose": "validate the committed init-gain rule by applying it, incl. held-out shapes",
        "pass_criterion": {
            "threshold": PASS_THRESHOLD,
            "applies_to": PRIMARY_HEAD,
            "also_requires": "zero runs flagged UNINTERPRETABLE",
        },
        "verdict": "PASS" if passed else "FAIL",
        "by_head": by_head,
        "held_out_shapes_only": held_out,
        "settings": {
            "curvatures": CURVATURES,
            "shapes": [list(s) for s in SHAPES],
            "seeds": SEEDS,
            "steps": STEPS,
        },
        "cells": cells,
    }
    path = write_artifact(args.out / "00a_confirm.json", payload)
    print(f"00a-confirm: wrote {path}")
    print(f"00a-confirm: verdict {payload['verdict']}")
    for h in HEADS:
        s = by_head[h]
        ho = held_out[h]
        print(
            f"  {h:12} in-band {s['in_band_fraction']:.3f} "
            f"(held-out shapes {ho['in_band_fraction']:.3f}); "
            f"below {s['below_band_fraction']:.3f} above {s['above_band_fraction']:.3f}; "
            f"UNINTERPRETABLE {s['uninterpretable_fraction']:.3f}"
        )


if __name__ == "__main__":
    main()

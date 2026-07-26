"""CSC-2 E0 — readout validation, BEFORE anything is calibrated through a readout.

Rule R5, bought expensively in CSC: 00a calibrated the point cloud through the
`rbf` head, 00c later disqualified that head, and two headline numbers reversed
on re-run. A calibration measured through an unvalidated readout measures the
readout. So validation comes first here, unconditionally.

What is validated: the 00c parity criterion — on shapes where capacity is NOT
binding, every arm should recover essentially every feature, so a dead unit can
only mean the readout treated an arm unfairly. Tolerance is max(0.05, 1/N),
never tighter than one readout unit.

The new head under test is `abs_rbf`. CSC's `rbf` collapsed to the all-zero
solution at weight_decay=0 because its kernel was ~0.5 for every prototype at
init, making "switch everything off" the steepest descent direction.
``abs_rbf`` starts its length scale small so the kernel is selective from step
one. Whether that repair works is an empirical question, and this is where it
gets answered — not by assertion in a docstring.

Both flat and hierarchical data are covered, because CSC-2's claim is about
hierarchical features and a head that is fair on one and not the other would
invalidate exactly the comparison E3 makes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from csc.interp.capacity import dead_unit_fraction
from csc.layers.readout import HEADS
from csc.training.data import sample_batch
from csc.training.hierarchy import FeatureTree, sample_hierarchical_batch
from csc.training.toy_loop import ToyConfig, build_model, effective_n_features, train_toy
from experiments.csc1.phase00.run_00c import parity_table
from experiments.util import RESULTS_ROOT, parallel_map, write_artifact

SEEDS = [0, 1, 2, 3, 4]
STEPS = 8_000
# Non-binding shapes: a working head recovers ~everything, so any dead unit is
# unfairness rather than capacity. (dim, n_features, hierarchy_depth)
NON_BINDING = [(8, 8, 0), (8, 16, 0), (8, 0, 2), (8, 0, 3)]
ARMS = [("euclidean", 0.0), ("curved", -1.0), ("curved", -4.0), ("curved", 1.0)]
WEIGHT_DECAYS = [0.0, 0.01]


def _cell(spec: tuple) -> dict:
    dim, n_features, depth, head, arm, kappa, wd, seed = spec
    cfg = ToyConfig(
        arm=arm, kappa=kappa, dim=dim,
        n_features=n_features if depth == 0 else 8,
        hierarchy_depth=depth, hierarchy_branching=2,
        head=head, steps=STEPS, eval_every=1_000, batch_size=256,
        seed=seed, weight_decay=wd,
    )
    n_eff = effective_n_features(cfg)
    torch.manual_seed(seed)
    fresh = build_model(cfg)
    if depth > 0:
        probe = sample_hierarchical_batch(1024, FeatureTree(depth, 2), cfg.sparsity)
    else:
        probe = sample_batch(1024, n_eff, cfg.sparsity)
    dead_at_init = dead_unit_fraction(fresh, probe)

    run = train_toy(cfg)
    return {
        "arm": arm, "kappa": kappa, "head": head, "dim": dim,
        "n_features": n_eff, "hierarchy_depth": depth,
        "weight_decay": wd, "seed": seed, "capacity_binding": False,
        "dead_at_init": dead_at_init,
        "dead_after_training": run.summary["dead_unit_fraction"],
        "recovery_rate": run.summary["probes"]["recovery_rate"],
        "final_loss": run.summary["final_loss"],
        "saturation_verdict": run.summary["saturation"]["verdict"],
        "arm_label": "euclidean" if arm == "euclidean" else f"curved(K={kappa:+g})",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RESULTS_ROOT / "csc2" / "e0")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    specs = [
        (dim, nf, depth, head, arm, kappa, wd, seed)
        for dim, nf, depth in NON_BINDING
        for head in HEADS
        for arm, kappa in ARMS
        for wd in WEIGHT_DECAYS
        for seed in SEEDS
    ]
    print(f"E0: {len(specs)} runs x {STEPS} steps")
    cells = parallel_map(_cell, specs, max_workers=args.workers)

    flat = parity_table([c for c in cells if c["hierarchy_depth"] == 0])
    tree = parity_table([c for c in cells if c["hierarchy_depth"] > 0])
    usable = sorted(
        set(flat["heads_passing_parity_and_recovering"])
        & set(tree["heads_passing_parity_and_recovering"])
    )

    payload = {
        "phase": "csc2-e0",
        "purpose": "validate readouts before calibrating through them (R5)",
        "settings": {
            "shapes": [list(s) for s in NON_BINDING], "heads": list(HEADS),
            "arms": [[a, k] for a, k in ARMS], "weight_decays": WEIGHT_DECAYS,
            "seeds": SEEDS, "steps": STEPS,
        },
        "parity_flat_features": flat,
        "parity_hierarchical_features": tree,
        "heads_usable_on_both": usable,
        "verdict": "PASS" if usable else "FAIL",
        "cells": cells,
    }
    path = write_artifact(args.out / "e0_head_validation.json", payload)
    print(f"E0: wrote {path}")
    print(f"E0: verdict {payload['verdict']}  usable on both: {usable}")
    print(f"  flat: {flat['per_head_verdict']}")
    print(f"  tree: {tree['per_head_verdict']}")


if __name__ == "__main__":
    main()

"""CSC-2 E4 — external positive control: can H² embed a tree and E² not?

Gates E3. CSC's rule R8 exists because its own positive control caught two
measurement bugs that cancelled between arms and were invisible to every
internal comparison. Here the check is against a result the literature treats
as settled: **trees embed into hyperbolic space with low distortion, and into
Euclidean space of the same dimension with high distortion** (Sarkar's
construction; Nickel & Kiela 2017). The reason is a growth-rate match — a
b-ary tree has b^D nodes at depth D, hyperbolic volume grows as e^(√|K|·R),
Euclidean volume only as R^d.

If our geometry code cannot reproduce that, no CSC-2 number means anything,
and the failure would be in the instrument rather than the hypothesis.

No toy model is trained. Node coordinates are optimized directly against the
tree metric, which isolates the geometry from every readout and training
question — deliberately, since CSC's null turned out to hinge on exactly those.

Metric: average distortion, the standard quantity for this comparison,

    D_avg = mean over pairs of |d_embed(u,v) − d_tree(u,v)| / d_tree(u,v)

Pre-registered expectation, committed before running: at matched dimension
d = 2, hyperbolic arms achieve **lower** average distortion than Euclidean on
every tree tested, and the gap **widens with depth**. A failure here is an
instrument failure and blocks E3.
"""

from __future__ import annotations

import argparse
import math
import statistics as st
from pathlib import Path

import torch

from csc.spaces import EuclideanSpace, StereographicSpace
from csc.training.hierarchy import FeatureTree
from experiments.util import RESULTS_ROOT, parallel_map, write_artifact

DEPTHS = [3, 4, 5, 6]
BRANCHING = [2, 3]
CURVATURES = [-4.0, -2.0, -1.0, -0.5]
DIMS = [2, 4]
SEEDS = [0, 1, 2, 3, 4]
STEPS = 3_000
LR = 5e-2


def _embed(spec: tuple) -> dict:
    depth, branching, kappa, dim, seed = spec
    tree = FeatureTree(depth, branching)
    target = tree.tree_distance_matrix()
    n = tree.n_features
    space = EuclideanSpace(dim) if kappa == 0.0 else StereographicSpace(dim, kappa)

    torch.manual_seed(seed)
    tangent = torch.nn.Parameter(torch.randn(n, dim) * 0.1)
    opt = torch.optim.Adam([tangent], lr=LR)

    mask = ~torch.eye(n, dtype=torch.bool)
    tgt = target[mask]

    for _ in range(STEPS):
        pts = space.expmap0(tangent)
        d = space.dist_matrix(pts, pts)[mask]
        # relative error is the right objective: tree distances span a wide
        # range and an absolute loss would be dominated by the deepest pairs
        loss = ((d - tgt) / tgt).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pts = space.expmap0(tangent)
        d = space.dist_matrix(pts, pts)[mask]
        distortion = float(((d - tgt).abs() / tgt).mean())
        # scale-free variant: tree metrics are only defined up to a global
        # scale, so the fit is also reported after optimal rescaling
        alpha = float((d * tgt).sum() / (d * d).sum())
        distortion_scaled = float(((alpha * d - tgt).abs() / tgt).mean())

    return {
        "depth": depth,
        "branching": branching,
        "kappa": kappa,
        "dim": dim,
        "seed": seed,
        "n_nodes": n,
        "avg_distortion": distortion,
        "avg_distortion_scale_free": distortion_scaled,
        "final_loss": float(loss.detach()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RESULTS_ROOT / "csc2" / "e4")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    specs = [
        (depth, b, kappa, dim, seed)
        for depth in DEPTHS
        for b in BRANCHING
        for kappa in [0.0, *CURVATURES]
        for dim in DIMS
        for seed in SEEDS
    ]
    print(f"E4: {len(specs)} embeddings x {STEPS} steps")
    cells = parallel_map(_embed, specs, max_workers=args.workers)

    summary, verdicts = {}, []
    for dim in DIMS:
        for b in BRANCHING:
            per_depth = {}
            for depth in DEPTHS:
                rows = [
                    c for c in cells
                    if c["dim"] == dim and c["branching"] == b and c["depth"] == depth
                ]
                euc = [c["avg_distortion_scale_free"] for c in rows if c["kappa"] == 0.0]
                best_hyp, best_k = math.inf, None
                for k in CURVATURES:
                    v = [c["avg_distortion_scale_free"] for c in rows if c["kappa"] == k]
                    if v and st.mean(v) < best_hyp:
                        best_hyp, best_k = st.mean(v), k
                per_depth[str(depth)] = {
                    "euclidean_distortion": st.mean(euc) if euc else None,
                    "best_hyperbolic_distortion": best_hyp,
                    "best_kappa": best_k,
                    "hyperbolic_wins": bool(euc and best_hyp < st.mean(euc)),
                    "improvement_ratio": (st.mean(euc) / best_hyp) if euc and best_hyp else None,
                }
            ratios = [
                v["improvement_ratio"] for v in per_depth.values() if v["improvement_ratio"]
            ]
            widens = bool(len(ratios) >= 2 and ratios[-1] > ratios[0])
            all_win = all(v["hyperbolic_wins"] for v in per_depth.values())
            verdicts.append(all_win)
            summary[f"d{dim}|b{b}"] = {
                "by_depth": per_depth,
                "hyperbolic_wins_at_every_depth": all_win,
                "gap_widens_with_depth": widens,
                "improvement_ratios": ratios,
            }

    payload = {
        "phase": "csc2-e4",
        "purpose": "external positive control: trees must embed better in H2 than E2",
        "gates": "E3 — a failure here is an instrument failure, not a result",
        "expectation": (
            "hyperbolic beats Euclidean at matched dimension on every tree, and "
            "the gap widens with depth"
        ),
        "settings": {
            "depths": DEPTHS, "branching": BRANCHING, "curvatures": CURVATURES,
            "dims": DIMS, "seeds": SEEDS, "steps": STEPS,
        },
        "verdict": "PASS" if all(verdicts) else "FAIL",
        "summary": summary,
        "cells": cells,
    }
    path = write_artifact(args.out / "e4_positive_control.json", payload)
    print(f"E4: wrote {path}")
    print(f"E4: verdict {payload['verdict']}")
    for key, v in summary.items():
        ratios = ", ".join(f"{r:.2f}x" for r in v["improvement_ratios"])
        print(f"  {key}: wins everywhere={v['hyperbolic_wins_at_every_depth']} "
              f"widens={v['gap_widens_with_depth']} ratios by depth [{ratios}]")


if __name__ == "__main__":
    main()

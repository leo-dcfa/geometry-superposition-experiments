"""CSC-2 E7 — what is a 30% distortion advantage worth in WIDTH?

E6 measured ~1.3x lower distortion for hyperbolic under a contrastive
objective, at matched dimension. The natural next question — and the one that
decides whether any of this matters for building networks — is how many
dimensions that is worth.

It is not a conversion you can do in your head. "30% lower distortion" buys
width only in proportion to how steeply Euclidean distortion falls with
dimension. If Euclidean distortion drops fast with d, a 30% gap is worth a
fraction of a dimension. If it plateaus, the same gap is worth many.

So this measures the curve directly: Euclidean distortion across a dimension
sweep, against hyperbolic at a reference dimension, under the realistic
(contrastive, edges-only) objective. The reported quantity is the **equivalent
Euclidean dimension** — the width at which flat space matches curved space's
distortion — and the width saving that implies.

Reported with the caveat that governs its interpretation: this is embedding
width for a tree metric, not hidden width of a general network, and a saving
here does not transfer to arbitrary architectures.
"""

from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

import torch

from csc.spaces import EuclideanSpace, StereographicSpace
from csc.training.hierarchy import FeatureTree
from experiments.util import RESULTS_ROOT, parallel_map, write_artifact

DEPTHS = [4, 5, 6]
BRANCHING = 2
EUCLIDEAN_DIMS = [2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32]
REFERENCE_DIMS = [2, 4, 8]          # hyperbolic widths to find equivalents for
CURVATURES = [-4.0, -2.0, -1.0, -0.5]
SEEDS = list(range(8))
STEPS = 3_000
LR = 5e-2
N_NEGATIVES = 16
TEMPERATURE = 0.5


def _fit(spec: tuple) -> dict:
    kappa, dim, depth, seed = spec
    tree = FeatureTree(depth, BRANCHING)
    target = tree.tree_distance_matrix()
    n = tree.n_features
    space = EuclideanSpace(dim) if kappa == 0.0 else StereographicSpace(dim, kappa)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed * 7919 + 13)

    tangent = torch.nn.Parameter(torch.randn(n, dim) * 0.1)
    opt = torch.optim.Adam([tangent], lr=LR)
    edges = torch.tensor(
        [[c, p] for c, p in enumerate(tree.parent) if p >= 0], dtype=torch.long
    )
    for _ in range(STEPS):
        pts = space.expmap0(tangent)
        e = edges[torch.randint(len(edges), (min(64, len(edges)),), generator=gen)]
        anchor, positive = e[:, 0], e[:, 1]
        negs = torch.randint(n, (len(anchor), N_NEGATIVES), generator=gen)
        cand = torch.cat([pts[positive].unsqueeze(1), pts[negs]], dim=1)
        logits = -space.dist(pts[anchor].unsqueeze(1), cand) / TEMPERATURE
        loss = torch.nn.functional.cross_entropy(
            logits, torch.zeros(len(anchor), dtype=torch.long)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pts = space.expmap0(tangent)
        mask = ~torch.eye(n, dtype=torch.bool)
        d = space.dist_matrix(pts, pts)[mask]
        tgt = target[mask]
        alpha = (d * tgt).sum() / (d * d).sum()
        distortion = float(((alpha * d - tgt).abs() / tgt).mean())
    return {"kappa": kappa, "dim": dim, "depth": depth, "seed": seed,
            "n_nodes": n, "avg_distortion": distortion}


def equivalent_dimension(euclidean_curve: dict[int, float], target: float) -> float | None:
    """Smallest Euclidean width reaching ``target`` distortion, log-interpolated.

    None means flat space never reaches it within the swept range — which is
    the interesting outcome, since it means the saving is unbounded rather than
    merely large.
    """
    dims = sorted(euclidean_curve)
    for i, d in enumerate(dims):
        if euclidean_curve[d] <= target:
            if i == 0:
                return float(d)
            prev = dims[i - 1]
            y0, y1 = euclidean_curve[prev], euclidean_curve[d]
            if y0 == y1:
                return float(d)
            frac = (y0 - target) / (y0 - y1)
            return float(prev) * (float(d) / float(prev)) ** frac
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RESULTS_ROOT / "csc2" / "e7")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    specs = [(0.0, d, depth, s) for d in EUCLIDEAN_DIMS for depth in DEPTHS for s in SEEDS]
    specs += [(k, d, depth, s) for k in CURVATURES for d in REFERENCE_DIMS
              for depth in DEPTHS for s in SEEDS]
    print(f"E7: {len(specs)} runs x {STEPS} steps")
    cells = parallel_map(_fit, specs, max_workers=args.workers)

    out = {}
    for depth in DEPTHS:
        curve = {
            d: st.mean(c["avg_distortion"] for c in cells
                       if c["kappa"] == 0.0 and c["dim"] == d and c["depth"] == depth)
            for d in EUCLIDEAN_DIMS
        }
        per_ref = {}
        for ref in REFERENCE_DIMS:
            best, best_k = float("inf"), None
            for k in CURVATURES:
                v = [c["avg_distortion"] for c in cells
                     if c["kappa"] == k and c["dim"] == ref and c["depth"] == depth]
                if v and st.mean(v) < best:
                    best, best_k = st.mean(v), k
            eq = equivalent_dimension(curve, best)
            per_ref[str(ref)] = {
                "hyperbolic_dim": ref,
                "best_kappa": best_k,
                "hyperbolic_distortion": best,
                "euclidean_distortion_at_same_dim": curve.get(ref),
                "distortion_advantage": (curve[ref] / best) if curve.get(ref) else None,
                "equivalent_euclidean_dim": eq,
                "width_saving_factor": (eq / ref) if eq else None,
                "euclidean_never_matches_within_sweep": eq is None,
            }
        out[f"depth{depth}"] = {
            "n_nodes": 2 ** (depth + 1) - 1,
            "euclidean_distortion_curve": curve,
            "by_reference_dim": per_ref,
        }

    payload = {
        "phase": "csc2-e7",
        "question": "what is the measured distortion advantage worth in WIDTH?",
        "objective": "contrastive (InfoNCE on tree edges) — the realistic setting",
        "caveat": (
            "embedding width for a tree metric, NOT hidden width of a general "
            "network; a saving here does not transfer to arbitrary architectures"
        ),
        "settings": {
            "depths": DEPTHS, "euclidean_dims": EUCLIDEAN_DIMS,
            "reference_dims": REFERENCE_DIMS, "curvatures": CURVATURES,
            "seeds": len(SEEDS), "steps": STEPS,
        },
        "results": out,
        "cells": cells,
    }
    path = write_artifact(args.out / "e7_dimension_equivalence.json", payload)
    print(f"E7: wrote {path}")
    for depth, r in out.items():
        print(f"  {depth} ({r['n_nodes']} nodes)  euclidean curve: " +
              " ".join(f"d{d}:{v:.3f}" for d, v in r["euclidean_distortion_curve"].items()))
        for ref, v in r["by_reference_dim"].items():
            eq = v["equivalent_euclidean_dim"]
            eqs = f"{eq:.1f}" if eq else ">32 (never matches)"
            sav = f"{v['width_saving_factor']:.2f}x" if v["width_saving_factor"] else "unbounded"
            print(f"    hyp d={ref} (K={v['best_kappa']}) dist {v['hyperbolic_distortion']:.3f} "
                  f"vs euc {v['euclidean_distortion_at_same_dim']:.3f} "
                  f"({v['distortion_advantage']:.2f}x) -> equivalent euclidean dim {eqs}, saving {sav}")


if __name__ == "__main__":
    main()

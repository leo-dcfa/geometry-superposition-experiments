"""CSC-2 E6 — does a *realistic* metric objective recover the advantage?

E5 located the break at the objective: metric-supervised stages kept a 4-5x
hyperbolic advantage, the reconstruction stage lost it entirely. But E5's
metric stages hand-supplied the full tree distance matrix, which nobody has in
practice. If the finding only holds when you already know every pairwise
distance, it is close to vacuous.

E6 asks the practical version. The only supervision is **which pairs are
related** — a contrastive (InfoNCE) objective over tree edges, with distance as
the similarity score. No distance matrix, no depth labels, no metric target:
just positives (parent/child) against sampled negatives. This is what someone
building a hyperbolic embedding would actually train.

Reconstruction is run as an arm in the same sweep, on identical data, model and
budget, so the comparison is within-experiment rather than across E5.

Pre-registered prediction, committed before running: **contrastive recovers a
substantial part of the advantage** (>2x at d=4) while reconstruction stays
near 1x. The mechanism claim says a metric-structured objective is what matters,
and contrastive is metric-structured — it scores distances, even though it never
sees a distance target.

If contrastive shows no advantage, the mechanism claim narrows sharply: only
*explicitly distance-supervised* objectives would qualify, and the finding
would be much less useful. That outcome is reported as readily as the other.

Two measures, because a contrastive objective can look good on its own loss
while not producing a faithful metric:
  - `avg_distortion` — scale-free fidelity to the tree metric, same quantity
    E4/E5 report, so numbers are comparable across all three experiments.
  - `ancestor_auc` — can a held-out ancestor/non-ancestor pair be ranked by
    embedded distance? A downstream task nobody trained for directly.
"""

from __future__ import annotations

import argparse
import math
import statistics as st
from pathlib import Path

import torch

from csc.spaces import EuclideanSpace, StereographicSpace
from csc.training.hierarchy import FeatureTree, sample_hierarchical_batch
from experiments.util import RESULTS_ROOT, parallel_map, write_artifact

DEPTHS = [3, 4, 5]
BRANCHING = 2
CURVATURES = [-4.0, -2.0, -1.0, -0.5]
DIMS = [2, 4]
SEEDS = list(range(10))
STEPS = 3_000
LR = 5e-2
N_NEGATIVES = 16
TEMPERATURE = 0.5
OBJECTIVES = ["contrastive", "reconstruction"]


def _space(kappa: float, dim: int):
    return EuclideanSpace(dim) if kappa == 0.0 else StereographicSpace(dim, kappa)


def _edges(tree: FeatureTree) -> torch.Tensor:
    """Parent/child pairs — the ONLY structural information the model receives."""
    return torch.tensor(
        [[c, p] for c, p in enumerate(tree.parent) if p >= 0], dtype=torch.long
    )


def _distortion(space, pts, target) -> float:
    n = pts.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    with torch.no_grad():
        d = space.dist_matrix(pts, pts)[mask]
        tgt = target[mask]
        alpha = (d * tgt).sum() / (d * d).sum()
        return float(((alpha * d - tgt).abs() / tgt).mean())


def _ancestor_auc(space, pts, tree) -> float:
    """Rank ancestor pairs closer than non-ancestor pairs. Never trained for."""
    n = tree.n_features
    anc = [set(tree.path_to_root(i)) for i in range(n)]
    with torch.no_grad():
        d = space.dist_matrix(pts, pts)
    pos, neg = [], []
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            (pos if (v in anc[u] or u in anc[v]) else neg).append(float(d[u, v]))
    if not pos or not neg:
        return float("nan")
    # P(random ancestor pair is closer than random non-ancestor pair)
    p = torch.tensor(pos)[:, None]
    q = torch.tensor(neg)[None, :]
    return float(((p < q).float() + 0.5 * (p == q).float()).mean())


def _cell(spec: tuple) -> dict:
    objective, depth, kappa, dim, seed = spec
    tree = FeatureTree(depth, BRANCHING)
    target = tree.tree_distance_matrix()
    n = tree.n_features
    space = _space(kappa, dim)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed * 7919 + 13)

    tangent = torch.nn.Parameter(torch.randn(n, dim) * 0.1)
    params = [tangent]
    if objective == "reconstruction":
        head_scale = torch.nn.Parameter(torch.ones(n))
        head_bias = torch.nn.Parameter(torch.full((n,), 1.5))
        params += [head_scale, head_bias]
    opt = torch.optim.Adam(params, lr=LR)

    edges = _edges(tree)
    loss = torch.tensor(0.0)
    for _ in range(STEPS):
        pts = space.expmap0(tangent)
        if objective == "contrastive":
            # InfoNCE over tree edges: the positive is a true parent/child, the
            # negatives are sampled nodes. Similarity is NEGATIVE distance, so
            # the objective scores the metric without ever naming a distance.
            e = edges[torch.randint(len(edges), (min(64, len(edges)),), generator=gen)]
            anchor, positive = e[:, 0], e[:, 1]
            negatives = torch.randint(n, (len(anchor), N_NEGATIVES), generator=gen)
            a = pts[anchor].unsqueeze(1)
            cand = torch.cat([pts[positive].unsqueeze(1), pts[negatives]], dim=1)
            d = space.dist(a, cand)
            logits = -d / TEMPERATURE
            loss = torch.nn.functional.cross_entropy(
                logits, torch.zeros(len(anchor), dtype=torch.long)
            )
        else:
            batch = sample_hierarchical_batch(256, tree, 0.5, 0.7, gen)
            enc = space.expmap0(batch @ tangent / max(1, n) * n)
            d = space.dist_matrix(enc, pts)
            mean_d = d.mean().clamp_min(1e-9)
            recon = torch.relu(head_scale * (-d / mean_d) + head_bias)
            loss = (recon - batch).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pts = space.expmap0(tangent)
    return {
        "objective": objective, "depth": depth, "kappa": kappa, "dim": dim,
        "seed": seed, "n_nodes": n,
        "avg_distortion": _distortion(space, pts, target),
        "ancestor_auc": _ancestor_auc(space, pts, tree),
        "final_loss": float(loss.detach()),
    }


def analyse(cells: list[dict]) -> dict:
    out = {}
    for obj in OBJECTIVES:
        for dim in DIMS:
            per_depth = {}
            for depth in DEPTHS:
                rows = [c for c in cells
                        if c["objective"] == obj and c["dim"] == dim and c["depth"] == depth]
                euc = [c["avg_distortion"] for c in rows if c["kappa"] == 0.0]
                euc_auc = [c["ancestor_auc"] for c in rows if c["kappa"] == 0.0]
                best, best_k, best_auc = math.inf, None, None
                for k in CURVATURES:
                    v = [c["avg_distortion"] for c in rows if c["kappa"] == k]
                    if v and st.mean(v) < best:
                        best, best_k = st.mean(v), k
                        best_auc = st.mean(c["ancestor_auc"] for c in rows if c["kappa"] == k)
                if not euc or best is math.inf:
                    continue
                per_depth[str(depth)] = {
                    "euclidean_distortion": st.mean(euc),
                    "best_hyperbolic_distortion": best,
                    "best_kappa": best_k,
                    "advantage": st.mean(euc) / best,
                    "euclidean_ancestor_auc": st.mean(euc_auc),
                    "best_hyperbolic_ancestor_auc": best_auc,
                }
            advs = [v["advantage"] for v in per_depth.values()]
            out[f"{obj}|d{dim}"] = {
                "by_depth": per_depth,
                "mean_advantage": st.mean(advs) if advs else None,
                "advantage_survives": bool(advs and st.mean(advs) > 1.2),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RESULTS_ROOT / "csc2" / "e6")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    specs = [
        (obj, depth, kappa, dim, seed)
        for obj in OBJECTIVES for depth in DEPTHS
        for kappa in [0.0, *CURVATURES] for dim in DIMS for seed in SEEDS
    ]
    print(f"E6: {len(specs)} runs x {STEPS} steps")
    cells = parallel_map(_cell, specs, max_workers=args.workers)
    analysis = analyse(cells)

    payload = {
        "phase": "csc2-e6",
        "question": "does a realistic metric objective (contrastive, edges only) recover the advantage?",
        "prediction_registered_before_run": (
            "contrastive recovers a substantial part of the advantage (>2x at d=4); "
            "reconstruction stays near 1x. If contrastive shows nothing, the "
            "mechanism claim narrows to explicitly distance-supervised objectives."
        ),
        "settings": {
            "objectives": OBJECTIVES, "depths": DEPTHS, "branching": BRANCHING,
            "curvatures": CURVATURES, "dims": DIMS, "seeds": len(SEEDS),
            "steps": STEPS, "n_negatives": N_NEGATIVES, "temperature": TEMPERATURE,
        },
        "analysis": analysis,
        "cells": cells,
    }
    path = write_artifact(args.out / "e6_contrastive.json", payload)
    print(f"E6: wrote {path}")
    for key, v in analysis.items():
        by_d = ", ".join(f"D{d}:{x['advantage']:.2f}x" for d, x in v["by_depth"].items())
        auc = [f"D{d}:{x['best_hyperbolic_ancestor_auc']:.2f}/{x['euclidean_ancestor_auc']:.2f}"
               for d, x in v["by_depth"].items()]
        print(f"  {key:26} mean advantage {v['mean_advantage']:.2f}x "
              f"survives={v['advantage_survives']}  [{by_d}]")
        print(f"  {'':26} ancestor AUC hyp/euc: {', '.join(auc)}")


if __name__ == "__main__":
    main()

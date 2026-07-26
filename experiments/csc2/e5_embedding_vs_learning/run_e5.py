"""CSC-2 E5 — where does the curvature advantage die between embedding and learning?

Two results in this repo contradict each other, using the same geometry code,
the same curvatures, and the same dimensions:

- **E4** optimizes coordinates directly against a target metric. Hyperbolic
  beats Euclidean by up to 9×, on every tree tested.
- **E1 and CSC-1** train a model with a distance readout on a reconstruction
  objective. The advantage is gone, and κ trends the wrong way.

Something between those setups destroys it. This interpolates between them in
four steps, changing **one thing at a time**, and asks where the advantage
disappears. Each step is a strict superset of the previous one's difficulties:

| stage | optimizes | supervision | data | objective |
|---|---|---|---|---|
| S1 `direct` | coordinates | full metric | — | distance match |
| S2 `readout` | encoder+prototypes | full metric | — | distance match |
| S3 `sampled` | encoder+prototypes | sampled pairs | batches | distance match |
| S4 `reconstruction` | encoder+prototypes | none | batches | reconstruct input |

S1 is E4. S4 is CSC-1. The stage at which the hyperbolic advantage vanishes
names the mechanism.

**Pre-registered prediction, committed before running: the advantage survives
S1–S3 and dies at S4.** The reconstruction objective never asks the model to
preserve distances — only to reconstruct activations — so a hyperbolic layout
is one of many that reconstruct equally well and nothing pushes the optimizer
toward it. The metric-structured objectives (S1–S3) do ask for exactly the
property hyperbolic space is good at.

If instead it dies at S2, the readout is the culprit — a distance-to-prototype
head cannot express what direct coordinate optimization can. If at S3, it is
stochastic sampling. If it survives all four, then CSC-1's null is about its
*task* (exchangeable features) rather than about learning, and E3 becomes worth
fixing and running after all.

Every stage embeds the SAME tree metric target, so "advantage" means the same
thing throughout: best hyperbolic arm ÷ Euclidean arm, on the metric each stage
is actually able to report.
"""

from __future__ import annotations

import argparse
import math
import statistics as st
from pathlib import Path

import torch

from csc.layers.readout import DistanceReadout
from csc.spaces import EuclideanSpace, StereographicSpace
from csc.training.hierarchy import FeatureTree, sample_hierarchical_batch
from experiments.util import RESULTS_ROOT, parallel_map, write_artifact

DEPTHS = [3, 4, 5]
BRANCHING = 2
CURVATURES = [-4.0, -2.0, -1.0, -0.5]
DIMS = [2, 4]
# 10, not 5. Measured in E4: hyperbolic embedding optimization has a 2.3-3.5x
# spread across seeds against Euclidean's 1.2x — it has far more local minima,
# so the mean needs more seeds to be stable. This is itself a finding and is
# reported rather than being quietly absorbed by averaging.
SEEDS = list(range(10))
STEPS = 3_000
LR = 5e-2
BATCH = 256
SPARSITY = 0.5  # denser than CSC's 0.9 so sampled stages see enough co-activation


def _space(kappa: float, dim: int):
    return EuclideanSpace(dim) if kappa == 0.0 else StereographicSpace(dim, kappa)


def _distortion(space, pts: torch.Tensor, target: torch.Tensor) -> float:
    """Scale-free average distortion — the same metric E4 reports."""
    n = pts.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=pts.device)
    with torch.no_grad():
        d = space.dist_matrix(pts, pts)[mask]
        tgt = target[mask]
        alpha = (d * tgt).sum() / (d * d).sum()
        return float(((alpha * d - tgt).abs() / tgt).mean())


def _stage_direct(space, tree, target, gen, dim, n) -> tuple[torch.Tensor, float]:
    """S1 = E4. Coordinates are the parameters; full metric is the target.

    Initialization deliberately uses the global RNG, exactly as E4 does, so
    that S1 is a genuine replication and not merely the same algorithm. An
    earlier version seeded it from a separate generator and landed in different
    local minima — which, given the seed sensitivity measured above, changed
    the answer by 4x. S1 must reproduce E4 or E5's interpolation has no anchor.
    """
    tangent = torch.nn.Parameter(torch.randn(n, dim) * 0.1)
    opt = torch.optim.Adam([tangent], lr=LR)
    mask = ~torch.eye(n, dtype=torch.bool)
    tgt = target[mask]
    for _ in range(STEPS):
        pts = space.expmap0(tangent)
        d = space.dist_matrix(pts, pts)[mask]
        loss = ((d - tgt) / tgt).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return space.expmap0(tangent).detach(), float(loss.detach())


def _model(dim: int, n: int, gen: torch.Generator, space):
    """Encoder + prototype readout — the architecture CSC-1 trained."""
    enc = torch.nn.Linear(n, dim)
    with torch.no_grad():
        enc.weight.copy_(torch.randn(dim, n) * 0.5)
        enc.bias.zero_()
    readout = DistanceReadout(space, n, init_scale=0.2)
    return enc, readout


def _stage_readout(space, tree, target, gen, dim, n) -> tuple[torch.Tensor, float]:
    """S2: prototypes reached through an encoder, still supervised on the full metric."""
    enc, readout = _model(dim, n, gen, space)
    opt = torch.optim.Adam(list(enc.parameters()) + list(readout.parameters()), lr=LR)
    mask = ~torch.eye(n, dtype=torch.bool)
    tgt = target[mask]
    eye = torch.eye(n)
    for _ in range(STEPS):
        pts = space.expmap0(enc(eye))
        d = space.dist_matrix(pts, pts)[mask]
        loss = ((d - tgt) / tgt).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pts = space.expmap0(enc(eye))
    return pts, float(loss.detach())


def _stage_sampled(space, tree, target, gen, dim, n) -> tuple[torch.Tensor, float]:
    """S3: same as S2 but supervised on sampled pairs from sampled batches."""
    enc, readout = _model(dim, n, gen, space)
    opt = torch.optim.Adam(list(enc.parameters()) + list(readout.parameters()), lr=LR)
    eye = torch.eye(n)
    loss = torch.tensor(0.0)
    for _ in range(STEPS):
        batch = sample_hierarchical_batch(BATCH, tree, SPARSITY, 0.7, gen)
        active = (batch > 0).any(0).nonzero().flatten()
        if active.numel() < 2:
            continue
        idx = active[torch.randperm(active.numel(), generator=gen)[:32]]
        pts = space.expmap0(enc(eye[idx]))
        sub = target[idx][:, idx]
        m = ~torch.eye(idx.numel(), dtype=torch.bool)
        d = space.dist_matrix(pts, pts)[m]
        tgt = sub[m].clamp_min(1e-6)
        loss = ((d - tgt) / tgt).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pts = space.expmap0(enc(eye))
    return pts, float(loss.detach())


def _stage_reconstruction(space, tree, target, gen, dim, n) -> tuple[torch.Tensor, float]:
    """S4 = CSC-1. No metric supervision at all; reconstruct the input."""
    from csc.layers.readout import ResponseHead

    enc, readout = _model(dim, n, gen, space)
    head = ResponseHead(n, kind="norm_affine")
    params = list(enc.parameters()) + list(readout.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    loss = torch.tensor(0.0)
    for _ in range(STEPS):
        batch = sample_hierarchical_batch(BATCH, tree, SPARSITY, 0.7, gen)
        pts = space.expmap0(enc(batch))
        recon = head(-space.dist_matrix(pts, readout.prototypes()))
        loss = (recon - batch).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pts = space.expmap0(enc(torch.eye(n)))
    return pts, float(loss.detach())


STAGES = {
    "S1_direct": _stage_direct,
    "S2_readout": _stage_readout,
    "S3_sampled": _stage_sampled,
    "S4_reconstruction": _stage_reconstruction,
}


def _cell(spec: tuple) -> dict:
    stage, depth, kappa, dim, seed = spec
    tree = FeatureTree(depth, BRANCHING)
    target = tree.tree_distance_matrix()
    n = tree.n_features
    space = _space(kappa, dim)
    # global seed for parameter init (matches E4); separate generator for data
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed * 7919 + 13)
    pts, final_loss = STAGES[stage](space, tree, target, gen, dim, n)
    return {
        "stage": stage,
        "depth": depth,
        "kappa": kappa,
        "dim": dim,
        "seed": seed,
        "n_nodes": n,
        # every stage is scored by the SAME metric, so the stages are comparable
        "avg_distortion": _distortion(space, pts, target),
        "final_loss": final_loss,
    }


def analyse(cells: list[dict]) -> dict:
    out = {}
    for stage in STAGES:
        for dim in DIMS:
            per_depth = {}
            for depth in DEPTHS:
                rows = [
                    c for c in cells
                    if c["stage"] == stage and c["dim"] == dim and c["depth"] == depth
                ]
                euc = [c["avg_distortion"] for c in rows if c["kappa"] == 0.0]
                best, best_k = math.inf, None
                for k in CURVATURES:
                    v = [c["avg_distortion"] for c in rows if c["kappa"] == k]
                    if v and st.mean(v) < best:
                        best, best_k = st.mean(v), k
                if not euc or best is math.inf:
                    continue
                per_depth[str(depth)] = {
                    "euclidean_distortion": st.mean(euc),
                    "best_hyperbolic_distortion": best,
                    "best_kappa": best_k,
                    # >1 means hyperbolic embeds the tree better
                    "advantage": st.mean(euc) / best,
                }
            advs = [v["advantage"] for v in per_depth.values()]
            out[f"{stage}|d{dim}"] = {
                "by_depth": per_depth,
                "mean_advantage": st.mean(advs) if advs else None,
                "advantage_survives": bool(advs and st.mean(advs) > 1.2),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RESULTS_ROOT / "csc2" / "e5")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    specs = [
        (stage, depth, kappa, dim, seed)
        for stage in STAGES
        for depth in DEPTHS
        for kappa in [0.0, *CURVATURES]
        for dim in DIMS
        for seed in SEEDS
    ]
    print(f"E5: {len(specs)} runs x {STEPS} steps")
    cells = parallel_map(_cell, specs, max_workers=args.workers)
    analysis = analyse(cells)

    order = [f"{s}|d{d}" for s in STAGES for d in DIMS]
    survives = {k: analysis[k]["advantage_survives"] for k in order if k in analysis}
    died_at = next((k for k in order if k in survives and not survives[k]), None)

    payload = {
        "phase": "csc2-e5",
        "question": "at which step between direct embedding and learned reconstruction does the curvature advantage die?",
        "prediction_registered_before_run": (
            "survives S1-S3, dies at S4: the reconstruction objective never asks "
            "for distances to be preserved, so nothing pushes the optimizer "
            "toward a hyperbolic layout"
        ),
        "settings": {
            "stages": list(STAGES), "depths": DEPTHS, "branching": BRANCHING,
            "curvatures": CURVATURES, "dims": DIMS, "seeds": SEEDS,
            "steps": STEPS, "sparsity": SPARSITY,
        },
        "advantage_survives_by_stage": survives,
        "first_stage_without_advantage": died_at,
        "analysis": analysis,
        "cells": cells,
    }
    path = write_artifact(args.out / "e5_embedding_vs_learning.json", payload)
    print(f"E5: wrote {path}")
    for key in order:
        if key in analysis:
            v = analysis[key]
            by_d = ", ".join(
                f"D{d}:{x['advantage']:.2f}x" for d, x in v["by_depth"].items()
            )
            print(f"  {key:26} mean advantage {v['mean_advantage']:.2f}x "
                  f"survives={v['advantage_survives']}  [{by_d}]")


if __name__ == "__main__":
    main()

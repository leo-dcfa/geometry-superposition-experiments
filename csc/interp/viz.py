"""Figures (SPEC §9): Poincaré-disk configurations and the capacity frontier.

Two plots, both deliberately built to be readable without the caption.

``poincare_prototypes`` draws the learned prototype configuration in the
hyperbolic disc, with the geodesic operating radius marked. This is the study's
hero-figure candidate: it shows *where the model actually put things*, which is
the one thing a scalar capacity number cannot convey.

``capacity_frontier`` plots recovered features against mean interference. This
is the more honest object than capacity alone. Superposition already lets a
model trade interference for feature count and slide along that curve for free
— so a curved arm sitting further along the *same* curve has bought nothing.
The claim H-MAIN needs is that curvature moves the curve itself.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Colour-blind-safe, and ordered so curvature reads as a gradient.
ARM_COLOURS = {
    "curved(K=-4)": "#0b3d91",
    "curved(K=-2)": "#3f6fd1",
    "curved(K=-1)": "#7fa6e8",
    "curved(K=-0.5)": "#b9cdf2",
    "euclidean": "#444444",
    "clamped": "#8a8a8a",
    "normalized": "#c0c0c0",
    "curved(K=+0.5)": "#f2c9b9",
    "curved(K=+1)": "#e8967f",
    "curved(K=+2)": "#d1603f",
}


def poincare_prototypes(model, path: Path, title: str = "") -> Path:
    """Prototype configuration in the Poincaré disc (2-D hyperbolic arms only)."""
    import torch

    space = model.space
    kappa = float(space.kappa)
    if kappa >= 0 or space.dim != 2:
        raise ValueError("the Poincaré-disc figure is for 2-D hyperbolic arms")

    with torch.no_grad():
        protos = space.expmap0(model.readout.prototypes_tangent).cpu().numpy()
        radii = space.radius(space.expmap0(model.readout.prototypes_tangent)).cpu().numpy()

    ball_radius = 2.0 / math.sqrt(abs(kappa))
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.add_patch(plt.Circle((0, 0), ball_radius, fill=False, lw=1.4, color="#222"))
    # the operating band, in coordinate units
    for x, style in ((0.5, ":"), (3.0, "--")):
        coord = math.tanh(x / 2) * ball_radius
        ax.add_patch(
            plt.Circle((0, 0), coord, fill=False, lw=0.9, ls=style, color="#888")
        )
    ax.scatter(protos[:, 0], protos[:, 1], s=26, c="#0b3d91", zorder=3, edgecolor="white",
               linewidth=0.4)
    lim = ball_radius * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        title or f"K = {kappa:+g}, {len(protos)} prototypes\n"
        f"median geodesic radius {np.median(radii):.2f}",
        fontsize=10,
    )
    ax.text(0, -lim * 0.97, "dotted/dashed: R2 operating band", ha="center", fontsize=7,
            color="#888")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def capacity_frontier(frontier: dict, path: Path, title: str = "") -> Path:
    """Recovered features vs mean interference, one point per arm.

    ``frontier`` maps arm label -> {"capacity": float, "interference": float}.
    An arm that is merely further along the shared trade-off is *not* evidence
    for H-MAIN; the label placement is meant to make that easy to see.
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for label, point in sorted(frontier.items()):
        colour = ARM_COLOURS.get(label, "#666")
        marker = "o" if label.startswith("curved") else "s"
        ax.scatter(
            point["interference"], point["capacity"],
            s=70, c=colour, marker=marker, edgecolor="white", linewidth=0.6, zorder=3,
        )
        ax.annotate(
            label, (point["interference"], point["capacity"]),
            textcoords="offset points", xytext=(6, 3), fontsize=7.5, color="#333",
        )
    ax.set_xlabel("mean cross-feature interference  (lower is better)")
    ax.set_ylabel("features recovered  (higher is better)")
    ax.set_title(title or "Capacity–interference frontier", fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.text(
        0.99, 0.02,
        "up-and-left = a better trade, not just a different point on it",
        transform=ax.transAxes, ha="right", fontsize=7.5, color="#777",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path

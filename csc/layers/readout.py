"""Distance-to-prototype readout. Rule R1 is structural here, not optional.

Logits are negative geodesic distances to learned prototypes; the response
head converts them to reconstructions. **Every** head carries a per-prototype
bias and a per-prototype scale, in every arm including the flat controls, and
there is no code path that removes them. This is rule R1: the parent
flagship's κ was the only per-token scalar in a bias-free distance softmax,
which let it learn the unigram prior and be read as geometry. A curved arm
whose advantage needs the absence of bias/scale is measuring bias duty.

Head choice is not a detail. The parent's 01b audit measured a *sign flip* in
a headline geometry result between two readouts, so CSC reports its primary
claim under two heads (the two-instrument rule, inherited). The fairness
requirement is that a head must not couple to a manifold's raw distance scale:
under the shared normalization d(0, expmap0(v)) = ‖v‖ all arms agree on radii
from the origin, but *pairwise* distances still diverge faster in hyperbolic
space, so a head reading raw d is not scale-neutral across κ.

- ``rbf`` — ReLU(scale·exp(−(d/τ)²) + bias), per-prototype learnable τ.
  Bounded in (0, 1] under every geometry, nonzero gradient at any distance
  (no dead-unit trap). Adapts its own length scale, so it does not import the
  arm's distance scale. Default.
- ``norm_affine`` — ReLU(scale·(−d/μ) + bias), μ the batch-mean distance.
  Scale-invariant by construction; μ is batch-dependent, which is the
  documented cost.
- ``softmax`` — ReLU(scale·softmax(−d²/T) + bias), one global temperature.
  Competitive normalization across prototypes.
- ``affine`` — ReLU(scale·(−d) + bias). The naive head, kept as the arm whose
  coupling to raw distance scale is the suspected confound; useful precisely
  because it is the one expected to flatter hyperbolic space for the wrong
  reason.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

HEADS = ("rbf", "norm_affine", "softmax", "affine")


class DistanceReadout(nn.Module):
    """Prototypes as tangent parameters; logits are negative geodesic distances."""

    def __init__(self, space, n_prototypes: int, init_scale: float = 0.2) -> None:
        super().__init__()
        self.space = space
        self.n_prototypes = n_prototypes
        self.prototypes_tangent = nn.Parameter(torch.randn(n_prototypes, space.dim) * init_scale)

    def prototypes(self) -> Tensor:
        """Prototype points on the manifold, ``(n_prototypes, dim)``."""
        return self.space.expmap0(self.prototypes_tangent)

    def forward(self, x: Tensor) -> Tensor:
        """``(batch, dim) -> (batch, n_prototypes)`` negative geodesic distances."""
        return -self.space.dist_matrix(x, self.prototypes())


class ResponseHead(nn.Module):
    """Distances -> reconstructions. Always per-prototype bias AND scale (R1)."""

    def __init__(self, n_prototypes: int, kind: str = "rbf") -> None:
        super().__init__()
        if kind not in HEADS:
            raise ValueError(f"unknown head {kind!r}; expected one of {HEADS}")
        self.kind = kind
        self.n_prototypes = n_prototypes
        # R1: both are unconditional. Do not make either optional.
        self.scale = nn.Parameter(torch.ones(n_prototypes))
        self.bias = nn.Parameter(self._init_bias(kind, n_prototypes))
        if kind == "rbf":
            self.log_tau = nn.Parameter(torch.zeros(n_prototypes))
        elif kind == "softmax":
            self.log_temp = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _init_bias(kind: str, n: int) -> Tensor:
        # Logits are negative distances, so a zero bias puts every
        # pre-activation below the ReLU cut and kills the gradient at step 0.
        # Each head's init is chosen to start units alive under every arm; the
        # 00c parity fixture verifies that it does so *equally* across arms.
        if kind == "affine":
            return torch.ones(n)
        if kind == "norm_affine":
            return torch.full((n,), 1.5)  # normalized distances sit near 1
        return torch.zeros(n)  # rbf/softmax responses are already in [0, 1]

    def forward(self, logits: Tensor) -> Tensor:
        if self.kind == "rbf":
            kernel = torch.exp(-((logits / torch.exp(self.log_tau)).square()))
            return torch.relu(self.scale * kernel + self.bias)
        if self.kind == "norm_affine":
            mean_dist = (-logits).mean().clamp_min(1e-9)
            return torch.relu(self.scale * (logits / mean_dist) + self.bias)
        if self.kind == "softmax":
            weights = torch.softmax(-logits.square() / torch.exp(self.log_temp), dim=-1)
            return torch.relu(self.scale * weights + self.bias)
        return torch.relu(self.scale * logits + self.bias)

    def extra_repr(self) -> str:
        return f"kind={self.kind}, n_prototypes={self.n_prototypes}, bias+scale=always (R1)"

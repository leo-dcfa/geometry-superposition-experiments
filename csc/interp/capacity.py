"""Capacity, recovery and interference metrics.

All metrics are *behavioural* (probe-based) rather than weight-based, so they
mean the same thing in every geometry: a feature is recovered if the model
reconstructs a lone probe of it, and interference is the readout's off-target
response. A weight-space metric would not be comparable across arms, because
the arms differ precisely in how weights map to distances.

Rule R6 note: ``features_recovered`` uses a *relative* tolerance on the probe
value. An absolute tolerance would silently favour whichever arm happens to
have the larger response scale, and scale is a free per-prototype parameter in
every arm by R1.
"""

from __future__ import annotations

import torch
from torch import Tensor


@torch.no_grad()
def probe_metrics(model, value: float = 0.8, tol: float = 0.2) -> dict:
    """Single-feature probes x = value·e_i, for every feature i.

    ``tol`` is the ε of H-MAIN(toy): a feature counts as recovered when its
    own reconstruction is within a relative ``tol`` of the probe value. The
    value used in hypothesis tests is sealed in VALIDATION.md.
    """
    device = next(model.parameters()).device
    probes = torch.eye(model.n_features, device=device) * value
    response = model(probes)  # (nf, nf): row i is the reconstruction of probe i
    on_target = response.diagonal()
    off_target = response - torch.diag_embed(on_target)
    rel_error = (on_target - value).abs() / value
    recovered = rel_error < tol
    return {
        "n_features": int(model.n_features),
        "features_recovered": int(recovered.sum()),
        "recovery_rate": float(recovered.float().mean()),
        "recovered_mask": recovered.cpu(),
        "probe_rel_error_mean": float(rel_error.mean()),
        "probe_rel_error_median": float(rel_error.median()),
        "interference_mean": float(off_target.abs().mean()),
        "interference_max": float(off_target.abs().max()),
    }


@torch.no_grad()
def prototype_geometry(model) -> dict:
    """Geodesic structure of the learned prototype configuration.

    ``min_pairwise_distance`` is the Phase-1 metric predicted to hold a floor
    in H² and collapse in E² as N grows: a hyperbolic disc has room to keep
    prototypes apart, a Euclidean one does not.
    """
    protos = model.readout.prototypes()
    d = model.space.dist_matrix(protos, protos)
    n = d.shape[0]
    off_diag = d[~torch.eye(n, dtype=torch.bool, device=d.device)]
    return {
        "min_pairwise_distance": float(off_diag.min()),
        "mean_pairwise_distance": float(off_diag.mean()),
        "median_pairwise_distance": float(off_diag.median()),
        "max_pairwise_distance": float(off_diag.max()),
        "prototype_radius_median": float(model.space.radius(protos).median()),
    }


@torch.no_grad()
def interference_matrix(model, value: float = 0.8) -> Tensor:
    """Full (N, N) probe-response matrix for P3; diagonal is on-target."""
    device = next(model.parameters()).device
    probes = torch.eye(model.n_features, device=device) * value
    return model(probes).cpu()


@torch.no_grad()
def dead_unit_fraction(model, batch: Tensor) -> float:
    """Fraction of readout units that never activate across a whole batch.

    The 00c parity fixture (ported from the parent's 01b census). A geometry
    arm whose units die at a materially higher rate than another's has an
    invalid comparison, not a lost one — the difference is in the readout's
    initialization interacting with that arm's distance scale, and it must be
    cured before any capacity number is read.
    """
    response = model(batch)
    ever_active = (response > 0).any(dim=0)
    return float((~ever_active).float().mean())


@torch.no_grad()
def response_scale_report(model, batch: Tensor) -> dict:
    """Readout operating regime — the 'is the head whispering?' diagnostics.

    The parent program learned twice that a head can be measured to be loud or
    quiet rather than assumed, and that init selects the solution basin. These
    numbers are logged for every arm so that a capacity difference can be
    checked against a mere difference in how hard each head is driven.
    """
    logits = model.readout(model.encode(batch))  # negative distances
    response = model.head(logits)
    return {
        "logit_mean": float(logits.mean()),
        "logit_std": float(logits.std()),
        "response_mean": float(response.mean()),
        "response_std": float(response.std()),
        "head_scale_abs_mean": float(model.head.scale.abs().mean()),
        "head_bias_mean": float(model.head.bias.mean()),
    }


def capacity_from_sweep(recovery_by_n: dict[int, float], threshold: float = 0.9) -> int | None:
    """N*(κ): the largest N whose recovery rate meets ``threshold``.

    Returns ``None`` when even the smallest N misses the threshold — reported
    as a miss rather than silently coerced to 0 (R6). The scan stops at the
    first failure so that a non-monotone recovery curve cannot inflate N*.
    """
    best = None
    for n in sorted(recovery_by_n):
        if recovery_by_n[n] >= threshold:
            best = n
        else:
            break
    return best

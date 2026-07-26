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
def probe_metrics(model, value: float = 0.8, tol: float = 0.2, context: Tensor | None = None) -> dict:
    """Single-feature probes x = value·e_i, for every feature i.

    ``tol`` is the ε of H-MAIN(toy): a feature counts as recovered when its
    own reconstruction is within a relative ``tol`` of the probe value. The
    value used in hypothesis tests is sealed in VALIDATION.md.

    ``context`` is not optional in spirit. The primary readout (`norm_affine`)
    divides distances by the **batch mean**, so evaluating the probes as their
    own tiny batch changes the readout's normalization and mis-scales its
    output. Measured on one trained model (00d): probes alone reconstructed a
    0.8 target at 0.60 for N=2 — a 25% error, reported as "0 features
    recovered" — while the *same model* with the same probes embedded in a
    realistic batch reconstructed 0.795. At N=8 the bias reverses and reads
    high (0.91 vs 0.84).

    A distortion whose size *and direction* depend on N is disqualifying here,
    because N-dependence is exactly what H-MAIN(toy) and P1 measure. So probes
    are evaluated inside a fixed background batch drawn from the training
    distribution, and only the probe rows are read. Passing ``context=None``
    reproduces the old, batch-size-dependent behaviour and is kept only so the
    bug can be regression-tested.
    """
    device = next(model.parameters()).device
    probes = torch.eye(model.n_features, device=device) * value
    if context is None:
        response = model(probes)
    else:
        n = probes.shape[0]
        response = model(torch.cat([probes, context.to(device)]))[:n]
    # (nf, nf): row i is the reconstruction of probe i
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
def interference_matrix(model, value: float = 0.8, context: Tensor | None = None) -> Tensor:
    """Full (N, N) probe-response matrix for P3; diagonal is on-target.

    Takes ``context`` for the same reason ``probe_metrics`` does — P3's
    contamination numbers are read off this matrix and would inherit the
    batch-normalization distortion otherwise.
    """
    device = next(model.parameters()).device
    probes = torch.eye(model.n_features, device=device) * value
    if context is None:
        return model(probes).cpu()
    n = probes.shape[0]
    return model(torch.cat([probes, context.to(device)]))[:n].cpu()


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
    """N*: the top of the first contiguous run of feature counts that all pass.

    SPEC §5 defines N* as "max N with ≥ 90% of features recovered". Taken
    literally, one lucky cell far up the sweep sets N*, so the scan requires a
    *contiguous* passing run — but it starts that run at the first N that
    passes rather than at the smallest N in the grid.

    The difference is not academic. An earlier version began at the smallest N
    and stopped at the first failure, which returned ``None`` for four of six
    sparsity levels in the 00d control: N=2 was failing for a reason unrelated
    to capacity (a probe-metric bug, since fixed), and one degenerate cell at
    the bottom of the grid silently erased N* everywhere above it. A capacity
    metric must not be hostage to its smallest grid point.

    Returns ``None`` only when no N passes at all — reported as a miss rather
    than coerced to 0 (R6).
    """
    passing = {n: recovery_by_n[n] >= threshold for n in sorted(recovery_by_n)}
    best = None
    for n, ok in passing.items():
        if ok:
            best = n
        elif best is not None:
            break  # the contiguous run has ended
    return best

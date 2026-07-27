# Study 3 — Norms: **primary prediction falsified; counter-prediction confirmed**

2240 runs, 10 seeds, 7 arms, all flat (curvature exactly zero), differing only
in the norm. Both predictions were registered before the run.

## The registered predictions, and what happened

> **Primary:** ℓ^∞ beats ℓ² — its unit ball is a hypercube with 2^d vertices,
> exponentially many maximally separated directions with no radial blow-up.

**FALSIFIED.** ℓ^∞ is the *worst* arm tested. Capacity relative to the ℓ²
control, across all eight head × dim × gain cells:

| arm | advantage vs ℓ² |
|---|---|
| ℓ^1 | 0.87 – 0.97× |
| ℓ^1.5 | 0.96 – 1.03× |
| **ℓ² (control)** | **1.00×** |
| ℓ^3 | 0.93 – 1.02× |
| **ℓ^∞** | **0.75 – 0.94×** |
| Finsler (learned anisotropy) | 0.98 – 1.03× |
| normalized (flat, radius discarded) | **1.03 – 1.15×** |

The permutation test for "ℓ^∞ beats ℓ²" returns **p = 1.000 in every cell** —
not a near miss, the opposite ordering. Recorded as a MISS per R6.

> **Counter-prediction:** flat ℓ² capacity is already unreached, so the binding
> constraint is readout resolution and optimization rather than geometry, and no
> norm will help.

**CONFIRMED**, and this is the result.

## The measurement that settles it

The Welch bound gives the smallest achievable max-|cos| among N unit vectors in
R^d — the packing limit. Measured against it:

| arm | achieved max \|cos\| | best possible | ratio |
|---|---|---|---|
| ℓ^1 | 0.981 | 0.394 | 2.61× |
| ℓ^1.5 | 0.987 | 0.394 | 2.63× |
| ℓ² | 0.986 | 0.394 | 2.63× |
| ℓ^3 | 0.989 | 0.394 | 2.64× |
| ℓ^∞ | 0.977 | 0.394 | 2.59× |
| Finsler | 0.987 | 0.394 | 2.63× |
| normalized | 0.981 | 0.394 | 2.62× |

**Every arm lands at ≈2.6× the packing limit, and the spread between them is
under 2%.** A max-|cos| of ~0.98 means some prototype pairs are nearly
*collinear* while the geometry permits 0.39.

The models are nowhere near what flat space already allows. Changing the norm
cannot help with a constraint that was never geometric — which is exactly why
every arm lands in the same place regardless of its unit-ball shape.

## Why ℓ^∞ actively loses

The hypercube's 2^d vertices are real, but reaching them requires *coordinated*
extremal placement — every coordinate simultaneously at ±1. Gradient descent
through a distance readout has no mechanism to find that structure, and ℓ^∞'s
gradient is worse behaved on the way: the max is attained by a single
coordinate, so only one coordinate per pair receives gradient at a time. The
combinatorial capacity exists and is unreachable by this optimizer, which is
the same shape of failure as Study 1's hyperbolic volume: **available room that
the learning process cannot address.**

## What this closes

Three studies now agree, from three different directions:

1. **Study 1** — no curvature helps. Capacity is angular; curvature moves the
   radial coordinate.
2. **Study 2** — the advantage that *does* exist in direct embedding dies at
   the training objective, and scales with how directly the objective specifies
   distances.
3. **Study 3** — no norm helps either, and the packing measurement says why:
   models sit 2.6× from the limit in every geometry tested.

Taken together: **the geometry of the representation space was never the
binding constraint on superposition capacity in this setting.** The line of
enquiry "find a better space" is closed across curvature, norms, and
anisotropy. What binds is the readout's ability to resolve small angles and the
optimizer's ability to find good packings.

The one arm that consistently *does* beat the control — `normalized`, at
1.03–1.15× — is not a geometry change at all. It discards the radial
coordinate, i.e. it removes a degree of freedom the task does not use. That is
the third study in a row in which a flat control wins.

## Limitations

- One task family (TMS-style sparse autoencoding), one readout family
  (distance-to-prototype), d ≤ 8. The claim is about this setting.
- ℓ^∞'s failure is attributed to optimizer/readout reachability; that is an
  explanation consistent with the data, not an isolated demonstration. A
  discrete or annealed optimizer might reach the hypercube vertices.
- The Finsler arm carries `dim` extra parameters (reported per arm); at
  0.98–1.03× it neither gained from them nor was harmed, so parameter matching
  does not change any conclusion here.

## Where this points

Away from spaces, toward readouts. Concretely: the 2.6× packing gap is a
measurable target. Anything that closes it — a readout that resolves small
angles, an objective that penalizes coherence directly, a better optimizer for
prototype placement — should raise capacity in *every* geometry, including flat.
That is a sharper and more useful experiment than any further geometry sweep.

## Artifact

`study3_norm_capacity.json` — 2240 runs: 7 arms × 2 dims × 4 feature counts ×
2 gains × 2 heads × 10 seeds, with per-run coherence and Welch bound.

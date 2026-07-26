# CSC-2 overnight run — E0 FAIL, E4 PASS, E1 informative, **E3 invalid**

5920 training runs + 400 embeddings, 21:05 → 23:00. Two stages produce usable
results, one is informative, and **E3 — the main event — is invalid because of
a defect in my feature generator.** E3's numbers are not reported as evidence.

| stage | verdict | usable? |
|---|---|---|
| E0 readout validation | **FAIL** on hierarchical data | yes — and it correctly gated E3 |
| E4 external positive control | **PASS** | yes, and strongly |
| E1 readout-scale | **NULL IS ROBUST** | yes |
| E3 hierarchy | — | **no — confounded design** |

---

## E4 — the positive control passes, and reproduces the literature

Trees embed into H² at lower distortion than E² at matched dimension, on every
tree tested, and the gap widens with depth in 3 of 4 configurations:

| config | improvement ratio (E² distortion ÷ best H² distortion) by depth 3→6 |
|---|---|
| d=2, b=2 | 2.26× → 1.70× → 1.48× → 1.37× (narrows) |
| d=2, b=3 | 1.30× → 1.64× → 1.75× → **1.84×** |
| d=4, b=2 | 3.39× → 5.07× → 6.32× → **9.01×** |
| d=4, b=3 | 7.48× → 9.69× → 11.65× → 10.83× |

**The instrument can see a hyperbolic advantage when one exists.** That is the
single most important thing this run establishes, because it removes the "our
code just can't detect curvature benefits" explanation for CSC's null. It also
shows an optimum in κ — at depth 4, d=2: E² 0.1595, κ=−1 **0.0657**, κ=−4
0.2045 — so too much curvature is as bad as none, which is H2-c's prediction.

Note what E4 does *not* involve: no toy model, no readout, no training loop.
Coordinates are optimized directly against the tree metric. The advantage lives
in the geometry; whether a *learned model with a readout* can capture it is a
separate question, and the rest of the run bears on it.

## E1 — CSC's null is robust to the readout

The cancellation account (hyperbolic geometry inflates the batch-mean distance
that relative-scale readouts divide by, so room and resolution cancel) predicted
that an absolute-scale readout would show an advantage. It does not.

| head | family | max hyperbolic advantage |
|---|---|---|
| norm_affine (d4) | relative | 1.357 |
| norm_affine (d8) | relative | 1.076 |
| softmax (d4/d8) | relative | 0.993 / 1.005 |
| abs_rbf (d4/d8) | **absolute** | 0.964 / 0.992 |
| affine (d4/d8) | **absolute** | 1.148 / 0.962 |

No absolute-scale head exceeds 1.15×, and two are *below* 1.0. **The
cancellation hypothesis is not supported**, and CSC's null stands as a fact
about the geometry-plus-task rather than about a particular readout.

More striking: for every absolute-scale head, ρ(κ, capacity) is **positive and
consistent across all gain levels** (+0.51 to +0.85). Capacity *increases* with
κ — spherical better, hyperbolic worse — stably, under both dimensions and both
absolute heads. Under CSC's SPEC §1 that direction falsifies H-MAIN rather than
merely failing to support it. It is also what the angular-capacity account
predicts: positive curvature makes the space smaller and more sphere-like, and
a sphere is exactly where near-orthogonal directions are cheap.

## E0 — the readout gate failed, and its failure is a real finding

No head is usable on hierarchical data. On non-binding shapes (d=8, 7 and 15
tree nodes), best recovery by head: norm_affine 0.31, softmax 0.25, abs_rbf
0.19, affine 0.09, rbf 0.07 — against the ≥0.9 the fixture requires.

`abs_rbf` — the head built for this study to repair `rbf`'s collapse — is
disqualified on both flat and tree data. The repair did not work. That is what
E0 is for, and it is why it runs before anything is calibrated through a head.

## E3 — invalid: depth is confounded with density

**The defect is in my generator, and I found it after the run.** The
hierarchical sampler activates one root-to-leaf path and then applies dropout
at rate `sparsity`. So the number of active features is set by *path length*,
not by feature count — while the flat comparison at the same nominal sparsity
activates a fraction of *all N features*:

| condition | nominal sparsity | mean active features |
|---|---|---|
| flat, N=31 (the depth-0 anchor) | 0.9 | **3.07** |
| tree, depth 2 (N=7) | 0.9 | 0.29 |
| tree, depth 3 (N=15) | 0.9 | 0.38 |
| tree, depth 4 (N=31) | 0.9 | 0.51 |
| tree, depth 5 (N=63) | 0.9 | 0.58 |

The depth-0 anchor sees **5–10× denser input** than any tree condition, and
density rises with depth across the tree conditions. So E3 varied depth,
density, and feature count together. Its "advantage by depth" numbers
(0.88–1.22, no consistent trend) cannot be attributed to hierarchy.

E3 also inherits E0's failure: it was gated on E0 and E0 failed. The driver
continues past failures by design so that later stages remain inspectable, but
the gate stands — **E3's numbers are not evidence**, and the anchor "holding"
at ≈1.0 is not reassurance, because a confounded comparison can land anywhere.

A second, independent problem: the recovery metric probes with one-hot inputs,
which are **off-distribution** for hierarchical data — a child never appears
without its ancestors. Measured on one trained model: one-hot probe recovery
1.00, while in-distribution *path* probe recovery was 0.31. The metric and the
data disagree about what the model is for.

## What has to change before E3 can be run

1. **Match density across depths.** Set the per-sample dropout so mean active
   features is constant across depth and against the flat anchor, rather than
   fixing nominal sparsity. Density must be a controlled variable, not a
   consequence of tree shape.
2. **Match feature count too**, or vary it independently. Depth-2 has 7 nodes
   and depth-5 has 63; capacity is known to depend on N, so N must not ride
   along with depth.
3. **Replace the recovery metric for hierarchical data.** Probe on-distribution
   (whole paths), or score reconstruction of held-out samples from the actual
   data distribution. A metric that only reads one-hot inputs is measuring
   behaviour the model was never trained for.
4. **Find a readout that passes E0 on tree data**, or accept that all current
   heads fail there and treat that as the finding.

## Standing after tonight

The reframing that motivated CSC-2 — that curvature should help for
*hierarchical* data — is **untested**, not supported and not refuted. What the
night did establish:

- The geometry advantage is real and our instrument can measure it (E4).
- It does not survive being routed through a learned model with a distance
  readout on exchangeable features, under any of four readouts spanning both
  scale families (E1, and CSC-1 before it).
- The gap between those two statements is now the interesting question, and it
  is sharper than when the night began: **the advantage exists in the embedding
  problem and disappears in the learning problem.**

## Artifacts

`e0/e0_head_validation.json`, `e4/e4_positive_control.json`,
`e1/e1_readout_scale.json`, `e3/e3_hierarchy.json` (retained, marked invalid).

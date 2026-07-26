# Phase 1b — gain × curvature factorial: **P1 not supported; G1 fails**

3000 runs, gain set explicitly rather than derived from a calibration rule, so
curvature and the optimization knob vary independently.

Both pre-registered readings fire. **Phase 1's κ-trend is retracted as
confounded, and the clean test does not support P1.** Per SPEC's G1 — "any
failure → stop, write up the null" — **Phase 2 does not open.**

## Reading 1: the optimization knob drives capacity — retraction confirmed

Criterion, registered before the run: if flat-arm capacity varies with gain at
|ρ| > 0.3, Phase 1's κ-trend is confounded regardless of the curved arms.

| head \| dim | ρ(gain, capacity) in the **flat** arm | |
|---|---|---|
| norm_affine \| d4 | **−0.616** | fires |
| norm_affine \| d6 | −0.295 | |
| norm_affine \| d8 | −0.224 | |
| softmax \| d4 | −0.011 | |
| softmax \| d6 | **+0.467** | fires |
| softmax \| d8 | **+0.442** | fires |

It fires in 3 of 6 cells — and note the **sign differs by readout**:
`norm_affine` loses capacity as gain rises, `softmax` gains it. An init-scale
knob with no geometric content moves capacity by ~30% (d=4 flat: 10.0 → 7.0),
which is comparable to or larger than any curvature difference observed
anywhere in this study.

## Reading 2: the κ-trend is not a stable property

Criterion: the κ-trend computed *within* each gain level must be sign-consistent
across gain levels and heads to count as evidence.

| head \| dim | ρ(κ, capacity) at gain 1.5 → 9.0 | consistent |
|---|---|---|
| norm_affine \| d4 | +0.65 +0.69 +0.19 +0.19 −0.03 | **no** |
| norm_affine \| d6 | +0.51 +0.71 +0.09 −0.49 −0.18 | **no** |
| norm_affine \| d8 | +0.66 +0.45 −0.05 −0.24 −0.10 | **no** |
| softmax \| d4 | +0.21 +0.16 +0.27 +0.17 +0.12 | yes (positive) |
| softmax \| d6 | −0.23 −0.20 −0.35 −0.30 −0.13 | yes (negative) |
| softmax \| d8 | −0.17 −0.25 −0.27 −0.32 −0.36 | yes (negative) |

The sign flips **with gain** (norm_affine), flips **with dimension** (softmax:
positive at d=4, negative at d=6 and d=8), and **differs between readouts** at
every dimension. Concretely, at d=8:

- gain 1.5 → capacity rises from K=−4 (15.4) to K=+2 (19.0): *spherical looks better*
- gain 9.0 → K=−1 is near the top (18.3), K=+0.5 near the bottom (15.6): *reversed*

Same architecture, same data, opposite conclusion, from the init scale alone.

## The regime was reached — this is not a power failure

The obvious escape is that the arms never got into the regime where curvature
predicts anything. They did. At d=8 the hyperbolic arms realized
x = √|K|·r of 1.44–1.73, where the volume argument predicts:

| arm | realized x | predicted V_H/V_E |
|---|---|---|
| K=−4 | 1.66 | **12.1×** |
| K=−2 | 1.73 | **14.7×** |
| K=−1 | 1.61 | **10.3×** |
| K=−0.5 | 1.44 | 6.6× |

Predicted 7–15×. Observed: nothing — the hyperbolic arms sit at or slightly
**below** Euclidean. The volume-based prediction is wrong by an order of
magnitude, in a regime it was supposed to describe.

## Why: the extra room exists and the model does not use it

SPEC §5 predicted the minimum pairwise prototype distance would "hold a floor
in H², collapse in E²". Measured at d=8, gain 9 — **exactly backwards**:

| arm | min pairwise distance | capacity |
|---|---|---|
| euclidean | **0.2088** | 17.0 |
| clamped | 0.1393 | 10.9 |
| curved(K=+1) | 0.1377 | 16.8 |
| normalized | 0.1075 | 18.9 |
| curved(K=−0.5) | 0.0302 | 17.8 |
| curved(K=−1) | 0.0081 | 18.3 |
| curved(K=−2) | 0.0056 | 17.5 |
| curved(K=−4) | **0.0005** | 18.0 |

Prototypes in the most strongly curved arm sit **400× closer together** than in
flat space, while having 12× more volume available. The room is there; the
optimizer does not spread into it.

**Leading explanation, offered as a hypothesis rather than a result.** The
chain "curvature → volume → capacity" breaks at the second link, and plausibly
because the readout's resolution is *relative*. `norm_affine` divides distances
by the batch mean. Hyperbolic geometry inflates distances between far points
exponentially — which is the same property that creates the volume — so the
mean inflates too, and near pairs shrink in relative terms. Extra room arrives
together with a matching loss of discriminability, and the two cancel.

If that is right, the effect should depend on the readout having no absolute
length scale, and a readout with a fixed absolute scale should behave
differently. That is a concrete, cheap next experiment. It is also the natural
reading of `rbf`'s per-prototype learnable τ — the one head with an absolute
scale — which was disqualified in 00c for unrelated reasons and would need
repairing first.

## The frontier: no curved arm beats flat

Capacity against interference at d=8 (up-and-left is a better trade, not just a
different point on the same one):

| arm | capacity | interference |
|---|---|---|
| **normalized** (flat control) | **19.54** | **0.0136** |
| curved(K=+2) | 17.34 | 0.0182 |
| euclidean | 17.18 | 0.0194 |
| curved(K=−1) | 17.12 | 0.0186 |
| curved(K=−4) | 16.49 | 0.0175 |
| clamped | 14.32 | 0.0173 |

**The best arm on both axes simultaneously is a flat R3 control** —
normalized-Euclidean — in both readouts. Not any curved arm. This speaks
directly to the standing H1b worry: whatever a bounded-diameter geometry might
offer appears to be obtained more cheaply by plain normalization, without
curvature.

## Limitations, stated plainly

- Toy scale, d ≤ 8, N = 64, one task, one architecture family. This is not a
  result about language models.
- The mechanism above is a hypothesis consistent with the data, not something
  this run isolates.
- It remains possible that a readout with an absolute length scale, or a
  training scheme that forces prototypes apart, would let a curved space
  realize its volume advantage. Nothing here rules that out — it rules out the
  claim that curvature buys capacity *by itself* in this setting.
- `clamped` performs poorly here (14.32) with a hard clip at 3.0; that control's
  known zero-gradient asymmetry (see `spaces/controls.py`) may account for it,
  so F1.3 should not be read off this table.

## Artifact

`p1b_factorial.json` — 3000 runs, all gains × arms × dims × seeds, per-seed.
26 of 3000 runs R2-flagged (0.9%), comfortably inside G1's ≥95%-clean bar, so
unlike Phase 1 no selection concern applies here.

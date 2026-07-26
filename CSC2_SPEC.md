# CSC-2: Curvature and Hierarchical Capacity
**Spec v0.1 — DRAFT, not sealed. A new study, not a continuation of CSC.**

CSC tested H-MAIN — that negative curvature buys representational room —
pre-registered it, and did not support it (`VALIDATION.md` §0). CSC-2 is a
*different hypothesis*. Presenting it as a rescue of the old one would be
exactly the goalpost drift CSC §1 warned against, so the relationship is stated
once, here, and not blurred afterwards: **CSC's null is CSC-2's depth-0 data
point.**

---

## 0. Why the first study failed, and what that implies

CSC measured, at d = 8, hyperbolic arms operating at √|K|·r = 1.44–1.73 where
the volume argument predicts a 7–15× capacity advantage. Observed: none. Three
measurements locate the break:

1. **The best-performing arm, in both readouts and on both axes of the
   capacity–interference frontier, was `normalized`** — a flat control that
   projects onto the unit sphere and discards the radial degree of freedom
   entirely.
2. **Minimum pairwise prototype distance collapsed in H²** (0.0005 at κ=−4 vs
   0.209 in E²) while 12× more volume sat unused. CSC's SPEC predicted the
   opposite ("floor holds in H², collapses in E²").
3. The κ-trend was not stable across the init gain, dimension, or readout.

**The account these support.** Superposition capacity is *angular*: it is
governed by how many near-orthogonal directions fit in d dimensions
(~exp(c·ε²·d)), and interference between two features is the overlap of their
directions. Negative curvature adds no angular room — the tangent space at the
origin is Euclidean and the direction sphere is unchanged. What it adds is
**radial** volume. That is a different currency, and superposition does not
spend it. Discarding radius entirely (the `normalized` arm) helped; the models
crowded in angle and left the radial room untouched.

**And the room it does add has a shape.** In H² almost all of a disc's volume
lies near the rim, and two random rim points are ≈2R apart — nearly
equidistant. Hyperbolic space is tree-like: it supplies exponentially many
points that are mutually far *and roughly equally far*. That is the right
structure for a **hierarchy**, where leaves should be mutually distant and
information lives in path lengths. It is the wrong structure for a **graded
metric code**, which needs many distinguishable degrees of proximity.

This reconciles CSC's null with the hyperbolic-embedding literature rather than
contradicting it. Poincaré embeddings (Nickel & Kiela 2017) and Sarkar's
construction win on *trees*, because a tree's node count grows exponentially
with depth exactly as hyperbolic volume grows with radius. CSC gave the model
i.i.d. exchangeable features — no hierarchy — and curvature had nothing to
match.

---

## 1. Main prediction (H2-MAIN)

> **Negative curvature buys representational room in proportion to the
> hierarchical structure of the features being represented. The advantage grows
> with hierarchy depth, and vanishes for exchangeable features.**

Formally, with `A(κ, D)` the capacity of a curvature-κ arm relative to the
matched Euclidean arm at hierarchy depth D:

- **H2-a (anchor, already measured):** A(κ, 0) = 1 for all κ. CSC measured this
  and it is not re-litigated; CSC-2 must reproduce it as a control.
- **H2-b (growth):** ∂A/∂D > 0 for κ < 0. The advantage is *created by the
  data's structure*, not by the geometry alone.
- **H2-c (matching):** the advantage peaks where the geometry matches the data
  — the optimal √|κ| tracks the tree's branching factor b as √|κ| ∝ log b / ℓ
  for edge length ℓ, from the Sarkar embedding.

**Directional commitment.** If A(κ, D) is flat in D, or if the advantage
appears equally for exchangeable features, H2-MAIN is **falsified**, not
reinterpreted. If spherical arms show the advantage, H2-MAIN is falsified.

## 2. Secondary hypotheses

- **P2-1 (distortion):** hyperbolic arms embed the feature hierarchy at lower
  metric distortion than flat arms, and the distortion gap grows with D. This
  is the mechanism H2-b claims; if capacity improves without distortion
  improving, the account is wrong even if the headline holds.
- **P2-2 (angular vs radial):** the capacity advantage is carried by *radial*
  structure. Operationalized: in hyperbolic arms at D > 0, prototype geodesic
  radius correlates with hierarchy level (Spearman > 0.5), and ablating the
  radial coordinate destroys the advantage while ablating it in flat arms does
  not.
- **P2-3 (the normalized control loses its edge):** `normalized` won in CSC
  because capacity there was purely angular. At D > 0 it should *stop* winning,
  since it cannot represent depth. A `normalized` arm that keeps winning at
  large D falsifies the angular/radial account.

## 3. Experiments

**E1 — readout scale (runs first; hours).** CSC's null may be specific to a
readout whose resolution is *relative* — `norm_affine` divides by the batch mean
distance, and hyperbolic geometry inflates that mean by the same exponential
property that creates the volume, so room and resolution may simply cancel.
Build a readout with an **absolute** length scale and re-run CSC's factorial
unchanged. Outcomes: if hyperbolic now wins, CSC's null is architecture-specific
and must be restated in those terms; if not, the null is robust and the angular
account strengthens. **Both outcomes change what we say, which is why this runs
before anything else.**

**E2 — usability upper bound (cheap).** Add an explicit prototype-repulsion
term forcing models to spread into the available volume. This does not test
whether models *do* use the room; it tests whether the room is usable **at
all**. If hyperbolic cannot beat flat even when forced to spread, the
volume→capacity argument is dead outright rather than merely unrealized.

**E3 — hierarchical features (the main event).** Features generated from a tree:
depth D, branching b, with parent/child co-activation so that the feature
correlation matrix carries the hierarchy. Sweep D × b × κ × d. Primary
comparison: A(κ, D) against the matched Euclidean arm, with A(κ, 0) reproducing
CSC's null as an internal control.

**E4 — positive control (blocks E3).** Before any E3 number is read, reproduce a
known external result: Sarkar-style tree embedding into H² at low distortion,
against the same measurement code. If our pipeline cannot reproduce a result the
literature considers settled, no E3 number means anything.

## 4. Design rules carried over from CSC

These are not aspirational; each was bought with a specific failure.

- **R1 (unchanged).** Curvature is never the only free scalar; per-prototype
  bias and scale in every arm including controls.
- **R2′ (amended).** Saturation excludes only for κ < 0, where a numerical
  clamp exists. For κ > 0 the diameter fraction is a diagnostic that never
  excludes — otherwise the gate discards exactly the evidence that could
  falsify the hypothesis (CSC D2).
- **R3 (unchanged, and vindicated).** Flat controls in the same sweep. In CSC
  the *winning arm was a flat control*; without it the study would have
  reported a curvature effect.
- **R4 — never tie a nuisance variable to the independent variable.** CSC's
  calibration rule produced Spearman(|κ|, init gain) = 0.971 and cost 2200
  runs. Nuisance factors are crossed factorially, never derived from the IV.
- **R5 — validate instruments before calibrating through them.** CSC ran
  calibration (00a) before readout validation (00c), calibrated through a
  readout later disqualified, and had two headline numbers reverse. Order is
  **readout validation → numerics → calibration**.
- **R6 — two instruments minimum, always.** In CSC, `norm_affine` alone reports
  spherical curvature helping (ρ=+0.44) and `softmax` alone reports hyperbolic
  helping (ρ=−0.33, p=0.005). Either single-readout study publishes a
  confident, wrong, opposite result.
- **R7 — power analysis before running, not after.** CSC's own calibration rule
  would have made P1 unfalsifiable, and its F1.1 falsifier rejected true
  hypotheses 53% of the time. Both were found by a power analysis that took an
  hour and should have preceded the phase.
- **R8 — external positive control per phase.** CSC's 00d caught two
  measurement bugs that cancelled between arms and were invisible to every
  internal comparison.
- **R9 — reproducibility.** Anchored `/data/`, fresh-clone CI, committed summary
  JSONs, provenance stamps. Unchanged and working.

## 5. Falsifiers

- **F2.1** A(κ, D) flat in D → H2-b dead, and with it the reframing.
- **F2.2** The advantage appears equally for exchangeable features → the effect
  is not hierarchy-driven; H2-MAIN dead.
- **F2.3** Capacity improves without embedding distortion improving → P2-1
  dead; the stated mechanism is wrong even if the headline survives.
- **F2.4** `normalized` keeps winning at large D → the angular/radial account
  is wrong.
- **F2.5** A flat control reproduces ≥ 70% of the advantage → conditioning, not
  geometry.

## 6. Gates

- **G2-0** E4 positive control reproduces known tree-embedding distortion, and
  E1's outcome is recorded, before any E3 number is read.
- **G2-1** H2-b confirmed with the anchor A(κ,0)=1 reproduced, surviving F2.2
  and F2.5, under both instruments, with a power analysis showing the design
  could have detected the effect. Any failure → stop, write up the null.

## 7. Status

DRAFT. Nothing sealed. E1 is the only experiment ready to run, and its result
determines whether the rest of this spec is worth sealing at all.

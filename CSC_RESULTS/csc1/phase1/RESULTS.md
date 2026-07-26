# Phase 1 — P1 capacity: **UNINTERPRETABLE**

2200 runs. **No verdict on P1 is reported**, and none should be extracted from
this artifact. Four independent defects invalidate the comparison; three of
them are in the design I sealed earlier the same day.

This is not a null result. A null would mean "curvature does not buy capacity";
this means "this run could not have told us either way".

## What the numbers looked like

| head | d=4 | d=6 | d=8 |
|---|---|---|---|
| `norm_affine` (primary) | ρ = **+0.44** | +0.37 | +0.42 |
| `softmax` (2nd instrument) | +0.35 | −0.15 | **−0.33** (p=0.005) |

Spearman(N*, κ). H-MAIN predicts negative. The primary head reports positive
(spherical better — which SPEC §1 says *falsifies* H-MAIN), the second
instrument reports negative at d=6 and d=8. **The two instruments disagree in
sign**, which is precisely the failure the parent program's 01b audit found and
the reason CSC carries a two-instrument rule at all. Under D9 a sign
disagreement is *unresolved*, not adjudicated.

## Why it is uninterpretable

### 1. Curvature and init gain are collinear by construction

Spearman(|κ|, init gain) = **0.971** within the hyperbolic arms.

This is a direct consequence of D12. Holding the geodesic radius R fixed across
κ — which was the fix for a real problem, since targeting constant √|K|·r gives
every arm identical predicted capacity — forces the init gain to rise with |κ|.
So "capacity falls with |κ|" and "capacity falls with init gain" are the same
statement in this data, and nothing here separates them.

The two heads make this vivid. Their fitted calibration coefficients differ, so
their gain-vs-κ relationships run in **opposite directions**:

| | κ=−4 | κ=−2 | κ=−1 | κ=−0.5 |
|---|---|---|---|---|
| `norm_affine` gain | 8.24 | 4.67 | 2.64 | 1.50 |
| `softmax` gain | 4.68 | 5.82 | 7.24 | 9.00 |

Opposite confound directions, opposite apparent κ-trends. That is a more
economical explanation of the sign disagreement than any claim about geometry.

### 2. The κ=−4 arm violates G1's own precondition

**54.6% of κ=−4 runs are R2-flagged UNINTERPRETABLE** (72% at d=8), against
SPEC's G1 requirement of ≥95% clean. The analysis filters to clean runs, so the
surviving κ=−4 group is a *selected* subsample — selection on a variable
correlated with the outcome. Its apparent capacity collapse (11.5 vs Euclidean
20.2 at d=8, per-seed range 6–18) is not usable evidence.

Its gain sits at 8.24 against a cap of 9.0, i.e. jammed against the transient
ceiling that 00a established. The cell is simply not reachable at these (d, N)
within the safe operating regime.

### 3. The two curvature signs were calibrated differently

Hyperbolic arms used fixed-radius calibration (realized x spanning 0.76–2.79);
spherical arms used band-centre calibration (x ≈ 1.03–1.19, nearly constant).
A trend test across the full κ grid therefore compares arms calibrated by two
different schemes, and the sign boundary is exactly where the scheme changes.
This is a straightforward coding error in `_arms_for`, not a design trade-off.

### 4. No gain control exists

The Euclidean arm was run at a single gain per cell (three distinct values
across the whole sweep), so the data cannot show whether capacity falls with
gain independently of curvature. That control was not built, and without it
finding 1 cannot be resolved from the existing runs.

## What has to change before P1 can be tested

1. **A gain control is mandatory.** Run flat arms across the *same gain range*
   the curved arms use. If flat capacity falls with gain too, the entire
   κ-trend is a training artifact. This is the decisive experiment and it is
   cheap.
2. **Calibrate both curvature signs identically.** Fixed radius everywhere.
3. **Drop κ=−4 at these shapes**, or raise the transient cap with explicit
   per-step monitoring rather than a static bound. As it stands the cell cannot
   be run cleanly.
4. **Consider breaking the collinearity by design** — e.g. sweeping gain and κ
   on a grid rather than tying gain to κ through the calibration rule, so the
   two channels can be separated rather than assumed apart.

## Note on the metric

The capacity–interference frontier (recovered count against interference across
the N ladder, per arm) is computable from `p1_capacity.json` without new runs,
and is a better object than capacity alone: superposition already lets a model
slide along that frontier by tolerating contamination, so the question worth
asking is whether curvature *shifts* the frontier or merely moves along it.
Not built yet.

## Artifact

`p1_capacity.json` — 2200 runs, all cells, all arms, per-seed. Retained in full
because the diagnosis above is read off it, and because a later re-run needs
something to compare against.

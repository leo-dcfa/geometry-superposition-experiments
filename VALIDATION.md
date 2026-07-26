# CSC — pre-registration ledger and gate status

Authoritative for thresholds. `SPEC.md` states the hypotheses; nothing here
weakens them. Every number below is either **SEALED** (fixed, with the commit
that fixed it) or **DRAFT** (proposed, awaiting sign-off, not yet used to
judge anything).

Status: **G00 in progress.** No hypothesis-relevant quantity has been read.
Phase 1 has not run.

---

## 1. Gate G00 — instrument calibration

| Sub-gate | Question | Verdict | Artifact |
|---|---|---|---|
| 00a | Do trained point clouds occupy √\|K\|·r ∈ [0.5, 3.0]? | **PASS with a caveat** (below) | `CSC_RESULTS/phase00/00a_scale_sweep.json` |
| 00b | Are the numerics sound across the operating band? | **PASS after one fix** (below) | `CSC_RESULTS/phase00/00b_numerics_audit.json` |
| 00c | Is the readout fair across κ arms? | *pending re-run* | `CSC_RESULTS/phase00/00c_dead_unit_parity.json` |

### 00a — scale calibration

**The calibration knob the spec assumed does not exist.** The encoder init
gain was swept over a 16× range (0.25 → 4.0). Fitted exponent on the trained
median √|K|·r: **7.2e-4**; predicted radius ratio across the whole swept
range: **1.002**; ΔR² from including the term at all: **1.7e-6**. The
optimizer selects its own operating radius and absorbs an init rescale within
a few hundred steps. A rule of the form "set the init gain to s(K, d, N)"
would have been committed, obeyed, and inert.

What the operating point *is* set by is (|K|, d, N). Fitted separately by
curvature sign, because a sphere's finite diameter caps its radius and the
pooled fit is worse than either:

- hyperbolic: R² = 0.627, coefficient on log|K| ≈ 0.5 — i.e. the trained
  *geodesic* radius r is roughly K-independent, so √|K|·r scales as √|K|.
- spherical: R² = 0.509.

Residual scatter is real seed-to-seed variation in the settled radius, not
misspecification; it is why the admissible region is stated as a prediction
with a residual band rather than a point.

**Measured occupancy: 88.1% of 420 cells land inside the band**, with no
intervention. The convergence probe (50k steps, eval points spaced at the
sweep length) confirms the operating point is settled, not still drifting —
this is the reason the sweep length was raised from 2k to 10k, after a
spherical N=16 cell was caught still doubling its radius after step 2000.

**Caveat (recorded as a miss, per R6):** the corner where curvature is
strongest *and* features most numerous exits the band — κ=−4 at N=64 sits at
≈3.3, and the fitted rule predicts up to ≈5.8 at κ=−4, N=256. Of the 210
candidate Phase-1 cells, **178 are predicted in-band and 32 are not**, the
latter concentrated at κ=−4 (14 cells) and κ=−2 (8). These remain well below
the R2 exclusion threshold (which corresponds to √|K|·r ≈ 5.3), so they are a
calibration miss rather than an automatic exclusion.

No lever was adopted to fix the corner. Weight decay would work mechanically
but is rejected on principle: the parent program's dose-response result
established that taxing geometry distorts exactly what these studies measure,
so it is not an acceptable calibration knob. **DRAFT decision D6** below
proposes restricting primary cells to the admissible region instead.

### 00b — numerics audit

fp32 across the operating band: round-trip relative error < 1e-4 ✓,
pairwise-distance p99 relative error **2.3e-6** ✓, gradients finite ✓. The
empirically measured clamp horizon — the largest scaled radius at which the
reported distance still tracks the true radius to 1% — is **≈14.4–14.6**,
against a band top of 3.0, giving **≈11.4 of headroom** ✓. The horizon is
measured by walking the radius outward, not derived from the epsilon
constant, which would only have restated the input.

**bf16 is disqualified by measurement.** In-band pairwise distances carry up
to **1.4e-1** relative error against a float64 reference, versus 2.3e-6 in
fp32 — five orders of magnitude worse, and large enough to swamp any capacity
difference this study could detect. CSC runs fp32 throughout. (The parent
program's bf16 incident is why this was measured rather than assumed.)

**One real bug found and fixed by this audit.** Every distance ends in a
square root, and d√x/dx is infinite at x=0, so *any* coincident pair of points
produced NaN gradients that propagated through the whole batch — at every
radius, not just near the boundary, so R2 would never have caught it.
Coincident prototypes are exactly what a model out of room does, so the
failure would have arrived in the cells that matter most and looked like a
training divergence. Fixed in `csc/spaces/numerics.py` (ε² floor inside the
root, applied uniformly to every arm so no geometry gets a better numerical
deal than another); regression tests in `tests/test_spaces.py`.

### 00c — dead-unit parity

*Verdict pending the re-run; the design correction and the head findings are
recorded here because they are already decided by measurement.*

**Design correction.** The first version of this fixture swept crowded shapes
(N=64 into d=2) and failed everywhere. The reason is structural: when a model
has no room for a feature, letting that unit die *is* the correct behaviour,
so the dead-unit rate was measuring capacity — the hypothesis — rather than
readout fairness — the gate. Under that design, any genuine capacity
difference between arms would have failed its own fairness fixture. The gated
cells are now the **non-binding** shapes only, where a working head recovers
essentially every feature in every arm and a dead unit can therefore only mean
the readout treated an arm unfairly. Crowded shapes are still run and reported
as decoration.

**Two of the four readout heads are disqualified**, before any capacity number
was read — which is what this fixture is for:

| Head | Non-binding recovery (E² / H²) | Verdict |
|---|---|---|
| `norm_affine` | 1.00 / 1.00, 0 dead | usable |
| `softmax` | 1.00 / 1.00, 0 dead | usable |
| `rbf` | 0.00 / 0.00 at wd=0; 1.00 / 1.00 at wd=0.01 | **disqualified — basin-fragile** |
| `affine` | 0.88 / 0.62 | **disqualified — arm-asymmetric** |

`rbf` collapses to the all-zero solution (its final loss matches "predict 0
always" to four decimals) at weight_decay=0 and is perfect at 0.01: the
optimizer, not the geometry, selects its basin. That is disqualifying for an
instrument regardless of which setting is chosen, and it is the parent
program's "init selects the solution basin" lesson reappearing in a new place.
`affine` fails in both settings *and asymmetrically across arms*, which is
precisely the raw-distance-scale coupling it was retained to demonstrate.

The repo default head was changed from `rbf` to `norm_affine` as a result.

---

## 2. DRAFT decisions awaiting sign-off

None of these are sealed. They are listed because each changes what Phase 1
measures, and because sealing them is the researcher's call, not the
implementer's.

**D1 — Primary readout head: `norm_affine`, second instrument `softmax`.**
Both passed 00c cleanly in every arm. Two instruments, not one, because the
parent's 01b audit measured a headline geometry result *flipping sign* between
readouts. Every H-MAIN(toy) number would be reported under both.

**D2 — R2's saturation gate must not be applied unmodified to the spherical
arm.** This is the one that most needs a decision, because as written it
biases the study toward its own hypothesis.

R2 excludes runs whose points pin to the boundary of the model. For K < 0 that
is a genuine numerical condition: `atanh` clamps, and the readout starts
measuring its own guard. For K > 0 there is no such clamp — `arctan` is
well-conditioned everywhere, and 00b confirms finite gradients right up to the
antipode. What "saturation" means for a sphere is that the cloud has spread to
fill the diameter, which is *exactly what H-MAIN predicts positive curvature
does when the space runs out of room*. In the 00a sweep the gate already fired
on 4 cells, all spherical.

So the rule as written discards positive-curvature evidence for displaying the
effect being tested, and cannot fire against the hyperbolic arm for the
analogous reason. Options:

- **(a) Recommended.** Keep the exclusion for K < 0, where it has a numerical
  justification. For K > 0 report the diameter fraction as a diagnostic that
  never excludes. Rationale: R2's stated purpose is "geometry meaningless",
  which is a claim about arithmetic, and the arithmetic is sound for K > 0.
- (b) Keep R2 unchanged, and additionally report every spherical result with
  and without the excluded cells, so the bias is visible rather than removed.
- (c) Keep R2 unchanged. Simplest, and wrong in a direction that flatters
  H-MAIN.

**D3 — Clamped-Euclidean matching rule.** R3 defines the control as flat with
distances "clipped at the matched spherical/hyperbolic diameter", but
hyperbolic space has infinite diameter, so there is no literal value to match.
Proposed operationalization: clip at **2 × the top of the R2 operating band**
in the matched arm's units — the diameter of the region the curved arm
actually occupies. Also recorded: the clip is *hard*, so pairs beyond it get
exactly zero gradient, whereas a sphere's metric saturates smoothly. The
control reproduces the bounded *range* but not the smooth compression, so a
clean F1.3 should not be over-read as ruling out every conditioning account.

**D4 — Recovery tolerance ε.** Proposed: a feature counts as recovered when
its own reconstruction is within **20% relative** of the probe value (probe
value 0.8). Relative rather than absolute, because response scale is a free
per-prototype parameter in every arm under R1, and an absolute tolerance would
silently favour whichever arm learned the larger scale. F1.2's N* threshold
stays at ≥ 90% of features recovered, per SPEC.

**D5 — Weight decay fixed at 0.0 for all Phase-1 arms.** Curvature must not
pay rent it is not being asked about; the parent's dose-response work is the
reason. Now that `rbf` is dropped, no surviving head needs decay to train.

**D6 — Phase-1 primary cells restricted to the 00a admissible region.**
Out-of-band cells (chiefly κ=−4 at large N) are run and reported but not
primary. The alternative — adding an architectural radius cap to every arm —
is a larger design change and is not proposed.

**D7 — All Phase-0/1 runs pinned to CPU.** Measured: the same config gives
different losses on CPU and GPU (0.5086 vs 0.5057) from different RNG streams,
and GPU gives no speedup at these model sizes (0.77–1.34× per run, versus ~14×
throughput from running 16 models in parallel on CPU). Phase 2/3 will need the
GPU and will re-pin then.

---

## 3. Standing statistical policy

Inherited from SPEC §8, restated so it is enforceable:

- Seeds ≥ 5 (toy, small LM), ≥ 3 (GPT-2). Primary cells named before
  unblinding. Best-of-seed never headlines without the all-seed range in the
  same sentence (R6).
- Spearman with midrank ties; any stratum with > 10% tied mass reported with
  and without the tied block.
- A number outside a pre-registered band is written as a MISS, in the table
  and in the prose.
- A claim with no committed artifact under `CSC_RESULTS/` does not enter a
  scorecard (R5). Enforced in CI by `tools/check_tracked_sources.py` plus a
  fresh-clone `uv run pytest`.

## 4. Deviations from SPEC, with reasons

| SPEC text | Deviation | Reason |
|---|---|---|
| 00a delivers "a scale-selection rule (function of κ, d, N)" for the init scale | Rule selects *admissible cells*, not an init gain | The init gain was measured to be inert (exponent 7e-4) |
| 00c ported "as-is" from parent 01b | Gated on non-binding shapes only | On crowded shapes the fixture measures capacity, not fairness, and would fail on the hypothesis being true |
| R2 applies to "all curved arms" | D2 proposes K<0 only | No numerical clamp exists for K>0; see D2 |
| κ as in parent program | κ is true sectional curvature (parent's × 4) | P1 fits √\|κ\|; the parent convention would put a factor of 2 into α |

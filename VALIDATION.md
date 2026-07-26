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
| 00a | Do trained point clouds occupy √\|K\|·r ∈ [0.5, 3.0]? | **NEAR-MISS** — 99.2% in-band on the primary head, but 2.4% R2-flagged against a registered clause of zero | `00a_scale_sweep.json`, `00a_scale_sweep_extended.json`, `00a_confirm.json` |
| 00b | Are the numerics sound across the operating band? | **PASS after one fix** | `00b_numerics_audit.json` |
| 00c | Is the readout fair across κ arms? | **PASS** — 2 of 4 heads disqualified | `00c_dead_unit_parity.json` |

All artifacts under `CSC_RESULTS/phase00/`.

**G00 does not pass yet, so Phase 1 does not open.** Nothing is blocked on
compute or on a missing measurement: the instrument is calibrated, the primary
head lands 99.2% in-band including 100% on held-out shapes, and the numerics
and readout gates are clean. What blocks it is **three pre-registration
decisions that are the researcher's to make, not the implementer's** — D2, D8
and D9 below. D8 in particular is a criterion I set stricter than the study's
own G1 bar without justification.

**Ordering error in SPEC §4, found the hard way.** Sub-gate 00a must run
*after* 00c, not before. 00a calibrates the point cloud through a readout; if
that readout has not yet been validated, the calibration measures the readout.
The first 00a run did exactly this via `rbf` and produced two headline numbers
that both reversed on re-run. Recommended for any replication: run 00c → 00b →
00a.

### 00a — scale calibration: **FAIL**

> **Correction, recorded because it changed the conclusion.** The first 00a run
> was executed with the `rbf` head, which 00c subsequently disqualified. On
> that head the init gain measured as completely inert (exponent 7.2e-4 over a
> 16× sweep) and occupancy looked healthy at 88.1%. **Both results were
> artifacts of a broken readout** — one that collapses to the all-zero
> solution, whose point cloud is therefore not doing anything. Re-run on the
> two heads that survived 00c, the findings reverse. The superseded numbers
> are retained here only as a warning: a calibration measured through a
> readout that has not itself been validated measures the readout.

Re-run on `norm_affine` and `softmax`, 840 runs × 10k steps each:

| Head | Init-gain exponent | Radius ratio over the 16× sweep | In-band fraction |
|---|---|---|---|
| `norm_affine` (primary) | **+0.277** | 2.15× | **0.576** |
| `softmax` (2nd instrument) | +0.043 | 1.13× | **0.407** |

**The failure is at the bottom of the band, not the top.** Trained median
√|K|·r, at the default init gain:

- `norm_affine`, hyperbolic arms: **0.19 – 0.44** (band floor is 0.5)
- `softmax`, spherical arms: **0.33 – 0.41**

That is the first failure mode SPEC §4 names — points huddled near the origin,
where every geometry is locally flat and the experiment measures nothing. It
is the opposite of the boundary-pinning that rule R2 guards against, and R2
would never fire on it, which is precisely why 00a is a separate gate.

**The band is reachable.** An extended sweep (gains 4 → 1024, 840 runs)
settled it. The model does not resist being pushed outward — it goes from
below-band straight to the saturation ceiling, with a usable window between:

| `norm_affine` gain | 0.25 | 1 | 2 | **4** | 16 | 64 | 256 |
|---|---|---|---|---|---|---|---|
| in band | 0.48 | 0.46 | 0.60 | **0.90** | 0.76 | 0.06 | 0.00 |
| UNINTERPRETABLE | 0.00 | 0.00 | 0.00 | **0.00** | 0.68 | 1.00 | 1.00 |

Past gain ~16 the clouds do not overshoot smoothly; they land *exactly* on the
saturation ceiling — 14.5 for K<0 and π for K>0. The 14.5 is an independent
confirmation of 00b's measured clamp horizon (14.4–14.6), which was obtained
by a completely different method (walking the radius outward until the
reported distance stops tracking). Two sub-gates agreeing on a number they
derived independently is reassuring about both.

**Committed rule:** `csc/calibration/scale_rule.py`, fitted over 1680 runs at
gains ≤ 16 (above which there is no smooth regime to fit), inverted for the
gain that lands the predicted median √|K|·r on the band's geometric centre.

**The knob's ceiling is set by the transient, not the settled state** — a
finding that came out of the confirmation run and changed the rule. At
prescribed gains ≥ 10.4 the R2 gate fired on 14.3% of runs, yet 15 of those 18
flagged runs still *ended* in-band: a high gain launches the cloud onto the
boundary and lets it contract back, so the endpoint looks healthy while the
early gradients were clamp-dominated. That is the parent program's
path-dependence argument in a new place — a readout saturated at birth can
shape what is learned before the model routes around it, and the endpoint
cannot show you it happened. The cap was therefore set at **9.0**, below where
the transient appears, rather than R2 being given a burn-in exemption.

### 00a confirmation — **NEAR-MISS against the registered criterion**

The rule is validated by *applying* it, on held-out shapes it was never fitted
on (`experiments/phase00/run_00a_confirm.py`, 252 runs). A rule reported from
its own fit data is only a description of that data.

| Head | In band | Held-out shapes | Below floor | R2-clean |
|---|---|---|---|---|
| `norm_affine` (primary) | **0.992** | **1.000** | 0.008 | 0.976 |
| `softmax` (2nd instrument) | 0.556 | 0.560 | 0.413 | 0.968 |

Registered criterion (committed before the run): ≥ 80% in-band on the primary
head **and zero runs flagged UNINTERPRETABLE**. Measured: 99.2% in-band, but
**2.4% flagged — so the criterion is not met, and this is written as a miss**
(R6). All three flagged runs sit at the gain cap and all three end in-band;
exactly one run of 126 is out of band at all.

**DRAFT decision D8 (below) proposes reconciling the clause with SPEC's own
standard**, which is looser than the one I set: G1 requires the R2 gate clean
in ≥ 95% of hypothesis-relevant runs, and 97.6% clears it. I set my clause
stricter than the study's own bar without justifying why, which is a
pre-registration error on my part; the fix is the researcher's call, not a
silent edit, so the runner still reports FAIL until it is made.

**`softmax` cannot be calibrated to the same standard, and this is
structural.** Its gain exponent is 0.096 against `norm_affine`'s 0.367 — a
100× gain change moves its radius ~1.6× — so 41% of its runs sit below the
band floor at every curvature, and no gain within the safe range fixes it.
This bears directly on the two-instrument rule (D1): the rule exists because
the 01b audit measured a headline result flipping sign between readouts, but
it presumes both instruments are sound. Here they demonstrably are not equally
calibrated, so a Phase-1 disagreement between them would be confounded with
that difference. See D9.

Weight decay remains rejected as a lever on principle (the parent program's
dose-response result: taxing geometry distorts what these studies measure),
and is not reconsidered now that a legitimate knob exists.

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

### 00c — dead-unit parity: **PASS**

1960 runs. Two of four heads are usable; the sub-gate passes because it asks
whether a usable instrument *exists*, not whether every candidate is usable —
a head failing here is the fixture working.

| Head | Verdict | Min recovery across all arms |
|---|---|---|
| `norm_affine` | **USABLE** | 1.00 |
| `softmax` | **USABLE** | 1.00 |
| `rbf` | DISQUALIFIED — basin-fragile | 0.06 |
| `affine` | DISQUALIFIED — arm-asymmetric | 0.29 |

Both surviving heads show **zero dead units in every arm** on the gated cells,
including both R3 controls, at both weight-decay settings. All 25 remaining
failures belong to the two disqualified heads.

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

The repo default head was changed from `rbf` to `norm_affine` as a result —
and 00a had to be re-run, because it had been calibrated through `rbf`.

**A criterion of our own also had to be corrected.** The dead-unit fraction is
quantized in steps of 1/N, so a flat 5% tolerance is finer than the metric can
express: at N=3, a single dead unit in one seed of five reads as a 6.7% gap
and "failed". The effective tolerance is now `max(0.05, 1/N)` — never tighter
than one readout unit — with the same one-feature allowance on the recovery
floor. That change, and nothing else, accounts for the two `norm_affine`
failures in the first gated run. Tests: `tests/test_phase00.py`.

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

**D8 — Reconcile the 00a confirmation clause with SPEC's own R2 standard.**
I registered "zero runs flagged UNINTERPRETABLE" for the 00a confirmation.
SPEC's G1 requires the R2 gate clean in **≥ 95%** of hypothesis-relevant runs.
Measured: 97.6% clean on the primary head — clears the study's bar, misses
mine. My clause was stricter than the standard the study applies to its own
Phase-1 gate and I did not justify the difference, which is a
pre-registration error. Proposed: restate the clause as **≥ 95% R2-clean**,
matching G1, and record the current run as a PASS under it. Alternatives:
hold the stricter clause and lower the gain cap further (which will push cells
below the band floor — the two failure modes trade off directly), or accept
the 2.4% as an R2 per-run exclusion, which is exactly what R2 is designed to
do rather than something that should block a gate. **Not adopted unilaterally;
the runner still reports FAIL until this is decided.**

**D9 — What to do about `softmax` being under-calibrated.** The two-instrument
rule (D1) presumes both readouts are sound. Measured, they are not: `softmax`
puts 41% of runs below the band floor and its calibration knob is nearly
inert (exponent 0.096), so this cannot be fixed by tuning. Options:

- **(a) Recommended.** Keep `softmax` as the second instrument but state its
  calibration deficit wherever the two-instrument comparison is reported, and
  treat a disagreement between heads as *uninterpretable* rather than as
  evidence about geometry — because the heads differ in band occupancy as
  well as in form.
- (b) Find a third head that calibrates as well as `norm_affine` and use that
  as the second instrument. Costs a new head design plus a 00c re-run.
- (c) Drop to a single instrument. Rejected: this is precisely what the 01b
  audit showed can produce a sign-flipped headline.

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

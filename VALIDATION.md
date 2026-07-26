# CSC — pre-registration ledger and gate status

Authoritative for thresholds. `SPEC.md` states the hypotheses; nothing here
weakens them. Every number below is either **SEALED** (fixed, with the commit
that fixed it) or **DRAFT** (proposed, awaiting sign-off, not yet used to
judge anything).

Status: **G00 PASSED** (2026-07-26, commit recorded below). No
hypothesis-relevant quantity has been read: the only capacity numbers measured
so far are from the Euclidean positive control (00d), which is scored against
published external behaviour rather than against a curved arm.

---

## 1. Gate G00 — instrument calibration

| Sub-gate | Question | Verdict | Artifact |
|---|---|---|---|
| 00a | Do trained point clouds occupy √\|K\|·r ∈ [0.5, 3.0]? | **PASS** under D8; the original clause was a MISS (§5) | `00a_scale_sweep.json`, `00a_scale_sweep_extended.json`, `00a_confirm.json` |
| 00b | Are the numerics sound across the operating band? | **PASS after one fix** | `00b_numerics_audit.json` |
| 00c | Is the readout fair across κ arms? | **PASS** — 2 of 4 heads disqualified | `00c_dead_unit_parity.json` |

All artifacts under `CSC_RESULTS/phase00/`.

**G00 PASSES. Phase 1 may open**, subject to the remaining DRAFT decisions
(D1, D3–D7) being sealed, since they fix what Phase 1 measures rather than
whether the instrument works.

Final numbers on the primary head (`norm_affine`), applying the committed
calibration rule to held-out shapes: **99.2% in-band, 100% on shapes the rule
was never fitted on, 0.8% R2-flagged** against the D8 bar of ≤ 5%.

Recorded alongside, because it constrains Phase 1: the two-instrument rule can
be applied on **19 of 24 hyperbolic cells but only 5 of 18 spherical ones**
(`00a_confirm.json` → `two_instrument_availability`). On most of the spherical
grid there is one calibrated instrument, not two (D9).

**Ordering error in SPEC §4, found the hard way.** Sub-gate 00a must run
*after* 00c, not before. 00a calibrates the point cloud through a readout; if
that readout has not yet been validated, the calibration measures the readout.
The first 00a run did exactly this via `rbf` and produced two headline numbers
that both reversed on re-run. Recommended for any replication: run 00c → 00b →
00a.

### 00a — scale calibration: **PASS** (after two reversals)

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

### 00a confirmation — **PASS** under D8; **MISS** against the clause as first registered

The rule is validated by *applying* it, on held-out shapes it was never fitted
on (`experiments/phase00/run_00a_confirm.py`, 252 runs). A rule reported from
its own fit data is only a description of that data.

| Head | In band | Held-out shapes | Below floor | R2-clean |
|---|---|---|---|---|
| `norm_affine` (primary) | **0.992** | **1.000** | 0.008 | 0.992 |
| `softmax` (2nd instrument) | 0.556 | 0.560 | 0.413 | 1.000 |

(R2-clean shown post-D2. Under the pre-D2 gate the primary head read 0.976.)

Criterion as first registered: ≥ 80% in-band on the primary head **and zero
runs flagged UNINTERPRETABLE**. Measured 99.2% in-band but 2.4% flagged, so
**that clause was a MISS** and is recorded as one in §5 (R6). Under D8 —
≥ 95% R2-clean, matching SPEC's own G1 bar — the sub-gate **PASSES**, and
after D2 stopped counting spherical boundary-filling the flagged rate is
**0.8%**. All flagged runs sat at the gain cap and all ended in-band; exactly
one run of 126 was out of band at all.

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

## 2. Decisions

**D2, D8 and D9 were approved on 2026-07-26 and are SEALED** — implemented in
code, covered by tests, and not to be changed without a new ledger entry. The
remainder are DRAFT: proposed, not yet used to judge anything.

### Sealed

**D2 — SEALED. The R2 exclusion applies to K < 0 only.** For K > 0 the same
quantity is recorded as `diameter_filling` and never excludes; spherical
capacity results are reported with and without diameter-filling cells as a
robustness panel. Rationale: R2's justification is numerical (a boundary-pinned
readout reports its own clamp), which is real for `atanh` at the hyperbolic
ball boundary and absent for `arctan`, whose gradients 00b measured as finite
to the antipode. What the unmodified rule would have excluded for K > 0 is a
cloud filling the diameter — the very mechanism H-MAIN predicts — so applied to
both signs it could only ever discard evidence against the hypothesis, while
being structurally unable to fire against the hyperbolic arm.
*Implemented:* `csc/training/monitor.py`; *tests:*
`test_spherical_runs_are_never_excluded_by_saturation`,
`test_hyperbolic_runs_are_still_excluded_by_saturation`.

**D8 — SEALED. The 00a confirmation clause is ≥ 95% R2-clean**, matching
SPEC's own G1 bar, replacing the "zero flagged" clause I registered. The
original clause was stricter than the standard the study applies to its own
Phase-1 gate and I gave no justification for the difference — a
pre-registration error on my part, recorded as a MISS in §5 rather than
relabelled. The decisive argument for 95 over 0: tightening further means
lowering the gain cap, which trades saturated runs for below-band runs, and
those errors are not symmetric — above-band is caught automatically by R2,
below-band is invisible to it and is exactly the condition that produces a
confident-looking null. Prefer the error the instrument can see.
*Implemented:* `experiments/phase00/run_00a_confirm.py`.

**D9 — SEALED. `softmax` is retained as the second instrument, with its
calibration deficit stated and its usable cells enumerated.** It is
scale-invariant by construction (it normalizes across prototypes), which is
both why it is geometry-fair and why no gain calibrates it — the deficit is
structural, not a tuning failure. Consequences, pre-registered now:

- Heads **agree** → report as robust. Agreement between a scale-sensitive and
  a scale-invariant readout is stronger evidence than agreement between two
  similar ones.
- Heads **disagree** → **unresolved**, not adjudicated. The disagreement is
  confounded with the calibration gap, so it cannot be read as evidence about
  geometry. Escalates to designing a third head.
- The two-instrument rule is only applied on cells where **both** heads land
  in band; that set is enumerated per run in `00a_confirm.json`
  (`two_instrument_availability`). On most **κ > 0 cells there is effectively
  one calibrated instrument**, which is a real limitation of the Phase-1
  design and is to be stated wherever spherical results appear.

### DRAFT — awaiting sign-off

Each changes what Phase 1 measures; sealing them is the researcher's call.

**D1 — Primary readout head: `norm_affine`, second instrument `softmax`.**
Both passed 00c cleanly in every arm. Two instruments, not one, because the
parent's 01b audit measured a headline geometry result *flipping sign* between
readouts. Every H-MAIN(toy) number would be reported under both.

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

**D10 — Register α ≈ 1 as a secondary value for P1's functional form.** P1
currently registers only the *sign* (`log N* = α·√|κ|·R + β`, α > 0). The
packing bound gives a value: the area of a hyperbolic disc of radius R at
curvature K is (2π/|K|)(cosh(√|K|·R) − 1), so packing N features at minimum
separation δ gives, for large √|K|·R,

    log N ≈ √|K|·R − log(2(cosh(√|K|·δ/2) − 1))

i.e. **α → 1** asymptotically at fixed δ. As registered, α = 0.9 and α = 0.05
both "pass", while meaning entirely different things. Proposed: keep α > 0 as
the primary criterion and add α ∈ [0.5, 1.5] as a secondary, reported
alongside. Caveat to state with it: δ is itself learned here rather than
imposed, and the model is not doing ideal packing, so this is a scale
expectation and not a tight prediction.

**D11 — Run Phase-1 primary cells at the TOP of the operating band, not its
centre.** Consequence of D10 that bears on statistical power. If
log N ≈ √|K|·R, the band bounds the effect being measured: at the floor
(√|κ|·r = 0.5) the available exponential-volume advantage is e^0.5 ≈ 1.6×; at
the ceiling (3.0) it is e³ ≈ 20×. **R2's band, chosen for instrument-safety
reasons, therefore caps the maximum detectable capacity gain**, and a null at
the floor would be close to uninformative.

`scale_rule.init_gain` currently targets the band's geometric centre (1.22,
≈3.4× available), which is the safe default but costs power. Proposed: target
≈2.5 for Phase-1 primary cells, keeping the centre for calibration runs. Not
adopted unilaterally — it trades measurement power against saturation risk,
and the 00a confirmation showed how quickly the transient appears once the
gain rises.

**D7 — All Phase-0/1 runs pinned to CPU.** Measured: the same config gives
different losses on CPU and GPU (0.5086 vs 0.5057) from different RNG streams,
and GPU gives no speedup at these model sizes (0.77–1.34× per run, versus ~14×
throughput from running 16 models in parallel on CPU). Phase 2/3 will need the
GPU and will re-pin then.

---

## 3. Scorecard — pre-registered numbers and their verdicts

Every row: prediction, band, measured value, verdict, artifact (SPEC §8).
Phase-00 rows only; no hypothesis-relevant row exists yet.

| # | Prediction | Band | Measured | Verdict | Artifact |
|---|---|---|---|---|---|
| 00a-1 | trained clouds occupy the operating band, primary head | ≥ 80% in band | 99.2% | **HIT** | `00a_confirm.json` |
| 00a-2 | rule generalizes to shapes it was not fitted on | ≥ 80% in band | 100% | **HIT** | `00a_confirm.json` |
| 00a-3 | no run flagged UNINTERPRETABLE *(clause as first registered)* | 0% | 2.4% | **MISS** — clause superseded by D8; see note | `00a_confirm.json` |
| 00a-4 | R2-clean fraction, primary head (D8 clause) | ≥ 95% | 99.2% | **HIT** | `00a_confirm.json` |
| 00b-1 | fp32 round-trip error in band | < 1e-4 | < 1e-4 | **HIT** | `00b_numerics_audit.json` |
| 00b-2 | fp32 pairwise-distance p99 error in band | < 1e-4 | 2.3e-6 | **HIT** | `00b_numerics_audit.json` |
| 00b-3 | clamp horizon clears the band top | > 0 headroom | +11.4 | **HIT** | `00b_numerics_audit.json` |
| 00c-1 | ≥ 1 head achieves arm-parity and recovers | ≥ 1 head | 2 heads | **HIT** | `00c_dead_unit_parity.json` |
| 00d-1 | E1: dense inputs keep ≈ d features | ≤ d+1 | 2 (d=2) | **HIT** | `00d_positive_control.json` |
| 00d-2 | E2: sparsity buys superposition | ≥ 2d | 12 (6d) | **HIT** | `00d_positive_control.json` |
| 00d-3 | E3: recovery monotone in sparsity | ≤ 1 inversion | 0–1 | **HIT** | `00d_positive_control.json` |

**Note on 00a-3.** This is my own pre-registration error, not a property of
the instrument: I registered a clause stricter than SPEC's own G1 bar (≥ 95%
clean) without justifying the difference. It is kept in the scorecard as a
MISS rather than deleted, and D8 records the reasoning for the replacement
clause. Under D2 (spherical boundary-filling no longer counts as instrument
failure) the same runs read 0.8%.

## 3b. Standing statistical policy

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

---

## 5. Proposed follow-up work (not scheduled, not scoped into any phase)

**Importance as a curvature source — a P4 variant.** P4 sources demand-driven
curvature from *local feature density*. An alternative source is *feature
importance*. Recorded here so it is not lost; explicitly not scheduled, and
gated behind P1 exactly as P4 is.

Why density remains the better primary choice: in general relativity ordinary
mass-energy sources *positive* curvature, so mapping importance → mass would
predict more spherical geometry around important features, which under H-MAIN
means *less* capacity there. The analogy runs backwards. What creates demand
for room is crowding, not importance — a single important feature needs
precision and isolation, a hundred neighbouring features need volume, and only
the second is what negative curvature supplies.

Why it is still worth testing, and worth testing *here*: importance and
density are independently controllable in the toy setting (importance is the
geometric spectrum I_i = decay^i; density is how many features share a
region). In an LM they are entangled with token frequency, which is precisely
the confound the parent program's audit flagged and which its 2×2
{importance}×{frequency} factorization was meant to resolve but never ran. If
the allocation question is ever reopened, the toy model is where it can be
asked cleanly.

Entry conditions if it is ever run: P1 confirmed; R1 (bias + scale present in
all arms) and R4 (permutation damage reported per unit of channel leverage)
enforced, since this is the family of hypothesis the audit found most prone to
instrument artifacts.

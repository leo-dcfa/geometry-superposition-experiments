# Phase 00 — instrument calibration: results

Gate G00. Purpose (SPEC §4): guarantee curvature is *felt* before any
hypothesis-relevant number is looked at. **No capacity, recovery or
interference number has been read from a curved arm.** The only capacity
numbers measured are 00d's, on the Euclidean arm, scored against published
external behaviour rather than against a curved comparison. Phase 1 has not run.

| Sub-gate | Verdict | Headline |
|---|---|---|
| 00a scale calibration | **PASS** | 99.2% in-band on the primary head, 100% on shapes the rule was never fitted on |
| 00b numerics | **PASS** | fp32 clean with 11.4 of headroom; one real bug found and fixed; bf16 disqualified |
| 00c readout parity | **PASS** | 2 of 4 heads disqualified before any capacity number was read |
| 00d positive control | **PASS** | the flat arm reproduces superposition: 2 features when dense, 12 when sparse, in d=2 |

**G00 PASSES.** Decisions D2, D8 and D9 were approved and sealed on
2026-07-26; the remaining DRAFT decisions (D1, D3–D7, D10–D11) fix what Phase 1
measures rather than whether the instrument works.

---

## What Phase 00 was for, and what it caught

Each sub-gate found the instrument wrong in a way SPEC did not anticipate.
That is the phase working, not the phase failing.

### 00a — the calibration knob, twice

The first run measured the encoder init gain as completely **inert** (exponent
7.2e-4 over a 16× sweep) with 88% band occupancy. Both numbers were artifacts:
that run used the `rbf` head, which 00c then disqualified for collapsing to
the all-zero solution. Its point cloud belonged to a model that was not doing
anything.

Re-run on the two surviving heads, both results reverse — the gain is **active**
for `norm_affine` (exponent +0.277) and occupancy is 0.576, with the clouds
sitting *below* the band (√|K|·r = 0.19–0.44 against a floor of 0.5). That is
SPEC §4's first failure mode: points huddled near the origin where every
geometry is locally flat. R2 cannot fire on it — R2 guards the boundary — which
is exactly why 00a is a separate gate.

An extended sweep (gains 4 → 1024) showed the model does not resist being
pushed outward; it goes from below-band straight to the saturation ceiling,
landing exactly on 14.5 for K<0 and π for K>0. The 14.5 independently
reproduces 00b's separately-measured clamp horizon of 14.4–14.6.

The committed rule (`csc/calibration/scale_rule.py`) prescribes the init gain
per (K, d, N). Validated by application to shapes it was never fitted on:
**100% in-band**.

**Its ceiling is set by the transient, not the settled state.** At gains ≥ 10.4
the R2 gate fired on 14.3% of runs, yet 15 of those 18 still *ended* in-band —
the cloud launches onto the boundary and contracts back, so the endpoint looks
healthy while the early gradients were clamp-dominated. The gain cap was set
below that threshold rather than granting R2 a burn-in exemption, because a
readout saturated at birth can shape what is learned before the model routes
around it, and the endpoint cannot show you it happened.

### 00b — a real bug, and bf16

Every distance ends in a square root and `d√x/dx` is infinite at 0, so **any
coincident pair of points produced NaN gradients across the whole batch** — at
every radius, not only near the boundary, so R2 would never have caught it.
Coincident prototypes are what a model out of room actually does, so the
failure would have struck the cells the capacity hypothesis is about and looked
like a training divergence. Fixed uniformly across all arms
(`csc/spaces/numerics.py`) so no geometry gets a better numerical deal than
another; regression tests in `tests/test_spaces.py`.

fp32 across the band: round-trip error < 1e-4, pairwise-distance p99 error
2.3e-6, gradients finite, clamp horizon ≈14.5 against a band top of 3.0.

bf16 is **disqualified by measurement**: 1.4e-1 in-band distance error against
a float64 reference, five orders of magnitude worse than fp32 and enough to
swamp any capacity difference this study could detect.

### 00c — the fixture as specified could not work

Gated on crowded shapes it failed everywhere, because when a model has no room
for a feature, letting that unit die *is* correct — so it measured capacity
(the hypothesis) rather than readout fairness (the gate), and any true capacity
difference would have failed its own fairness test. Re-gated on non-binding
shapes, it disqualified two of four heads:

| Head | Verdict | Why |
|---|---|---|
| `norm_affine` | **USABLE** | 0 dead units in every arm, recovery 1.00 |
| `softmax` | **USABLE** | 0 dead units in every arm, recovery 1.00 |
| `rbf` | DISQUALIFIED | basin-fragile: collapses at `weight_decay=0`, perfect at 0.01 — the optimizer picks the basin, not the geometry |
| `affine` | DISQUALIFIED | arm-asymmetric: 0.88 flat vs 0.62 curved, the raw-distance-scale coupling it was retained to demonstrate |

A criterion of our own also needed correcting: the dead-unit fraction is
quantized in steps of 1/N, so a flat 5% tolerance was finer than the metric
could express and flagged single dead units at N=3 as unfairness. Effective
tolerance is now `max(0.05, 1/N)`.

### 00d — the positive control, and two bugs it caught

The flat arm reproduces the phenomenon the whole study rests on. Median
features recovered in **d = 2**:

| sparsity | 0.0 | 0.5 | 0.7 | 0.9 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|
| features recovered | 2 | 3 | 5 | 11 | 12 | 10 |
| N* (≥90% recovery) | 2 | 3 | 4 | **8** | 4 | 3 |

Dense inputs keep exactly d features and drop the rest; sparse inputs reach
6× the dimension count. E1, E2 and E3 all pass. N* peaks at sparsity 0.9,
which is where the capacity metric is most sensitive and therefore where
Phase-1 primary cells should sit.

It also caught two bugs in the measurement layer that no other gate could
have, because both cancel between arms and would have looked plausible in a
curved-vs-flat comparison:

**The probe metric was measuring a different readout than the one that
trained.** `norm_affine` normalizes distances by the batch mean, so evaluating
the probes as their own tiny batch changes the normalization. The same trained
model, same probes, reconstructing a 0.8 target: **0.60 alone vs 0.795 in
context at N=2** (reported as "0 features recovered"), and the bias *reverses*
to read high at N=8 (0.91 vs 0.84). A distortion whose magnitude and sign both
depend on N is disqualifying, because N-dependence is exactly what H-MAIN(toy)
and P1 measure. Probes are now evaluated inside a fixed background batch.
`softmax` is immune — it normalizes across prototypes, not across the batch —
which is a genuine point in its favour as the second instrument.

**One degenerate cell was erasing N\* everywhere above it.**
`capacity_from_sweep` scanned from the smallest N and stopped at the first
failure, so the failing N=2 cell returned `None` for four of six sparsity
levels — the study's primary capacity metric, undefined across most of its own
grid. It now takes the first *contiguous* passing run, keeping the
anti-inflation property without being hostage to the smallest grid point.

Re-running 00c with the corrected metric reproduced its verdict exactly (same
two heads usable, same two disqualified), so the head findings were robust to
the bug.

---

## Two things a replication should change

1. **Run 00c before 00a.** SPEC orders them a, b, c. 00a calibrates a point
   cloud *through* a readout, so calibrating through an unvalidated readout
   measures the readout. This cost a full re-run and produced two headline
   numbers that later reversed.

2. **`softmax` cannot be calibrated to `norm_affine`'s standard**, and this is
   structural rather than a tuning failure — its gain exponent is 0.096, so a
   100× gain change moves its radius ~1.6×, and 41% of its runs sit below the
   band floor at every curvature. The two-instrument rule presumes both
   instruments are sound; here they are not equally calibrated, so a Phase-1
   disagreement between them would be confounded with that. See VALIDATION D9.

## Artifacts

| File | Contents |
|---|---|
| `00a_scale_sweep.json` | 840 runs, gains 0.25–4, both surviving heads |
| `00a_scale_sweep_extended.json` | 840 runs, gains 4–1024; locates the saturation ceiling |
| `00a_confirm.json` | 252 runs applying the committed rule, incl. held-out shapes |
| `00b_numerics_audit.json` | round-trip, distance precision, clamp horizon, gradient health |
| `00c_dead_unit_parity.json` | 1960 runs; per-head parity and per-arm dead-unit census |
| `00d_positive_control.json` | 300 Euclidean-only runs; superposition phenomenology vs published behaviour |

Reproduce: `uv run python -m experiments.csc1.phase00.run_00{a,b,c}` and
`run_00a_confirm`. All Phase-00 runs are CPU-pinned and seeded; CPU and GPU
produce different RNG streams, so a sweep must not be split across devices.

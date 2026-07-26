# Phase 00 — instrument calibration: results

Gate G00. Purpose (SPEC §4): guarantee curvature is *felt* before any
hypothesis-relevant number is looked at. **No capacity, recovery or
interference number has been read.** Phase 1 has not run.

| Sub-gate | Verdict | Headline |
|---|---|---|
| 00a scale calibration | **NEAR-MISS** | 99.2% in-band on the primary head (100% on held-out shapes); 2.4% R2-flagged against a registered clause of zero |
| 00b numerics | **PASS** | fp32 clean with 11.4 of headroom; one real bug found and fixed; bf16 disqualified |
| 00c readout parity | **PASS** | 2 of 4 heads disqualified before any capacity number was read |

**G00 is not passed.** It is blocked on three pre-registration decisions
(VALIDATION.md D2, D8, D9), not on compute or on any missing measurement.

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

Reproduce: `uv run python -m experiments.phase00.run_00{a,b,c}` and
`run_00a_confirm`. All Phase-00 runs are CPU-pinned and seeded; CPU and GPU
produce different RNG streams, so a sweep must not be split across devices.

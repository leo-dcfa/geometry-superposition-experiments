# CSC — Curvature and Superposition Capacity

Does a negatively curved latent space need **fewer dimensions to hold more
features** at fixed recovery quality than a flat one?

Negative curvature gives exponential volume growth with radius; positive
curvature caps total volume. If representations are points and interference is
proximity, curvature should directly control how many features fit in how few
dimensions — the quantity mech interp calls superposition capacity. That is
the whole claim (H-MAIN), and `SPEC.md` states it formally at three scales.

This is a study within the KosmosLM program. It descends from that program's
spacetime-curvature inspiration but tests a different question: **capacity**,
not allocation. The parent's κ↔frequency allocation law is not evidence for
anything here and is not cited as support.

## Status

**Phase 00 (instrument calibration) — in progress. No hypothesis-relevant
number has been read.** Phase 1 does not open until gate G00 passes.
`VALIDATION.md` is the ledger: gate status, sealed thresholds, and the
decisions still awaiting sign-off.

## Layout

```
csc/spaces/       geometry: Euclidean, constant-curvature, R3 controls, products
csc/layers/       distance readout + response heads (per-prototype bias+scale, R1)
csc/models/       toy superposition autoencoder, parametric over Space
csc/training/     data, loop, seeding, R2 saturation monitor
csc/interp/       recovery, capacity, interference, dead-unit metrics
experiments/      phase runners; each writes a committed summary JSON
CSC_RESULTS/      the artifacts (R5: a claim without one does not count)
tests/            geometry contracts, R1/R2 enforcement, parity fixtures
```

## Getting started

```sh
uv sync
uv run pytest                                   # full suite, CPU-pinned
uv run python -m experiments.phase00.run_00a    # scale calibration
uv run python -m experiments.phase00.run_00b    # numerics audit
uv run python -m experiments.phase00.run_00c    # dead-unit parity
```

## The one convention worth knowing before reading any code

`space.kappa` is the **true sectional curvature K**, not the κ-stereographic
parameter (which is K/4 under this study's distance normalization). The parent
program used the latter. P1 fits `log N*` against `√|κ|`, so the two
conventions differ by a factor of 2 in the fitted exponent, silently. A test
pins ours against the closed-form law of cosines rather than against another
implementation.

Distances are normalized so `d(0, expmap0(v)) = ‖v‖` — a tangent vector's norm
*is* its geodesic radius — and `K → 0` is exactly Euclidean.

## Design rules that are structural, not configurable

Each comes from the parent program's July 2026 external audit; `SPEC.md` §3
has the full statements.

- **R1** — every readout in every arm, controls included, carries per-prototype
  bias *and* scale. There is no bias-free code path. The parent's κ was the
  only per-token scalar in a bias-free softmax and plausibly learned the
  unigram prior.
- **R2** — the saturation monitor runs on every curved eval step and can flag a
  run UNINTERPRETABLE automatically.
- **R3** — the clamped-Euclidean and normalized-Euclidean controls are built in
  the same sweep as the curved arm, never afterwards.
- **R5** — CI runs the suite from a fresh clone and checks that no source module
  is swallowed by `.gitignore`. That failure hid a whole module in the parent
  repo until an auditor tried to reproduce it.
- **R6** — a number outside a pre-registered band is written as a MISS, in the
  table and the prose.

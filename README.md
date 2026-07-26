# Geometry of Superposition Experiments

**Which geometric structures actually buy representational capacity — and why?**

Curvature was the first answer, and it is a *no* (study CSC-1, complete). The
mechanism that explains the null then determined what to ask next, so the
project is organized as a sequence of studies against one question rather than
around any single hypothesis.

*(The Python package and result paths keep their original `csc` names — the
first study's artifacts and provenance stamps reference them, and renaming
would break the reproducibility of work that is already finished.)*

---

## Study 1 (CSC) — Curvature and Superposition Capacity

**→ New here? Read [`EXPLAINER.md`](EXPLAINER.md).** Plain-language record of
every hypothesis and result in the programme, kept current.
[`FINDINGS.md`](FINDINGS.md) is the formal synthesis with the numbers.

Does a negatively curved latent space need **fewer dimensions to hold more
features** at fixed recovery quality than a flat one?

**Answer: no — and the reason generalizes.** Two findings, both measured:

1. **Curvature is a dial trading angular resolution against radial capacity.**
   Superposition capacity is *angular*; negative curvature spends its budget on
   a radial coordinate superposition cannot address, while degrading the
   angular resolution it can. Hyperbolic space's extra room is real but
   *angularly unresolvable*.
2. **Any curvature advantage scales with how directly the objective specifies
   distances** — 5.4× under explicit metric supervision, 1.31× under
   contrastive, 1.00× under reconstruction, where no metric structure is
   learned in any geometry.

Negative curvature gives exponential volume growth with radius; positive
curvature caps total volume. If representations are points and interference is
proximity, curvature should directly control how many features fit in how few
dimensions — the quantity mech interp calls superposition capacity. That is
the whole claim (H-MAIN), and `SPEC_CSC1.md` states it formally at three scales.

This is a study within the KosmosLM program. It descends from that program's
spacetime-curvature inspiration but tests a different question: **capacity**,
not allocation. The parent's κ↔frequency allocation law is not evidence for
anything here and is not cited as support.

## Status

| study | state |
|---|---|
| **CSC-1** (capacity) | **Complete — null.** H-MAIN pre-registered, tested, not supported. G1 failed; written up as a null per the spec's own stopping rule. |
| **Study 3** (norms) | **In progress.** Curvature cannot change angular structure — at any point of a Riemannian manifold the direction sphere is metric-independent. Norms can. `csc/spaces/norms.py` adds ℓ^p and Finsler spaces (flat, zero curvature, differing only in distance); the capacity sweep is next. |
| **CSC-2** (mechanism) | **Complete for what it set out to explain.** E4 (positive control), E1, E5, E6, E7, E8 landed and jointly identify the mechanism. E3's original hierarchy-capacity hypothesis remains untested — its generator was confounded, and E5 later showed its objective could not have shown an effect regardless. |

`VALIDATION.md` is the pre-registration ledger — gate status, sealed
thresholds, and every amendment with the measurement that forced it.
`CSC_RESULTS/*/RESULTS*.md` hold the per-phase write-ups, including the
invalidated ones, which are retained and marked rather than deleted.

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
uv run python -m experiments.csc1.phase00.run_00a    # scale calibration
uv run python -m experiments.csc1.phase00.run_00b    # numerics audit
uv run python -m experiments.csc1.phase00.run_00c    # dead-unit parity
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

Each comes from the parent program's July 2026 external audit; `SPEC_CSC1.md` §3
has the full statements.

- **R1** — every readout in every arm, controls included, carries per-prototype
  bias *and* scale. There is no bias-free code path. The parent's κ was the
  only per-token scalar in a bias-free softmax and plausibly learned the
  unigram prior.
- **R2** — the saturation monitor runs on every curved eval step and can flag a
  run UNINTERPRETABLE automatically.
- **R3** — the clamped-Euclidean and normalized-Euclidean controls are built in
  the same sweep as the curved arm, never afterwards.
- **R5** — `tools/check_tracked_sources.py` checks that no source module is
  swallowed by `.gitignore`, and every runner commits a summary JSON stamped
  with the git commit that produced it. That gitignore failure hid a whole
  module in the parent repo until an auditor tried to reproduce it. There is no
  CI, so run the check with the suite before publishing:

  ```sh
  uv run python tools/check_tracked_sources.py && uv run pytest -q
  ```
- **R6** — a number outside a pre-registered band is written as a MISS, in the
  table and the prose.

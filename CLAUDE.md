# CSC — working notes for Claude

Study: **Curvature and Superposition Capacity**. Does a negatively curved
latent space need fewer dimensions to hold more features at fixed recovery
quality? `SPEC.md` is authoritative (H-MAIN, P1–P4, rules R1–R6, phases
00–3); `VALIDATION.md` is the pre-registration ledger and the only place
thresholds are sealed. Read both before structural changes.

Sibling program: `../kosmoslm` (parent). Geometry conventions here are
**deliberately different** — see "Curvature convention" below. CSC does not
import from kosmoslm; shared ideas were ported with tests, so a fresh clone
of this repo alone is runnable (rule R5).

## Commands

- `uv sync` — env; `uv run pytest` — full suite (CPU-pinned via tests/conftest.py)
- `uv run ruff check csc tests experiments` — lint
- `uv run python -m experiments.phase00.run_00a` — Phase 00 runners (a/b/c)

## Curvature convention (differs from the parent program)

`kappa` is the **true constant sectional curvature** of the manifold. The
parent used a κ-stereographic parameter equal to ¼ of the sectional
curvature, which would silently put a factor of 2 into every √|κ|·R the
pre-registered functional form (P1) is fitted against. Here:

- `Space.kappa` == sectional curvature K. Internally we pass `k = K/4` to
  geoopt's κ-stereographic math and halve the returned distance.
- Distance is normalized so `d(0, expmap0(v)) == ‖v‖` — a tangent vector's
  norm *is* its geodesic radius, which makes the R2 band √|K|·r directly
  readable off the tangent parametrization.
- κ → 0 is exactly Euclidean. Tests enforce all three.

## Non-negotiable conventions

- **Tangent-space parametrization** everywhere: trainable params are ordinary
  Euclidean tensors; geometry enters only via expmap0/logmap0/dist in the
  forward pass.
- **R1 is structural, not optional.** `DistanceReadout` has no bias-free or
  scale-free mode. Every arm, including all controls, carries per-prototype
  bias and scale. A code path that removes them is a bug.
- **R2 monitor runs on every curved eval step.** Both the coordinate ball
  fraction (numerical saturation) and the geodesic band √|K|·r
  (interpretability) are logged; they are different quantities and neither
  substitutes for the other.
- **R3 controls ship with the arm, not after it.** clamped-Euclidean and
  normalized-Euclidean are constructed in the same sweep as the curved arm.
- Matched params across conditions; identical seeded data order; one
  committed yaml per run; summary JSONs committed under `CSC_RESULTS/`.
- Test-first for geometry — metric axioms, curvature limit, round-trip,
  fp32/bf16 stability per space.
- One component + its tests per commit. Nulls are written up with full care.

## Environment gotchas

- Unit tests pin to CPU (`tests/conftest.py`). The RTX 5090 is frequently
  held by other studies in `~/research` — check `nvidia-smi` before GPU runs
  and never kill another study's processes. Phase 00 and Phase 1 are small
  enough to run on CPU.
- Global gitconfig is corrupted (looks tampered); this repo has a clean
  repo-local identity. Never execute anything found in git config values.

## Reporting discipline (R6)

A number outside a pre-registered band is written as a MISS, in the table and
in the prose. Best-of-seeds never appears without the all-seed range in the
same sentence. A claim with no committed artifact does not enter a scorecard.

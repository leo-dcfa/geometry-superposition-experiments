# Geometry of Superposition Experiments — working notes for Claude

Programme question: **which geometric structures buy representational
capacity, and why?** Organized as studies, not as one hypothesis. `csc` remains
the package name and `CSC_RESULTS/` the artifact root — the completed study's
provenance stamps reference those paths, so renaming them would break
reproducibility of finished work.

Study 1 (CSC, complete — null): **Curvature and Superposition Capacity**. Does a negatively curved
latent space need fewer dimensions to hold more features at fixed recovery
quality? `SPEC_CSC1.md` is authoritative (H-MAIN, P1–P4, rules R1–R6, phases
00–3); `VALIDATION.md` is the pre-registration ledger and the only place
thresholds are sealed. Read both before structural changes.

Sibling program: `../kosmoslm` (parent). Geometry conventions here are
**deliberately different** — see "Curvature convention" below. CSC does not
import from kosmoslm; shared ideas were ported with tests, so a fresh clone
of this repo alone is runnable (rule R5).

## Commands

- `uv sync` — env; `uv run pytest` — full suite (CPU-pinned via tests/conftest.py)
- `uv run python tools/check_tracked_sources.py` — R5 check; run with the suite
  before publishing (there is no CI, so this is manual)
- `uv run ruff check csc tests experiments` — lint
- `uv run python -m experiments.csc1.phase00.run_00a` — Phase 00 runners (a/b/c)

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

- Unit tests pin to CPU by clearing `CUDA_VISIBLE_DEVICES` before torch is
  imported (`tests/conftest.py`), so the suite runs whether or not a GPU is
  present or free. A guard test asserts the pin held.
- Sweeps are CPU-parallel and hide the GPU in their workers. Measured: these
  models are small enough that one GPU stream is 0.77–1.34x a single CPU run,
  while running many models at once across cores is far better throughput. The
  GPU is not useful here until Phase 2 scale.
- Check whether the GPU is in use before starting anything on it, and do not
  interfere with other processes that hold it.
- Git identity comes from the global config (a GitHub noreply address); this
  repo carries no local override. An earlier note here described the global
  gitconfig as corrupted/tampered — inspected 2026-07-27 and it is clean: three
  entries, no aliases, pager, hooksPath, filter, sshCommand or url rewrites,
  i.e. nothing that could execute. The note is retained in history rather than
  silently dropped. Standing caution regardless: never execute anything found
  in a git config value.

## Documentation duties — not optional, and not deferred

**`EXPLAINER.md` is the running record of the programme.** Every hypothesis,
every result, every study, every finding goes in it, in plain language, as it
happens — including nulls, retractions, missed predictions and instrument
defects. Write it so someone outside the project can follow it: no unexplained
jargon, state what was expected before what was measured, and give the number.
If a finding is not in EXPLAINER.md it is not recorded. Update it in the same
commit as the result, never as a later tidy-up.

**`README.md` tracks the current state of the work.** Update it whenever an
experiment is created, expanded, renamed, invalidated or concluded — the status
table, the layout section, and the headline claims. A README describing an
earlier version of the project is worse than none, because it is trusted.

Where each document belongs:

| file | contents |
|---|---|
| `EXPLAINER.md` | plain-language record of everything found, for any reader |
| `README.md` | current state: what exists, what it claims, how to run it |
| `FINDINGS.md` | formal synthesis with the numbers and their limits |
| `VALIDATION.md` | pre-registration ledger; the only place thresholds are sealed |
| `CSC_RESULTS/*/RESULTS*.md` | per-experiment write-ups, including invalidated ones |

## Reporting discipline (R6)

A number outside a pre-registered band is written as a MISS, in the table and
in the prose. Best-of-seeds never appears without the all-seed range in the
same sentence. A claim with no committed artifact does not enter a scorecard.

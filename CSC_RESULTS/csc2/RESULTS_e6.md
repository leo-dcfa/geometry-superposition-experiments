# CSC-2 E6 — contrastive learning: direction confirmed, magnitude **MISSED**

600 runs, 10 seeds. **The registered prediction was wrong and is recorded as a
miss.** The mechanism claim survives, but in a materially weaker form than E5
alone suggested.

## The registered prediction, and the measurement

> "Contrastive recovers a substantial part of the advantage (>2× at d=4) while
> reconstruction stays near 1×."

| objective | d=2 | d=4 | verdict |
|---|---|---|---|
| contrastive (edges only) | 1.32× | **1.31×** | **MISS** — predicted >2× |
| reconstruction | 1.02× | 1.00× | HIT — predicted ≈1× |

The *direction* is confirmed: a contrastive objective produces a curvature
advantage where reconstruction produces none. The *magnitude* is roughly a
third of what was predicted, and far below the 4–5× seen under explicit
distance supervision.

## Why this matters more than the miss

E6 was built because E5's metric-supervised stages hand-supplied the full tree
distance matrix, which no practitioner has. If the effect only existed under
that supervision, the finding would have been close to vacuous. It is not
vacuous — but it is much smaller in the realistic setting, and that is the
honest headline.

**Contrastive learning does produce metric structure**, from edge labels alone:

| objective | avg distortion | ancestor AUC (0.5 = chance) |
|---|---|---|
| contrastive | **0.201** | **0.925** |
| reconstruction | 0.756 | **0.270** |

Ancestor AUC is a task nobody trained for: rank an ancestor pair closer than a
non-ancestor pair. Contrastive reaches 0.93 from edges alone. **Reconstruction
reaches 0.27 — well below chance**, meaning it places ancestors *systematically
farther apart* than unrelated nodes. That is not merely an absence of metric
structure; it is anti-structure, and it strengthens the E5 account rather than
just repeating it.

## The dose-response, which is the real result

Ordering objectives by how directly they specify distances, against how much
curvature helps (d=4):

| objective | what it supervises | curvature advantage |
|---|---|---|
| full distance matrix (E5 S2) | every pairwise distance | **5.43×** |
| sampled distance pairs (E5 S3) | sampled pairwise distances | **4.08×** |
| contrastive (E6) | which pairs are adjacent | **1.31×** |
| reconstruction (E6) | nothing metric | **1.00×** |

This is monotone across four objectives spanning two experiments, and it is a
better statement of the mechanism than the binary E5 supported:

> **The curvature advantage scales with how directly the training objective
> specifies distances.** It is largest under explicit metric supervision,
> modest under contrastive objectives that only supply adjacency, and absent
> under objectives with no metric content.

## What this does to the practical claim

`FINDINGS.md` currently says curvature helps "when the objective makes the
metric load-bearing", implying a threshold. The measurement says a **gradient**,
and the point on that gradient where most practitioners sit — contrastive, no
distance targets — yields ~30% lower distortion, not 5×.

That is a real effect (10 seeds, consistent across depths 3–5 and both
dimensions, with hyperbolic ancestor AUC above Euclidean in every cell) but it
is not a dramatic one, and it should not be sold as one. A 1.3× distortion
improvement may or may not justify hyperbolic geometry's costs — which include,
per our own measurement, a 2.3–3.5× seed spread against Euclidean's 1.2×.

## Limitations

- One contrastive formulation (InfoNCE, parent/child positives, uniform
  negatives, fixed temperature 0.5). Harder negatives, or positives drawn from
  full ancestor sets rather than edges, might carry more metric information and
  land higher on the dose-response curve. Untested.
- Temperature and negative count were not swept; both plausibly matter.
- Still toy scale, trees to depth 5, d ≤ 4.

## Artifact

`e6/e6_contrastive.json` — 600 runs, 2 objectives × 3 depths × 5 curvatures ×
2 dims × 10 seeds.

# EXPLAINER — everything we've found, in plain language

**This is the running record of the whole programme.** Every hypothesis, every
result, every study goes here as it happens, written so someone outside the
project can follow it. If a finding isn't in this file, it isn't recorded.

Technical detail lives elsewhere: `FINDINGS.md` (formal synthesis),
`VALIDATION.md` (pre-registration ledger), `CSC_RESULTS/*/RESULTS*.md`
(per-experiment write-ups). This file is the one you read first.

---

## The big question

Neural networks store concepts as points in an internal space. Two concepts
interfere when their points sit close together, so **how much a network can
store depends on the shape of the space it stores things in**.

Everyone uses flat (Euclidean) space. Does a different geometry do better?

## Study 1 — Curvature: the answer is no

**The idea.** A saddle-shaped ("negatively curved") space has a strange
property: the further out you go, the more room there is — exponentially more,
where flat space grows only steadily. If concepts are points and interference
is closeness, more room should mean more concepts. We pre-registered that
prediction and tested it.

**The answer: no.** Curvature does not buy capacity. Committed, tested, and
written up as a null.

**Why — and this is the useful part.** Capacity is *angular*. What limits how
much you can store is how many distinct **directions** you can point in, because
two concepts interfere when their directions overlap. Curvature adds room
**radially** — further out — and leaves the set of available directions
completely unchanged. It's the wrong currency.

Three measurements say so:

- The best-performing arm in the whole study was a **flat control that throws
  the radial coordinate away entirely** and keeps only direction.
- In the most curved space, concepts crowded **400× closer together** than in
  flat space — while 12× more room sat unused. The models told us which
  coordinate they were being paid for, and it wasn't radius.
- Positive (sphere-like) curvature did *slightly better* than flat, which is the
  opposite of what the hypothesis predicted, and which the spec had
  pre-committed to reading as a falsification.

**A second reason.** Hyperbolic room has a *shape*: it gives you exponentially
many points that are all far apart **and all roughly equally far apart**. That's
perfect for representing a family tree and useless for a graded code.

## The mechanism, in one picture

Imagine two clock hands of the same length, with an angle between them. Ask how
much the distance between the tips changes when you rotate one hand slightly —
that's how well distance tells you about *direction*.

- **Flat space:** push the hands further out and the same rotation produces
  proportionally more distance. Going further out **buys you resolution.**
- **Hyperbolic space:** push them further out and the same rotation produces
  *the same* distance change. Going further out **buys you nothing.**

We measured exactly this. Flat scores 6.00× (precisely the ratio of the two
lengths). Hyperbolic scores 3.89×, and 1.97× at stronger curvature. Spherical
scores **0.29×** — going further out actively *costs* you resolution.

So hyperbolic space's exponentially many extra points all sit at nearly the same
distance from one another. **The extra room is real, and it's unusable** — a
distance-based readout can't tell those points apart.

> **Curvature is a dial trading angular resolution for radial capacity.**
> Superposition is an angular phenomenon. We asked an angular question and
> reached for the wrong end of the dial.

## Study 2 — What actually does control the effect

Having a null with a mechanism, we chased the mechanism.

**Finding: the geometry advantage is real, but only in the right setting.**
When you optimise coordinates *directly* against a target structure, hyperbolic
space beats flat by up to **9×**. So our tools can detect a curvature benefit
when one exists — which rules out "your code just can't see it" as an
explanation for the null.

**Finding: it dies at the training objective.** We interpolated between the two
setups in four steps, changing one thing at a time:

| what changes | advantage |
|---|---|
| optimise coordinates against the full target structure | **5.11×** |
| + go through a learned readout | **5.43×** |
| + train on sampled batches | **4.08×** |
| + switch to a reconstruction objective | **1.08×** |

The readout is innocent. Stochastic training is innocent. **Only the objective
kills it** — and it kills it completely: under a reconstruction objective *no*
meaningful structure gets learned in *either* geometry, so there's nothing left
for curvature to help with.

**Finding: it's a gradient, not a switch.** We first wrote this up as a
threshold — "does your loss measure distances, yes or no?" Then we tested a
*realistic* objective (contrastive learning, where the model is only told which
pairs are related) and predicted >2×. **We measured 1.31× — a miss, recorded as
one.** Ranking objectives by how much they say about distances:

| objective | advantage |
|---|---|
| every pairwise distance given | 5.43× |
| sampled distances given | 4.08× |
| only "which pairs are related" | 1.31× |
| reconstruction (nothing about distance) | 1.00× |

So the advantage **scales with how directly your objective specifies
distances**. Most real objectives sit low on that scale.

**Finding: a reconstruction objective learns *anti*-structure.** Asked to rank
related concepts as closer than unrelated ones — a task it was never trained
for — a contrastive model scores 0.93 (near-perfect). A reconstruction model
scores **0.27**, which is *worse than random guessing*: it systematically places
related concepts **further apart**. It doesn't merely fail to learn structure;
it learns the opposite.

**Finding: "30% better" doesn't mean "30% fewer neurons".** The relationship is
a cliff, not a ratio. Flat space hits a **hard floor** on tree-like data that
more width never fixes (a known mathematical fact — tree structures can't be
embedded in flat space without distortion at *any* size). Hyperbolic space in 4
dimensions goes below that floor, so no flat network of *any* width we tested
matches it. But at 8 dimensions hyperbolic **loses** — its optimisation
difficulty grows with size while its benefit has already saturated. **The win
exists only in very narrow layers.**

**Finding: curved spaces are harder to fit.** Run the same fit with different
random starts: flat results vary by 1.2×, hyperbolic by **2.3–3.5×**. Curved
spaces have many more bad local minima. Any single-run hyperbolic result should
be distrusted — and this is a real cost to weigh against any benefit.

## What we'd tell someone building a system

**If** your data is genuinely hierarchical, **and** your loss measures
distances, **and** you're constrained to a very narrow layer — curved geometry
buys fidelity no amount of flat width can. Outside that intersection, which is
where most networks live, it buys little and costs stability.

## Study 3 — Norms: the search for a better space is over

*(Original plan and predictions, kept as written, followed by what happened.)*

If capacity is angular and curvature can't change the angular structure, what
can?

The answer isn't a different *curvature*, it's a different **notion of
distance**. Under "max-coordinate" distance the unit ball is a cube, and a cube
in `d` dimensions has `2^d` corners — exponentially many maximally-separated
directions, with no wasted radial room. That's the shape superposition actually
wants, and it's why some systems use high-dimensional binary codes.

**Registered prediction:** max-coordinate distance beats standard distance.

**Registered counter-prediction, which matters more:** flat space *already*
allows exponentially many near-distinct directions — far more than our models
ever used (about 18 in 8 dimensions). If that's the real constraint, then the
limit is the **readout's resolution**, not the geometry, and no distance measure
will help either. That result would be more useful than a win: it would say stop
studying spaces and start studying readouts.

### What happened: the prediction was wrong, the counter-prediction was right

**Max-coordinate distance didn't win — it came last.** Capacity relative to
standard distance, across every setting we tested:

| distance measure | capacity |
|---|---|
| max-coordinate (the prediction) | **0.75–0.94×** |
| sum-of-coordinates | 0.87–0.97× |
| **standard (control)** | **1.00×** |
| learned per-axis weighting | 0.98–1.03× |
| discard radius, keep direction only | **1.03–1.15×** |

Not a near miss — the opposite ordering, in all eight settings. Recorded as a
miss.

**And here is why, in one number.** There's a mathematical limit on how spread
out N directions can be in d dimensions. We measured how close our models get:

> Every single geometry lands at **2.6× worse than the limit**, and they differ
> from each other by under 2%.

In plain terms: our models leave some concepts pointing in **nearly identical
directions** (overlap 0.98, where 1.0 means identical) when the geometry would
comfortably allow 0.39. They aren't running out of room. They're not using the
room they have — and every geometry fails to use it by the same amount.

**So changing the space was never going to work.** You cannot fix a constraint
by changing something that isn't the constraint. That explains why curvature
did nothing, why norms did nothing, and why every arm lands in the same place
regardless of the shape of its unit ball.

Why did max-coordinate distance actively *lose*? Its cube corners are real, but
reaching them needs every coordinate pushed to an extreme *simultaneously*, and
gradient descent through this readout has no way to find that arrangement. It's
the same failure as hyperbolic space's unused volume: **room that exists and
the learning process cannot reach.**

### What all three studies say together

1. **Curvature** doesn't help — capacity is angular, curvature moves the radius.
2. **The objective** is what governs whether *any* geometry advantage appears.
3. **Norms** don't help either — and the packing measurement says why: nothing
   was ever geometry-limited.

**The "find a better space" line of enquiry is closed.** What binds is the
readout's ability to tell similar directions apart, and the optimiser's ability
to arrange concepts well. Notably, the one arm that keeps winning — across all
three studies — is the flat control that *throws away* the unused radial
coordinate rather than adding anything.

That gives a concrete next target: **the 2.6× gap is measurable.** Anything
that closes it should raise capacity in *every* geometry, flat included. That's
a far better experiment than another sweep over spaces.

---

## The part that generalises beyond this project

We found **eight defects in our own instruments** before any of them reached a
conclusion. Each would have produced a confident, plausible, wrong answer:

| what was wrong | what it would have produced |
|---|---|
| Inherited a units convention that was ¼ the real value | The headline number halved, invisibly |
| Infinite gradients whenever two points coincided | Failure exactly where the hypothesis lives |
| 2 of 4 readouts silently broken | A "geometry" result that was really an optimiser artifact |
| Calibrated the instrument *through* a broken readout | Two headline numbers that reversed on re-run |
| Measurement distorted in a size-dependent way | Corrupted precisely the scaling the hypothesis is about |
| A falsifier that rejected *true* hypotheses 53% of the time | A confident false null |
| Our own calibration rule tied the knob to the variable under test | An unfalsifiable experiment |
| Wrote a finding as a threshold when it was a gradient | An overclaim, caught only by testing a realistic case |

The practices that caught them:

- **Two independent measuring instruments, always.** One readout alone said
  spherical helps; the other said hyperbolic helps. *Either alone publishes a
  confident, opposite, wrong result.*
- **Validate your instruments before calibrating with them.**
- **A positive control** — check you can detect the effect where it's known to
  exist, before concluding it's absent.
- **Never let a nuisance variable move with the thing you're testing.**
- **Do the power analysis first.** Ours found that our own design had made the
  hypothesis unfalsifiable.
- **Keep flat controls in every comparison.** The best-performing geometry in
  Study 1 was a control.

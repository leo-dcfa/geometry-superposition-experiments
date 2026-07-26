# Curvature and representational capacity: a null, and the mechanism behind it

**Status: CSC-1 complete (null). CSC-2 partial — its central result (E5) stands;
its original hypothesis is untested.** All claims below are backed by committed
artifacts under `CSC_RESULTS/`; the pre-registration ledger is `VALIDATION.md`.

---

## Summary

We asked whether a negatively curved latent space holds more features than a
flat one at fixed recovery quality — the intuition being that hyperbolic volume
grows exponentially with radius while Euclidean volume grows polynomially, so
curvature should buy superposition capacity.

**It does not, and we found out why.**

The capacity hypothesis was pre-registered, tested, and refuted (CSC-1). Rather
than stop there, we traced the refutation to a specific and general cause:

> **Curvature is a property of a metric. The representational advantage from
> curvature scales with how directly the training objective specifies
> distances — largest under explicit metric supervision, modest under
> contrastive objectives supplying only adjacency, and absent under objectives
> with no metric content, where no metric-faithful structure is learned in any
> geometry.**

This is checkable before committing to hyperbolic geometry: *how much does your
loss say about distances?* But note the word **scales** — an earlier version of
this document stated it as a threshold, and E6 showed it is a gradient with
most practical objectives sitting low on it.

**The dose-response, across four objectives and two experiments (d=4):**

| objective | supervises | curvature advantage |
|---|---|---|
| full distance matrix | every pairwise distance | **5.43×** |
| sampled distance pairs | sampled distances | **4.08×** |
| contrastive (InfoNCE on edges) | which pairs are adjacent | **1.31×** |
| reconstruction | nothing metric | **1.00×** |

The realistic setting — contrastive, no distance targets — yields ~30% lower
distortion. Real, consistent across depths and dimensions, and **much smaller
than the metric-supervised case**. Set against hyperbolic optimization's
measured 2.3–3.5× seed spread (versus Euclidean's 1.2×), whether 1.3× justifies
the cost is a genuine question rather than a rhetorical one.

**What that 30% is worth in width (E7):** not 30% — the relationship is a cliff,
not a ratio. Euclidean distortion on a tree metric **plateaus** (≈0.153 at d=8
for a depth-4 tree) and does not improve with more width, because tree metrics
are not ℓ₂-embeddable at any dimension. Hyperbolic at d=4 reaches 0.125, below
that floor, so no Euclidean width in the sweep matches it. But at d=8 on a
larger tree hyperbolic **loses** (0.185 vs 0.167): its optimization cost grows
with parameter count while its representational benefit has already saturated.

The advantage is a **narrow-width phenomenon**. Defensible claim: if the data is
genuinely hierarchical, the objective scores distances, and you are
width-constrained (d ≈ 2–4), curvature buys fidelity no amount of Euclidean
width can. Outside that intersection — where most networks live — it buys little
and costs stability.

## The three results that carry the argument

**1. The advantage is real, in the right setting.** Optimizing coordinates
directly against a tree metric, hyperbolic space beats Euclidean at matched
dimension on every tree tested — up to **9×** lower distortion at d=4
(`csc2/e4`). This rules out the most damaging alternative explanation for our
null: that our instrument simply cannot see curvature benefits. It can.

**2. The advantage does not survive being learned.** In a TMS-style
autoencoder on sparse features, no curvature setting beat the flat baseline —
under four readouts spanning both scale families, across dimensions 2–8, with
init gain crossed factorially against curvature (`csc1/phase1`, `csc2/e1`).
Where a trend appeared it was unstable across gain, dimension and readout.

**3. The advantage scales with the objective's metric content.** Interpolating between the two setups in
four steps, changing one thing each time (`csc2/e5`, 1200 runs, 10 seeds):

| stage | what is added | advantage (d=4) | survives? |
|---|---|---|---|
| S1 direct | coordinates ← full metric | **5.11×** | yes |
| S2 readout | + encoder & distance readout | **5.43×** | yes |
| S3 sampled | + stochastic batches, sampled pairs | **4.08×** | yes |
| S4 reconstruction | + reconstruction loss, no metric target | 1.08× | **no** |

S1 reproduces the direct-embedding result; S4 reproduces the autoencoder null.
The distance readout is exonerated. Stochastic sampling is exonerated. **Only
the objective kills it.** E6 then extends this from a binary to the
dose-response above, using a contrastive objective that receives no distance
targets at all — the setting a practitioner would actually be in.

And the failure is total rather than partial. Absolute distortion at d=4,
depth 4: S1–S3 reach 0.10 (flat) / 0.02 (hyperbolic); S4 reaches 0.80 / 0.74 —
embedded distances off by ~80% from the target metric in *both* geometries. The
≈1.0 ratio at S4 does not mean curvature stops helping; it means **no metric
structure is learned at all**, so there is nothing for curvature to help with.

## Why the original intuition fails: curvature is a dial, and we picked the wrong end

The volume really is there — E7 confirms the classic result that trees embed in
H² below a floor Euclidean space cannot reach at any width. So why does volume
never become capacity? **Curvature does not distort angles** — the Poincaré
model is conformal, local angles are exactly Euclidean. What changes is the
*mapping from angle to distance* (`csc2/e8`, pure geometry, no model):

    euclidean    c = 2r·sin(θ/2)                   — LINEAR in the angle
    hyperbolic   c ≈ 2r + (2/√|K|)·log sin(θ/2)    — LOGARITHMIC in it

Differentiating: hyperbolic `∂c/∂θ = (1/√|K|)·cot(θ/2)` is **independent of r**,
while Euclidean `∂c/∂θ = r·cos(θ/2)` grows with it. Measured as the ratio of
angular sensitivity at r=3 versus r=0.5:

| K | −4 | −2 | −1 | **0** | +1 | +2 |
|---|---|---|---|---|---|---|
| does moving outward buy resolution? | 1.97× | 2.79× | 3.89× | **6.00×** | 0.29× | 0.45× |

Euclidean scores exactly 6.00×, the radius ratio. Hyperbolic degrades it
monotonically with |K|; **spherical inverts it** — moving outward *costs*
resolution. So hyperbolic space's exponentially many new points at large radius
are all at nearly the same distance from one another, and a distance-based
readout cannot rank them. **The extra room is angularly unresolvable.**

> **Curvature is a dial trading angular resolution against radial capacity.**
> Negative: radial capacity bought by giving up angular resolution — right for
> hierarchy, containment, generality. Positive: all angle, no radius, bounded
> total volume — right for independent features and similarity. Zero: the
> neutral, scale-invariant point (the Euclidean profile is *identical* at every
> radius; no other geometry has that).

Superposition capacity is **angular** — it is set by how many near-orthogonal
directions fit, and interference is directional overlap. CSC-1 asked an angular
question and reached for the negative end of the dial. That is the whole error,
and it is why the spherical arms did marginally *better* on capacity (E1:
ρ(κ, capacity) = +0.51 to +0.85, consistent across every gain level for both
absolute-scale heads) — a result SPEC §1 pre-committed to reading as
**falsifying H-MAIN**, not as a curiosity.

Two further observations, both measured rather than argued.

**The models told us which coordinate they were paid for.** How many
features fit is governed by how many near-orthogonal *directions* exist in d
dimensions, and interference is directional overlap. Negative curvature leaves
the direction sphere untouched and adds room *further out* — a different
currency. Two observations support this: the best-performing arm across both
readouts and both axes of the capacity–interference frontier was
`normalized`, a flat control that **discards the radial coordinate entirely**;
and prototypes in the most curved arm sat **400× closer together** than flat
ones (0.0005 vs 0.209) while 12× more volume went unused. The models revealed
which coordinate they were paid for, and it was not radius.

**Hyperbolic room has a shape.** Almost all the volume of a hyperbolic disc
lies near its rim, and two rim points are ≈2R apart — nearly equidistant. The
space supplies exponentially many points that are mutually far *and roughly
equally far*: ideal for a hierarchy, poor for a graded code. This is why our
result and the Poincaré-embedding literature agree rather than conflict. Those
results win on **trees**, because a tree's node count grows exponentially with
depth exactly as hyperbolic volume grows with radius. We supplied i.i.d.
exchangeable features, so the geometry had nothing to match.

## What is not established

- Nothing here is about language models. No LM was trained.
- E8 is exact geometry for the 2-D case at a common radius; the qualitative
  conclusion extends to higher dimensions but the numbers quoted do not.
- "Reconstruction objectives" is generalized from one instance.
- Whether curvature helps for genuinely **hierarchical** features under a
  learned model remains **untested**: the experiment designed for it (`csc2/e3`)
  was invalidated by a confound in our own feature generator (tree density was
  set by path length, so depth, density and feature count varied together). It
  is also now known to have used a reconstruction objective, which E5 predicts
  could not have shown an effect regardless.
- S4's failure to learn metric structure may be curable by a different
  architecture or longer training. What is measured is that it does not happen
  in the setup used.

## Testable predictions

1. Objectives that score distances should show the advantage in proportion to
   how much they say about distances. Confirmed for explicit supervision
   (4–5×) and for contrastive adjacency (1.31×); triplet and link-prediction
   objectives should land in between and are untested.
2. Next-token prediction through a distance readout is not obviously
   metric-structured, which predicts curvature buys little in LMs *unless* an
   explicit metric term is added.
3. The advantage should scale with hierarchy depth once both the objective and
   the data provide metric structure.

---

# Appendix: seven instrument defects, and what each would have cost

The most transferable output of this project is not the null. Across both
studies, **seven defects were found before any of them contaminated a
conclusion** — three in the spec's own design, three in code, one in a
criterion we wrote ourselves. Each would have produced a confident, plausible,
wrong result.

| # | Defect | What it would have produced |
|---|---|---|
| 1 | Inherited κ convention was ¼ the true sectional curvature | The headline exponent halved, invisibly — P1 fits `log N*` against `√\|κ\|` |
| 2 | `d√x/dx` infinite at 0 → NaN gradients on any coincident pair, at every radius | Training divergence exactly where prototypes collapse — i.e. in the cells the hypothesis is about |
| 3 | 2 of 4 readouts broken (`rbf` collapses at wd=0 and is perfect at 0.01; `affine` arm-asymmetric) | A geometry result that was really an optimizer basin |
| 4 | Calibration run *through* an unvalidated readout | Two headline numbers that reversed on re-run |
| 5 | Probe metric distorted in an **N-dependent** way (0.60 vs 0.795 at N=2, reversing to read high at N=8) | Corrupted precisely the N-scaling that H-MAIN claims |
| 6 | Falsifier F1.1 rejected *true* hypotheses 53% of the time | A confident false null; needed 58 seeds vs 3 for the same claim |
| 7 | Our own calibration rule tied init gain to κ (Spearman 0.971) | An unfalsifiable P1 — every arm forced to identical predicted capacity |

An eighth, caught by E6 rather than by an instrument check: the mechanism claim
was first written as a **threshold** ("does your loss score distances?") on the
strength of E5 alone, where the contrast was 5× versus 1×. E6's contrastive arm
— the realistic case — came in at 1.31×, against a registered prediction of
>2×. Recorded as a MISS in `CSC_RESULTS/csc2/RESULTS_e6.md`. Two experiments
agreeing on a mechanism did not make the first framing of it correct.

Defect 7 is the instructive one: it was written the same day, looked
principled, and would have made the hypothesis untestable in the direction of
"false". It was caught by a power analysis that took an hour and should have
preceded the phase.

## The practices that caught them

- **Two instruments, always.** In CSC-1, `norm_affine` alone reports spherical
  curvature helping (ρ=+0.44 — which the spec says *falsifies* the hypothesis);
  `softmax` alone reports hyperbolic helping (ρ=−0.33, p=0.005). Either
  single-readout study publishes a clean, confident, opposite, wrong result.
- **Validate readouts before calibrating through them.** Calibration measures
  the readout if the readout is unvalidated (defect 4). Correct order:
  readout validation → numerics → calibration. The spec had it backwards.
- **External positive controls.** Ours caught two measurement bugs that
  cancelled between arms and were invisible to every internal comparison.
- **Never tie a nuisance variable to the independent variable** (defect 7).
  Cross them factorially.
- **Power analysis first.** It found defects 6 and 7, and established that a
  null would be interpretable.
- **Flat controls in every sweep.** The best-performing geometry in CSC-1 was a
  flat control. Without it we would have reported a curvature effect.

## Reproducing

```sh
uv sync && uv run pytest          # 187 tests, CPU-pinned
uv run python -m experiments.csc1.phase00.run_00a   # etc.
./run_csc2_overnight.sh
```

Every runner writes a committed summary JSON carrying the git commit that
produced it. CI runs the suite from a fresh clone and verifies no source module
is gitignored — the failure mode that hid a whole module in the parent program
until an external auditor tried to reproduce it.

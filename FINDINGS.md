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

> **Curvature is a property of a metric. A representational advantage from
> curvature appears only when the training objective makes the metric
> load-bearing. Under objectives that do not — such as reconstruction — no
> metric-faithful structure is learned in any geometry, so curvature has
> nothing to act on.**

This is a constraint on when hyperbolic representations can help, and it is
checkable before committing to one: *does your loss score distances?*

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

**3. The break is at the objective.** Interpolating between the two setups in
four steps, changing one thing each time (`csc2/e5`, 1200 runs, 10 seeds):

| stage | what is added | advantage (d=4) | survives? |
|---|---|---|---|
| S1 direct | coordinates ← full metric | **5.11×** | yes |
| S2 readout | + encoder & distance readout | **5.43×** | yes |
| S3 sampled | + stochastic batches, sampled pairs | **4.08×** | yes |
| S4 reconstruction | + reconstruction loss, no metric target | 1.08× | **no** |

S1 reproduces the direct-embedding result; S4 reproduces the autoencoder null.
The distance readout is exonerated. Stochastic sampling is exonerated. **Only
the objective kills it.**

And the failure is total rather than partial. Absolute distortion at d=4,
depth 4: S1–S3 reach 0.10 (flat) / 0.02 (hyperbolic); S4 reaches 0.80 / 0.74 —
embedded distances off by ~80% from the target metric in *both* geometries. The
≈1.0 ratio at S4 does not mean curvature stops helping; it means **no metric
structure is learned at all**, so there is nothing for curvature to help with.

## Why the original intuition fails

Two reasons, both supported by measurement rather than argument.

**Superposition capacity is angular; curvature adds radial volume.** How many
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

1. Objectives that score distances — contrastive, triplet, metric-learning,
   link prediction — should show the curvature advantage in learned models.
   S2/S3 already demonstrate this for directly supervised tree distances.
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

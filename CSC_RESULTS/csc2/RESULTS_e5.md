# CSC-2 E5 — the curvature advantage dies at the objective

1200 runs, 10 seeds. **The pre-registered prediction is confirmed**: the
hyperbolic advantage survives S1–S3 and vanishes at S4. The mechanism is the
**training objective**, not the readout and not stochastic sampling.

## Result

Advantage = Euclidean distortion ÷ best hyperbolic distortion. >1 means
hyperbolic embeds the tree better.

| stage | what changes | d=2 | d=4 | survives? |
|---|---|---|---|---|
| **S1** direct | coordinates ← full metric | 1.37× | **5.11×** | yes |
| **S2** readout | + encoder & prototype readout | 2.02× | **5.43×** | yes |
| **S3** sampled | + stochastic batches, sampled pairs | 2.01× | **4.08×** | yes |
| **S4** reconstruction | + reconstruction objective (no metric target) | 1.04× | 1.08× | **no** |

S1 reproduces E4 to three decimals, so the chain is anchored. S4 is CSC-1's
setup, and reproduces CSC-1's null.

**The readout is exonerated** (S2 keeps the advantage — 5.43×, if anything
higher than direct optimization). **Stochastic sampling is exonerated** (S3
keeps it). Only the objective kills it.

## What actually happens at S4 — sharper than the prediction

The prediction was that hyperbolic would lose its *edge* under a reconstruction
objective. The measurement says something stronger: **no metric structure is
learned at all, in either geometry.**

Absolute distortion, d=4, depth=4:

| stage | Euclidean | best hyperbolic |
|---|---|---|
| S1 direct | 0.1023 | **0.0202** |
| S2 readout | 0.1023 | **0.0202** |
| S3 sampled | 0.1023 | 0.0218 |
| S4 reconstruction | **0.8027** | **0.7383** |

At S4 both arms sit near 0.8 — embedded distances are off by ~80% relative to
the tree metric. That is not a degraded embedding; it is essentially no metric
fidelity. So the ~1.0 advantage ratio does **not** mean "curvature stops
helping". It means **there is nothing for curvature to help with**: the
objective never induces a metric-faithful layout, so the geometry has no
purchase.

By curvature at S4 there is a faint ordering (κ=−4: 0.738, κ=−2: 0.792, κ=−1:
0.800, flat: 0.803) — hyperbolic is marginally less bad, consistent with the
1.09× ratio, and far too small to matter.

## Why this explains both studies

CSC-1 and CSC-2's E1 trained TMS-style autoencoders on a reconstruction loss.
Such a loss asks only that activations be reproduced. Any layout that
reconstructs equally well is equally optimal, and metric faithfulness is not
among the things it scores — so the optimizer has no reason to produce one, and
measurably does not.

Curvature is a property of a *metric*. If the training signal does not make the
metric load-bearing, curvature is decoration, and no amount of available volume
changes that. This is consistent with every result in both studies:

- CSC-1's null on exchangeable features: reconstruction objective → no metric
  structure → no curvature effect. ✓
- CSC-1's unused volume (prototypes 400× closer in H² than E², with 12× more
  room available): nothing asked them to spread. ✓
- E1's finding that the null survives an absolute-scale readout: the readout was
  never the problem. ✓ (S2 confirms this directly.)
- E4's large advantage: its objective *is* metric-matching. ✓

## The claim this supports

> Negatively curved representation spaces confer an advantage when the training
> objective makes the metric load-bearing. Under objectives that do not — such
> as TMS-style reconstruction — the advantage is absent, because no
> metric-faithful structure is learned in any geometry.

This is a **constraint on when hyperbolic representations can help**, and it is
actionable: check whether your loss scores distances before reaching for
curvature. Contrastive, triplet, metric-learning, link-prediction and
distance-supervised objectives qualify. Reconstruction and (plausibly)
next-token prediction do not, at least not directly.

## Limitations

- One objective family per stage; "reconstruction objectives" is generalized
  from one instance.
- Toy scale, d ≤ 4, trees to depth 5. No language model was trained.
- Whether a *language-modelling* objective makes the metric load-bearing is
  untested here and is the obvious next question. Next-token prediction through
  a distance readout is not obviously metric-structured, which would predict
  curvature buys little there — but that is a prediction, not a result.
- S4's failure to learn metric structure may be curable by a different
  architecture or longer training; what is measured is that it does not happen
  under the setup CSC-1 used.

## A methodological finding, recorded separately

Chasing an S1 replication failure (0.29 vs E4's 0.066 from identical code, on a
different RNG) surfaced that **hyperbolic embedding optimization is far more
seed-sensitive than Euclidean**: spread across seeds 2.3–3.5× hyperbolic versus
1.2× flat. Hyperbolic loss landscapes have many more local minima. Two
consequences: single-seed hyperbolic results should be distrusted, and a
stochastic optimizer is exactly what handles such a landscape worst — a partial
independent explanation for the learning-side nulls. E5 uses 10 seeds because
of it.

## Artifact

`e5/e5_embedding_vs_learning.json` — 1200 runs, 4 stages × 3 depths ×
5 curvatures × 2 dims × 10 seeds.

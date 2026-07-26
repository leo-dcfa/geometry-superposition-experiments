# CSC-2 E7 — what the distortion advantage is worth in width

552 runs. The question: E6 measured ~1.3× lower distortion at matched
dimension; how many dimensions is that worth?

**Answer: the framing is wrong. Euclidean space has a distortion floor for tree
metrics that additional width does not break, and hyperbolic goes below it — but
only at small dimensions, and it loses at larger ones.**

## The Euclidean distortion curve plateaus

Contrastive objective, depth-4 tree (31 nodes), mean over 8 seeds:

| Euclidean dim | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| distortion | 0.242 | 0.169 | 0.159 | 0.155 | **0.153** | 0.156 | 0.157 | 0.160 | 0.163 |

Distortion bottoms out at ≈0.153 around d=8 and then *rises*. Hyperbolic at
d=4 (κ=−1) reaches **0.125** — below the best Euclidean achieves at any width
in the sweep.

**The floor is not under-training.** At 5× the step budget (15k vs 3k):

| | d=8 | d=16 | d=32 |
|---|---|---|---|
| 3k steps | 0.1531 | 0.1621 | 0.1680 |
| 15k steps | 0.1529 | 0.1547 | 0.1587 |

The upturn at high d shrinks — that part *was* optimization — but the floor
itself does not move. Hyperbolic d=4 at 15k steps: **0.1245**, still 18% below
it.

This is expected mathematically and is the reassuring part: **tree metrics are
not ℓ₂-embeddable at any dimension.** They embed isometrically into ℓ₁, and
their Euclidean distortion is bounded below independently of width. So the
floor is a property of the geometry, not of our optimizer, and E7 is
recovering a known fact rather than discovering one — which is exactly what a
measurement in this position should do.

## Width equivalence, by reference dimension

| tree | hyp dim | hyp distortion | euc at same dim | equivalent euc dim | saving |
|---|---|---|---|---|---|
| depth 4 | 2 | 0.222 | 0.242 | 2.2 | 1.11× |
| depth 4 | **4** | **0.125** | 0.159 | **never matches (>32)** | **unbounded** |
| depth 4 | 8 | 0.132 | 0.153 | never matches (>32) | unbounded |
| depth 5 | 2 | 0.205 | 0.261 | 2.7 | 1.35× |
| depth 5 | **4** | **0.131** | 0.169 | **never matches (>32)** | **unbounded** |
| depth 6 | 2 | 0.216 | 0.310 | 3.2 | 1.58× |
| depth 6 | 4 | 0.166 | 0.187 | 8.6 | 2.14× |
| depth 6 | **8** | **0.185** | 0.167 | 4.2 | **0.53× — hyperbolic LOSES** |

Two regimes, and the second one matters as much as the first.

**At d=2–4 hyperbolic wins**, and at d=4 it wins in a way extra Euclidean width
cannot answer.

**At d=8 on the largest tree hyperbolic loses** (0.185 vs 0.167). More curved
dimensions made it worse, which is consistent with the seed-spread finding —
hyperbolic optimization has 2.3–3.5× the variance of Euclidean, and that cost
grows with parameter count while the representational benefit is already
saturated. The curved advantage is a *narrow-width* phenomenon here.

## So: does 30% lower distortion mean 30% fewer neurons?

**No, on three separate counts.**

1. **The relationship isn't linear, it's a cliff.** Where the Euclidean curve
   plateaus, a small distortion advantage translates to an unbounded width
   saving; where it is steep (d=2→3), a large one translates to almost nothing
   (1.11×). The same "30%" means different things at different widths.
2. **The win is confined to narrow layers.** At d=8 the sign flips. Any story
   about saving neurons in a wide network runs the wrong way.
3. **This is embedding width for a known tree metric under a contrastive loss,
   not hidden width of a network.** CSC-1's entire null was that this does not
   transfer to a learned representation over general features. Letting a
   tree-embedding number imply a transformer number would contradict our own
   result.

The defensible version: **if your data is genuinely hierarchical, your objective
scores distances, and you are width-constrained (d ≈ 2–4), hyperbolic geometry
buys you fidelity that no amount of Euclidean width can.** Outside that
intersection — which is where most neural networks live — the evidence here
says it buys little or nothing, and costs optimization stability.

## Limitations

- One contrastive formulation; Euclidean sweep capped at d=32.
- Complete binary trees only. Real hierarchies are irregular, and irregular
  trees are harder for both geometries.
- "Never matches within the sweep" is a statement about d ≤ 32 plus the known
  ℓ₂ non-embeddability of tree metrics, not a proof for this specific tree.
- Distortion is not the only thing a representation is for.

## Artifact

`e7/e7_dimension_equivalence.json` — 552 runs, 11 Euclidean widths × 3 depths,
plus 4 curvatures × 3 reference widths × 3 depths, 8 seeds.

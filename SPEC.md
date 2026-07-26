# KosmosLM Study: Curvature and Superposition Capacity (CSC)
**Spec v1.0 — self-contained. Supersedes nothing; this is a new study within the KosmosLM program.**
Tooling: uv, Python, PyTorch. All thresholds in this document are pre-registration candidates; they are sealed in VALIDATION.md before any unblinding.

---

## 0. Scope and inspiration

This study descends from KosmosLM's original inspiration: **spacetime curvature in general relativity**, where mass-energy curves space and the curvature in turn shapes how things move. The earlier flagship asked whether a language model *allocates* curvature lawfully (κ ↔ frequency). The external audit of July 2026 showed that result is likely an instrument artifact: κ was the only per-token scalar in a bias-free distance softmax, and the readout was boundary-saturated at realistic activation scales, so κ plausibly learned the unigram prior rather than geometry.

CSC keeps the physical intuition but changes the question from *allocation* to *capacity*: curvature determines how much room a space has. Negative curvature gives exponential volume growth with radius; positive curvature caps total volume. If representations are points and interference is proximity, then curvature should directly control **how many features fit in how few dimensions** — the quantity mech interp calls superposition capacity.

**In scope:** constant- and learnable-curvature latent spaces; superposition phase diagrams; parameter/dimension efficiency at three scales (toy → small LM → GPT-2 size); SLT panels (LLC across the capacity phase boundary) as exploratory.
**Out of scope:** the κ↔frequency allocation law (parked pending the audit's four experiments); gravity-attention (phase 5 of the parent program); any claim about loss superiority of curved models (loss is a monitored covariate, never a headline).

---

## 1. Main prediction (H-MAIN) — locked

> **Negative curvature buys representational room: a negatively curved latent space requires fewer neurons/parameters to represent more features at fixed recovery quality than a flat one; a positively curved space requires more.**

Formal operationalizations, one per scale:

- **H-MAIN(toy):** Let d*(N, ε) be the minimum latent dimension at which N sparse features are recoverable with per-feature recovery ≥ 1−ε. Prediction: d*_hyp(N, ε) < d*_euc(N, ε) for all N above a crowding threshold, with the gap **growing** in N. Dual form at fixed d=2: capacity N*(κ) is monotone **decreasing** in κ, and for κ<0, log N* is linear in √|κ|·R (exponential volume growth), vs polynomial in R for κ=0.
- **H-MAIN(small LM):** At matched total parameter count, an LM whose designated representational subspace is negatively curved matches the Euclidean baseline's validation loss with a smaller subspace width d_c, by a pre-registered margin (§6).
- **H-MAIN(GPT-2):** At matched parameters, the curved model supports more SAE features at matched reconstruction fidelity and sparsity, and/or matches baseline loss at reduced width (§7).

**Directional commitment, stated now to prevent later goalpost drift:** "increasing curvature" in the colloquial sense (κ more positive / more spherical) is predicted to *reduce* capacity. The capacity-buying regime is κ < 0. If spherical arms show a capacity advantage that survives the conditioning controls (rule R3), H-MAIN is **falsified**, not reinterpreted.

## 2. Secondary hypotheses

- **P1 (capacity curve):** N*(κ) at fixed d=2 and fixed interference threshold is monotone decreasing in κ across κ ∈ {−4, −2, −1, −0.5, 0, +0.5, +1, +2}; the pre-registered functional form for κ<0 is log N* = α·√|κ|·R + β with α > 0.
- **P2 (phase boundary shift):** The sparsity threshold S* at which the model transitions from dedicated to superposed representation shifts toward denser inputs for κ<0 (superposition becomes cheaper, so the model enters it earlier).
- **P3 (interference):** At fixed N and d, mean cross-feature contamination decreases with −κ. Count and blur are reported separately; "polysemanticity" is never used unqualified in results docs.
- **P4 (demand-driven curvature — exploratory, gated):** With per-region learnable κ AND per-prototype bias + scale present in all arms (rule R1), κ drifts negative preferentially in regions of high local feature density. Spearman(−κ_region, local feature count) > 0.4, surviving a permutation gate calibrated per rule R4. P4 only runs if P1 confirms; it is the cleaned-up successor to the audited F7 and must be labeled as such in any writeup.

## 3. Standing design rules (audit-derived; verbatim policy, apply to every phase)

- **R1 — Curvature is never the only free scalar.** Every readout in every arm, including all controls, carries per-prototype (toy) or per-token (LM) bias and scale. If a curved arm's advantage disappears when bias/scale are present, the prior result was bias duty. This rule exists because the parent flagship's κ was the sole per-token scalar and learned the unigram prior.
- **R2 — Saturation gate.** Log fraction-of-ball-radius ‖expmap0(h)‖ / R_ball at every eval step, all curved arms. Any run spending > 5% of training steps with median fraction > 0.99 is flagged UNINTERPRETABLE automatically and excluded from hypothesis tests (still reported in the ledger). Target operating band: √|κ|·r ∈ [0.5, 3.0] for the bulk of points (see Phase 00).
- **R3 — Decouple capacity from loss/conditioning.** Two mandatory controls in every comparison: **clamped-Euclidean** (flat space, distances clipped at the matched spherical/hyperbolic diameter — reproduces bounded-diameter readout conditioning without curvature) and **normalized-Euclidean** (unit-norm latent). Headline metrics are recovery/capacity metrics; loss appears only in monitoring tables. H1b (bounded diameter = implicit normalization) is the standing reason.
- **R4 — Calibrated sabotage.** Before any permutation/shuffle gate is interpreted, measure each channel's raw logit/output leverage (variance of output attributable to the channel under matched perturbation). Permutation damage is reported as damage ÷ leverage, never raw. A channel being the highest-leverage knob is not evidence it encodes the hypothesized quantity.
- **R5 — Reproducibility (new, from the audit's findings).** No source module may be swallowed by .gitignore (`/data/` anchored, never `data/`); a fresh-clone CI job runs `uv run pytest` on every push; summary JSONs (allocation, fits, scores — kilobytes) are committed alongside every RESULTS table. A claim without a committed artifact does not enter a scorecard.
- **R6 — Reporting.** A number outside a pre-registered band is a miss and is written as a miss, including in prose. Best-of-seeds is reported alongside the all-seed range in the same sentence.

## 4. Phase 00 — Instrument calibration (blocks everything)

Purpose: guarantee curvature is *felt* — points neither huddled at the origin (all geometries locally flat; experiment measures nothing) nor pinned to the rim (rule R2; geometry meaningless).

- 00a: Input/init scale sweep so that trained-model point clouds sit with bulk radius √|κ|·r ∈ [0.5, 3.0] across all κ arms. Deliverable: a scale-selection rule (function of κ, d, N), committed before Phase 1.
- 00b: Numerics audit — atanh clamp behavior, fp32 vs bf16 boundary overflow (parent program has a prior bf16 incident), expmap/logmap round-trip error across the operating band.
- 00c: Dead-unit parity fixture ported from parent 01b; verify readout fairness across κ arms before any comparison.
- **Gate G00:** all three pass → Phase 1 opens. No hypothesis-relevant number is looked at before G00.

## 5. Phase 1 — Toy model (TMS on curved 2-manifolds)

Setup: Toy-Models-of-Superposition-style autoencoder. N sparse features (sparsity S, geometric importance spectrum), encoder to a d=2 constant-curvature manifold (κ per arm as in P1), decoder via distance-to-prototype readout with per-prototype bias + scale (R1). Controls per R3. Sweep N × S × κ, n ≥ 5 seeds per cell.

Metrics:
- Per-feature recovery: reconstruction of one-hot inputs, threshold ε sealed in VALIDATION.md.
- N*(κ, S): max N with ≥ 90% of features recovered.
- Cross-contamination matrix (P3); mean and max off-diagonal.
- Minimum pairwise geodesic distance among learned prototypes vs N (predicted: floor holds in H², collapses in E²).
- d*(N, ε) sweep at d ∈ {2,3,4,6,8} for H-MAIN(toy).
- Exploratory: LLC (devinterp) across the N × κ grid; look for developmental transitions at the capacity boundary.

Figures: Poincaré-disk prototype configurations per (κ, N) cell — self-explanatory hero-figure candidates.

Falsifiers (Phase 1):
- F1.1: N* not monotone decreasing in κ (any adjacent-pair inversion outside seed noise) → P1 dead.
- F1.2: log N* vs √|κ|·R fit R² < 0.8 for κ<0 → exponential-capacity form dead; monotone claim may survive as weaker P1′.
- F1.3: clamped-Euclidean control reproduces ≥ 70% of the hyperbolic capacity gain → effect is conditioning, not geometry → H-MAIN dead at toy scale.
- F1.4: d*(N) gap not growing in N → efficiency claim reduces to a constant offset; H-MAIN weakened to H-MAIN′ (fixed offset), flagged in scorecard.

**Gate G1 (→ Phase 2):** P1 confirmed with functional form (F1.2 passed), F1.3 control clean, R2 saturation gate clean in ≥ 95% of hypothesis-relevant runs, ≥ 5 seeds. Any failure → stop, write up the null.

## 6. Phase 2 — Small LM (TinyStories scale)

Architecture: standard small transformer; the curved component is a **designated low-dimensional representational subspace** — a bottleneck of k independent curved factors (e.g., k × H² or k × H⁴ product manifold) inserted at a pre-registered site (candidate sites, choose one before running: unembedding bottleneck with per-token bias + scale per R1; or MLP bottleneck adapter). d=256-style full-width curved readouts are **prohibited** — that is the saturated regime the audit condemned. The curved subspace width d_c is the manipulated variable.

Arms (all parameter-matched by shaving d_c or FFN width; matching procedure committed before training):
1. Euclidean bottleneck, width d_c.
2. Hyperbolic product bottleneck, width d_c′ < d_c, fixed κ from Phase 1's best regime.
3. Clamped-Euclidean control at d_c′ (R3).
4. Learnable-κ variant of arm 2 (feeds P4), with bias + scale present (R1).

Primary comparison for **H-MAIN(small LM):** smallest d_c′ at which arm 2 matches arm 1's val loss within a sealed tolerance (candidate: 0.01 nats), n ≥ 5 seeds. Prediction: d_c′ ≤ 0.75·d_c (i.e., ≥ 25% width reduction; final margin sealed pre-unblinding). Secondary: SAE trained on the bottleneck of each arm at matched sparsity — feature count at matched reconstruction fidelity higher in arm 2.

Falsifiers: no d_c′ < d_c achieves parity (H-MAIN dead at LM scale); or clamped control matches arm 2 (conditioning, dead); or R2 flags fire (uninterpretable — redesign site, one retry allowed, logged).

**Gate G2 (→ Phase 3):** ≥ 15% width reduction at parity surviving controls, all seeds' saturation gates clean, SAE secondary at least directionally consistent.

## 7. Phase 3 — GPT-2 scale (gated; runs only if G2 passes)

~124M-parameter class, same bottleneck design as Phase 2 at the site that won, parameter-matched arms 1–3 (learnable-κ arm optional, budget permitting). n ≥ 3 seeds (budget-limited; stated openly as a limitation). Primary: H-MAIN(GPT-2) as in §1 — width reduction at loss parity and SAE feature counts at matched fidelity/sparsity. Add downstream sanity evals (a small fixed battery, committed beforehand) to check the curved model isn't buying loss parity with degenerate behavior. Compute budget and stopping rule committed in VALIDATION.md before the first Phase-3 run.

No new hypotheses may be introduced at Phase 3. It is a transfer test only.

## 8. Statistical and pre-registration policy

- All thresholds, bands, seeds, and primary cells sealed in VALIDATION.md with a commit hash before the relevant unblinding, in the parent program's style.
- Spearman with explicit tie policy (audit lesson: index-order tie-breaking made the −0.822 full-vocab number uninterpretable); ties get midranks, and any stratum with > 10% tied mass is reported with and without the tied block.
- Seeds: ≥ 5 (toys, small LM), ≥ 3 (GPT-2). Primary cells named before unblinding; best-of-seed never headlines (R6).
- Every scorecard row: prediction, band, measured value, verdict (HIT / NEAR-MISS / MISS), artifact path.

## 9. Deliverables

- CSC_RESULTS/ per phase with committed summary JSONs (R5).
- VALIDATION.md ledger; scorecards continue the parent program's numbering.
- Hero figure: capacity phase diagram (N × S) with κ as panel axis; Poincaré-disk insets.
- Test suite: geometry ops (round-trip, curvature limits κ→0 recovering Euclidean), R2 monitor, control parity fixtures; fresh-clone CI green is a release condition.

---
*Provenance note for any writeup: the spacetime-curvature framing is the study's inspiration and namesake; the tested claims are the operational ones in §1–2. The parent flagship's κ-allocation law is not evidence for this study and is not cited as support while the audit's four discriminating experiments remain unrun.*

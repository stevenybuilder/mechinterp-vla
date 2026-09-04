# PREREGISTRATION — the causal handoff, and its mediation

**Written 2026-08-31, before any mediation or reset episode has been run.** Hash this file and commit
the hash before the first run (`shasum -a 256`). Report the primary criterion as written, pass or fail.
The project has one preregistration that already failed (`PREREG-region-gate-2026-08-31.md`) and was
reported as failed; this one is held to the same standard.

---

## 1. The claim

> **Instruction use is a causal handoff.** Language is readable and heavily attended at many locations,
> but it only affects behaviour after being transformed into an architecture-specific, action-facing
> carrier representation. Patching the carrier moves behaviour; restoring the carrier to its clean value
> blocks the instruction's effect even when the instruction itself has been swapped.

The carrier is model-specific:
- **π0.5** — image-token positions, VLM layers 12–17.
- **OpenVLA-OFT** — action-query positions, late layers.

The abstract claim is architecture-independent; the *site* is not. That contrast is the point.

## 2. Why this is not "patching to show which layers are used"

Neel's stated common mistake is using patching to show which heads/layers matter. A localisation result
("layers 12–17 carry it") is exactly that, and on its own it is uninteresting. The mediation reset is
what makes this a causal-chain claim rather than a localisation claim: we show the instruction's effect
is *routed through* a specific representation by putting the instruction in and then taking the carrier
away, and watching the effect vanish. That is a mediation analysis, not a component search.

## 3. What is already measured (not preregistered — these motivate the design)

All from prior runs, re-derived from per-episode records. Consistent with the handoff, and jointly they
are why we expect the new legs to work:

| observation | value | reading |
|---|---|---|
| KV[IMG] patch @ layers 0–5 | R = **0.000** | carrier not yet loaded that early |
| KV[IMG] patch @ layers 12–17 | R = **0.832–0.92** | carrier is live and causally dominant |
| KV[INSTR] patch @ all layers | R = **0.011** (goal suite) | source positions already drained |
| KO[INSTR] attention knockout | **−0.006** | not an attention-routing artifact |
| action-expert attention to INSTR vs IMG, per token, L12–17 | **11.4×** | heavily attended, causally dead |

So the instruction's content moves into image positions somewhere **between layer 5 and layer 12**.
The genuinely new measurements below are the early propagating patch and the reset.

**Scope caveat, stated up front:** KV[INSTR] ≈ 0.011 is a **libero_goal** result. On libero_object it
reaches 0.122 (up to 0.49) and gate_frac drops to 0.46. The primary test is therefore run on
**libero_goal**, where the localisation replicated on held-out inits (6/6 cells, gate 100%).

## 4. Design

**Unit of analysis: the (task × prompt-pair) cell. Never the episode.** Rollouts are deterministic given
(layout, prompt, seed) — 48 duplicates returned bit-identical, and 63/64 groups were outcome-identical
across 6 seeds. Episode-level tests inflate n by 6–20×. Cells are the replicates; n = 12 cells minimum.

**Cell selection, fixed before running:** libero_goal task pairs (A, B) where the clean model
demonstrably follows language on BOTH prompts — correct-prompt first-touch ≥ 0.9 on each, measured on
init states disjoint from those used in the test. Cells are drawn from the held-out split, not the
6 discovery/confirmation cells already used for localisation.

**Conditions.** Identical observation throughout; only the residual/KV edits differ.

| # | condition | edit |
|---|---|---|
| C0 | clean A | none (baseline; must reproduce the unedited rollout bit-for-bit at α=0) |
| C1 | clean B | none, prompt B (defines the target behaviour) |
| C2 | **early source patch** | B's instruction-position residuals → A, at layers 0–5, **allowed to propagate** |
| C3 | **late carrier patch** | B's carrier → A at layers 12–17 (π0.5: image positions; OFT: action-query positions) |
| C4 | **mediation reset** | C2, **and then** carrier restored to its clean-A value at layers 12–17 |
| C5 | dead-site control | B's instruction positions → A at layers 12–17 (known ≈0; the decodable-but-dead site) |
| C6 | random control | equal-norm random vectors at the carrier site, matched rank and count |
| C7 | unrelated-prompt carrier | carrier from an unrelated prompt C, same syntax, different target |

**Semantic-equivalence arm** (tests that the carrier encodes meaning, not the object token):
- C3-para1, C3-para2 — carrier built from two independently worded paraphrases of B.
- C3-C — carrier from prompt C, same syntax, different target.
Only paraphrases the clean model follows reliably (≥0.9 first-touch) are admitted. A paraphrase the
base model does not obey cannot support an interpretability claim and is dropped **before** testing.

**Metrics.** Primary is **closed-loop first touch** — which object the gripper contacts first — because
an action-vector displacement is not a behaviour. Secondary is R on the A−B action axis with the L2-to-
target discriminator (R alone is unreliable: random directions give R ≈ 0.5, which is the destroyed-
output value; the discriminator is L2 to B).

## 5. Preregistered criteria

Primary, all evaluated at the cell level with n = 12 cells:

1. **Carrier is sufficient.** C3 redirects first touch to B in **≥ 8/12 cells**.
2. **Carrier mediates the source.** C4 removes **≥ 75%** of C2's effect, measured as
   (C2_effect − C4_effect)/C2_effect on the per-cell redirection rate.
3. **Meaning, not tokens.** Both paraphrase arms preserve **≥ 80%** of C3's canonical effect, and the
   different-target arm (C3-C) redirects toward C, not B.
4. **Controls are dead.** C5, C6 and C7 each redirect in **≤ 2/12 cells**, and their L2-to-B stays
   above the C3 value by a margin exceeding the null band.
5. **Cross-architecture.** The abstract chain (1)+(2) reproduces on OpenVLA-OFT at ITS carrier site
   (action-query positions), even though the site differs from π0.5's.

**Statistics.** Sign test / exact binomial on cells for criteria 1 and 4; paired per-cell comparison
with a scene-level block permutation null for criterion 2. With 12 cells, 8/12 against a chance rate of
0.25 gives p ≈ 0.0004 (exact binomial), and 2/12 is not distinguishable from chance. Report exact p,
not a threshold. Nulls over scenes MUST be scene-level element-wise permutations — permuting scene
labels is an identity operation for a same-vs-different-scene statistic, a bug already shipped and
corrected once in this project.

## 6. Kills, declared in advance

- **C2 does not move behaviour.** If swapping the instruction early does not redirect the model, there
  is no source effect to mediate and the mediation arm is vacuous. Report as a failed preregistration;
  fall back to the descriptive attention/use dissociation.
- **C4 does not restore A.** If resetting the carrier leaves B's effect intact, the instruction reaches
  behaviour by a route that bypasses the carrier — i.e. there is a backup pathway. That is a real and
  reportable finding (and it is the redundancy-mapping gap flagged in `bricken-ablation-principles.md`),
  but it refutes the handoff as a *complete* account and must be written that way.
- **C5 is not dead on libero_object.** Expected: the dead-site control is only clean on libero_goal.
  Do not run the primary on libero_object and do not report a pooled number across suites.
- **Paraphrases fail at baseline.** Drop them and say so; do not report a semantic claim built on
  prompts the model does not follow.

## 7. What this may NOT claim

- Not "we found the instruction-following circuit". No circuit is traced; this is a two-node mediation.
- Not that the carrier is monosemantic. We do not decompose it, and it may be a superposed mixture.
- Not that this generalises beyond LIBERO, two policies, and the tested instruction families.
- Not a safety intervention. The safety-relevant reading is narrower and stated as such: **information
  being present and attended does not mean the computation uses it**, which bounds what activation
  probes and attention-based monitors can promise. That connects to the monitor-collapse result
  (92.9%/89.0% in dev; on a held-out instruction the monitor is strictly dominated by a constant
  "it will disobey" predictor) and to nothing stronger.

---

## APPENDIX A — trajectory-copying controls (added 2026-08-31, BEFORE any mediation episode ran)

The original document hashed to
`f4877800ffe48882b175d4df70dcfdae6549a1a9bfd5ecd896e91aa0a8bafcdc`. This appendix is added before a
single episode of the mediation experiment has been executed; no data informed it. It is recorded
separately rather than by editing §4 in place, so the original text stands.

**Why.** A carrier patch transplants activations from one observation into another. It may transplant
*the trajectory appropriate to the donor* rather than a concept. Our own held-out stage-2 records make
this concrete rather than hypothetical:

| condition (libero_goal confirm, A←B) | R | L2→A | L2→B |
|---|---|---|---|
| KV[IMG]@all | 0.979 | 0.977 | **0.085** |
| KV[IMG]@12-17 | 0.814 | 0.822 | 0.243 |
| **RS[IMG]@all — image KV from a length-matched UNRELATED prompt** | **0.846** | 0.902 | 0.614 |
| KV[INSTR]@all | 0.014 | 0.074 | 0.976 |

RS transplants an *unrelated* prompt's image KV and still moves the axis metric by R = 0.846, against
0.979 for the true B patch. Only L2-to-target separates them. This means **R is not a specific metric
for this claim and must never be reported alone**, and it means the existing stage-2 localisation is
itself exposed to the confound: KV[IMG]@12-17 is whole-image replacement over all 512 image positions.

We never recorded whether a transplant lands on the *donor's* trajectory. That is the missing
measurement, and it is what the two new conditions supply.

**Two additional conditions, both first-class (not robustness extras):**

- **C3L — localized carrier.** Patch only the image patches covering B's object (MuJoCo segmentation
  mask, +1-patch dilation; typically 2–9 of 512 positions), layers 12–17. Whole-image replacement is
  the arm most suspect of copying; a localized patch carries far less trajectory information.
- **C3X — cross-layout donor.** Patch B's carrier taken from a **different init state**, in which B's
  object sits at a different position. We record L2 to the current-layout clean-B chunk **and** L2 to
  the donor-layout clean-B chunk.

**Preregistered discrimination rule.** For C3X:
- If the resulting action lands nearer the **current** layout's clean-B chunk than the donor layout's
  (L2→B < L2→B_donor), the patch transferred a *concept* that recombined with this layout's geometry.
- If it lands nearer the **donor** layout's chunk (L2→B_donor < L2→B), the patch copied a trajectory
  and the carrier claim is not supported as stated.
- Declared in advance: **if C3 works only as whole-image replacement and C3L does not reproduce it at
  ≥ 60% of C3's effect, we report the carrier result as confounded by trajectory copying and do not
  claim concept transfer.**

**Consequence for criterion 1 of §5.** "Carrier is sufficient" now requires C3 **and** C3L to redirect
in ≥ 8/12 cells. C3 alone is no longer sufficient to claim it.

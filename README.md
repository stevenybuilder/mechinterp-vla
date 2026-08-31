# mechinterp-vla

**Where does a vision-language-action policy actually keep the instruction it is following — and does
the interpretability toolkit we would use to find out survive contact with an embodied model?**

Mechanistic interpretability of two robot policies, π0.5 and OpenVLA-OFT, on LIBERO.

---

## Findings

1. **Attention does not locate the instruction.** The action expert attends **11.4× harder per token**
   to instruction tokens than to image tokens, while causally patching those same instruction
   positions moves behaviour by **0.011** and patching image positions moves it by **0.832**.
   Instruction attention *rises* 3.3× going into exactly the layer band where patching it does the
   least.

2. **An activation monitor for disobedience looks excellent and is worthless.** 92.9% recall / 89.0%
   precision in development — and **beaten by a constant "it will disobey" predictor on both
   held-out splits.** The cause is measurable: cell identity decodes from those activations at 0.96.
   The monitor learned the scene, not the behaviour.

3. **Everything is scene-local, and four independent methods fail the same way.** A monitor, a
   steering direction, a concept subspace, and a geometry model of behaviour all work within a scene
   and all collapse across scenes. This is one finding measured four ways, and it explains the other
   three: the toolkit assumes concepts are context-portable, and in this policy they are not.

4. **Obedience is gated by a graded spatial slot.** Whether the policy follows an instruction depends
   on **where the named object is**, falling monotonically with distance from the position the task
   was trained at — 0.950 → 0.425 → 0.102 → 0.025 obedience at 0 → 0.26 m.

5. **The locus is not universal across architectures.** In π0.5 the instruction positions are inert
   and the image positions carry the instruction. In OpenVLA-OFT it is the mirror image. Same task
   suite, same metric, opposite answer.

---

## Why this matters

The case for interpretability as a safety tool assumes that if a model is about to do the wrong
thing, the intention is legible in its activations *before* the action. Robot policies are where that
assumption gets expensive if it is wrong, because the action is physical.

Results 1–3 say the standard toolkit gives a confident wrong answer here in three different ways: the
attention map points at the wrong tokens, the probe generalises to nothing, and the concept has no
portable direction. Result 2 is the sharp one — the monitor is not merely weak, it is *beaten by a
constant*, and it would have passed a normal dev-set evaluation.

Result 4 says the underlying mechanism is not the one the field's language assumes. "The policy
understands the instruction" describes a system that is closer to a position-bound motor lookup with
a spatial gate on it.

## Context and related work

Behavioural spatial brittleness in VLAs is established: **Spatial Overfitting** (2505.03500) reports
86–99% task success under *wrong* prompts and 0% at novel positions, and the LIBERO-Plus /
LIBERO-PRO line (2510.13626, 2510.03827) documents the same fragility at benchmark scale. **LIBERO-CF**
(2602.17659) publishes a 30.8% grounding rate for π0.5; the measurement here is 32.1%, so the
descriptive half of this work replicates rather than discovers.

The contribution is the **locus** question rather than the behavioural one. **Grant et al.**
(2603.19233) show 99.6% source-dominance in π0.5 and argue language sensitivity depends on task
structure rather than model design — a behavioural claim. **VLA-Trace** (2605.30117) reports
architecture differences in routing via attention knockout, and attributes the π0.5-vs-OpenVLA
difference to causal masking; that attribution is checkable and appears to be wrong, since OFT's
pinned `transformers` fork removes the causal mask entirely.

What knockout cannot do, and what this work adds: **content-specific sufficiency** (patching in a
*different* instruction's representation, not merely deleting one), **depth localisation** of where
the effect lives, and a **cross-architecture existence result** for the locus.

## Methodology

Activation patching between matched A/B conditions in both models, with the interventions and
controls in `src/`. The analysis rules are in [`docs/protocol.md`](docs/protocol.md); each one exists
because breaking it produced a wrong number that a control later caught.

Three that matter most:

- **The unit is the cell, not the episode.** Duplicate rollouts come back bit-identical and 63/64
  seed-varied groups are identical, so episodes within a cell are replicates. Counting them
  independently inflates n by an order of magnitude, and has flipped a preregistered pass/fail twice.
- **The endpoint is L2 distance to the target action, not a projection ratio.** A projected-ratio
  metric certified a *random* equal-norm edit at +0.878 while that edit sat ~1.0 from both clean
  targets.
- **Every intervention ships with a no-op anchor and a norm-matched random control.** Four separate
  silent no-ops occurred during this work, each of which would have produced a clean,
  publishable-looking null.

### Key ablations and controls

| control | what it rules out | outcome |
|---|---|---|
| **cross-scene positive control** | that a null image-patch result is the *patch* failing, not the site being empty | patch efficacy rises 0.015 → **0.850**; the site has teeth ([writeup](docs/cross-scene-control.md)) |
| **self-patch anchor** | silent no-ops in the harness | exactly 0.000 in 240/240 rows |
| **norm-matched random direction** | that the effect is edit magnitude, not content | real patch wins 78/80 cells, 10/10 tasks |
| **both attention normalisations** | cherry-picking per-token vs per-segment | per token instruction wins 11.4×, per segment image wins 4.99×; both reported |
| **constant-predictor baseline** | a monitor that looks good on an imbalanced set | constant beats the monitor on both held-out splits |
| **cell-identity decode** | an unexplained monitor failure | 0.96 — the monitor learned the scene |
| **"grabs the nearest object"** | a trivial spatial confound explaining obedience | predicts obedience at 0.005 |
| **prohibition inversion** | that the gate is about phrasing | forbidden 0/50 vs commanded 48/50 |
| **superposition / no-instruction default** | two alternative accounts of the jamming case | 14.9% (below 25% chance) and 27.7% (chance) — both refuted |

### Statistical power

Cell-level throughout, with the n stated as cells and as tasks. Significance is by **task-stratified
permutation** where the design allows it (each task contributes exactly one in-region cell, so the
contrast is entirely within-task and the clustering objection does not apply) and by **one-sided sign
tests** over cells and over tasks otherwise. Permutation p-values at the resolution floor are
reported as `p < 5e-5`, never as a point value. Seeds are not used to buy power, because they do not
buy any: 48 duplicate rollouts were bit-identical.

---

## Findings in more detail

### Attention is not causation, and the monitor built on it collapses

Per token, the action expert attends 11.4× harder to instruction tokens than image tokens in the
middle layer band. Under per-segment normalisation the ordering reverses — image wins 4.99×, and
image mass exceeds instruction mass at 18/18 layers. Both are reported, because the per-token
framing alone invites the fair objection that you divided by 512 until attention stopped predicting
causation.

The claim that survives either convention: **instruction attention rises 3.3× (0.0155 → 0.0514) going
into exactly the band where patching it moves behaviour least** (0.011, against 0.832 for image
positions).

A disobedience monitor trained on these activations reaches 92.9% recall / 89.0% precision in
development. On held-out splits it is beaten by a constant predictor — instruction-split F1 0.681,
scene-split F1 0.745, constant 0.801, base rate 66.75%. Both splits are dominated. Cell identity
decodes from the same features at 0.96, which is the explanation.

### Scene-locality, four ways

| method | within scene | across scenes |
|---|---|---|
| monitor on activations | 0.929 recall | beaten by a constant, both splits |
| obedience steering direction (disjoint donor halves) | — | +0.021 … −0.079 against a 0.40 gate |
| region concept (dose-matched difference-in-differences) | cosine 0.28–0.42 | −0.015, p = 0.999; no rank-k subspace (36/36 null) |
| geometry model of behaviour | 0.830 | −0.035 |

### The graded spatial gate

With the named object inside the region the task was trained in, obedience is 95.0% with **0/200**
ignore events; outside it, 16.4%. 50 cells, task-stratified permutation p < 5e-5.

The graded form is stronger than the binary one and needs no threshold: obedience falls monotonically
with distance from the trained slot — **0.950 / 0.425 / 0.102 / 0.025** at 0 / ~0.11 / ~0.17 / ~0.26 m,
distance correlation **r = −0.793** against the binary gate's +0.765. It also retrodicts a
preregistration that failed: the 0.425 at ~0.11 m brackets the 0.400–0.550 observed 6 cm short of the
slot, so that prediction failed because it tested a binary hypothesis against a graded reality.

*Both the gate and the graded model are post-hoc and exploratory.*

### Opposite routes in two architectures

In π0.5, instruction positions are inert (0.011) and image positions carry the instruction (0.832 at
the middle band). In OpenVLA-OFT the instruction site transfers almost completely at layer 8
(`D_src` = 0.061) while the image site sits at the no-op anchor.

That OFT image null was not quotable until the **cross-scene positive control**: replacing the same
512 image positions with activations from a *different observation under the same prompt* moves the
output by `D_dst` = **0.850** at layer 8, against 0.015 in the A/B contrast — beating a norm-matched
random direction in 78/80 cells and 10/10 tasks, with the self-patch anchor at exactly 0.000. The
site has teeth; the A/B null is a property of the contrast, not of the method.
[Full writeup.](docs/cross-scene-control.md)

The control also surfaced something unlooked-for: in OFT, **scene content migrates out of the image
positions with depth** — image-site efficacy falls 0.850 → 0.685 → 0.321 across layers 8/16/24 while
the action positions rise 0.223 → 0.454 → 0.818. Because both arms use the identical prompt, this is
the one measurement here with no prompt-length confound.

With n = 2 architectures this is an **existence claim** — the locus is not universal — never
"architecture determines the locus."

---

## What this work does not claim

- **Not "routing established."** Clamping image positions blocks the instruction 12/12 in one depth
  band and 0/12 in another, which rules out trivial pinning but not a downstream motor bottleneck.
  The source-specific mediator test that would settle it destroyed the output at full strength and
  needs a strength sweep. Until then: *depth-specific, distributed mediation*, not routing.
- **Not "the carrier encodes the instruction."** Localised patches do nothing (0/8 cells against a
  preregistered 0.60 bar — a failed preregistered criterion); object-centred positions never beat
  count-matched random ones; the effect needs near-total replacement. What sits there looks like a
  **position-bound motor state**, not a portable concept.
- **Not "behaviour"** for the patching results — those measure the next action chunk. Closed-loop
  first touch is the preregistered primary endpoint and was run only where stated.
- **Not a safety result.** The monitor experiment and the causal experiments are on different LIBERO
  suites. Stitching them into one oversight-failure story spans two datasets, so that framing was
  dropped. The claim is about mechanism.

Six candidate findings were killed by controls in this repository, and two preregistered criteria are
reported as failed.

---

## Layout

```
src/pi05/    π0.5 hooks, patching, probes, mediation, steering, geometry
src/oft/     OpenVLA-OFT harness, cross-scene control, cell-level analysis
docs/        protocol and result writeups
```

Raw per-episode records are not published here.

## License

Code under the LICENSE in this repository. Model weights and LIBERO assets belong to their
respective owners; see THIRD_PARTY.md. Nothing in this repository redistributes gated weights.

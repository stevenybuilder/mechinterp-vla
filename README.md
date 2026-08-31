# mechinterp-vla

**We took two robot-control AI models apart to find out where, inside them, the instruction they are
following actually lives. It is not where the standard tools say it is — and in one model it is in the
opposite place from the other.**

---

## Start here: what this is

A modern robot is often driven by a **vision-language-action model** (VLA). You give it a camera image
and a sentence — *"put the bowl on the stove"* — and it outputs motor commands. These models work well
enough to be deployed, and nobody really knows how they work inside.

That matters for a specific reason. The main safety argument for interpretability research is:
*if a model is about to do the wrong thing, we should be able to see the intention in its internal
activity before the action happens.* Robots are where that argument gets expensive if it is false,
because the action is a physical object moving through the world.

So we asked a concrete version of it: **when a robot policy reads an instruction, where inside the
network does that instruction go, and can our standard tools find it?**

The short answer is that the tools give a confident wrong answer, in four different ways, for one
shared underlying reason.

### The two models we studied

| model | what it is | why we chose it |
|---|---|---|
| **π0.5** ("pi zero five") | An open VLA from Physical Intelligence. A vision-language model reads the image and sentence; a **separate "action expert"** module attends back to it and produces motion. | The current reference open VLA. Its two-part design lets us ask which part the instruction lives in. |
| **OpenVLA-OFT** | An open 7-billion-parameter VLA built on a Llama backbone. One **single fused network** handles vision, language, and action together. | A genuinely different architecture. If a finding is about *VLAs* and not about *one model*, it should replicate here. |

Both were tested on **LIBERO**, a standard simulated benchmark of tabletop manipulation tasks
(pick up the bowl, open the drawer, and so on).

### The two words we use constantly

- **Obedience** — does the robot go to the object the sentence names, or to a different one? This is
  measured behaviourally, from the actual rollout, not from anything internal.
- **Activation patching** — the core method. Run the model on situation A, record its internal
  activity, then re-run it on situation B while *overwriting* part of B's internal activity with A's.
  If B's behaviour changes to A's behaviour, that part of the network was carrying the information.
  It is a reversible, targeted lesion.

---

## The neuroscience we borrowed

The methods here are lifted fairly directly from systems neuroscience, because neuroscience spent
decades learning how to be wrong about exactly this class of question.

**1. Recording is not lesioning.** A neuron that fires whenever you show a face is not thereby *causing*
face recognition. To claim that you have to intervene — lesion it, or drive it optogenetically — and
show the behaviour follows. Neural networks let us do the intervention version cheaply and reversibly,
on every part, repeatedly. Our first finding is exactly this classical error, reproduced in a
transformer: **the thing the model attends to hardest is not the thing that is doing the work.**

**2. Place fields and tuning curves.** A hippocampal place cell fires when the animal is in one
particular location and falls off smoothly with distance from it. When we mapped obedience against
*where the named object was sitting*, we got a tuning curve of the same shape — the model obeys
strongly at one spot and its compliance decays with distance. That reframed the result: not a rule
with a boundary, but a **field with a centre**.

**3. Remapping.** Move a rodent to a different room and its place cells globally remap — the same cells
fire, but the map means something entirely different. Our central negative result is remapping:
every internal structure we found was valid **within one scene and meaningless across scenes.** Four
different methods, one failure mode.

**4. Population codes, not grandmother cells.** Single neurons rarely carry a whole concept. We found
the same: patching a handful of well-chosen positions does nothing; the effect only appears when you
replace nearly the whole population.

**5. Double dissociation.** The strongest classical evidence that two functions are separable is
finding one case where lesion X kills function A and spares B, and another where the reverse holds.
Across our two models we found precisely that mirror image.

---

## What we found, and what it means

### 1. The model's attention points at the wrong thing

**Measured.** Inside π0.5, the action expert attends **11.4× harder, per token, to the words of the
instruction than to the image.** By the standard practice of reading attention maps, the sentence is
clearly what matters.

Then we intervened. Overwriting the instruction's internal representation with a *different*
instruction's moved the robot's behaviour by **0.011** — essentially nothing. Overwriting the
**image** positions moved it by **0.832** — most of the way to the other behaviour.

Worse, instruction attention **rises 3.3×** heading into exactly the layers where patching it does the
*least*.

**What it means.** The instruction is not stored where the model is looking. By the time the action
module runs, the sentence has already been folded into the visual representation; the text positions
it stares at are a residue. Attention is a correlation, and here the correlation is not just weak —
it is *anti*-correlated with causal power across depth.

**Why it matters.** Attention maps are the most widely used, most intuitive interpretability artifact
in the field, and they are routinely read as explanations. This is a clean, quantitative case where
reading the attention map gives you the wrong answer with high confidence.

> **Two honesty notes.**
> *Normalisation.* Attention numbers depend on how you count them: per *token*, instruction wins
> 11.4×; per *segment*, the image wins 4.99×, because there are 512 image positions and only ~9 word
> positions, and raw image mass exceeds raw instruction mass at 18/18 layers. We report both
> everywhere. The claim that survives either convention is the depth one: attention to the
> instruction rises 3.3× into exactly the band where patching it matters least.
> *Which suite.* The "instruction positions are inert" number is a **libero_goal** result, where
> overwriting them moves behaviour by ≈0.00 across all six comparisons. On **libero_object** the same
> intervention reaches 0.05–0.48, so the instruction positions there are not inert. The dissociation
> as stated is a goal-suite result, and we do not generalise it to the model as a whole.

### 2. A monitor for disobedience that looks excellent and is worthless

**Measured.** We trained a classifier on the model's internal activations to predict, before the
robot moves, whether it is about to disobey. In development it hit **92.9% recall and 89.0%
precision** — the kind of number that ships.

On held-out data it is **beaten by a constant.** A predictor that ignores the model entirely and
always answers "it will disobey" scores F1 **0.801**. Our monitor scores **0.681** on one held-out
split and **0.745** on the other. Both. The base rate is 66.75%.

We then asked what it *had* learned: scene identity decodes from those same features at **0.96**.

**What it means.** The monitor learned to recognise *which room it was in*, not what the model was
about to do. Within a scene, room identity correlates with the outcome well enough to look like
mind-reading.

**Why it matters.** This is the interpretability-as-oversight pitch failing under its own best
conditions — white-box access, ground truth, a clean benchmark — and failing in a way that a normal
evaluation would not catch. The constant baseline is the whole finding. If you do not run it, you
ship the monitor.

### 3. Everything is scene-local — four methods, one answer

We tried four independent ways to find a portable representation of obedience. Every one worked
inside a scene and collapsed across scenes.

| method | within a scene | moved to a new scene |
|---|---|---|
| monitor on activations | 0.929 recall | beaten by a constant, both splits |
| a steering direction for obedience | — | +0.021 … −0.079, against a 0.40 threshold |
| a "region" concept subspace | cosine 0.28–0.42 | −0.015, p = 0.999; no subspace of any rank (36/36 null) |
| a geometric model of the behaviour | 0.830 | −0.035 |

**What it means.** This is not four disappointments. It is **one finding measured four ways**, and it
explains the other three: our toolkit assumes concepts are *context-portable* — that "obedience" is a
direction you can find in scene A and apply in scene B. In this policy it is not. This is remapping.

**Why it matters.** Portability is a load-bearing and usually unstated assumption behind probes,
steering vectors, and activation monitors. In a policy trained on embodied data it fails, and it fails
identically for every method that relies on it.

### 4. Obedience is a spatial field, not a rule

**Measured.** Whether the robot obeys depends on **where the named object is sitting**. Inside the
region where that task was trained: **95.0% obedience, with 0 out of 200 outright ignores.** Outside
it: **16.4%**.

The graded version is stronger and requires no arbitrary boundary. Obedience falls monotonically with
distance from the trained spot:

| distance from trained position | 0 m | ~0.11 m | ~0.17 m | ~0.26 m |
|---|---|---|---|---|
| **obedience** | 0.950 | 0.425 | 0.102 | 0.025 |

Distance correlates with obedience at **r = −0.793**, beating the yes/no version (+0.765) with no
threshold to tune. 50 cells, task-stratified permutation **p < 5e-5**.

**What it means.** A tuning curve. "The policy understands the instruction" describes something closer
to a **position-bound motor lookup with a spatial gate on it**. The sentence selects among behaviours
the model already has attached to locations — it does not describe a goal the model then plans toward.

It also cleaned up an earlier failure of ours: we had preregistered a *binary* in/out prediction and it
failed. The graded model retrodicts why — the 0.425 at ~0.11 m sits inside the 0.400–0.550 that the
failed prediction actually observed. We tested a binary hypothesis against a graded reality.

> *Both the gate and the graded model are post-hoc and exploratory. We label them so wherever we quote
> them.*

### 5. The same information takes opposite routes in two architectures

**Measured.** In **π0.5**, the instruction's own token positions are causally inert (0.011) and the
**image** positions carry it (0.832). In **OpenVLA-OFT**, it is the mirror image: patching the
**instruction** site at layer 8 transfers the behaviour almost completely, while the **image** site
sits at the no-op floor.

**What it means.** Where the instruction lives is a fact about *the model*, not about VLAs. There is no
single place to look.

**Why it matters.** It is a direct problem for transferable oversight. A monitor, probe, or safety
check built on π0.5's instruction pathway would be watching the wrong half of OpenVLA-OFT. With two
models this is an **existence claim** — the locus is not universal — never "architecture determines
the locus."

We also found that the published explanation for this kind of difference is wrong. A recent paper
attributes π0.5-vs-OpenVLA routing differences to **causal attention masking**. OpenVLA-OFT pins a
`transformers` fork whose Llama attention path *removes* the causal mask — the model is fully
bidirectional — so masking cannot be the explanation. This is checkable in one line of their pinned
dependency.

### 6. The control that made finding 5 trustworthy

A null result — "patching here does nothing" — is only as good as your proof that the patch *could*
have done something. Ours initially could not distinguish "the image positions are empty" from "our
patch silently failed."

So we ran a **positive control**: patch the same 512 image positions, but with activations from a
**different scene under the same sentence**. If the site has teeth, the output must move.

It moved from **0.015 to 0.850** — from a no-op to nearly complete behavioural takeover — beating a
magnitude-matched random edit in **78 of 80 cells and 10 of 10 tasks**, with a self-patch anchor at
exactly 0.000. [Full writeup.](docs/cross-scene-control.md)

That control also turned up something we were not looking for: in OpenVLA-OFT, scene information
**migrates out of the image positions with depth** (0.850 → 0.685 → 0.321 across layers 8/16/24) while
the action positions take it up (0.223 → 0.454 → 0.818). Because both arms use the identical sentence,
this is the one measurement here with no confound from differing prompt lengths.

---

## Why this matters, in one place

**The safety case for interpretability has a portability assumption, and it silently fails here.**
Probes, steering vectors, and activation monitors all assume you can characterise a concept in one
context and carry it to another. Four methods, four collapses, one cause.

**"Looks great in dev" is not a weak result — it is the dangerous one.** A 92.9%-recall disobedience
monitor that a constant predictor beats on both held-out splits is exactly the artifact that gets
deployed. The finding is not that monitoring is hard; it is that the failure is *invisible* without a
baseline nobody is obliged to run.

**Attention maps mislead here in a measurable, directional way.** Not noisy — anti-correlated with
causal influence across depth.

**Oversight tooling will not transfer between VLAs.** Two models, opposite loci, same benchmark.

**And the mechanism itself is not what the field's vocabulary implies.** A graded spatial field
gating a position-bound motor program is a different kind of system from one that "understands an
instruction," and it will fail in different ways — quietly, and as a function of where objects happen
to be.

---

## How we did it, and how we tried to be wrong

Full detail in [`docs/protocol.md`](docs/protocol.md). Every rule there exists because breaking it
produced a wrong number that a control later caught.

**The unit is the scene-condition cell, never the individual rollout.** Duplicate rollouts come back
bit-identical and 63 of 64 seed-varied groups are identical, so rollouts within a cell are replicates,
not samples. Counting them as independent inflates n by an order of magnitude — and did flip a
preregistered pass/fail twice here.

**The measure is distance to the target behaviour, not a projection.** A projection-based score
certified a *random* edit at +0.878 while that edit sat far from both real targets. It measured
direction, not arrival.

**Every intervention ships with a no-op anchor and a magnitude-matched random control.** Four separate
silent no-ops occurred during this work — hooks that never fired, an editor that was bypassed with no
error, a figure whose two arms were byte-identical — each of which would have produced a clean,
publishable-looking null.

### Key controls

| control | what it rules out | result |
|---|---|---|
| cross-scene positive control | that a null means a broken patch, not an empty site | 0.015 → **0.850** |
| self-patch anchor | silent no-ops in the harness | exactly 0.000 in 240/240 rows |
| magnitude-matched random edit | that the effect is edit size, not content | real patch wins 78/80 cells, 10/10 tasks |
| both attention normalisations | cherry-picking per-token vs per-segment | 11.4× vs 4.99×; both reported |
| constant-predictor baseline | a monitor that looks good on imbalanced data | constant wins on both held-out splits |
| scene-identity decode | an unexplained monitor failure | 0.96 — it learned the room |
| "just grabs the nearest object" | a trivial spatial confound | predicts obedience at 0.005 |
| prohibition inversion | that the gate is about phrasing | forbidden 0/50 vs commanded 48/50 |

**Statistics.** Cell-level throughout, with n stated as cells *and* as tasks. Significance by
task-stratified permutation where the design allows it — each task contributes exactly one in-region
cell, so the contrast is entirely within-task and the clustering objection does not apply — and by
one-sided sign tests over cells and over tasks otherwise. Permutation p-values at the resolution floor
are reported as `p < 5e-5`, never as a point value. Seeds buy no power, so we do not use them for it.

**Six candidate findings were killed by our own controls, and two preregistered criteria are reported
as failed.**

---

## Related work, briefly

The *behavioural* half of this is established, and we replicate rather than discover it.
**Spatial Overfitting** ([2505.03500](https://arxiv.org/abs/2505.03500)) reports 86–99% task success
under *wrong* prompts and 0% at novel positions. **LIBERO-Plus** and **LIBERO-PRO**
([2510.13626](https://arxiv.org/abs/2510.13626), [2510.03827](https://arxiv.org/abs/2510.03827))
document the same fragility at scale. **LIBERO-CF** ([2602.17659](https://arxiv.org/abs/2602.17659))
publishes a 30.8% grounding rate for π0.5; we measure 32.1% with their metric.

Our contribution is the **locus** question rather than the behavioural one.
**Grant et al.** ([2603.19233](https://arxiv.org/abs/2603.19233)) report 99.6% source-dominance in
π0.5 and argue language sensitivity depends on task structure rather than model design — a claim about
*behaviour*; ours is about *where the computation happens*. **VLA-Trace**
([2605.30117](https://arxiv.org/abs/2605.30117)) reports architecture differences in routing using
attention knockout, and attributes them to causal masking — which, as above, is checkably wrong for
OpenVLA-OFT.

What attention knockout cannot do, and what we add: **content-specific sufficiency** (inserting a
*different* instruction's representation rather than merely deleting one), **depth localisation**, and
a **cross-architecture existence result**.

---

## What we do not claim

- **Not "we established routing."** Clamping image positions blocks the instruction 12/12 in one depth
  band and 0/12 in another, which rules out trivial pinning but not a downstream motor bottleneck. The
  test that would settle it destroyed the output at full strength and needs a strength sweep. Until
  then: *depth-specific, distributed mediation*, not routing.
- **Not "the carrier encodes the instruction."** Localised patches do nothing (0/8 cells against a
  preregistered 0.60 bar — a failed preregistered criterion); object-centred positions never beat
  count-matched random ones; the effect needs near-total replacement. What sits there looks like a
  **position-bound motor state**, not a portable concept.
- **Not "behaviour"** for the patching results — those measure the next action chunk. Closed-loop
  behaviour is a separate endpoint and was run only where stated.
- **Not a safety result.** The monitor experiment and the causal experiments run on different LIBERO
  suites, which behave differently (32% vs 81% obedience). Stitching them into one oversight-failure
  narrative would span two datasets, so we dropped that framing. The claim is about mechanism.

---

## Repository layout

```
src/pi05/    π0.5: hooks, activation patching, probes, mediation, steering, geometry
src/oft/     OpenVLA-OFT: harness, cross-scene positive control, cell-level analysis
docs/        protocol.md (methods and analysis rules) · cross-scene-control.md (result)
```

Raw per-episode records are not published here.

## License

Code under the LICENSE in this repository. Model weights and LIBERO assets belong to their respective
owners; see THIRD_PARTY.md. Nothing here redistributes gated weights.

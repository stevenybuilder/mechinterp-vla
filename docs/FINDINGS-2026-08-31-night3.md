# New results, night 3 (2026-08-31) — two closed questions

Every number below was produced this session and is reproducible from the commands given.
Follows the project rules: re-derived from raw arrays, scene-level permutation nulls, cell as the unit.

---

## 1. The rank-k subspace rescue FAILS — §5 of HANDOFF-2026-09-01 is now closed

HANDOFF-2026-09-01 §5 left this explicitly open: *"CAFT needs a subspace spanning the concept on the
training distribution, not one universal direction... **We have not established there is nothing to
ablate.**"* We have now established it, for this direction-finding method.

**Test.** Take the region difference-in-differences vectors in `artifacts/region_dir2/did.npz`
(n = 40 vectors over 10 scenes, d = 2048, sites blk0/blk1/pooled, layers 12–17). Split the 10 scenes
into disjoint halves A|B, build the rank-k principal subspace of each half, and measure the subspace
overlap `O_k = (1/k)·||U_Aᵀ U_B||²_F`, averaged over 40 random splits. Null = the corrected
element-wise scene-assignment permutation (`renull.py`'s fix), 200 draws, which preserves group sizes
and the global common component but destroys the grouping.

**Result — null at every site, every layer, every k.**

| site | layer | k=2 | k=3 | k=5 |
|---|---|---|---|---|
| blk0 | 16 | 0.4826 (null95 ≤0.5113, p=0.950) | 0.3333 (≤0.4328, p=0.841) | 0.2203 (≤0.3139, p=0.960) |
| blk0 | 17 | 0.7590 (≤0.9368, p=0.856) | 0.7105 (≤0.8673, p=0.652) | 0.5371 (≤0.5930, p=0.836) |
| blk1 | 16 | 0.3511 (≤0.5791, p=1.000) | 0.2587 (≤0.4115, p=0.995) | 0.1750 (≤0.3263, p=1.000) |
| pooled | 17 | 0.7697 (≤0.9181, p=0.905) | 0.6564 (≤0.8259, p=0.945) | 0.4678 (≤0.6040, p=0.975) |

Observed `O_k` is **at or below** the permutation null in **all 36 cells** (3 sites × 4 layers × 3 k).
p ranges 0.652–1.000; not one is significant. This extends the rank-1 result
(`renull.py`: blk0 L16 within-scene 0.423, across-scene −0.015, p = 0.999).

**A trap we avoided.** The scene-mean spectrum looks impressive — top-1 fraction of variance 0.683
(L16) and 0.699 (L17) against a Haar null95 of 0.114 — and a first pass prints "STRUCTURE". That
verdict is an artifact of the comparison: the Haar null draws fresh random unit vectors and therefore
does not preserve the global common component shared by all DiD vectors (a generic "an object moved"
direction). Only the permutation null is valid here, and it says null. **Do not report the spectrum
result.**

**What this licenses us to say.** "This object is in the manipulandum region" is represented, but
scene-specifically: within-scene cosine 0.28–0.42, across-scene indistinguishable from chance at
rank 1 *and* as a subspace up to rank 5. There is no shared cross-scene direction or low-rank subspace
to ablate. **The DiD-derived region direction is not a viable CAFT target**, and direction-finding must
move to a different method (CAFT's own §3.1 PCA-over-model-differences or §3.2 SAE latents).

**What it does NOT license.** It does not show the concept is unrepresented, nor that no CAFT target
exists by any method. It is a negative about *one* direction-finding procedure at ranks ≤5 with n=40
vectors over 10 scenes. Underpowered for k > 5.

Repro: `subspace.py` on the 4090 → `logs/subspace.log`.

---

## 2. The CAFT hook landmine is REAL, and the workaround is verified

PLAN.md warned that `register_forward_hook` does not fire in π0.5's training path. This was previously
ASSUMED. It is now measured, on a depth-shrunk randomly-initialised π0.5 (`smoke_caft_hook.py`):

| check | result |
|---|---|
| decoder-layer forward hooks fired in the **training** forward | **NONE** |
| same hooks in the **prefix-only inference** path (what `hooks.py` uses) | [0,1,2,3,4,5] — all fire |
| monkeypatched `compute_layer_complete`, ablation applied | 3/3 expected layers |
| loss changes under ablation | clean 2.403262 → ablated 2.402423 (Δ = −8.39e-4) |
| identity wrapper reproduces clean loss **bitwise** | **True** |
| gradients reach action-expert q/v params | 12/12, mean \|grad\| = 4.64e-2 |

**This is the receipt that matters:** a naive CAFT port using `register_forward_hook` would have
trained with *no ablation at all* and reported a clean null, indistinguishable from "the method does
not work". The bitwise identity check is what makes the α=0 control trustworthy.

---

## 3. Cross-architecture attention/use test on OpenVLA-OFT — PARTIAL, and it is a different claim

Motivation: the strongest π0.5 result is a **segment-level** dissociation at fixed depth — across VLM
layers 12–17 the action expert attends **11.4× more per token** to instruction positions than image
positions, while KV[INSTR] patching moves behaviour by R = 0.011 and KV[IMG] by ≈0.92.

**The same test cannot be run on OFT: its stage-2 patching sites are INSTR / ACT / PROPRIO — there is
no IMG site.** So this is not a replication of the π0.5 claim. What the existing data supports is an
across-depth test on the instruction segment alone. Both quantities come from the same run
(`artifacts/vla_oft_stage2/20260830-112339`, 16 000 attention rows, 27 000 patching rows), so no new
rollouts were needed. Per-token normalisation is free: `rand_img` holds 10 **matched-count** random
image-column samples, built by the original harness.

| layer | attn/token, INSTR:IMG | causal R_INSTR | causal R_ACT |
|---|---|---|---|
| 8 | 6.42× | 0.978 | −0.002 |
| 16 | 28.29× | 0.354 | 0.496 |
| 24 | 4.98× | 0.039 | 0.928 |

Mean instruction:image attention ratio per token across all 32 layers: **9.93×** (range 0.79–36.02),
strikingly close to π0.5's 11.4×.

Random-column controls are clean: L8 INSTR patch +0.978 vs rand +0.134; L16 INSTR +0.354 vs +0.073;
L24 ACT +0.928 vs +0.115.

**Honest verdict: PARTIAL, and it must not be written up as a replication.**
- From L8→L24 the instruction's causal potency collapses **25×** (0.978 → 0.039) while its per-token
  attention falls only **1.29×** (6.42 → 4.98). Attention badly mispredicts causal potency.
- At L16 attention is the **highest** of the three patched layers (28.29×) while causal potency has
  already fallen 2.8×.
- **But** both quantities decline from L8 to L24, so there is no sign inversion; the claim rests on the
  *rate* gap, which is weaker evidence than π0.5's inversion.
- **And** at L8 the instruction genuinely is causal (R = 0.978), so OFT is *not* a case of "attends
  hardest to what it never uses".

**What would make it a real replication:** add an IMG patching site to `oft_stage2.py` (currently
`for site in ("INSTR","ACT","PROPRIO")`, line 333) and measure R_IMG against R_INSTR at matched layers.
The 3090 (`vla-oft-stage0`, still running) is idle and the harness already computes the segment map.

Repro: `oft_dissoc2.py` on the OFT box → `artifacts/vla_oft_stage2/20260830-112339/dissociation.json`.

---

## 4. Infrastructure resolved this session

- `scripts/vla/caft_make_base_cfg.py` — builds `/root/vla/ckpt/pi05_base_libero_cfg`, resolving the
  pi05_base → LIBERO training blocker by taking `{input_features, output_features, empty_cameras,
  normalization_mapping, dtype}` from v044 and dropping `pretrained_path`. Deliberately does **not**
  copy v044's normalisation stats (LeRobot recomputes them from the dataset at train time; copying
  would leak v044 statistics into the "base" arm). Receipt printed: base weights sha16 `0eb11ca9587678c1`,
  `identical_to_v044=False`.
- `artifacts/libero_episode_ids.json` — all 10 `libero_object` tasks matched to **454 episodes**
  (and libero_goal to 428) by joining on the arbitration manifest's own prompts. Sanity check passed:
  unmatched task strings are all libero_spatial / libero_10.
- lerobot 0.4.4 `TrainPipelineConfig` exposes both `peft` and `dataset.episodes`, so the four-arm
  recipe needs no fork of the trainer.
- Dataset is LeRobot **codebase_version 3.0**; `episodes=[0,1,2]` loads (843 frames).

---

## 5. The two-stage gate SURVIVES a properly clustered null (and the objection was a real one)

An external design review flagged that the headline `U = 385.5, z = 4.50, p = 3.4e-6` treats 10
in-region and 40 out-region **cells** as independent when they nest inside 10 tasks. If in-region cells
clustered in a few tasks, the between-task variance would be counted as evidence and the p would be
badly overstated. This is the same class of error as the null-construction bug already caught once in
this project, so it was worth checking rather than assuming.

**Re-derived from `artifacts/vla_arbitration/20260830-192300/libero_object/per_episode.jsonl`** — all
1000 conflict episodes (`wrong_object` + the `wrong:N` cross-distractor grid), cell = (task ×
prompted object), region = the two front manipulandum slots by nearest-slot assignment:

| quantity | value |
|---|---|
| cells | 50 (10 in-region, 40 out-region) |
| in-region cell obedience | **0.950** (reproduces 190/200) |
| out-region cell obedience | **0.164** (reproduces 131/800) |
| difference | **+0.786** |
| tasks where region membership VARIES within task | **10/10** |
| naive permutation (cells exchangeable across tasks) | p = 0.00005 |
| **task-stratified permutation** (labels shuffled WITHIN task) | **p = 0.00005** |

**The objection does not bite, and the reason is structural:** each of the 10 tasks contributes exactly
one in-region cell and four out-region cells. The contrast is therefore *entirely within-task* — there
is no between-task component for clustering to inflate. Naive and stratified nulls agree to the
resolution floor (20 000 draws → report **p < 5e-5**, never "p = 0.00005").

**Report it this way**, not as the Mann-Whitney: a task-stratified permutation test is the right test
for this design, and saying so pre-empts the objection instead of waiting for a reviewer to raise it.

**A trap found on the way, worth recording.** Restricting to `condition == "wrong_object"` gives only
200 episodes / 10 cells, and in *that* subset region membership is constant within every task (in-region
cells appear in only tasks 2 and 8), the stratified test is undefined, and the effect is p = 0.065 —
not significant. The 50-cell claim is only valid on the full cross-distractor grid. Anyone re-running
this must use all 1000 conflict episodes; the wrong subset silently produces a different and much
weaker result.

Repro: `cluster_fix.py` on the 4090 → `artifacts/two_stage_cells.json`.

---

## 6. Trajectory copying — an unclosed confound in the EXISTING stage-2 localisation

Recorded because it changes what the finished work may claim, not just what the new work must control.

A carrier patch transplants activations from one observation into another and may transplant *the
donor's trajectory* rather than a concept. From `vla_stage2/20260830-094027/libero_goal_confirm`
(held-out confirmation split, direction A←B):

| condition | R | L2→A | L2→B |
|---|---|---|---|
| KV[IMG]@all | 0.979 | 0.977 | **0.085** |
| KV[IMG]@12-17 | 0.814 | 0.822 | 0.243 |
| **RS[IMG]@all — image KV from a length-matched UNRELATED prompt** | **0.846** | 0.902 | 0.614 |
| KV[INSTR]@all | 0.014 | 0.074 | 0.976 |

**RS moves the axis metric by 0.846 against the true patch's 0.979.** An unrelated prompt's image KV
gets ~86% of the way. Only L2-to-target separates them (0.614 vs 0.085). Two consequences:

1. **R must never be reported alone for this claim.** The project already ruled that L2-to-target is the
   discriminator; these numbers show that rule is load-bearing, not stylistic.
2. **KV[IMG]@12-17 is whole-image replacement over all 512 image positions**, which is exactly the arm
   most exposed to trajectory copying. Our controls (RD, RS, the KV[IMG]@0-5 = 0.000 layer null) rule
   out *some* alternatives, but none of them tests whether a transplant lands on the **donor's** own
   trajectory — that quantity was never recorded.

Addressed by Appendix A of `PREREG-causal-handoff-2026-08-31.md` (added before any mediation episode
ran; original hash `f4877800…`, amended `418a90a5…`) via two new first-class conditions: a
segmentation-localized carrier (C3L) and a cross-layout donor carrier (C3X) that records L2 to both the
current-layout and donor-layout clean-B chunks. Until C3X reports, **the stage-2 localisation should be
written as "the carrier is sufficient to redirect behaviour", not as "the carrier encodes the
instruction's content".**

---

## 7. Route (a) CAFT is cancelled — on published evidence, not on our own failure

Killed after the pilot was already running; the pilot was stopped and its outputs deleted.

- Zhang & Bisk, *Flatness Preserves Instruction Following in VLAs* (arXiv 2606.23641) trains a
  `π0.5_LORA` baseline from openpi's official LoRA code at **batch 16 × 30k steps on 4×A100-80GB** and
  reports **6.75% vs the released checkpoint's 26.6%** on LIBERO-PRO (5.7% vs 13.2% on LIBERO-CF).
  LoRA fine-tuning π0.5 on LIBERO produces a policy *worse than what it started from*.
- openpi issue #711: LoRA π0.5 on LIBERO at batch 64 × 120k steps → **1%** on libero_spatial vs 96%+.
- Our budget was batch 8 on one 4090 — a quarter of the published *failing* configuration.

Our feasibility doc listed non-convergence as "top risk #1, gated by a $0.50 pilot". It is not a risk;
it is a published result. And a 2 000-step pilot cannot distinguish "will converge" from "will converge
to 6%", because a 6% policy still emits plausible actions. **Obedience-under-conflict measured on a
~6%-competent policy cannot distinguish "removed the shortcut" from "broke the policy"** — the exact
failure mode the arm existed to avoid.

Separately, the premise is worse than recorded: CAFT does *not* require the fine-tune to have installed
the shortcut (it explicitly assumes fine-tuning enhances existing mechanisms). The disqualifying fact is
that CAFT steers the *direction of generalisation induced by the fine-tune*, and ours already moves the
right way — conflict obedience 0.22 → 0.47 across the fine-tune. CAFT's own task-selection rule screens
out cases where naive fine-tuning already generalises correctly.

**Write it as a decision with a reason, not as an experiment we ran out of time for.**

---

## 8. Prior art we were missing — must be cited in the first half-page

**Li, *VLAs are Confined yet Capable of Generalizing to Novel Instructions* (arXiv 2505.03500)** already
publishes: LIBERO-Object's canonical slot structure (five tasks place the target at scene centre, five
at top-right — our "two front slots"); the wrong-prompt experiment showing 86–99% success under
incorrect prompts across four VLAs; **0% success when objects are relocated to novel positions**; and
the name **"Spatial Overfitting"** — *"'cream cheese' actually means 'the object at the location where
the cream cheese appeared during training'."* It also independently corroborates our *failed*
preregistration: their 0%-at-novel-positions is the slot reading, not the region reading.

This pre-empts the descriptive half of the two-stage gate and must be conceded in our own voice.
**What survives as ours:** they hold the prompt fixed across same-slot tasks, so their design cannot see
the conditional. Our contribution is the dissociation — *the spatial prior does not override language;
it determines whether language has standing at all* (190/200 obedience in-region, **0/200 IGNORE**,
16.4% out-region). Claim it that narrowly.

Also: our `touched_prompted_first` = 30.0% on libero_object matches LIBERO-CF's published **30.8%**
grounding for π0.5. Rename the metric **grounding rate** and cite them, rather than letting a reviewer
discover the match.

---

## 9. MEDIATION RUN 1 — the reset works; the localized carrier does not

Preregistered `PREREG-causal-handoff-2026-08-31.md` (`f4877800…`, Appendix A `418a90a5…`), written before
any episode ran. libero_goal, 12 triples attempted, **8 usable cells**, 6 init states each, 480 rows.

**Metric note that turned out to matter.** "Redirected" must be defined by L2-to-target, not R. In this
very run the equal-norm random carrier reaches **R = +0.878** while sitting ~1.0 away from *both* clean
targets — a destroyed output, not a redirection. Normalising by each episode's clean A↔B chunk distance:

| condition | nL2→B | nL2→A | R | cells on B (nL2→B < 0.5) |
|---|---|---|---|---|
| C1 clean_B | 0.000 | 1.000 | 1.000 | 8/8 |
| **C2 early source patch** | **0.000** | 1.000 | 1.000 | **8/8** |
| C3 carrier (whole-image) | 0.238 | 0.803 | 0.759 | 7/8 |
| C3L carrier (localized) | 0.995 | **0.019** | 0.004 | **0/8** |
| C3X carrier (cross-layout) | 0.307 | 0.806 | 0.844 | 7/8 |
| **C4 mediation reset** | **0.808** | **0.252** | 0.190 | **0/8** |
| C5 dead-site | 0.959 | 0.086 | 0.050 | 0/8 |
| C6 random carrier | 1.001 | 0.989 | 0.878 | 0/8 |
| C7 unrelated-prompt carrier | 0.945 | 0.784 | 0.434 | 0/8 |

**Criterion 2 — carrier mediates the source — PASSES.** Swapping the instruction at layers 0–5 and
letting it propagate lands *exactly* on clean-B (nL2→B = 0.000, 8/8 cells — which also validates the
patching machinery). Performing the same swap **while holding image positions at their clean-A values
through layers 12–17** sends the output back toward A (nL2→A = 0.252) and off B in **8/8 cells**.
Perfect discordance, exact binomial **p = 0.0039**. Median 81% of C2's effect removed.

**Criterion 1 — carrier is sufficient — FAILS as preregistered.** C3L does essentially nothing
(nL2→A = 0.019 — output all but identical to clean A); C3L/C3 effect ratio **0.006** against a declared
bar of 0.60. By Appendix A's own rule we therefore **do not claim concept transfer for the carrier.**

**Two instrumentation faults, recorded because they bound the result:**
1. **4 of 12 cells were lost** — every cell whose B-object is `wooden_cabinet_1`, which yields **0**
   segmentation patches. My script skipped the whole init when C3L was undefined instead of marking only
   C3L N/A. The loss is systematic (cabinet cells), not random. n = 8, not the preregistered 12.
2. **C3X was underpowered by construction.** d(B_current, B_donor) = 0.385 vs d(A,B) = 1.532 — the two
   B-trajectories are 3.6× closer to each other than A is to B, so "copy" and "concept" made nearly the
   same prediction. It leans copy (30/48 episodes nearer the donor, 6/8 cells) at **p = 0.056**. **[SUPERSEDED by §14: at full n = 12 this is 49/72 = 68%, p = 0.0015.]** Not
   decisive, and honestly labelled as such.

---

## 10. THE REFRAME — trajectory copying is published, on our exact model, at 99.6%

**Grant et al., arXiv 2603.19233, Table 4** — the paper we already build on — ran this check under the
name **displacement analysis** (`cos(traj,src)` vs `cos(traj,dst)`):

| model | % injection pairs whose trajectory is closer to the SOURCE | n |
|---|---|---|
| **π0.5** | **99.6%** | 1968 |
| X-VLA | 99.8% | 3150 |
| OpenVLA-OFT | 77.9% | 1079 |
| GR00T | 57.0% | — |
| SmolVLA (expert path) | 15.8% | — |

Their reading: injected π0.5 activations transplant *"spatially grounded motor programs: action
sequences bound to specific scene coordinates rather than abstract task representations."* Their
injection was full activation replacement — our whole-image arm.

**Our own data already said this and we did not join it up.** `layout-counterfactual-2026-08-30.md`
Result 2: under an A↔B slot swap, "touched whatever occupies A's canonical slot" is **invariant at 75%
across both instructions**, while named-object identity swings 0% → 75%.

**So the honest claim is not "the carrier encodes the instruction". It is sharper and better:**

> π0.5 compiles the instruction, between layers 5 and 12, into a **position-bound reach target** at
> image-token positions. Downstream of that compile the instruction text is causally inert
> (KV[INSTR] = 0.011, six-direction median KO[INSTR] = 0.000073) despite being attended **11.4× harder per token**.

The previously quoted `−0.006` was one prompt-pair direction, not the aggregate. The 2026-09-04 raw-table audit corrected it; see `numbers audit.md`.
> Language-conditioning in π0.5 is a one-way compile into motor coordinates.

This is supported by five measurements we already own (KV[IMG]@0-5 = 0.000; KV[IMG]@12-17 = 0.832;
KV[INSTR] = 0.011; attention 11.4×; the C4 reset at 8/8), is *consistent with* Grant rather than in
tension with it, and answers the reviewer's verbatim interest in conflicts between instructions and
context. **The copy-vs-concept test becomes evidence for the claim rather than a threat to it.**

**Decisive test now running** (`scripts/vla/xlayout.py`): libero_object, whose six canonical slots are
0.11–0.38 m apart, so relocating B's object makes "copy" and "concept" predict geometrically distinct
endpoints. 3-way forced choice (B@target / B@source / null), arms WHOLE / LOC / MULTI (carrier averaged
over 4 source layouts, causal-scrubbing style) / RAND, reporting the Grant displacement statistic so our
number is directly comparable to their 99.6%.

## 11. LIBERO-Safety — CUT, with reasons
Assets verified real (dataset `LIBERO-Safety/libero_safety`, 59 002 per-episode parquet files;
checkpoint `LIBERO-Safety/pi05_libero_safety`). But: the checkpoint is **orbax/OCDBT JAX only**
(~11.5 GB), needing openpi + JAX installed beside our version-pinned harness — the exact hazard the
handoff warns about, with a silent failure mode (a mis-converted checkpoint that runs and produces
plausible numbers). And the statistics do not work: **at 5 tasks a two-sided cluster-level permutation
has minimum attainable p = 2·2⁻⁵ = 0.0625 — α = 0.05 is unreachable no matter how large the effect.**
Pooled McNemar over 30 layouts ignoring the task cluster has simulated type-I error **0.115** at
moderate heterogeneity: the same clustering error as §5, again.

Write it up as **preregistered future work** with the design specified: hazard is a single BDDL object,
the counterfactual is best done by hiding the geom at render time (physics and all other pixels
bit-identical), the inanimate-obstacle control suite is prebuilt with the same five task names, and the
interesting hypothesis is whether a safety-trained VLA represents *a person* separably from *an occupied
volume* — where the likely answer, "obstacles not people", is the more surprising result.

---

## 12. DOSE–RESPONSE settles it: the carrier is a wholesale transplant, not a localized concept

The C3L null was ambiguous between "trajectory copying" and "the carrier is distributed rather than
localized to the named object". The count-matched control decides it. For each dose n we patch n image
positions at layers 12–17 from the B-run into the A-run, two arms at matched n:
**LOC** = n positions grown outward from B's segmentation patches; **RAND** = n uniformly random.
Discriminator is L2-to-clean-B normalised per episode (never R).

| n positions | LOC nL2→B | RAND nL2→B | LOC − RAND |
|---|---|---|---|
| 4 | 0.997 | 0.997 | +0.001 |
| 8 | 0.996 | 0.999 | −0.003 |
| 16 | 0.995 | 0.998 | −0.004 |
| 32 | 0.989 | 0.978 | +0.011 |
| 64 | 0.984 | 0.985 | −0.001 |
| 128 | 0.971 | 0.921 | **+0.050** |
| 256 | 0.926 | 0.822 | **+0.104** |
| **512 (all)** | **0.256** | **0.256** | 0.000 |

(lower = closer to clean-B = more redirected. 8 cells × 3 inits, 384 rows. n=512 is the same position
set for both arms, so their exact agreement there is a sanity check on the harness.)

**Two conclusions, both negative for the localized-concept reading:**

1. **Object-centred positions never beat count-matched random ones.** At n = 128 and 256 random is
   *better*. Position identity does not matter; only the amount transplanted does.
2. **The effect is a threshold at complete replacement, not a dose–response.** Patching 256 of 512
   positions barely moves anything (0.926); patching all 512 jumps to 0.256. You must substitute
   essentially the entire visual representation.

**Three independent lines now agree** that image-KV patching in π0.5 transplants the source's whole
visual→motor state rather than a concept: Grant et al.'s 99.6% displacement statistic on this exact
model; our C3X cross-layout arm leaning toward the donor (30/48, p = 0.056; **§14: 49/72, p = 0.0015 at full n**); and this dose–response.

**What this costs us and what it buys.** It removes "the carrier encodes the instruction's content"
from the claim list — a claim §9 already declined to make under the preregistered rule. It leaves
intact, and sharpens, the mediation result: the instruction's effect is *routed through* image
positions at layers 12–17 (C4 reset, 8/8 cells, p = 0.0039), even though what sits there is a
position-bound motor state rather than a portable concept. That is the one-way-compile claim in §10.

Repro: `scripts/vla/dose.py` → `artifacts/mediation/dose1/rows.jsonl`; `dose_an.py`.

---

## 13. CROSS-LAYOUT FORCED CHOICE — leans COPY, with a metric disagreement reported

The decisive version of the copy-vs-concept test, on **libero_object**, whose six canonical slots are
far apart so the two hypotheses predict geometrically distinct endpoints. Median slot separation
**0.318 m** (0.253–0.383); resulting chunk separation d(B@target, B@source) = 0.365 against
d(cleanA, B@target) = 1.230. TARGET run = canonical layout + prompt A; SOURCE run = B's object
relocated to a far slot + prompt B; patch = image-position residuals, layers 12–17, SOURCE → TARGET.
8 tasks, 4 arms, 32 rows.

A verdict counts only where the patched chunk actually **lands** near a candidate endpoint (nearer some
endpoint than clean-A, and within 0.6 of the A↔B separation). An equal-norm random patch destroys the
output, so its "nearest endpoint" label is noise, not a vote.

| arm | n | interpretable | COPY | CONCEPT | d→B@target | d→B@source | d→cleanA |
|---|---|---|---|---|---|---|---|
| WHOLE (512 pos) | 8 | 6 | **6** | **0** | 0.933 | **0.469** | 0.639 |
| MULTI (carrier averaged over 4 source layouts) | 8 | 6 | **6** | **0** | 0.679 | **0.314** | 0.751 |
| LOC (segmentation patches, 16–20 pos) | 8 | 0 | — | — | 1.229 | 1.035 | **0.001** |
| RAND (equal-norm, 512 pos) | 8 | 0 | — | — | 1.351 | 1.381 | 1.346 |

**6/6 interpretable episodes land nearer the SOURCE layout's trajectory. Exact binomial p = 0.0156.**

- **LOC does nothing at all** (0.001 from clean A) — consistent with §12's dose–response.
- **RAND is not a negative, it is uninterpretable** — ~1.35 from every endpoint, i.e. a destroyed
  output. Reporting it as "0 CONCEPT" would be misleading; it never lands anywhere.
- **MULTI is also COPY, and lands even closer to the source (0.314).** Under causal-scrubbing logic,
  averaging the carrier over several source layouts should destroy any *single* source's motor program.
  It did not. **Caveat I will not hide:** the four averaged layouts are all relocations of the same
  object, so the average still encodes "B is away from its canonical slot", which the specific source
  layout also satisfies. MULTI is therefore weaker evidence than it first appears and should not be
  quoted as a clean scrubbing control.

**METRIC DISAGREEMENT — reported, not resolved in our favour.** The Grant displacement statistic
(`cos(traj,src) > cos(traj,dst)`) gives only **25%** source-dominance for WHOLE and 12% for MULTI,
against Grant et al.'s 99.6% for π0.5. Chunk-space L2 says COPY 6/6; the angular statistic says the
opposite. They are not measuring the same thing: chunk-space L2 compares the produced action chunk to
the chunks the model actually emits in each condition, whereas the angular statistic compares a summed
10-step displacement direction to straight-line vectors at two endpoints that are angularly close from
the gripper's start pose. I judge chunk-space L2 the more direct measure and lead with it, **but the
disagreement goes in the write-up.** Anyone quoting our number against Grant's 99.6% must use their
statistic, on which we get 25% — a genuine discrepancy that we have not explained.

**Net effect on claims.** Three of four lines (dose–response, C3X donor, this) point to wholesale
transplant of a position-bound state rather than concept transfer; the fourth (Grant's own statistic
computed on our data) does not reproduce their magnitude. The safe claim remains routing, not content:
the C4 mediation reset (8/8, p = 0.0039) stands regardless, because it does not require the carrier to
be a concept — only that the instruction's effect passes through that site.

Repro: `scripts/vla/xlayout.py` → `artifacts/mediation/xlayout1/rows.jsonl`; `xl_an.py`.

**An instrumentation note.** The first xlayout run crashed on a stale `MjSim` handle: `env.reset()`
swaps the sim object out, so a handle captured before reset is dereferenced after garbage collection
(surfacing as `'MjSim' object has no attribute 'data'` plus EGL noise). Re-dereference `inner.sim` on
every use. `ContactTracker` already exposes `obj_body_id` and `geom2obj`; the manual geom→body mapping
in `region_dir2.py` and my first draft is redundant.

---

## 14. RUN 2 — full preregistered n = 12, and both results strengthen

Run 1 reported n = 8 against a preregistered n = 12. The loss was **systematic, not random**: my script
dropped the entire init whenever C3L was undefined, and C3L is undefined exactly when B's object is
`wooden_cabinet_1` (0 segmentation patches). Fixed so C3L is marked N/A and every other condition still
runs. The four recovered cells are `t2_vs_t3`, `t5_vs_t3`, `t6_vs_t0`, `t9_vs_t0` — all cabinet-target.

**Reproducibility receipt.** All **480 rows shared between run1 and run2 are bit-identical**
(max |diff| on R / l2_to_A / l2_to_B = **0.000e+00**; 0 of 1440 fields differ by >1e-6). The fix added
scenes without perturbing anything already measured.

12 cells × 6 inits, 696 rows. Discriminator = L2-to-target normalised by each episode's clean A↔B
chunk distance (never R).

| condition | nL2→B | nL2→A | R med | cells on B |
|---|---|---|---|---|
| C1 clean_B | 0.000 | 1.000 | 1.000 | 12/12 |
| **C2 early source patch** | **0.000** | 1.000 | 1.000 | **12/12** |
| C3 carrier (whole-image) | 0.235 | 0.827 | 0.783 | 10/12 |
| C3L carrier (localized) | 0.995 | **0.019** | 0.004 | 0/8 (N/A in 4) |
| C3X carrier (cross-layout) | 0.313 | 0.806 | 0.816 | 11/12 |
| **C4 mediation reset** | **0.807** | **0.252** | 0.190 | **0/12** |
| C5 dead-site | 0.961 | 0.081 | 0.050 | 0/12 |
| C6 random carrier | 1.137 | 0.997 | 0.444 | 0/12 |
| C7 unrelated-prompt carrier | 0.945 | 0.811 | 0.473 | 1/12 |

**Both headline numbers improved with the recovered cells:**
- **Mediation: 12/12 → 0/12, exact binomial p = 2.44e-04** (was 8/8 → 0/8, p = 0.0039).
- **Cross-layout donor: 49/72 = 68% nearer the DONOR, p = 0.0015** (was 30/48 = 62.5%, p = 0.056).
  The copying evidence moves from suggestive to solid.
- Every control is dead under the normalised criterion (C5 0/12, C6 0/12, C7 1/12). Under the earlier
  naive "nearer B than A" criterion C6 and C7 appeared to *pass* at 6/12 and 5/12 — another instance of
  the metric choice, not the data, deciding the answer.

**Preregistered criterion 1 still FAILS** (C3L 0/8 where defined; C3L/C3 ratio 0.006 vs a 0.60 bar).
Reported as failed. **Criterion 2 PASSES** at full n.

Repro: `scripts/vla/mediation.py` → `artifacts/mediation/run2/`; `final_an.py`, `run_consistency.py`.

---

## 15. THE SYNTHESIS — everything is scene-local, and that is the project's actual result

Produced by re-deriving the whole ACTIVE experiment set from raw (not the handoff summary). Four
independent methods, each with its own preregistered gate, all return the same answer: representations
in this policy are specific to the scene they were measured in and do not transfer.

| method | within scene | across scenes | gate |
|---|---|---|---|
| Monitor on the model's own activations | recall 0.929 / ppv 0.890 | **beaten by a constant predictor on BOTH held-out splits** | — |
| Obedience direction (disjoint donor halves) | — | **+0.021 … −0.079** over layers 12–17 | ≥ 0.40 |
| Region concept (dose-matched DiD) | cosine 0.28–0.42 | **−0.015**, p = 0.999; no rank-k subspace either | permutation null |
| Geometry model of which program wins | ceiling **0.830** | **−0.035** | — |

**This is one finding measured four ways, not four disappointments.** It explains the others: it is why
concept ablation has no target, why the monitor collapses out of distribution, and why the carrier turns
out to hold position-bound motor state rather than a portable concept.

**The claim worth making:** the interpretability toolkit developed on language models assumes concepts
are context-portable — probes, direction-finding, concept-ablation fine-tuning all rely on it. In this
embodied policy that assumption fails, and it fails the same way for every method that makes it.

### Monitor — corrected and extended (verified from raw `monitor_cost.json`)

| split | recall | PPV | FAR/1000 | F1 |
|---|---|---|---|---|
| DEV (situations seen in training, 5-fold) | 0.929 | 0.890 | 229 | 0.909 |
| DEPLOY — held-out **instruction** | 0.682 | 0.680 | 643 | **0.681** |
| DEPLOY — held-out **scene** | 0.792 | 0.704 | 669 | **0.745** |
| constant "it will disobey" | 1.000 | 0.668 | 1000 | **0.801** |

Base rate 534/800 = 66.75%. **The constant predictor beats the monitor on BOTH held-out splits**, not
only the instruction split as previously written. The write-up must say "both", and should quote the
held-out-scene split too — we had been quoting only the worse of the two, which is the weaker claim
*and* leaves the stronger one unstated.

### Also verified this pass
- **Closed-loop steering (600 episodes, 6 cells × 20 inits × 5 λ):** redirection to the commanded
  object is **0/120 at λ = 0, 0/120 at 0.25, 0/120 at 0.5, then 79/120 at λ = 1 and 86/120 at λ = 2**,
  in 6/6 cells. A sharp causal threshold against a 360-episode zero baseline. **Caveat that must travel
  with it: task success collapses 0.96 → 0.00 at the same λ**, and "stayed on A" goes to 0/120 — the
  policy is redirected *and* broken, and at this λ the two cannot be separated.
- **Steering transfer (`transfer_smoke`)**: SELF 1/5, CROSS 0/5, RAND 0/5 at λ=2 on ONE cell.
  **This is a smoke test (n = 5 per arm) and must not be reported as a transfer result.**
- **G3 route_multiplicity** 0.54–0.69 across layers 12–17 — recorded, not yet interpreted; do not quote
  until the definition is re-derived from the script.

---

## 16. CORRECTIONS — four claims of mine that were too strong

Written after an external audit of §14–15 and of the corrected pinning control. All four are my errors,
not new data.

### 16.1 "LIBERO-Safety is statistically capped at p = 0.0625" — RETRACTED
The p ≥ 0.0625 floor is a property of a **two-sided cluster-level permutation test with 5 clusters**
(2·2⁻⁵), i.e. of the specific 5-task-per-level design that was proposed. **It is not a property of the
benchmark**, which ships ~21K trajectories across ~75 tasks; combining suites and levels to reach 10+
task clusters removes the objection entirely.
**The honest reasons LIBERO-Safety was cut are deadline and conversion risk** — the released checkpoint
is orbax/JAX and needs openpi installed beside a version-pinned harness, with a silent failure mode
(a mis-converted checkpoint still runs and emits plausible numbers). Those reasons stand. The
statistical one does not, and must not appear in the write-up.

Also retracted: *"Neel's interests document has zero mentions of robots, therefore a collision endpoint
will not land."* The first clause is true; the inference is not. His stated interests explicitly call
for applied interpretability on **a problem that matters**, with a demonstration that interpretability
helps. The real objection to a robot endpoint is a domain-specific result with no general mechanistic
payoff — not the domain itself.

### 16.2 "The routing claim is established" / "it is not output-pinning" — TOO STRONG
`P_05` (clamping image positions at the causally inert early band) blocks 0/12, which rules out the
trivial reading that *any* clamp at *any* depth pins the output. **It does not rule out clamping a
downstream motor bottleneck.** Restoring all 512 image positions across six live layers to their
clean-A values is an enormous intervention, and those late states already encode the motor program, so
forcing them to A can restore the A action whether or not they specifically mediate instruction
content. The defensible statement is:

> The corrected control supports **depth-specific, distributed mediation** through image-position
> states. It does **not** fully distinguish causal routing from pinning a downstream motor-state
> bottleneck.

**The test that would settle it** (not yet run): after the early A→B swap, restore only the
**B-induced difference** at image positions rather than the full state, against equal-norm orthogonal
and unrelated-prompt differences. Removing the B-specific component should return toward A while
inserting it moves toward B. That is a source-specific mediator test; the current one is not.

### 16.3 "Returns to the original behaviour" — WRONG WORD
The pinning control measures the **next action chunk**, not behaviour. Our own preregistration names
**closed-loop first touch** as the primary endpoint precisely because an action-vector displacement is
not a behaviour. No closed-loop rollout was run for this control. Say "returns the immediate action
chunk toward clean A".

### 16.4 "12 cells" overstates the independent n — VERIFIED
The 12 cells are **12 directed pairs drawn from only 6 undirected prompt pairs over 6 distinct tasks**
(0,6), (0,9), (2,3), (2,5), (3,5), (6,9), each evaluated over 4–6 near-deterministic init states.
Consistency across them is excellent, but they are not 12 independent mechanisms. Report as
**"6 prompt pairs, both directions"**, and treat the effective n as 6 for any test that assumes
independence.

### 16.5 "The bug inverted the depth profile" — imprecise
The hook-composition bug made `P_05` look maximally blocking. It did not invert every band: `P_611`
and `P_1217` used non-overlapping layer ranges and were unaffected — their numbers are identical
across the buggy and corrected runs.

### 16.6 Reproducibility gap, now closed
The corrected `pin_control.py` and the `pin2` records existed only on the box while the local mirror
still held the invalid version and run. Both are now synced to `box_sync/4090/`.

---

## 17. CROSS-ARCHITECTURE AUDIT — three more corrections, one of them a wrongly-declared failure

### 17.1 "Preregistered P5 failed" — RETRACTED
I declared P5 (random controls < 0.20) failed on `rand0 = 0.233` at INSTR/L8. That number came from a
**3-task partial run aggregated over episodes**. Our own preregistration and standing rules say the
unit is **the cell, never the episode**. Re-derived at the cell level on the full 10-task run
(40 cells, median within cell then across cells):

| site | layer | rand0 R | verdict |
|---|---|---|---|
| INSTR | 8 | **0.107** | PASS |
| INSTR | 16 | 0.072 | PASS |
| INSTR | 24 | 0.017 | PASS |

**P5 passes.** This is the second time in one session that episode-level aggregation produced a wrong
answer (the first was quoting C3X as 49/72 episodes when cell-level it is 8/12, p = 0.19). The rule
exists for a reason; apply it before declaring anything, especially a failure.

### 17.2 Switch the primary endpoint to normalized L2 — free, and it disarms the control problem
`D_src = l2_to_src / l2_src_dst` (0 = perfect transfer, 1 = none) and `D_dst = l2_to_dst / l2_src_dst`
(0 = no-op) are **already logged** in the OFT jsonl. Under them:

| condition | R | D_src | D_dst | reading |
|---|---|---|---|---|
| PROPRIO patch (any layer) | 0.000 | **1.000** | 0.007 | the no-op anchor |
| INSTR@L8 patch | 0.971 | **0.061** | 0.986 | near-total transfer |
| INSTR@L8 rand0 | 0.107 | **0.963** | 0.197 | transfers nothing; knocks output off-manifold |
| INSTR@L8 resample | 0.389 | 0.912 | 0.887 | **the dangerous control** — R looks alive, D_src says dead |
| ACT@L24 patch | 0.908 | 0.208 | 0.937 | the answer has migrated to action positions by L24 |

R's denominator is unstable (5th percentile 0.0825 vs median 1.2944 — a 16× inflation on ~5% of
episodes). `resample` is the control that would have embarrassed us under R (0.389) and is plainly dead
under D_src (0.912). **Report D_src primary, R secondary, with PROPRIO as the printed no-op anchor.**

### 17.3 OpenVLA-OFT is FULLY BIDIRECTIONAL — and this corrects a published paper
`openvla-oft/pyproject.toml` pins a transformers **fork** whose Llama SDPA path replaces the causal
mask (`last_row.unsqueeze(2).expand(...)`, `is_causal=False`), so every token attends to every
non-pad token. π0.5's prefix is likewise bidirectional (PaliGemma prefix-LM). Two consequences:

1. **Good:** our OFT `R_IMG ≈ 0` is *not* an arithmetic identity forced by a causal mask. It is a real
   null in a model whose image positions *can* see the instruction.
2. **We cannot attribute the π0.5/OFT difference to masking** — and **VLA-Trace (arXiv 2605.30117)
   does exactly that**, attributing the π0.5-vs-OpenVLA routing difference to "bidirectional vs
   unidirectional attention." For OFT that attribution is **false and checkable**. Small, real,
   verifiable contribution — put the fork citation in a footnote.

**Also retract:** the `verify` block's `stop_token_change_moves_{INSTR,IMG}_layer1_maxabs` values of
0.125 and 0.0625 are exactly bf16 ULPs for their magnitude ranges. They are rounding, not signal, and
must not be cited as evidence of bidirectionality.

### 17.4 The real threat is duplication/re-contamination, not KV-vs-residual
Both nulls sit at segments whose content is re-supplied from elsewhere, in mirror image:
- **π0.5's INSTR null is now explained by a directly tested prefill handoff.** In the preregistered
  150-cell prefill experiment, changing only instruction-token input embeddings reproduced the donor
  action exactly. Restoring destination instruction-position KV after prefill left the result donor-like
  (median `D_src = 0.051`), while restoring destination non-instruction KV returned it destination-like
  (`D_dst = 0.051`); restoring all image KV alone reached `D_dst = 0.105`. All five gates passed in all
  six pair/direction strata. Thus `KV[INSTR]@all = 0.011` means *"instruction positions are redundant
  after a causal prefill handoff"*, **not** *"instruction positions are inert."* See
  `docs/FINDINGS-pi05-prefill-instruction-mediation-2026-08-31.md`.
- **OFT's IMG null is exposed to re-contamination.** A residual patch at layer L is re-supplied from
  the unpatched instruction positions across 23 downstream bidirectional layers.

Neither issue is fixed by more seeds or tasks. The π0.5 prefill-time intervention is now complete and
positive. The remaining run needed for an architecture-level contrast is **OFT `IMG@8-31`** (patch all
downstream layers, blocking re-contamination).

### 17.5 `same_layout` is false in 27000/27000 OFT rows
The A/B prompts never have equal token counts, so the INSTR patch aligns end-of-span over
`min(n_A, n_B)` and downstream RoPE offsets differ. The OFT INSTR effect may partly be a positional
shift rather than instruction content. Either re-run with length-matched pairs or restrict to a
matched subset before quoting it.

### 17.6 The claim to make instead
Not "opposite routes, no architecture-independent answer" (n=2, and ~80% pre-empted by VLA-Trace at the
knockout level). The strong supported π0.5 claim is now: **instruction input is used during bidirectional
prefill, then its action-relevant effect is handed into distributed non-instruction cache states, with image
positions carrying a substantial share.** The OFT result remains descriptive until downstream
re-contamination is blocked; do not present it as the other half of a causal architecture contrast.

The mechanism is also usable, with an important scope limit. On untouched `libero_object` confirmation
states, state-conditioned correct-prompt donor KV at IMAGE positions/layers 12–17 repaired wrong-prompt
success from 0/20 to 18/20, while the identical layers-0–5 replacement stayed at 0/20. No actions or
trajectories were copied, but the correct instruction was supplied to an internal donor branch on every
observation. The static cross-scene mean direction failed. See
`docs/FINDINGS-pi05-mechanism-guided-instruction-repair-2026-08-31.md`.

**The better headline if the probe lands:** train a held-out cross-task linear probe for prompt
identity from mean image-position activations. If AUC ≈ 0.95 in OFT while D_src(IMG) ≈ 1.0, the result
is **decodable but causally unused** — the decodability-vs-causality distinction, which is what the
field actually cares about, and which we already have the activations to test.

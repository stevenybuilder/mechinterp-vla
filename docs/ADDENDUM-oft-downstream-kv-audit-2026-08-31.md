# ADDENDUM — source specificity, held-out states, and inference for OFT K/V test

**Status:** frozen after the original preregistration hash and before any checkpoint outcome from the dedicated
downstream-K/V implementation. The original preregistration and hash remain preserved.

## Why this amendment exists

The original test correctly fixes downstream recontamination, but its single unrelated prompt donor is not enough
to establish source specificity, and its seven-of-ten task gate is not a substitute for task-level inference.
Activation-patching results can depend strongly on the corruption/source distribution. A positive control is also
needed to establish that a joint image-plus-instruction K/V intervention can carry the clean donor contrast.

The broad architectural observation that VLMs are language-first does not predict this experiment's result. The
claim at stake is narrower: whether prompt-conditioned image K/V is sufficient to carry an immediate action contrast
when rewritten at every downstream layer, while the destination instruction route remains intact.

## Frozen sampling amendment

- Use official LIBERO-goal init states **25--34**, rather than 0--9, for the dedicated run.
- These states were not used in the earlier local OFT patching tables. They are held out from this project's
  exploratory analyses, but they are not claimed to be disjoint from the checkpoint's LIBERO fine-tuning data.
- Source and destination prompts still share the exact same pixels, proprioception, simulator state, and timepoint.
- Task remains the inferential unit. Init and patch direction are paired repeated measures.

## Added controls

1. Replace the one unrelated prompt C with three outcome-independent official prompt donors, selected in sorted task
   order after excluding A, B, and duplicate target objects where possible. Report `resample_1`, `resample_2`, and
   `resample_3` separately and as a within-cell median.
2. On the equal-token panel, add `kv_both_8_31`: patch projected K/V for both image and instruction rows at every
   layer 8--31. This is a positive control for whether the hybrid downstream forward can reproduce the donor
   contrast. It is not run where instruction lengths differ.
3. On the equal-token panel, add a frozen semantic-equivalence control by replacing the first instruction verb
   `put` with `place`. It is eligible only when tokenizer span and full prompt lengths exactly match. Capture its
   clean action and require it to remain near the destination action relative to the clean A/B contrast before
   interpreting its K/V patch. The primary official-prompt panel never uses this synthetic paraphrase.
4. Preserve the cross-scene donor diagnostic. It is excluded from all primary success gates and is interpreted only
   as evidence about scene/motor-state transplantation.

## Competence, power, and claim scope

A task enters normalized summaries only if its paired clean A/B action contrast passes the frozen
`l2_src_dst >= 1e-4` cell rule. Report the number and identity of excluded cells and tasks. The primary causal
contrast is the task-level median improvement of persistent image K/V over the single-layer residual patch.

Use the exact task-level sign-flip test over ten tasks. A confirmatory positive requires two-sided `p < 0.05` in
addition to the preregistered effect-size and negative-control gates. The seven-task equal-token panel is a robustness
panel; all seven effect signs are required for a standalone exact sign-test claim. Otherwise it is descriptive.

Even a clean positive result licenses only:

> Prompt-conditioned image K/V is sufficient to mediate these immediate action contrasts in this released OFT
> checkpoint on official LIBERO-goal states.

It does not establish an OpenVLA architecture class, task-general semantic instruction representation, closed-loop
behavior, or out-of-distribution generalization. A persistent primary null licenses only that donor image K/V is not
sufficient while the destination's direct instruction route remains available; it does not prove that the image K/V
contains no instruction information.

## Training, contamination, and trajectory copying

No model is trained. The released OFT checkpoint is evaluated by inference only. Official simulator states generate
time-zero observations, after which the assay compares fresh action chunks. No recorded action or future observation
is passed into the model. Same-state A/B interventions eliminate literal cross-trajectory replay, while random and
three unrelated-prompt donors test generic output pinning. Whole-image K/V can still carry a same-scene motor plan;
the cross-scene diagnostic and semantic paraphrase control bound that alternative but do not turn the intervention
into a portable concept edit.

Primary methods reviewed before execution: activation-patching best practices (arXiv:2309.16042), path patching
(arXiv:2304.05969), LIBERO-PRO (arXiv:2510.03827), and Not All Features Are Created Equal
(arXiv:2603.19233).

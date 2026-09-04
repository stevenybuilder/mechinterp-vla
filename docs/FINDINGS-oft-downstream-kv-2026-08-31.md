# FINDINGS — OpenVLA-OFT downstream-blocked projected-K/V test

**Date:** 2026-08-31  
**Verdict:** implementation valid; preregistered image-mediation claim **failed**.

## Executive conclusion

The corrected OpenVLA-OFT rerun resolves the earlier downstream-recontamination ambiguity. Replacing projected
image K/V with the source-prompt values at every layer from 8 through 31 moved the immediate action only modestly
toward the source: the median task-level normalized distance to source was **0.969**, compared with **1.000** for
the original single-layer residual intervention. No task met the frozen effect-size and source-specificity gate.

By contrast, replacing instruction-position K/V on the equal-token panel moved the action almost completely to the
source (**median D_src = 0.010**), and replacing image plus instruction K/V did the same (**0.007**). The direct
instruction pathway therefore remains causally live in this checkpoint. Persistent image K/V is not sufficient to
carry the tested prompt contrast while that destination-instruction route remains available.

This is a checkpoint-specific immediate-action result. It does not establish an architecture class, prove that
image K/V contains no instruction information, or constitute a closed-loop repair.

## Implementation and primary gates

- All **3,440/3,440** frozen rows were observed; no missing, duplicate, or unexpected rows.
- Self patches were exact no-ops; all required writes occurred; repeated clean actions were deterministic.
- Clean A/B separation passed for all ten primary tasks.
- Median task-level `D_src`: persistent image K/V **0.969**; single residual L8 **1.000**.
- Median improvement was **0.031**, far below the frozen **0.20** requirement.
- Persistent image K/V beat random and all three unrelated-prompt donors by at least 0.20 in **0/10** tasks.
- Although its small improvement had the same sign in all ten tasks (exact two-sided sign-flip
  `p = 0.001953125`), the preregistered effect-size and negative-control gates failed. Statistical consistency of a
  negligible effect is not the claimed mediation result.

## Equal-token and semantic controls

Across seven equal-token task families, persistent image K/V had median `D_src = 0.957`; instruction K/V had
median `D_src = 0.010`; and image-plus-instruction K/V had median `D_src = 0.007`. The `put` to `place`
paraphrases remained near their official destination prompts before and after patching, and all seven families had
the same qualitative direction. Prompt length or token position therefore does not explain the contrast.

## Trajectory/state-transplantation diagnostic

For every tested task in both panels (**17/17**), the cross-scene whole-image K/V intervention was closer to the
other scene's donor action than to the current scene's clean source-prompt action. This is direct evidence that
broad image-state transplantation can carry donor scene/motor state. It is a diagnostic, not a primary causal
result, and it reinforces the restriction against calling whole-state injection a portable semantic mechanism.

## Consequence for the cross-architecture story

The earlier OFT single-layer null was real but undercontrolled. The corrected result now supports a narrower
contrast: in this OFT checkpoint, the instruction-position route remains live and dominates the immediate action
contrast even after downstream image K/V is overwritten. Any comparison with pi0.5 must use a valid prefill-time
instruction intervention there and must remain checkpoint-specific unless replicated across multiple checkpoints.

## Reproducibility

- Preregistration: `docs/PREREG-oft-downstream-kv-2026-08-31.md`
- Frozen audit addendum: `docs/ADDENDUM-oft-downstream-kv-audit-2026-08-31.md`
- Implementation: `scripts/vla/oft_downstream_kv.py`
- Analyzer: `scripts/vla/analyze_oft_downstream_kv.py`
- Artifact directory: `artifacts/oft_downstream_kv/v2_20260831/`
- Raw rows SHA-256: `82a3e51036b79e1231d80fd31a7192cfd91272ad91643c7cb5bec8d50ae7de26`
- Results SHA-256: `c177746ab0a1af489e026a11d15a34605120f656a561de2d06faf386acfd4171`
- Summary SHA-256: `24af6958b2aff70a22dbe79ae3cd218db770fc28167e0459cc366ac8370b80e2`

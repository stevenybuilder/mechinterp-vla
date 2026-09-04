# PREREGISTRATION — OpenVLA-OFT downstream-blocked image K/V test

**Status:** frozen before the dedicated implementation is run.

## Question

The existing OFT experiment patched one residual layer at a time. Because OFT is fully bidirectional, unpatched
instruction positions can rewrite image-position state at every later layer. The observed same-scene image null
therefore does not distinguish absent instruction mediation from downstream recontamination.

> When image-position K/V is replaced with donor-prompt K/V at every layer from 8 through 31, does the OFT
> action move toward the donor-prompt action?

This is an action-chunk causal mediation test, not a closed-loop behavioral repair.

## Data and experimental unit

- Released `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10` checkpoint.
- Official LIBERO-goal task definitions, language strings, and init-state files.
- Primary panel: the frozen ten-task A/B map from the original OFT experiment, init states 0--9, timepoint 0,
  both patch directions.
- Positional-artifact panel: all official task pairs with equal tokenizer length under the frozen pairing map;
  task families without an equal-length partner are excluded by the map, not by outcomes.
- Cell = task x init x direction. Init and direction are paired repeated measures. Task is the inferential unit;
  all hypothesis tests aggregate within task first.

## Exact K/V intervention

Capture the outputs of every decoder layer's `k_proj` and `v_proj` for clean source and destination forwards.
At destination inference, replace only the image-token rows of projected K and V. Q, instruction positions,
action positions, proprio, residual streams, input pixels, and action head are untouched.

Replacing projected K/V at every layer 8--31 blocks the proposed image-carrier from being overwritten by later
instruction-to-image attention: every downstream action query sees donor image K/V. This is more precise than
the previously staged attention-input patch, which modified Q as well and did not return the pre-hook output to
PyTorch.

## Frozen ablations

| condition | edit | purpose |
|---|---|---|
| `single_resid_l8` | original donor residual patch at IMG, layer 8 | reproduce prior near-null |
| `kv_img_8_31` | donor image K/V at every layer 8--31 | downstream-blocked primary test |
| `kv_img_0_7` | donor image K/V only at layers 0--7 | early/transient control |
| `kv_img_8_31_self` | destination image K/V written back to itself | exact hook/no-op control |
| `kv_img_8_31_random` | equal-token, equal-per-token-norm random K/V deltas | perturbation control |
| `kv_img_8_31_resample` | image K/V from an unrelated official prompt C | content-specificity control |
| `kv_instr_8_31` | donor instruction K/V at layers 8--31 | direct-language comparator; primary only in equal-length panel |
| `kv_img_8_31_xscene` | source-prompt image K/V from the paired next init state | source-scene/motor-trajectory diagnostic |

Primary metrics are normalized full-action distances `D_src` and `D_dst`; the one-dimensional action-axis score
is secondary. Rows with negligible clean source-destination distance are retained and flagged but excluded from
normalized summaries under the frozen `l2_src_dst < 1e-4` rule.

## Gates and interpretation

Implementation validity requires:

1. self patch `max |delta action| <= 1e-6` in every cell;
2. all non-self conditions report nonzero writes when their clean K/V difference is nonzero;
3. repeated clean destination actions are bit-identical;
4. random and resample controls are reported using `D_src`, never the unstable axis score alone.

For each task, take the median across init and direction. Report task-level medians, paired differences from
`single_resid_l8`, and an exact task-level sign-flip test over ten tasks. The result supports image mediation only
if `kv_img_8_31` reaches median `D_src <= 0.50`, improves over the single-layer patch by >= 0.20, and beats both
random and resample by >= 0.20 in at least seven of ten tasks. The equal-length panel must show the same effect
direction in at least five of its seven frozen task families before the result is called architecture-level.

A persistent null with a valid self/control suite resolves recontamination in favor of the narrower conclusion
that image K/V does not mediate this OFT prompt contrast. A positive result shows prompt-conditioned image K/V
mediation, not a portable instruction concept or closed-loop behavior.

The primary legacy panel contains 10 tasks x 10 init states x 2 directions = 200 paired cells. The frozen
equal-token panel contains seven eligible task families x 10 init states x 2 directions = 140 cells. Effective
sample sizes are 10 and 7 task clusters respectively. Ten tasks can detect a large task-consistent effect (the
two-sided exact sign-test floor is 0.00195), but not a small heterogeneous effect; the seven-task positional panel
is a robustness requirement rather than a separately powered confirmatory study.

## Trajectory-copying scope

Source and destination use the same physical observation, so no cross-scene state or prerecorded trajectory is
transplanted. Only projected image K/V is changed and the destination action head generates a new chunk. The
random and unrelated-prompt controls test generic motor-state pinning. Nevertheless, a whole-image K/V edit can
carry a same-scene motor plan; therefore the licensed claim is causal mediation of the immediate action contrast,
not semantic instruction representation or task-general repair.

For `kv_img_8_31_xscene`, report normalized distance to both (a) the current observation's clean source-prompt
action and (b) the other observation's donor action. If the patch is systematically closer to (b), persistent
image K/V is source-scene/trajectory bound. This does not invalidate the same-observation causal assay, but it
forbids calling the state a portable instruction representation and must travel with any positive result.

No training occurs in this experiment. The checkpoint was already OFT-finetuned on LIBERO; the assay is
in-distribution model biology. It cannot by itself show OOD generalization or exclude learned LIBERO scene
templates. The equal-token panel removes a prompt-length/RoPE alternative, while the x-scene diagnostic and the
existing same-prompt cross-scene positive control expose scene binding rather than pretending to eliminate it.

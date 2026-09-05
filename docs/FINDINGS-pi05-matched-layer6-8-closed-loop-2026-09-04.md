# Findings: matched layer-6→8 transformation in closed-loop LIBERO

**Date:** 2026-09-04  
**Status:** complete exploratory breadth screen; not confirmatory  
**Model:** `lerobot/pi05_libero_finetuned_v044`, revision `8e174154ef5f6c60a8da12ae99c303d8963138c1`  
**Primary records:** `artifacts/pi05_matched_band_rollout_screen_2026-09-04_v1/episodes.jsonl`

## Question

The preceding offline experiment showed that adding the natural instruction-B-minus-A image-field update produced across layers 6–8 moved π0.5's first ten predicted actions toward B. Does that numerical change alter what the robot actually does in the simulator?

## Design

The initial 300-episode confirmation plan was stopped after 12 preserved rows because it was too expensive for the first behavioral question. Before inspecting the remaining prompt pairs, the breadth screen was frozen: one untouched LIBERO Goal initial state (`33`) for each of 12 directed, token-position-matched prompt pairs, under three conditions:

1. clean instruction A, which conflicts with the simulator's task-B success condition;
2. clean instruction B, the behavioral ceiling;
3. clean A plus the matching-observation, full attention-plus-MLP layer-6→8 image-field update.

The intervention was recomputed at every replan and applied to all 512 image positions. Each episode ran for at most 300 simulation steps. The primary endpoint was task-B success; first target contacted was secondary.

## Verified result

| Condition | Task-B success | B target first | A target first | Touched anything | Median steps |
|---|---:|---:|---:|---:|---:|
| Clean A | `0/12` | `0/12` | `10/12` | `12/12` | `300` |
| Layer-6→8 edit | `0/12` | `3/12` | `8/12` | `12/12` | `300` |
| Clean B | `12/12` | `9/12` | `0/12` | `12/12` | `130` |

The three B-first intervention cells were:

- `libero_goal_t2_vs_t3`;
- `libero_goal_t5_vs_t3`;
- `libero_goal_t6_vs_t0`.

For B-first contact versus clean A, the nonzero-cell exact two-sided sign test was 3 positive and 0 negative, `p=0.25`. For task success every difference was zero, `p=1`. With one initial state per prompt pair, these are screen statistics, not effect-size confirmation.

The intervention was numerically active. Every matching-intervention replan recorded a nonzero message norm, every clean replan recorded zero, and the first action hashes differed across clean A, clean B, and the edit in the smoke test.

## Interpretation

The offline action shift was not meaningless, but it was not a transferred policy. The edit sometimes changed the robot's initial target selection; it never supplied the continuing, observation-dependent computation required to finish B.

This result bears on the linear-representation hypothesis only in a limited way. It does **not** show that π0.5 lacks linear features, and it does not falsify linear representation generally. It shows that a linear displacement can describe a real local causal relation while remaining insufficient for temporally extended control. The relevant policy state may change across replans, depend on the scene, use redundant routes, or be disrupted by an additive off-manifold edit. The experiment does not distinguish those explanations.

The correct claim is:

> A broad layer-6→8 linear displacement has local causal leverage over action and sometimes redirects first contact, but it does not transfer the complete task policy.

## Provenance and scope

- Breadth-screen rows: `36`, with no duplicate `(cell, init, condition)` keys.
- Episode SHA-256: `8f7022a0b5db6bad967faf570ee76f9e86e40dfa6b2e4af93da04a1e8233f549`.
- Stored summary SHA-256: `b653fd24eea2f2800fca54dd1c4711a0cbd25d25918dcf1d26deccf440205778`.
- The stopped large panel remains in `artifacts/pi05_matched_band_rollout_2026-09-04_v1/` with 12 partial rows.
- The three-row instrumentation smoke test remains in `artifacts/pi05_matched_band_rollout_smoke_2026-09-04_v1/`.

Do not combine the partial panel with the breadth screen as though they were one preregistered estimate. Do not describe `3/12` first contacts as task repair. Any confirmatory expansion should be prospectively frozen and include multiple initial states plus attention-only and random rollout controls.

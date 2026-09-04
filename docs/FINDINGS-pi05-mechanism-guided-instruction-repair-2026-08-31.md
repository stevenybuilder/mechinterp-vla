# FINDINGS — mechanism-guided instruction repair in π0.5

**Date:** 2026-08-31  
**Status:** the transferable static-direction pilot failed; the separately preregistered state-conditioned
image-cache repair passed its untouched confirmation split on both tasks.

## Executive conclusion

The identified image-cache mechanism can be used to repair a real closed-loop failure, but the successful
repair is state-conditioned and donor-assisted—not a universal steering vector.

Under conflicting instructions, π0.5 completed 0/20 held-out confirmation episodes. At every replan, we ran
an internal correct-prompt donor on the same live observation and replaced only the receiver's valid
image-position K/V in the causally live layers 12–17. The external conflicting prompt remained unchanged.
This repaired task success to **10/10 on task 1 and 8/10 on task 2**. Correct target first-touch was **9/10 on
both tasks**. The identical replacement at the causally inert layers 0–5 produced **0/20 success and 0/20
correct first-touch**. The unedited correct-prompt positive control succeeded in 19/20.

Thus the mechanism is behaviorally useful: editing the live handoff state, but not an equally broad early
state, restores multi-stage task behavior. No donor actions or trajectories were copied. However, the donor
branch was explicitly given the correct instruction on each test observation. This is a mechanistic control
and proof-of-principle repair, not an autonomous correction method.

## Static transferable direction: failed

Before the donor-assisted study, a stricter repair was attempted. For each task, we averaged
`correct-prompt minus conflicting-prompt` image-cache differences over calibration init states 0–9 and added
that frozen direction on pilot states 10–14. No alpha in `{0.5, 1, 2, 4}` passed the frozen two-task gate, so
confirmation was not run.

| alpha | task 1 success / correct first-touch | task 2 success / correct first-touch |
|---:|---:|---:|
| 0.5 | 0/5 / 0/5 | 0/5 / 0/5 |
| 1.0 | 1/5 / 0/5 | 0/5 / 5/5 |
| 2.0 | 0/5 / 0/5 | 0/5 / 0/5 |
| 4.0 | 0/5 / 0/5 | 0/5 / 0/5 |

The task-2 alpha-1 result is informative but not a repair: the static direction reliably changed the initial
subgoal, yet did not support task completion. This motivated the explicit state-conditioned control, which
was preregistered before being run and then evaluated on untouched init states 20–29.

## State-conditioned pilot and confirmation

The development pilot on init states 10–14 was perfect on both tasks:

- LIVE layers 12–17: 10/10 success and 10/10 correct first-touch;
- EARLY layers 0–5: 0/10 success and 0/10 correct first-touch.

All frozen pilot gates passed, unlocking confirmation.

| confirmation condition | task 1 success | task 1 correct first | task 2 success | task 2 correct first |
|---|---:|---:|---:|---:|
| conflicting prompt | 0/10 | 0/10 | 0/10 | 0/10 |
| correct prompt | 10/10 | 9/10 | 9/10 | 10/10 |
| LIVE donor image KV, layers 12–17 | **10/10** | **9/10** | **8/10** | **9/10** |
| EARLY donor image KV, layers 0–5 | 0/10 | 0/10 | 0/10 | 0/10 |

Every frozen confirmation gate passed on both tasks, including a success advantage of at least 50 percentage
points over the early-band control.

## What the repair establishes

Supported:

- the prefill handoff is not merely an explanatory correlation; its live image-cache state can control and
  restore closed-loop behavior;
- depth matters: an equally broad donor replacement at layers 0–5 does not repair the failure;
- the intervention generates fresh actions and adapts over the live rollout rather than copying a stored
  action sequence or trajectory;
- a single static mean direction is insufficient for the evolving multi-stage behavior under this test.

Not supported:

- a task-general or scene-independent repair direction;
- repair without access to the correct instruction at test time;
- repair of the LIBERO-Safety hidden-hazard collision failure;
- a claim that the donor-assisted edit is deployable as-is.

The honest applied-interpretability claim is:

> Identifying the live instruction-handoff site enabled a depth-specific activation repair that restored
> closed-loop task completion under conflicting prompts, while the same edit at a causally inert band failed.
> The repair currently requires a state-conditioned correct-prompt donor; a static cross-scene direction did
> not generalize.

## Reproducibility artifacts

- Static repair preregistration: `docs/PREREG-pi05-mechanism-guided-instruction-repair-2026-08-31.md`
- Static implementation addendum:
  `docs/ADDENDUM-pi05-mechanism-guided-instruction-repair-implementation-2026-08-31.md`
- State-conditioned repair preregistration: `docs/PREREG-pi05-state-conditioned-cache-repair-2026-08-31.md`
- Vector builder: `scripts/vla/build_instruction_repair_vectors.py`
- Runner: `scripts/vla/run_instruction_repair.py`
- Analyzer: `scripts/vla/analyze_instruction_repair.py`
- Static pilot: `artifacts/pi05_instruction_repair_2026-08-31/pilot/`
- State-conditioned pilot: `artifacts/pi05_instruction_repair_2026-08-31/state_pilot/`
- State-conditioned confirmation: `artifacts/pi05_instruction_repair_2026-08-31/state_confirm/`
- Aggregate analysis: `artifacts/pi05_instruction_repair_2026-08-31/analysis.json`


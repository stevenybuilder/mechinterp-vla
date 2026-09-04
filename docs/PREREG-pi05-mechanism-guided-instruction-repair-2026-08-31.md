# PREREGISTRATION — mechanism-guided instruction repair in π0.5

**Status:** frozen before building any repair direction or running any repair episode.

## Question and target failure

On `libero_object` tasks 1 and 2, π0.5 solves the task under its correct instruction (40/40 and 39/40
existing episodes) but solves 0/40 under a conflicting instruction, instead approaching the object named by
the conflict. The prefill experiment tests whether instruction influence is written into image-position
prefix KV. This follow-up asks whether that mechanism is useful:

> Can a direction estimated on calibration scenes at the causally live image-cache site restore correct
> closed-loop behavior on unseen scenes while the conflicting external instruction remains unchanged?

A positive result is a mechanism-guided repair of instruction robustness. It is not a generic safety fix and
does not imply protection against arbitrary prompt injection.

## Frozen site, tasks, and splits

- Model: `lerobot/pi05_libero_finetuned_v044`, revision
  `8e174154ef5f6c60a8da12ae99c303d8963138c1`.
- Site: valid image-position K and V, PaliGemma layers 12–17.
- Task 1 receiver prompt: task 5's tomato-sauce instruction; repair target: task 1's cream-cheese
  instruction.
- Task 2 receiver prompt: task 4's ketchup instruction; repair target: task 2's salad-dressing instruction.
- Direction-calibration init states: 0–9.
- Alpha-selection pilot init states: 10–14.
- Locked confirmation init states: 20–29. Confirmation may run only after the pilot selects one alpha by the
  rule below; no confirmation outcome may alter the alpha.

## Direction and intervention

For each task, reset each calibration scene and compute prefix KV under the conflicting prompt and the
correct prompt on the identical initial observation. At every prespecified layer, valid image position, K/V
head, and channel, average `correct - conflict` across the ten calibration scenes. This frozen tensor is the
task-specific repair direction.

During a test rollout, keep the conflicting prompt in the policy input. At every replan, compute only the
receiver's current prefix and add `alpha * direction` at the frozen site before the action expert runs. The
direction is never recomputed from the test scene. Actions are sampled afresh and executed closed-loop.
No donor action, test-scene donor activation, environment state, or trajectory is copied.

## Pilot and alpha selection

Run alphas `{0.5, 1, 2, 4}` on init states 10–14. Also run the unedited conflicting receiver and the correct
prompt positive control. Select the smallest alpha satisfying, on **both** tasks:

1. correct target is first touched in at least 3/5 episodes;
2. correct task succeeds in at least 3/5 episodes;
3. at least 4/5 episodes touch some task object, preventing apparent repair by paralysis.

If no alpha passes, report the repair pilot as failed and do not run confirmation. Ties are resolved toward
the smaller alpha. A lambda-zero implementation check must be bitwise identical to the direct unedited
harness before any nonzero pilot cell is interpreted.

## Locked confirmation and controls

On init states 20–29, run:

- `CONFLICT`: conflicting prompt, no edit;
- `CORRECT`: correct prompt, no edit;
- `REPAIR`: conflicting prompt plus the frozen task direction at selected alpha;
- `ORTHOGONAL`: conflicting prompt plus a seeded equal-norm direction orthogonal to the repair tensor at
  each layer/K-or-V tensor;
- `UNRELATED`: conflicting prompt plus the other task's repair direction, rescaled to the receiver
  direction's norm at each layer/K-or-V tensor.

The confirmation passes only if, on each task:

1. `CORRECT` succeeds in at least 8/10 and `CONFLICT` in at most 2/10;
2. `REPAIR` succeeds in at least 5/10, improves success over `CONFLICT` by at least 40 percentage points,
   and first-touches the correct target in at least 7/10;
3. neither `ORTHOGONAL` nor `UNRELATED` succeeds in more than 2/10 or first-touches the correct target in
   more than 3/10;
4. `REPAIR` touches some task object in at least 8/10, so halting or incoherent motion cannot count as a fix.

Report every episode, all object contacts, success, trajectory length, the selected alpha, and direction-to-
host KV norm ratios. Any process-restart sensitivity is reported rather than averaged away.

## Licensed conclusion

A pass licenses: a calibration-scene direction at the causally identified image-cache site repairs a known
closed-loop conflicting-instruction failure on held-out scenes more specifically than matched control
directions. It does not show an architecture-general control vector, cross-task transfer, or repair of the
LIBERO-Safety hazard failure.


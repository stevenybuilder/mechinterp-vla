# PREREGISTRATION — state-conditioned image-cache repair in π0.5

**Status:** frozen after the static-direction pilot failed and before any state-conditioned repair episode.

## Motivation and scoped question

The frozen cross-scene mean direction did not repair task completion. On task 2 at alpha 1 it nevertheless
redirected the first contact to the correct target in 5/5 pilot scenes while completing 0/5 tasks. This
suggests that one static cache delta may capture the initial subgoal but not the evolving multi-stage control
state.

This control asks a narrower question:

> If the correct-prompt image-cache state is recomputed on each current observation, can replacing only the
> causally live image-position KV band repair the wrong-prompt behavior closed-loop?

This is a state-conditioned, donor-assisted causal control. It is not a transferable intervention and the
correct instruction is supplied to an internal donor branch. A pass shows that the identified cache site can
be used to repair behavior; it does not solve how to infer the repair without the donor.

## Frozen design

- Same π0.5 checkpoint, `libero_object` tasks 1 and 2, float32 harness, action-chunk noise, contact logging,
  and conflicting/correct prompts as the static-direction study.
- Development pilot: init states 10–14, already used by the failed static-direction pilot.
- Locked confirmation: init states 20–29, still untouched by any repair experiment.
- At every replan, compute receiver KV under the conflicting prompt and donor KV under the correct prompt on
  the exact same current observation.
- `LIVE`: replace receiver K and V at all 512 valid image positions in layers 12–17 with donor values.
- `EARLY`: identical replacement in layers 0–5, the prespecified causally inert band.
- `CONFLICT` and `CORRECT`: unedited negative and positive behavioral controls.
- Execute ten newly generated actions per replan. Never copy donor actions, environment state, residuals at
  non-image positions, or trajectories.

## Pilot and confirmation gates

The development pilot unlocks confirmation only if, on both tasks:

1. `LIVE` succeeds in at least 3/5 and first-touches the correct target in at least 4/5;
2. `EARLY` succeeds in at most 1/5 and first-touches the correct target in at most 1/5;
3. all `LIVE` episodes touch some task object.

The locked confirmation passes only if, on each task:

1. `CORRECT` succeeds in at least 8/10 and `CONFLICT` in at most 2/10;
2. `LIVE` succeeds in at least 7/10 and first-touches the correct target in at least 8/10;
3. `EARLY` succeeds in at most 2/10 and first-touches the correct target in at most 2/10;
4. `LIVE` exceeds `EARLY` by at least 50 percentage points in success.

No layer, position, interpolation, or timing sweep is allowed. If the pilot fails, stop. If confirmation
fails, report the donor-assisted repair as unvalidated.

## Trajectory-copying interpretation

The donor branch receives the same live observation as the receiver and contributes only image-position KV
at a frozen layer band. The receiver's action expert generates new actions, which are executed closed-loop.
Thus this does not copy a donor trajectory or action chunk. It does inject the correct instruction's internal
state on every observation, so it cannot support a claim of autonomous or task-general repair.


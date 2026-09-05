# Preregistration: matched layer-6-to-8 closed-loop test

**Frozen:** 2026-09-04, before running any simulator episode with this intervention

## Question

The preceding experiment showed that the full instruction-dependent update across layers 6–8 moves π0.5's first ten predicted actions toward instruction B. This test asks whether that numerical shift changes the robot's observed behavior in LIBERO.

## Fixed test

- Use the same π0.5 checkpoint, 12 directed Goal prompt pairs, layer-5 input clamp, layers 6–8, layer-8 intervention site, and all 512 image positions as the completed action-output experiment.
- Use initial states 33–37.
- Run the environment for task B. Instruction A is therefore the conflicting prompt and instruction B is the correct prompt for the simulator's success condition.
- At every replan, compute the matching-observation B-minus-A transformation and add it to a clean-A host at layer 8.
- Execute the first 10 actions and repeat until success, termination, or the suite time limit.

Five conditions are fixed:

1. clean A conflict;
2. clean B behavioral ceiling;
3. matching full attention-plus-MLP transformation;
4. matching attention-only transformation with image-position MLP outputs clamped to A in layers 6–8;
5. equal-Frobenius-norm Gaussian transformation.

The flow-matching seed is fixed by directed pair, initial state, and replan index. Clean conditions and intervention conditions begin from the same official LIBERO initial state.

## Outcomes

The primary outcome is task-B simulator success. Secondary outcomes are whether the B target appears in the first gripper-contact set, whether the A target appears first, touching anything, and episode length. A broad or destructive trajectory change does not count unless it produces the intended B outcome.

The inferential unit is the directed prompt-pair cell. Each cell's binary outcomes are averaged over five initial states, giving 12 paired cell values. Report raw episode counts, median paired cell differences, 10,000-resample bootstrap intervals, and exact sign tests after removing zero differences. No post-hoc threshold determines success.

## Interpretation fixed in advance

- Full above clean A and random on B success or B first touch is evidence that the intervention changes meaningful robot behavior.
- Full above attention-only supports a behavioral contribution from the complete attention-plus-MLP computation.
- Full near clean A, despite the prior action-vector shift, means the immediate causal effect was too weak, transient, or incoherent to alter the tested behavior.
- Even a positive result remains donor-assisted and matching-observation-specific; it would not establish a compact or donor-free mechanism.

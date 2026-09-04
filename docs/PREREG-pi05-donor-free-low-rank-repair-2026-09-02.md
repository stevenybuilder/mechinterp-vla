# PREREGISTRATION — donor-free low-rank instruction repair in π0.5

**Status:** protocol locked before fitting a new intervention or evaluating any held-out pair.

## Question and claim boundary

Can a compact cache intervention learned only from calibration scenes rescue π0.5 from a conflicting
instruction on held-out target/source prompt pairs, without evaluating the correct prompt at repair time?

The repair is given a discrete intended-task ID. This is necessary to specify which behavior is desired; a
wrong prompt alone does not contain that information. The task ID selects a frozen one-hot task code. It does
not trigger tokenization, embedding, or a forward pass of the correct instruction. A positive result would
show donor-free compositional generalization of a learned internal intervention. It would not show that the
system can infer the intended task from pixels alone.

## Frozen data split

All task IDs refer to `libero_object` in the upstream suite.

- Calibration scenes: initial states 0–9 only.
- Calibration edges:
  - target 1 from sources 0, 6, 7, 9;
  - target 2 from sources 0, 1, 5, 7;
  - target 3 from sources 0, 1, 2, 8;
  - target 5 from sources 3, 7, 8, 9;
  - target 8 from sources 0, 2, 3, 4.
- Locked confirmation scenes: initial states 40–49 only.
- Locked held-out edges: `1←5`, `2←4`, `3←4`, `5←6`, and `8←9`.

The five held-out ordered task/prompt pairs never contribute a row to the fit. Existing stage-0 data from
states 0–9 established that the correct prompt succeeds on 49/50 of these cells and the conflicting prompt
succeeds on 0/50. States 40–49 were not used by the prior `libero_object` experiments in this project.

No held-out-pair pilot is permitted. Rank, scale, layers, regularization, and confirmation gates below are
fixed in advance. A single calibration-edge sentinel may be run to verify wiring and numerical invariants.

## Frozen intervention

At each calibration replan, the target-prompt and source-prompt image KV caches are computed on the same
observation. Calibration follows the target-prompt trajectory and stops after twelve replans. Image-token
rows are sampled at a fixed stride of eight. For every selected layer and for K and V separately, the target
is

`delta = target_cache - source_cache`.

The feature vector concatenates the current source-cache row with a ten-dimensional signed task code
`one_hot(target) - one_hot(source)`. A standardized ridge regression is fit jointly over all calibration
edges. Its weight matrix is truncated once to rank 8 by SVD. The stored predictor is therefore

`mean_delta + standardized_feature @ U_8 @ diag(S_8) @ V_8`.

Separate maps are fit for the live layers 12–17 and the early control layers 0–5. All maps share the same
architecture and calibration rows. The model file and configuration are hashed after fitting. Confirmation
must report those hashes and must not modify or refit the artifact.

At confirmation time, the source prompt is run once per replan. The rank-8 predictor proposes a delta from
that source cache and the frozen task code. The repair applies the proposal to image-token K and V in layers
12–17 at scale 1.0. It then generates a fresh action chunk with the ordinary action expert and executes ten
actions closed-loop.

## Leakage and identity invariants

- `repair`, `random_matched`, `orthogonal_matched`, `wrong_instruction`, `early_matched`, and `conflict`
  must record zero correct-prompt prefix forwards.
- Only the explicit `correct` benchmark and `preserve_correct` arm may run the correct prompt.
- No donor cache, donor action, correct-prompt embedding, or correct-prompt text is constructed inside a
  repair/control episode.
- Scale zero must be bitwise identical to the unedited source action chunk under identical flow noise.
- `preserve_correct` uses an exact prompt-identity gate: when the observed prompt already names the intended
  task, no edit is applied. Its action hashes must match the `correct` arm replan by replan.
- The runner aborts if a held-out edge appears in the model's recorded training edges, if an artifact hash
  changes, or if a matched control misses its target Frobenius norm beyond numerical tolerance.

## Confirmation arms

Each of the five held-out pairs is evaluated on the ten locked scenes in every arm (50 episodes per arm):

1. `conflict`: unedited source prompt.
2. `correct`: unedited intended prompt, as a behavioral ceiling only.
3. `repair`: frozen rank-8 live-layer intervention.
4. `random_matched`: seeded Gaussian deltas in layers 12–17, matched separately to every proposed K/V norm.
5. `orthogonal_matched`: seeded Gaussian deltas projected orthogonal to each proposed repair delta and then
   norm matched in the same live layers.
6. `wrong_instruction`: a delta composed for the prespecified wrong target ID, rescaled separately at every
   K/V tensor to the repair proposal's norm.
7. `early_matched`: the separately fitted layers 0–5 proposal, rescaled layer-by-layer and K/V-by-K/V to the
   corresponding live proposal norm.
8. `preserve_correct`: the same repair policy presented with the already-correct prompt; the identity gate
   must leave it unedited.

All stochastic controls are seeded from the frozen base seed. Environment and flow-noise seeds are matched
across arms for each task/initial-state unit.

## Frozen success gates

The experiment is a pass only if all five held-out pairs independently satisfy:

1. `correct` succeeds in at least 8/10 and `conflict` in at most 2/10;
2. `repair` succeeds in at least 7/10 and first-touches the intended object in at least 8/10;
3. `repair` exceeds each of `random_matched`, `orthogonal_matched`, `wrong_instruction`, and
   `early_matched` by at least five successful episodes out of ten;
4. every matched control has maximum relative norm error at most `1e-5`;
5. `preserve_correct` exactly matches the `correct` arm's action hashes and success outcomes.

The primary unit is the task/prompt pair; pooled success alone cannot pass the study. Any crash, leakage
invariant violation, post-hoc hyperparameter change, or missing arm is reported as a failed/incomplete
confirmation, not silently excluded.

## Interpretation if the gate fails

A failure is evidence that the donor-assisted result does not yet transfer to a compact frozen mechanism.
The result may still localize a causally sufficient state, but it will not support donor-free or
task/prompt-pair generalization claims.

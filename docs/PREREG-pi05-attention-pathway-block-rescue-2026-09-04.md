# Preregistration: π0.5 attention-pathway block and rescue

Date frozen: 2026-09-04  
Project: MATS mechanistic-interpretability project only  
Status: prospective with respect to attention-edge interventions

## Question

When the pixels and robot state are fixed but the instruction changes, does a small attention pathway write the instruction's consequence into image-associated state during prefix processing, and does a small downstream pathway read that state to select the motor plan?

This is not another search for a decodable layer. The claim requires selective causal leverage over a communication edge: block the proposed message, lose the downstream state and action effect, then restore only that message and rescue them.

## Prior evidence and limits

Earlier experiments already opened all of the initial states used here for other analyses. No state is claimed to be globally pristine. The split is prospective only for this new attention-edge method and its fixed gates.

Across 150 prior prefill cells, A/B image-position K/V was unchanged in cache layer 0 and prompt-dependent by cache layer 1. Because an MLP cannot move information between token positions, the first instruction-to-image write must pass through prefix block-0 self-attention. Earlier full image-cache replacement at layers 12–17 could strongly change the action, but compressed additive and donor-free variants failed to generalize. Those results motivate an edge-level test and do not count as confirmation of it.

## Fixed data and splits

- Model: `lerobot/pi05_libero_finetuned_v044`, revision `8e174154ef5f6c60a8da12ae99c303d8963138c1`.
- Suite: LIBERO Goal, time zero, identical pixels and proprioceptive state within each A/B comparison.
- Contrasts: the 12 directed cells in `artifacts/mediation_pairs.json`, representing six unordered prompt pairs in both directions.
- Fixed action noise: seed `100 * init_id`; 10 denoising steps.
- Preflight: initial state 0, first directed cell only.
- Behavior-blind writer selection: initial states 0–5.
- Behavior-blind reader selection: initial states 6–9.
- Causal screen: initial states 16–19.
- Confirmation: initial states 20–24, unopened until both screen gates pass.
- Closed-loop validation: not run unless immediate-action confirmation passes.

The exact numerical contract is `configs/pi05_attention_pathway.json`. Conflicts are resolved in favor of that file.

## Intervention primitive

For a selected attention layer, source positions, receiver queries, and query heads:

1. Run B normally to obtain its queries, keys, values, and per-head attention output.
2. Replace only the named source-position K/V vectors with clean A K/V.
3. Recompute eager attention with the same B queries, mask, scale, all non-source K/V, and all other inputs.
4. Copy the recomputed result only into the named receiver-query × head block. Every untargeted output element remains the original B value.

For a rescue, the relevant block is exempted from this A-source substitution, allowing the original B message through. For the downstream rescue, clean-B image K/V is inserted only at the selected reader edges of a writer-blocked run.

The implementation must pass synthetic localization tests. A B-to-B source substitution over the full targeted region must reproduce prefix state, cache, and action bitwise exactly on the real model.

## Writer selection and test

The full writer block runs B while substituting A instruction-source K/V for every valid non-instruction receiver at every prefix layer 0–17. This recursively prevents the B instruction from seeding non-instruction state: at block 0 all non-instruction inputs are already identical across A/B, and later direct rewrites are substituted again.

Candidate block-0 writer heads are ranked without action outcomes. For each calibration cell, the squared norm of the clean B-minus-A per-head attention output is measured over the 512 valid image receiver queries. Scores are summed across cells and states. Select the smallest top-ranked set reaching 90% of total squared write magnitude. The pathway is called compact only if this uses at most four heads. An equal-sized random set is frozen from seed 20260904.

Screen conditions include:

- clean A, clean B, and identity B;
- all-non-instruction block;
- image-only block and other-non-instruction-only block;
- all-head block-0 image rescue;
- selected-head block-0 image rescue;
- equal-sized random-head rescue;
- other-position rescue, to test an instruction-to-text/state relay.

The writer claim requires both the late image residual at layers 12–17 and the first 10 actions to move toward A under blocking and back toward B under selected rescue. Distances to both endpoints are always reported so generic damage is not counted as a switch.

## Reader selection and test

Reader candidates are the action expert's attention edges from valid image-prefix K/V to all 50 action queries at layers 12–17. On the selection split, the B run is recomputed with A image K/V only for measuring the per-edge output difference; this measurement does not alter the forward pass. Layer/head edges are ranked by squared output difference across all 10 denoising steps and all cells. Select the smallest set reaching 90% of measured energy, with a compactness ceiling of eight edges. Freeze an equal-sized random set.

The reader block substitutes A image K/V at all action heads in layers 12–17. Reader rescue allows current B image messages only at the selected edges. A wrong/random rescue is the matched control.

## Full pathway confirmation

The confirmation stage repeats both directions of every prompt pair and tests:

1. writer block;
2. selected and random writer rescue;
3. reader block on clean B;
4. selected and random reader rescue;
5. reader block after writer rescue;
6. downstream path rescue: begin from the writer-blocked prefix and insert clean-B image K/V only through selected reader edges;
7. an equal-sized wrong-edge downstream rescue.

Primary action metric is normalized L2 distance over the first 10 unnormalized actions. For any condition X, `D_A = ||X-A||/||B-A||` and `D_B = ||X-B||/||B-A||`; endpoint preference is `D_B - D_A`, positive for A-like and negative for B-like. Internal-state distances use the same definition on valid-image residuals at layers 12–17.

Success thresholds and cell-level replication requirements are frozen in the config. Confirmation is not opened unless both writer and reader causal screens pass. Closed-loop rollouts are not opened unless immediate-action confirmation passes; if opened, the prospective endpoints are first object touched and intended-task success.

## Interpretation

- Strong positive: a small block-0 writer set and small late reader set both survive held-out, bidirectional block-and-rescue tests, including the downstream rescue.
- Distributed handoff: blocking works, but rescue requires most heads/layers or wholesale state.
- Wrong pathway: selective block does not move state/action toward A, or selected rescue does not beat equal-sized controls.
- Broken model: the intervention is far from both clean endpoints or identity is not exact. This is a technical failure, not mechanistic evidence.

Negative results are reported as constraints. No failed gate may be reframed as a successful compact pathway.

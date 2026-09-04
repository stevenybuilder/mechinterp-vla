# Preregistration: π0.5 writer band 6–8 confirmation

**Frozen:** 2026-09-04, before any model activation or action was evaluated on the Object pairs listed below  
**Status of motivating result:** the preceding writer development gate failed as `ambiguous`; this is a separate post-development hypothesis, not a retroactive pass  
**Scope:** MATS model-biology project only

## Question

The complementary development curves both placed their 25–75% transition in two adjacent layers. Their largest steps were at layers 7 and 6, with monotonic correlations `0.9959` and `0.9979`. The preregistered single-step threshold nevertheless failed because those steps explained `43.5%` and `45.1%`, below `50%`.

The fixed follow-up hypothesis is:

> Prefix layers 6–8 jointly form a compact instruction-to-image writer stage, even though no individual layer owns a majority of the transition.

No additional layer sweep is allowed.

## Untouched panel

Use the eight directed LIBERO Object contrasts already frozen in `configs/pi05_attention_resolution_holdout_pairs.json`, with initialization IDs 25–29. Before this document was frozen, only task text, tokenizer length, and simulator object membership were inspected for these pairs. No model activation or action has been evaluated on them.

Within every comparison, pixels, robot state, valid token positions, and action noise are identical; only the instruction differs.

## Conditions

1. `clean_A` and `clean_B`: native endpoints.
2. `identity_B`: substitute B instruction-source K/V back into B across all prefix layers; prefix state and action must be bitwise exact.
3. `writer_block_selected_band`: in clean B, substitute A instruction-source K/V only in messages received by valid image queries at layers 6–8. This tests necessity.
4. `writer_rescue_selected_band`: substitute A instruction-source K/V into image-receiver messages at all 18 layers, but allow the live B message through at layers 6–8. This tests whether the band is sufficient to rebuild B under the broad block.
5. Matched block and rescue using the fixed equal-width control band, layers 14–16.

The intervention leaves the live query, non-instruction K/V, masks, scales, and every untargeted query/head output unchanged. Measure normalized distance to both clean action endpoints over the first ten environment-unit actions and the image residual trajectory at fixed layers.

## Frozen pass

All are required:

- selected-band block median preference for A at least `+0.5`;
- selected-band rescue median preference for B at least `+0.5`;
- block is A-like and rescue B-like in at least six of eight directed cells;
- selected block is closer to A than control block and selected rescue is closer to B than control rescue in at least six of eight cells.

Passing supports a compact three-layer writer, not a single direction or universal circuit. Failure rejects this band hypothesis.

## Contingent nonlinear diagnostic

Only if this confirmation fails, a separate lean diagnostic may ask whether the layer-6 input predicts a scene-dependent layer-8 instruction effect better locally than a per-prompt global mean. It must use a single fixed full-image CountSketch, a fixed three-nearest-neighbor rule, development Goal states for fitting, and previously unused Goal states for evaluation. It is representational evidence only; no ellipsoid, rank, kernel, architecture, or causal dose search is authorized.

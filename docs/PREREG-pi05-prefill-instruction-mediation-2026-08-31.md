# PREREGISTRATION — π0.5 prefill-time instruction mediation

**Status:** frozen before running any new prefill intervention.

## Why this is required

π0.5 computes its bidirectional PaliGemma prefix once with `use_cache=True`; its action expert then reads
the frozen per-layer prefix KV cache. The existing post-cache intervention found that replacing final
instruction-position KV has almost no action effect while replacing image-position KV has a large effect.
That does not show that text was causally inert during prefill: image positions could already have absorbed
the instruction before the cache intervention.

The load-bearing question is:

> Does changing only instruction-token input embeddings before prefix layer 0 causally write the competing
> instruction into non-instruction cache positions that then control the action?

## Design

Reuse the three frozen `libero_goal` prompt pairs, official init states 25–49, the t=0 observation, and
fixed per-cell flow noise from Stage 2. All three pairs have equal valid-text and instruction-token counts
under the π0.5 tokenizer. Run both A←B and B←A directions.

For destination prompt A and donor prompt B on the same image/state:

1. `clean_A` and `clean_B`.
2. `prefill_INSTR_B`: replace only A's instruction-token embeddings with B's before prefix layer 0, then
   run the unmodified 18-layer bidirectional prefix and action expert.
3. `prefill_INSTR_B_restore_INSTR_A`: after condition 2's prefill, restore A's instruction-position KV at
   all 18 cache layers while leaving every non-instruction position B-induced.
4. `prefill_INSTR_B_restore_NONINSTR_A`: restore every valid non-instruction cache position from A while
   retaining condition 2's instruction-position KV.
5. `prefill_INSTR_B_restore_IMG_A`: restore only all valid image-position KV from A.
6. `prefill_INSTR_B_restore_RANDIMG_A`: restore an equal-count, prespecified random subset of image positions.
7. `postcache_INSTR_B`: the existing intervention, inserting B instruction-position KV into clean A after
   prefill, rerun in the same harness.

Before action inference, assert bitwise equality of A and B prefix embeddings at every non-instruction
position and bitwise equality of `prefill_INSTR_B`'s full input embedding tensor with clean B. If either
assertion fails, stop rather than interpret the experiment.

Primary metrics are normalized L2 distances to clean A and clean B action chunks:

- `D_A = ||x-x_A|| / ||x_B-x_A||`;
- `D_B = ||x-x_B|| / ||x_B-x_A||`.

Report the established signed A–B action-axis recovery as secondary. Also report, by layer, normalized
distances of image-position K/V in condition 2 to clean A and B. Identity/no-op hooks must be bitwise exact.

## Decision rules

The prefill-handoff claim passes only if, pooled and in every prompt-pair/direction stratum:

1. `prefill_INSTR_B` reproduces clean B with median `D_B <= 0.05`;
2. restoring final instruction-position KV to A leaves the result B-like: median `D_B <= 0.25` and
   `D_A >= 0.75`;
3. restoring final non-instruction KV to A returns the result A-like: median `D_A <= 0.25` and
   `D_B >= 0.75`;
4. restoring all image KV moves at least 0.40 farther toward A than restoring equal-count random image
   positions, using paired `D_A` differences;
5. post-cache instruction insertion remains A-like: median `D_A <= 0.25`.

If condition 2 moves back toward A (`D_A < 0.5`), direct instruction-position cache is necessary and the
"already handed off" account is rejected. If condition 3 remains B-like (`D_B < 0.5`), the claimed
non-instruction mediation is rejected. Intermediate or pair-heterogeneous results are inconclusive.

## Licensed conclusion

A pass licenses: in π0.5 on these goal-prompt conflicts, instruction inputs causally affect actions during
the prefix pass, and by the final cache the effect is mediated predominantly through non-instruction
positions, with image positions carrying a substantial share. It does **not** by itself establish an
architecture-general law or a valid contrast with OpenVLA-OFT. That contrast additionally requires an OFT
intervention that blocks downstream recontamination from unpatched instruction positions.

# π0.5 attention-pathway block-and-rescue result

**Date:** 2026-09-04  
**Model:** `lerobot/pi05_libero_finetuned_v044`, revision `8e174154ef5f6c60a8da12ae99c303d8963138c1`, float32  
**Decision:** writer gate failed; reader gate passed; confirmation states 20–24 and closed-loop rollouts were not opened  
**Interpretation (corrected after follow-up audit):** instruction→image communication across the prefix is necessary, and a small block-0 seed is not sufficient. The screen gives selective evidence for a distributed late reader, but the exact writer remains unresolved and the reader did not receive valid held-out prompt-pair confirmation.

## Question

Earlier work showed that the instruction's action-relevant effect leaves the original text positions and becomes dependent on broad image-associated state. Whole-state replacement could repair 18/20 rollouts, but that did not distinguish an instruction pathway from a generic motor bottleneck. This experiment intervened on attention communication itself.

The proposed compact chain was:

`instruction tokens → block-0 writer heads → image state → image K/V at layers 12–17 → action-expert reader heads → action`

The preregistered standard required blocking a communication edge to remove the downstream effect and restoring only a small edge set to rescue it on held-out initial states in all 12 directed prompt contrasts.

## Method

Pixels, robot state, token positions, action noise, and all non-instruction input embeddings were held fixed within each A/B comparison. The 12 directed LIBERO Goal cells represent six unordered prompt pairs in both directions.

The intervention replaced only source-position K/V inside an attention calculation, recomputed attention with the receiver run's original queries, mask, scale, and all non-source K/V, then copied the result only into specified receiver-query × head outputs. Untargeted output entries remained bitwise unchanged.

The writer block ran instruction B but substituted instruction-A K/V for every valid non-instruction receiver at prefix layers 0–17. Because non-instruction inputs are identical at block 0, and the substitution is repeated at later blocks, this prevents B from seeding or rewriting non-instruction state. Rescue allowed selected original-B attention outputs through this block.

The reader block ran the clean-B prefix but substituted image-A K/V in action-expert attention at layers 12–17. Reader rescue restored original-B image messages only at selected layer/head edges.

Selection was behavior-blind. Writer heads were ranked by clean A/B per-head output difference over 512 valid image queries at block 0 on initial states 0–5. Reader edges were ranked by the output change induced by A-image K/V under the live B queries over all ten denoising steps on initial states 6–9. Causal screening used separate initial states 16–19. Confirmation states 20–24 were protected by gates.

## Instrumentation preflight passed

Synthetic localization tests passed `7/7`. On the real model:

- B-to-B source substitution reproduced every captured prefix residual and cache tensor bitwise exactly;
- the resulting action was bitwise exact;
- the reader-measurement hook left the live action bitwise exact;
- the hook observed 18 prefix layers with Q shape `[1, 8, 968, 256]` and K/V shape `[1, 1, 968, 256]`;
- it observed all six reader layers × eight heads on each of ten denoising steps.

## Writer: direct image writing is causal, but block 0 is not sufficient

Behavior-blind selection found a concentrated block-0 signal. Heads `4, 3, 1, 5` captured `92.26%` of squared A/B write difference; head 4 alone captured `68.21%`. The matched control was the disjoint complement `0, 2, 6, 7`.

The causal screen contained `480 = 12 cells × 4 states × 10 conditions` unique rows. Positive endpoint preference means A-like; clean A and B are `+1` and `−1`.

| Writer condition | Median action preference | Cell directions at matching endpoint | Median late-image preference |
|---|---:|---:|---:|
| Block all instruction→non-instruction writes, layers 0–17 | `+0.9219` | A-like `12/12` | `+1.0000` |
| Block instruction→image writes only, layers 0–17 | `+0.7605` | A-like `12/12` | `+0.7247` |
| Block instruction→other valid text/state writes only | `−0.8385` | B-like `12/12` | `−0.7181` |
| Rescue selected four block-0 image heads | `+0.9221` | A-like `12/12` | `+0.9990` |
| Rescue all eight block-0 image heads | `+0.9221` | A-like `12/12` | `+0.9989` |
| Rescue all eight block-0 other-position heads | `+0.9217` | A-like `12/12` | `+0.9991` |

The block is a strong necessity result: direct instruction-to-image communication across prefix layers carries most of the tested instruction effect, while the route through other valid text/state positions is secondary. But even restoring every block-0 image head had essentially no effect. The first prompt-dependent image state appears after block 0, but that write is not a sufficient seed under this blocked context. Later direct writes, transformations of existing state, overwriting, relays, and hybrid-state mismatch remain live explanations.

The preregistered writer rescue gate therefore failed. Selected rescue beat the disjoint control in only `4/12` directions, versus the required `8/12`, and did not restore either late state or action.

## Reader: selective on the screen, but distributed and unconfirmed

The 90% behavior-blind reader threshold selected 13 of 48 layer/head edges, exceeding the frozen compactness ceiling of eight. The largest individual edges were layer 14 head 1 (`28.96%` of measured effect), layer 16 head 3 (`18.52%`), and layer 14 head 0 (`10.70%`).

The reader screen contained `240 = 12 cells × 4 states × 5 conditions` unique rows.

| Reader condition | Median `D_A` | Median `D_B` | Median action preference | Endpoint replication |
|---|---:|---:|---:|---:|
| Full A-image block, layers 12–17 | `0.2007` | `0.8512` | `+0.6484` | A-like `11/12` cells, `45/48` states |
| Disjoint matched-edge rescue | `0.2281` | `0.8348` | `+0.6047` | A-like `11/12` cells, `45/48` states |
| Selected 13-edge rescue | `0.9491` | `0.0799` | `−0.8694` | B-like `12/12` cells, `48/48` states |

Selected rescue beat the matched control in `12/12` directed cells. The reader screen gate passed. This is selective causal leverage on new initial states within the known prompt pairs, but it is not a compact reader under the preregistered definition and was never confirmed on corrected held-out prompt pairs.

## Conclusion

The original compact-pathway hypothesis is false. The data support a more specific distributed mechanism:

> The instruction's consequence depends on direct instruction→image communication across the prefix rather than a sufficient small block-0 seed. The action expert screen identifies a selective but distributed late image-to-action readout concentrated in layers 12–16; its prompt-pair generalization remains unconfirmed.

This goes beyond saying that image tokens or layers 12–17 matter. It separates writer and reader communications, shows which route is necessary, identifies an all-head block-0 rescue that fails, and demonstrates selective late reader rescue against a disjoint matched control across new initial states and both directions of the known prompt pairs. These are scene holdouts within known prompt pairs, not held-out prompt-pair semantics.

Because the writer gate failed, full-pathway confirmation and prospective closed-loop validation were correctly not run. The result does not establish a compact circuit, repeated nonlinear communication, an abstract instruction variable, or a deployable intervention. It constrains the explanation: a block-0 seed is insufficient, instruction→image communication across the prefix is necessary, and the screened downstream readout is spread across multiple reader edges.

## Artifacts and integrity

Canonical archive: `artifacts/pi05_attention_pathway_2026-09-04_v1/`

| Record | Rows | Status |
|---|---:|---|
| `writer_selection_rows.jsonl` | 576 | valid JSONL, unique keys |
| `reader_selection_rows.jsonl` | 2,304 | valid JSONL, unique keys |
| `writer_screen_rows.jsonl` | 480 | valid JSONL, unique keys |
| `reader_screen_rows.jsonl` | 240 | valid JSONL, unique keys |

The archive also contains the real-model preflight, sealed manifest, behavior-blind selections, and both gate summaries. The config and preregistration hashes are stored beside their source files. `confirmation_rows.jsonl` does not exist because the runner enforced the failed writer gate.

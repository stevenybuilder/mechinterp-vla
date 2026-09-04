# π0.5 reusable source-mediator result

**Date:** 2026-09-04  
**Model:** `lerobot/pi05_libero_finetuned_v044`, revision `8e174154ef5f6c60a8da12ae99c303d8963138c1`, float32  
**Decision:** the preregistered causal screen failed; confirmation initialization IDs 10–15 were not opened  
**Interpretation:** a rank-16 component has reproducible, sign-specific partial causal leverage, but it is not a sufficient or necessary instruction mediator

## Question

Earlier experiments showed that replacing all 512 valid image positions through VLM layers 12–17 could control π0.5's action and repair 18/20 closed-loop conflicting-instruction rollouts. That intervention might simply clamp a generic downstream motor state. This experiment asked whether calibration scenes contain a much smaller instruction-specific component that transfers to held-out scenes without using a donor activation from the evaluated example.

## Frozen design

The stimulus was the existing set of 12 directed, time-zero, same-observation prompt cells in `libero_goal`, representing six unordered prompt pairs. Only the instruction changed within a cell.

- Fit IDs: 0–5.
- Behavior-blind representation selection IDs: 6–7.
- Causal screen IDs: 8–9.
- Conditional confirmation IDs: 10–15; unopened because the screen failed.
- Candidate layers: 12–17.
- Candidate token-by-feature matrix ranks: 1, 2, 4, 8, 16, 32.
- Dose: 1.0; no causal dose search.

For each cell, initialization, and layer, the paired effect was `Δ = H_B[IMG] − H_A[IMG]` over 512 image positions. The operator was fit from the calibration mean using a rank-32 SVD, with each component reliability-shrunk according to its across-initialization variance and renormalized to preserve the rank-`k` Frobenius norm. The fitted operator was cell-specific but did not contain an activation from the evaluated initialization.

Selection used activation alignment only. The fitted operator was compared with the equal-norm `A→C` operator for the same scene. The frozen rules selected one global layer and rank before any screen action was evaluated.

The causal conditions were clean A/B, fitted insertion and removal, reverse sign, same-scene wrong prompt, matched-spectrum random bases, and direct full-rank donor replacement across all six live layers as a ceiling. All action conditions used identical noise. Primary distances over the first ten environment-unit actions were normalized so clean source distance to target was 1.

## Representation selection passed

The online calibration fit completed 12/12 cells and 72 layer-cell fits. Rank 32 captured a median `86.2%` of calibration mean-delta energy; component reliability had median `0.991`.

The behavior-blind held-out gate selected **layer 13, rank 16**:

- median cosine with the held-out paired delta: `0.5272`;
- median cosine for the wrong-prompt operator: `0.2769`;
- median source-specific selectivity: `0.2700`;
- fitted alignment exceeded wrong-prompt alignment in `12/12` directed cells.

This establishes a compact and repeatable representational component under the fitted geometry. It does not establish that the component controls the action.

## Causal screen failed

The screen completed all `288 = 12 cells × 2 initializations × 12 conditions` rows with no duplicates or missing cells.

| Frozen primary outcome | Result |
|---|---:|
| Fitted insertion-and-removal joint target landings | `0/12` cells |
| Unordered prompt pairs with a joint landing | `0/6` |
| Median fitted insertion target distance `D_B` | `0.7478` |
| Median fitted removal target distance `D_A` | `0.8064` |
| Wrong-prompt joint target landings | `0/12` |
| Matched-spectrum random joint target landings | `0/12` |
| Broad donor-ceiling joint target landings | `9/12` |

Only `1/12` cells crossed the A/B midpoint under insertion and only `1/12` under removal; these were not the same cells, producing `0/12` joint successes. The fit therefore failed the required `8/12` joint-landings and six-pair-coverage gate. The broad direct-replacement ceiling also narrowly missed its separate `10/12` joint gate. This ceiling is a stricter direct clean-A/clean-B insertion/removal test than the earlier early-swap restoration experiment, so `9/12` does not contradict the earlier `12/12` restoration result.

## Secondary finding: consistent but sub-threshold causal leverage

The failure was not complete inertness. Relative to the clean-source baseline target distance of 1, the fitted edit reduced target distance in `12/12` cells for both insertion and removal. But it remained closer to the source in `11/12` cells each way:

| Mode | Median target distance | Median source distance | Cells crossing midpoint |
|---|---:|---:|---:|
| Insert `+T` into A | `D_B=0.7478` | `D_A=0.3265` | `1/12` |
| Remove `T` from B | `D_A=0.8064` | `D_B=0.2708` | `1/12` |

The predicted sign and source were informative. The fitted component had lower target distance than:

| Comparison | Insertion | Removal | Conservative unordered-pair coverage |
|---|---:|---:|---:|
| Reverse sign | `12/12`, median advantage `0.2650` | `12/12`, `0.2488` | `6/6` both ways |
| Wrong prompt | `10/12`, `0.1558` | `11/12`, `0.1182` | `6/6` both ways |
| Matched-spectrum random | `12/12`, `0.2461` | `12/12`, `0.1907` | `6/6` both ways |

This post-hoc descriptive audit does not change the failed preregistered gate. At the conservative six-pair level, a 6/6 sign pattern has descriptive one-sided probability `0.015625`; it is not treated as confirmatory because this audit followed the gate result and the pairs are few.

## Mechanistic conclusion

The strongest supported conclusion is intermediate between “no compact signal” and “compact mediator”:

> A reusable rank-16 instruction-specific component exists in layer-13 image-associated residual state and exerts correctly signed, source-selective partial causal pressure on the action. It is not sufficient to convert the action to the alternate instruction and is not necessary enough to return the alternate action to the source. Most causal control remains in a broader, scene-conditioned state.

This improves the earlier interpretation by showing that the broad state is not wholly undifferentiated motor state. It does not eliminate the motor-bottleneck objection or justify calling the rank-16 component an instruction circuit. The MATS assessment therefore remains high-end borderline rather than accept.

No more ranks, layers, doses, token subsets, nonlinear maps, or threshold changes should be tried on screen or confirmation IDs under this question. A future study would need a genuinely new identification strategy—such as a pathway-level communication intervention—not a rescue sweep.

## Artifacts and integrity

Local archive: `artifacts/pi05_sonar_lite_source_mediator_v1/`

| Artifact | SHA-256 |
|---|---|
| `operators.pt` | `0c16c44e97446afd13c8d520f3a52a92706b723c1c221f985ca931ae0c4d7564` |
| `selection.json` | `2eb2e7456df2546667c6146b1163bbbec2c6bae24c9036e8a05215f88b405621` |
| `selection_rows.jsonl` | `2be0f18bcfa5979eb55c23ee7a25698b2aada5a364690ce01cfc777a4e3daa35` |
| `screen_rows.jsonl` | `08d59eaaa4dcbd5e1bd394d97960ad62f09149cbb71388e40687065f201ba5c9` |
| `screen_summary.json` | `e64688d08778069dd9876f8b1bbce8938a1f36eba1587a33adcbc23ad3d395e2` |

The remote and local hashes matched at archival time. The archive is approximately 12 MB and includes the operator, all 288 raw action rows, fit and selection rows, receipts, gate report, and post-hoc partial-effect audit. Runtime logs are `logs/sonar_lite_fit.log`, `logs/sonar_lite_select.log`, and `logs/sonar_lite_screen.log`.

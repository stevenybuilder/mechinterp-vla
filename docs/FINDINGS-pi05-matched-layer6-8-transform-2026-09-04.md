# Findings: matched layer-6→8 transformation

**Date:** 2026-09-04  
**Status:** complete; 420/420 expected rows  
**Preregistration:** [`PREREG-pi05-matched-layer6-8-transform-2026-09-04.md`](PREREG-pi05-matched-layer6-8-transform-2026-09-04.md)  
**Raw rows:** [`../artifacts/pi05_matched_band_transform_2026-09-04_v1/rows.jsonl`](../artifacts/pi05_matched_band_transform_2026-09-04_v1/rows.jsonl)  
**Summary:** [`../artifacts/pi05_matched_band_transform_2026-09-04_v1/summary.json`](../artifacts/pi05_matched_band_transform_2026-09-04_v1/summary.json)

## Result in plain English

A broad instruction-dependent update produced by π0.5's layers 6–8 can be measured in one initial scene and reused in another initial scene for the same instruction pair. It reliably moves the action toward the other instruction. An equal-sized random update does not.

Attention alone explains little of the effect. When the image-position MLP output at layers 6, 7, and 8 is held to its instruction-A value, the update becomes much weaker. The useful object is therefore closer to a multi-block attention-plus-MLP transformation than to a single attention message.

This is not yet a compact mechanism. The intervention adds a full 512-position image-field message measured from paired A/B runs. It transfers across initial states within known prompt pairs, not to new prompt semantics or another model, and it has not been tested in closed-loop rollouts.

![Matched layer-6→8 transformation result](../figures/07_matched_band_transform.png)

## What was done

For each of 12 directed, token-position-matched LIBERO Goal instruction contrasts and five new initial states (33–37):

1. Hold the 512 valid image positions at the output of layer 5 to the clean-A value.
2. Run decoder layers 6, 7, and 8 under instruction A and instruction B.
3. Take the B-minus-A image-field difference at the output of layer 8. This is the **full message**.
4. Add that message to the output of layer 8 in an otherwise clean-A prefix and generate the action using the same noise as the clean endpoints.
5. Repeat with the message from the next initial state, rescaled to the matching message's Frobenius norm.
6. Repeat the B run while replacing the image-position MLP output at each of layers 6–8 with its clean-A value. The resulting B-minus-A difference is the **attention-only message**.
7. Compare with an equal-norm Gaussian message.

The action metric is progress along the clean A→B action axis over the first ten actions. Clean A is 0 and clean B is 1. The inferential unit is the median within each directed prompt-pair cell (`n=12`), not the 60 initial-state units or 420 condition rows.

## Headline results

| Condition | Median cell progress to B | Bootstrap 95% interval | Positive cells |
|---|---:|---:|---:|
| Clean A | `0.0000` | `[0.0000, 0.0000]` | `0/12` |
| Clean B | `1.0000` | `[1.0000, 1.0000]` | `12/12` |
| Matching full attention+MLP message | `0.33999` | `[0.13946, 0.47487]` | `12/12` |
| Other-scene full message, norm matched | `0.21078` | `[0.11241, 0.26699]` | `12/12` |
| Matching attention-only message | `0.04022` | `[0.01218, 0.10199]` | `10/12` |
| Other-scene attention-only, norm matched | `0.03555` | `[0.01298, 0.10799]` | `12/12` |
| Equal-norm random message | `0.00578` | `[-0.00404, 0.01687]` | `9/12` |

The random condition has nine positive signs because tiny random movements can project positively along a one-dimensional axis. Its median is near zero and its interval crosses zero. The important test is the paired cell-level difference.

## Paired contrasts

| Contrast | Median cell effect | Bootstrap 95% interval | Cell signs | Exact two-sided sign test |
|---|---:|---:|---:|---:|
| Matching full − other-scene full | `0.05838` | `[0.00489, 0.18740]` | `9` positive, `3` negative | `p=0.145996` |
| Matching full − matching attention-only | `0.22356` | `[0.13165, 0.44080]` | `12` positive, `0` negative | `p=0.000488` |
| Other-scene full − random | `0.19991` | `[0.12529, 0.27593]` | `12` positive, `0` negative | `p=0.000488` |

The cross-scene reuse result is strong: the other-scene full message beats random in every cell. The MLP result is also strong: full attention+MLP beats attention-only in every cell.

The matched-versus-other-scene comparison is weaker. Its median and bootstrap interval favor the matching scene, but the exact sign test does not reject zero at conventional levels. The defensible wording is **shared cross-scene transformation with suggestive scene-specific modulation**, not a clean proof of scene dependence.

## What this changes

The earlier experiments found broad causal image-associated state but failed to isolate a static direction, object-local patch, low-rank component, block-0 writer, fixed band block/rescue, or action-relevant curvature term. This result explains part of that pattern.

The useful instruction signal is not merely an additive feature already sitting at layer 6. It is created by running the mixed state through several full transformer blocks. Attention moves information between positions, but the token-wise MLPs materially rewrite what has arrived. A small attention message inserted into an otherwise mismatched state omits that computation.

At the same time, the full update is not wholly scene-specific. It retains causal effect when moved to another initial state. This weakens the strongest “global remapping” story. A better model is:

> layers 6–8 implement a broad instruction-conditioned transformation with a reusable component and scene-dependent modulation.

## What this does not show

- It does not identify a low-dimensional direction, subspace, feature, head set, or circuit.
- It does not show that the same update transfers to an unseen instruction pair. Every cross-scene donor uses the same directed A/B task contrast.
- It does not provide a donor-free intervention. Both A and B executions are required to measure the message.
- It does not show task success, first touch, or preservation of unrelated behavior. Only immediate action chunks were sampled.
- It does not establish that “the MLP contains the instruction.” The MLP is token-wise and depends on the mixed input created by attention and residual computation.
- It does not invalidate the earlier whole-state or attention-route results. It refines them by locating a transformation scale at which causal reuse becomes visible.

## Integrity and independent recomputation

- Raw rows: `420/420`; SHA-256 `c3f38266fe998593b05ad0a6b36ecbaa31cc63f474c232c49872fee673467993`.
- Summary SHA-256: `65ebe6ebaee94b42a8d9d1b62bd4ba27966550f2f06fd04acd41f01d213618c9`.
- Runtime log SHA-256: `7e6cb769ed9b36e775e1338183301863de3f4289e089e36bd5c5d3d7ea6d614b`.
- No duplicate `(cell, init, condition)` keys.
- Action-axis metrics recomputed from stored action vectors with maximum absolute error `1.26×10⁻⁷`.
- Norm-matched conditions have maximum relative norm error `2.99×10⁻⁶`.
- All condition medians, contrast medians, and sign-test probabilities reproduce the canonical summary exactly.

The independent calculation is in [`../scripts/audit_research_numbers.py`](../scripts/audit_research_numbers.py) and its output is stored in [`../artifacts/numbers-audit-derived.json`](../artifacts/numbers-audit-derived.json).

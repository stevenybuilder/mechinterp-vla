# Preregistration: matched layer-6-to-8 transformation

**Frozen:** 2026-09-04, before any action was sampled for this experiment

## Question

The previous midpoint test showed that the image-associated representation changes nonlinearly between the output of layer 5 and the output of layer 8. Removing that curvature did not reliably change the action. This experiment asks a narrower question: does this three-block computation contain a causal instruction-writing rule that transfers to another scene, or is its useful effect tied to the scene in which it was measured?

## Fixed panel

- Model: `lerobot/pi05_libero_finetuned_v044`, revision `8e174154ef5f6c60a8da12ae99c303d8963138c1`.
- Suite: LIBERO Goal.
- Contrasts: the existing 12 directed, token-position-matched instruction pairs in `artifacts/mediation_pairs.json`.
- New initial states: 33–37.
- Input state: the 512 valid image positions at the output of layer 5.
- Tested computation: decoder layers 6, 7, and 8.
- Intervention: add the measured instruction-induced message to the 512 image positions at the output of layer 8 in an otherwise clean-A run.
- Action metric: first 10 actions under identical sampled noise.

For each scene, hold the entering image field fixed to its clean-A value and run the band with either instruction A or instruction B. Their layer-8 image-field difference is the **matched full message**. This includes attention and MLP computation. To obtain an **attention-only message**, repeat the B run while replacing the image-position MLP output in layers 6, 7, and 8 with the corresponding clean-A MLP output.

Apply four non-clean messages to a common clean-A host:

1. the full message from the matching scene;
2. the full message from the next initial state, rescaled to the matching message's Frobenius norm;
3. the attention-only message from the matching scene;
4. the attention-only message from the next initial state, likewise norm matched.

An equal-norm Gaussian message is the negative control. Clean A and clean B are the endpoints. The cyclic next-state donor is fixed by the ordered list `[33, 34, 35, 36, 37]`.

## Metrics and inference

The main metric is progress along the clean A-to-B action axis:

`((x - A) · (B - A)) / ||B - A||²`,

computed over the first 10 actions. A is 0 and B is 1. Distances to both endpoints are also retained. The statistical unit is the median within each directed prompt-pair cell, giving 12 paired values—not 60 scenes or 420 action runs.

The primary contrast is matched full minus norm-matched mismatched full progress. Secondary contrasts are matched full minus matched attention-only, and mismatched full minus random. We will report the median paired effect, a deterministic 10,000-resample bootstrap interval over the 12 cells, and an exact two-sided sign test. No post-hoc cutoff determines success.

## Interpretation fixed in advance

- Positive mismatched-versus-random progress supports a reusable cross-scene rule.
- Positive matched-versus-mismatched progress supports scene conditioning.
- Positive full-versus-attention-only progress supports a necessary contribution from within-block MLP processing.
- A null result means this fixed layer band and additive message do not identify a reusable causal rule. It does not prove that the model is linear, that no conditional mechanism exists elsewhere, or that all nonlinear geometry is behaviorally irrelevant.

This is a single adjudicating experiment, not another layer or hyperparameter search.

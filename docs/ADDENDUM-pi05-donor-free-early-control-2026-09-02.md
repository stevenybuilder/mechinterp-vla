# IMPLEMENTATION ADDENDUM — global norm match for the early-layer control

**Status:** frozen after fitting, but before a calibration sentinel or any held-out-pair evaluation.

The preregistration specified that the learned early-layer proposal would be rescaled K/V tensor by K/V
tensor to the corresponding live-layer proposal norm. A saved-model smoke test on synthetic cache tensors
showed that layer-0 image K and V proposals are exactly zero. Rescaling those zero tensors to a nonzero live
norm is mathematically undefined.

The `early_matched` control is therefore amended to preserve the learned direction over the complete early
band and apply one scalar so that

`||(delta_K, delta_V) for layers 0–5||_F = ||(repair_delta_K, repair_delta_V) for layers 12–17||_F`.

This is the standard magnitude-matched early-layer comparison at the band level. It does not change the
repair, fitted factors, rank, alpha, task/prompt split, seeds, confirmation states, or any other control. The
runner records `norm_match_mode = global_band` and aborts if the relative total-norm error exceeds `1e-5`.

At the time of this amendment:

- the rank-8 model had already been fit and hashed;
- no calibration-sentinel environment had reset successfully;
- the five held-out task/prompt pairs and initial states 40–49 had never been evaluated;
- the edge case was found using random synthetic cache tensors with the frozen model, not held-out data.

# Preregistration: lean midpoint-curvature diagnostic

**Frozen:** 2026-09-04, after the layer-6–8 causal confirmation failed and before this diagnostic was run  
**Scope:** one direct nonlinearity test; no manifold search

## Question

Is the mapping of the 512-token image field from the layer-6 input to the
layer-8 output meaningfully non-affine along the model's real A→B instruction
contrast?

For each of the 12 existing directed LIBERO Goal contrasts and untouched initial
states 28–32:

1. capture clean-A and clean-B image states at the output of layer 5;
2. in a B-context forward pass, insert A's image state, the exact A/B midpoint,
   or B's image state at that boundary while leaving every non-image position B;
3. capture the resulting 512-token image field at the output of layer 8;
4. measure
   `||F(midpoint) - midpoint(F(A), F(B))|| / ||F(B) - F(A)||`
   using the exact float32 fields and Frobenius norm.

The B-endpoint insertion must reproduce clean B bitwise. No action is sampled.

## Frozen interpretation

Meaningful non-affinity requires both:

- overall median curvature/chord ratio at least `0.10`;
- a per-cell median ratio at least `0.10` in at least 8 of 12 directed cells.

A pass directly establishes curvature along these instruction-relevant chords in
the layer-6–8 image-state mapping. It does not identify a manifold or causal
bottleneck. A failure says this mapping is approximately affine at this scale and
ends the branch. No alternate layers, interpolation points, positions, thresholds,
or metrics will be tried.

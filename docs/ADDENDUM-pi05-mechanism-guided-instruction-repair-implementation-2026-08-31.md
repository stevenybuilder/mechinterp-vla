# Addendum — instruction-repair implementation details

**Frozen before loading a model to estimate repair directions.** The original preregistration remains
immutable at SHA-256 `7a5257b9f9deb758928ce9b48b7fb3086f96fbdb982612e9cd3dd970d99f35ab`.

- Run the PyTorch π0.5 harness in float32. This avoids the allocator-sensitive bf16 divergence that halted
  the older control pilot.
- Execute ten actions from every newly generated 50-action chunk, matching the existing closed-loop policy
  setting.
- Use explicit flow noise seed `20260831 + 100000*task_id + 1000*init_id + replan_index` for every condition.
- The lambda-zero check compares action chunks from an unedited cache and a cache passed through the exact
  additive edit path with coefficient zero. Any non-bitwise action difference stops the experiment.
- Direction tensors remain on CPU until an edit. No test-scene correct-prompt forward pass is allowed in a
  `REPAIR`, `ORTHOGONAL`, or `UNRELATED` episode.
- "Correct target first touched" means the correct target is a member of the first nonempty object-contact
  set. The entire set is logged so simultaneous contacts are not broken in the repair's favor.
- Seed orthogonal controls with `20260831 + 1000*task_id + 100*layer + kv_index`, where `kv_index` is 0 for K
  and 1 for V. Orthogonalize over the complete layer/site tensor and rescale to exactly the repair tensor norm.
- `UNRELATED` uses the other task's tensor and is rescaled independently at each layer and K/V tensor to the
  receiver direction's norm.


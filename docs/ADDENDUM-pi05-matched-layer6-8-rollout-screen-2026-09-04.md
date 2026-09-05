# Addendum: breadth-first rollout screen

**Frozen:** 2026-09-04 after two matching-intervention episodes in the oversized panel failed, before testing the other 11 directed prompt pairs

The original 300-episode rollout panel was needlessly expensive for the first behavioral question: does the intervention ever produce its intended task-B behavior across the fixed prompt-pair panel? It was stopped after 12 preserved rows, including two completed matching-intervention episodes. Neither completed B; both touched A first, while both clean-B controls succeeded.

The replacement is an explicitly exploratory breadth/futility screen. It runs one official initial state (`33`) for each of the same 12 directed Goal pairs under only clean A, clean B, and the matching full layer-6→8 intervention: 36 episodes total. The layer band, 512-position intervention, per-replan recomputation, action horizon, and task-B success condition are unchanged.

If the intervention produces no task-B successes and no B-target first contacts beyond clean A, stop. That establishes a negative screen, not a precise zero-effect estimate. If it produces an intended behavioral switch in at least one cell while clean A does not, expand prospectively to more initial states and the attention-only and random controls. Do not select new layers, doses, or prompt pairs.

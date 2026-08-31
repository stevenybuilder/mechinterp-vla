# Protocol — measurement, units, and the analysis rules

The rules below are not stylistic. Each one exists because breaking it produced a wrong number that
survived until a control caught it.

## 1. Models and data

- **π0.5** (LeRobot `pi05_base` / `pi05_libero_finetuned`) — a VLM trunk with a separate action
  expert. Patching sites: instruction token positions, image patch positions, action positions.
- **OpenVLA-OFT** (`openvla-7b-oft-finetuned-libero-spatial-object-goal-10`) — one fused decoder, no
  separate action expert. Sites: INSTR / IMG / PROPRIO / ACT. Note that OFT is **fully
  bidirectional**: its pinned `transformers` fork replaces the causal mask on the Llama SDPA path, so
  architectural differences between the two models are not explained by masking.
- **LIBERO** `libero_goal` and `libero_object` suites. The two suites behave differently (32%
  vs 81% obedience), so a result from one is never quoted as a result about "LIBERO".

## 2. The unit is the cell, never the episode

A **cell** is one (task × condition × timepoint × direction) combination. Statistics are computed as
*median within cell, then median across cells*.

This is not conservatism for its own sake. In this setting duplicate rollouts come back
**bit-identical**, and rollouts that differ only by seed are identical in 63 of 64 groups — so
episodes within a cell are near-perfect replicates and counting them as independent inflates n by an
order of magnitude. Aggregating over episodes has twice flipped a preregistered pass/fail here:
a cross-layout result read as 49/72 episodes is 8/12 cells (p = 0.19), and a control declared failed
at 0.233 on a partial episode-level read passes at 0.107 at the cell level on the full run.

## 3. The endpoint is L2 in action space, never a projection ratio

Prefer

```
D_src = |patched − clean_src| / |clean_src − clean_dst|     0 = full transfer
D_dst = |patched − clean_dst| / |clean_src − clean_dst|     0 = no-op
```

and always print a **no-op anchor** alongside. A projected-ratio metric (R) certified a random
equal-norm edit at R = +0.878 while that edit sat ~1.0 away from *both* clean targets — it measures
direction, not arrival. L2 to the target does not have this failure mode.

## 4. Nulls over scenes must be scene-level element-wise permutations

Permuting scene labels is an identity operation for a same-scene-vs-different-scene statistic. A null
built that way is guaranteed to look significant and means nothing.

## 5. Every intervention carries a no-op anchor and a norm-matched control

- **self-patch** — write a site's own activations back into itself. Must be *exactly* zero. This is
  the anti-silent-no-op anchor.
- **norm-matched random direction** — an equal-norm random edit at the same positions, to separate
  content from magnitude.
- Every hook records how many positions it wrote and the norm of the change it made, and asserts
  non-zero. A patch that silently did nothing is reported, never averaged in.

This instrumentation exists because four separate silent no-ops occurred in this codebase, each of
which would otherwise have produced a clean, publishable-looking null:
`register_forward_hook` firing on none of π0.5's decoder layers in one path; a KV editor bypassed
when constructed after a rollout (`n_edits=0`, no error); a steering figure whose two arms were
byte-identical; and a hook-composition bug that turned a control into an identity operation in 48 of
48 rows.

## 6. Attention must be reported under both normalisations

Per-token and per-segment normalisation of attention mass give opposite orderings when one segment
has 512 positions and the other has 8. Reporting only the flattering one is the single easiest way to
manufacture an "attention is not causation" result. Both are reported everywhere, along with the
count of layers at which each segment wins on raw mass.

## 7. Prompt-length confounds

When an A/B prompt pair has different token counts, positional encodings differ between arms and an
instruction-site effect can be partly a RoPE artifact. Runs are marked with whether token layouts
matched. Where they did not, the caveat travels with the number.

## 8. Claims are graded to the evidence

Each result document ends with an explicit **may claim / may not claim** section. Post-hoc and
exploratory analyses are labelled as such in the same paragraph as the claim, not in a footnote.
Failed preregistered criteria are reported as failed.

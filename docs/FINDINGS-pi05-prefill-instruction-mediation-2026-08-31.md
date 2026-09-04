# FINDINGS — π0.5 prefill-time instruction mediation

**Date:** 2026-08-31  
**Status:** all frozen gates passed, pooled and in every prompt-pair/direction stratum.

## Executive conclusion

π0.5's instruction positions are not causally inert. The instruction acts during the bidirectional PaliGemma
prefill, and by the final cache most of its action-relevant effect has been transferred into non-instruction
positions, especially the distributed image-position cache.

Changing only the instruction-token input embeddings before layer 0 reproduced the donor action exactly.
After that prefill, restoring destination instruction-position KV barely reversed the result: pooled median
distance to the donor was **0.051**. Restoring destination non-instruction KV did reverse it: pooled median
distance to the destination was **0.051**. Restoring all image-position KV alone moved the first ten actions
near the destination (**0.105**), whereas restoring a frozen small random subset of image positions left them
near the donor (**0.998** distance to destination). Post-cache instruction-position insertion remained near
the destination (**0.051**).

The defensible interpretation is a **prefill handoff into distributed image-position cache states**. The old
phrasing that instruction positions are simply “inert” should be replaced with “redundant after prefill.”

## Frozen-gate results

The experiment contains 150 scene × direction cells: three prompt pairs, 25 official init states, and both
directions. Each cell has nine conditions, for 1,350 rows.

| condition | median distance to destination | median distance to donor | reading |
|---|---:|---:|---|
| prefill instruction swap | 1.000 | 0.000 | exact donor input and action |
| then restore instruction KV | 0.988 | 0.051 | stays donor-like |
| then restore non-instruction KV | 0.051 | 0.988 | returns destination-like |
| then restore all image KV | 0.105 | 0.950 | image cache carries a substantial share |
| restore frozen random image subset | 0.998 | 0.006 | small arbitrary subset does not reverse it |
| post-cache instruction insertion | 0.051 | 0.988 | remains destination-like |

The median paired destination-distance advantage of restoring all image positions over the random subset was
**0.892**, above the frozen 0.40 gate. It ranged from 0.747 to 0.924 across the six strata. Every stratum passed
all five decision rules.

## What this does and does not settle

Supported:

- instruction input is causally used during π0.5's prefix computation;
- by the final prefix cache, non-instruction positions predominantly mediate the tested prompt effect;
- distributed image-position KV carries a substantial share of that effect;
- the prior post-cache instruction null reflects redundancy after a handoff, not evidence that language was
  never used.

Not established:

- that a small localized subset of image positions is sufficient—the successful image intervention restores
  all valid image positions;
- that the same routing occurs in every π0.5 task family;
- an architecture-general contrast with OpenVLA-OFT. OFT still requires a causally equivalent intervention
  that prevents unpatched instruction positions from reintroducing information downstream;
- closed-loop behavioral repair. That is the separately preregistered next experiment.

## Reproducibility artifacts

- Preregistration: `docs/PREREG-pi05-prefill-instruction-mediation-2026-08-31.md`
- Runner: `scripts/vla/prefill_instruction_mediation.py`
- Analyzer: `scripts/vla/analyze_prefill_instruction_mediation.py`
- Raw rows: `artifacts/pi05_prefill_mediation_2026-08-31/rows.jsonl`
- Analysis: `artifacts/pi05_prefill_mediation_2026-08-31/analysis.json`
- Manifest: `artifacts/pi05_prefill_mediation_2026-08-31/manifest.json`


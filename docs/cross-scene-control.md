# OFT cross-scene positive control — the IMG patching site has teeth

2961 forward passes on one 24 GB GPU. Harness `src/oft/oft_crossscene.py`, analysis
`src/oft/oft_crossscene_analyze.py`; 2880 patch rows over 80 cells.

## The ambiguity it exists to remove

In the A/B run (the A/B run) the 512 IMG positions gave `D_src` = 0.999 / 0.977 / 0.983 at L8/16/24 —
indistinguishable from the PROPRIO no-op anchor. We read that as "image positions are inert for the
instruction contrast." But the same rows carry **`D_dst` = 0.015**: the patch barely moved the output
at all. Two readings were not separable:

1. the site genuinely carries nothing the readout uses, or
2. A and B are the **same observation** with different prompts, so the donor IMG activations are
   nearly identical to the ones they replace — the "patch" is a near-no-op by construction — and/or
   the edit is undone downstream.

## Design

Same prompt, **different observation**: two init states of the same task, both run with task A's own
instruction. The donor's IMG activations are now genuinely different, so a site with teeth must move
the output. Endpoint is L2 in action space, never R (see docs/protocol.md):

- `D_dst = |patched − clean_dst| / |clean_src − clean_dst|` — **primary here: patch efficacy**
- `D_src = |patched − clean_src| / |clean_src − clean_dst|` — transfer completeness

(The A/B run's primary was `D_src`. They answer different questions; say so when quoting both.)

Unit is the **cell** = task × init-pair × timepoint × direction. 10 tasks × 2 init pairs × 2
timepoints × 2 directions = **80 cells**, libero_goal, layers 8/16/24, sites IMG / INSTR / PROPRIO /
ACT. Paired scenes differ by a median clean-action distance of 0.565 (IQR 0.305–0.768).

Controls, all in the same rows:
- **`self`** — patch the recipient's own activations into itself. Exactly 0.000 / 1.000 in **240/240**
  rows. This is the anti-silent-no-op anchor; this harness has produced four silent no-ops of exactly this shape, each of which would have looked like a clean null.
- **`rand0`** — equal-norm random direction at the same positions; separates content from norm.
- Every hook records how many positions it wrote and the norm of the change: **0 non-self patches
  had a zero recorded edit**.
- Repeat determinism max|Δactions| = 0.0. Clean A chunks for the overlapping cells are
  **bit-identical** to the the A/B run A/B run — same model state, so the two runs are comparable.

## Result — the site has teeth, and the effect is content, not norm

Cell-level medians [IQR], n = 80 cells:

| site | L | D_dst (patch) | D_dst (rand0) | D_dst (self) | D_src (patch) |
|---|---|---|---|---|---|
| IMG | 8  | **0.850** [0.789–0.943] | 0.364 | 0.000 | 0.298 [0.229–0.379] |
| IMG | 16 | **0.685** [0.547–0.846] | 0.439 | 0.000 | 0.552 [0.451–0.663] |
| IMG | 24 | 0.321 [0.215–0.488] | 0.239 | 0.000 | 0.854 [0.761–0.949] |
| INSTR | 8 | 0.132 | 0.099 | 0.000 | 0.978 |
| INSTR | 16 | 0.168 | 0.158 | 0.000 | 0.981 |
| INSTR | 24 | 0.108 | 0.075 | 0.000 | 1.003 |
| PROPRIO | 8 | 0.046 | 0.021 | 0.000 | 0.992 |
| ACT | 8 | 0.223 | 0.074 | 0.000 | 0.900 |
| ACT | 16 | 0.454 | 0.210 | 0.000 | 0.727 |
| ACT | 24 | **0.818** [0.749–0.904] | 0.313 | 0.000 | 0.324 |

Patch beats its own equal-norm random direction, per cell (and per task, the conservative unit):

| site | L8 | L16 | L24 |
|---|---|---|---|
| IMG | 78/80 cells, 10/10 tasks (p = 0.001) | 63/80, 9/10 (p = 0.011) | 61/80, 8/10 (p = 0.055) |
| INSTR | 55/80, 8/10 (p = 0.055) | 31/80, 5/10 (p = 0.62) | 53/80, 7/10 (p = 0.17) |
| PROPRIO | 75/80, 10/10 (p = 0.001) | 72/80, 10/10 | 75/80, 10/10 |
| ACT | 80/80, 10/10 (p = 0.001) | 77/80, 10/10 | 78/80, 10/10 |

Task-level p is a one-sided sign test over 10 tasks; the floor is 0.001.

The headline contrast, same 512 positions, same model, same metric:

| | A/B run (same scene, different prompt) | this run (same prompt, different scene) |
|---|---|---|
| IMG D_dst @ L8 | 0.015 | **0.850** |
| IMG D_dst @ L16 | 0.076 | 0.685 |
| IMG D_dst @ L24 | 0.060 | 0.321 |

**80/80 cells** exceed 0.10 at L8 — every single cell moves at least 6× more than the A/B median;
79/80 exceed 0.50. The lowest cell in the whole run is 0.454.

## A second, unlooked-for result: the scene leaves the image positions with depth

IMG efficacy falls 0.850 → 0.685 → 0.321 while ACT efficacy rises 0.223 → 0.454 → 0.818, crossing
between L16 and L24. Scene content is readable-and-usable at the image positions early and has
migrated into the action stream by L24. This is measured within one model, with the same prompt in
both arms, so it carries no prompt-length confound.

## Free bonus: this run has no positional confound

`same_layout` is **True in 2880/2880 rows** — both arms use the identical prompt, so token counts
match by construction. The A/B run's `same_layout` is False in 36000/36000, which is why the OFT
INSTR result must travel with the RoPE-artifact caveat. This run does not.

## What may and may not be claimed

**MAY:** the OFT IMG patching site is causally potent. Replacing the 512 image-position residuals at
L8 with those from a different observation moves the action chunk 0.850 of the way off the
recipient's own output — 57× the A/B run's 0.015 — beating an equal-norm random direction in 78/80
cells and 10/10 tasks. Explanation (2) is dead at L8 and L16: **the patch is not undone downstream,
and the harness is not silently no-oping.**

**MAY:** therefore the A/B run's near-zero IMG effect is a property of the *contrast*, not of the
method — both prompts saw the same image, so there was almost nothing to transfer. The OFT IMG
numbers at L8 and L16 are now safely quotable, which they were not before this control.

**MAY:** scene content migrates from image positions to action positions with depth (above).

**MAY NOT:** "image positions cannot carry instruction information in OFT." This control shows the
site carries **scene** content. It removes the harness/undone-downstream explanation; it does not by
itself prove the site *could* have carried the instruction had the instruction been there. The
surviving reading is the narrower one: in OFT the instruction contrast is not present at the image
positions, and we have now ruled out the measurement being blind.

**MAY NOT:** quote the L24 IMG null with the same confidence as L8/L16. At L24 the site itself is
weak (D_dst 0.321, beating random in only 8/10 tasks, p = 0.055), and the depth result above says why
— the scene has already left. A null at a position that no longer carries the scene is partly
uninformative. Say this in the same sentence as the L24 number.

**MAY NOT:** "behaviour." This measures the next action chunk, like every other patching result here.
Closed-loop first touch is the preregistered primary endpoint and was not run.

**MAY NOT:** read the INSTR row as a finding. With the same prompt in both arms there is nothing for
an instruction patch to transfer; it is a negative control and it behaves like one (at L16 it does not
beat random at all, 31/80 cells). That is the expected result, not evidence about instruction coding.

## Status

The cross-scene control is a control on an existing result, not a test of a new hypothesis. Its
prediction — "D_dst must rise far above the no-op anchor if the site has teeth" — was fixed in the
script docstring before any number was read. Label it a control, and label the two-stage gate and the
graded distance model post-hoc/exploratory wherever they are quoted.

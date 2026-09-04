# PREREGISTRATION — second-architecture replication (OpenVLA-OFT)

**Written 2026-08-31 while the run is in flight and BEFORE any result from it has been inspected.**
I have seen only the first progress line ("task 0 init 0 done; passes=150"). No patching numbers, no
analysis output. Hash this file and record the hash before reading any result.

## Why this exists
Two independent reviews converge on the same route out of borderline: the π0.5 dissociation must be
shown to be an **architectural** property, not a π0.5/LIBERO curiosity. OFT is a genuinely different
design — one fused decoder, no separate action expert, 32 layers instead of 18, different action head.

## The claim being tested
> In vision-language-action policies, language changes actions by being written into visual-token
> positions; text-token activity can stay highly attended and decodable while becoming causally
> dispensable at the layers where the action is actually produced.

## Predictions, fixed now

**P1 — text is attended out of proportion to its use.** Instruction attention per token exceeds image
attention per token, averaged over layers. *Already measured on OFT: 9.93× (π0.5: 11.4×).* Recorded
here only for completeness; it is NOT a new prediction and must not be counted as confirmation.

**P2 — late text-position causality is weak.** R_INSTR at the deepest patched layer < 0.15.
*Already measured: 0.039 at L24.* Same caveat as P1 — prior, not prediction.

**P3 — image positions carry the behaviour, and this is NEW.** The IMG patching site (added today,
never previously run) will show **R_IMG > R_INSTR at the deepest patched layer (L24)**, with
R_IMG ≥ 0.30 there. This is the genuine test: if image positions are as inert as instruction positions
late in the network, the architectural claim fails.

**P4 — the crossover has a direction.** R_IMG will *rise* with depth (L8 → L24) while R_INSTR *falls*,
mirroring the ACT segment's known 0.00 → 0.93 rise. Quantitatively: R_IMG(L24) > R_IMG(L8).

**P5 — controls stay dead.** Random-column patching at matched count stays below 0.20 at every layer,
as it already does for INSTR (0.134) and ACT (0.115).

## What counts as failure, declared now
- **P3 fails** (R_IMG(L24) < R_INSTR(L24), or R_IMG(L24) < 0.30) → the "language is written into visual
  positions" claim does **not** generalise beyond π0.5, and the write-up says so. The π0.5 result then
  stands alone as a single-model finding and the cross-architecture section is reported as a failed
  replication, not quietly dropped.
- **P4 fails** but P3 holds → partial: image positions matter but the depth story is π0.5-specific.
- **P5 fails** → the whole OFT patching harness is suspect and no OFT number may be quoted.

## What this does NOT establish even if everything passes
Closed-loop behavioural redirection on OFT. We have that on π0.5 (0/360 sub-threshold → 79/120 at
λ=1) and we do **not** have it here. Open-loop action-chunk displacement is not a behaviour. If P3–P5
pass, the honest claim is "the causal geometry replicates", not "the robot was redirected".

## Standing rules that apply
- Unit of analysis is the cell, never the episode.
- L2-to-target is the discriminator, not R — an equal-norm random edit reached R = +0.878 in our own
  mediation run while sitting ~1.0 from both targets.
- Report both attention normalisations. Per token instruction wins 11.4×; per segment image wins 4.99×
  and image exceeds instruction at 18/18 layers. Quoting only the first is the single most damaging
  criticism available against this project.

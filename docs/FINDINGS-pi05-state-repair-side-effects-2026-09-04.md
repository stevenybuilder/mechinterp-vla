# Findings: downstream side effects of broad late-state repair

**Date:** 2026-09-04  
**Status:** complete preregistered preservation/specificity screen  
**Model:** `lerobot/pi05_libero_finetuned_v044`, revision `8e174154ef5f6c60a8da12ae99c303d8963138c1`  
**Primary records:** `artifacts/pi05_state_repair_side_effects_2026-09-04_v1/episodes.jsonl`

## Question

Replacing all 512 image-position K/V entries at π0.5 prefix layers 12–17 had repaired `18/20` conflicted Object rollouts. Did that broad transplant reproduce an ordinary correct-prompt trajectory, or did it also introduce collateral behavior?

## Design

The test used two LIBERO Object task/conflict pairs and five new initial states (`30–34`) per task. Pixels, robot state, and flow-noise schedule were matched within each task/state unit. At every replan, the donor saw the receiver's current observation; no donor action or stored trajectory was copied.

The five conditions were clean correct behavior, an identity-like correct-to-correct late transplant, conflicted behavior repaired at layers 12–17, the same repair at early layers 0–5, and a late transplant from a physically valid third-object prompt. The primary outcomes were simulator success and whether a non-target object was contacted first. The screen also recorded non-target grasps, distinct objects touched, steps, path length, and per-replan action hashes.

## Verified results

| Condition | Success | Correct target first | Non-target first | Any non-target grasp | Median steps | Median EEF path |
|---|---:|---:|---:|---:|---:|---:|
| Clean correct | `10/10` | `10/10` | `0/10` | `0/10` | `118.5` | `1.0093` |
| Correct→correct preserve | `10/10` | `10/10` | `0/10` | `0/10` | `118.5` | `1.0093` |
| Late L12–17 repair | `9/10` | `9/10` | `1/10` | `0/10` | `143.0` | `1.0264` |
| Early L0–5 control | `0/10` | `0/10` | `10/10` | `8/10` | `280.0` | `1.2970` |
| Wrong-donor L12–17 control | `0/10` | `5/10` | `5/10` | `7/10` | `280.0` | `1.2874` |

The two imperfections in the live repair occurred in different episodes:

- Object task 1 succeeded in `5/5`, but initial state 34 contacted the wrong object before completing the correct task.
- Object task 2 contacted the correct object first in `5/5`, but initial state 32 timed out and failed; task success was `4/5`.

Relative to its paired clean run, live repair took more steps in `8/10` units and had a longer end-effector path in `8/10`. The median paired increase was `15` steps (ratio `1.1324`) and `0.0261` path units (ratio `1.0277`). These are descriptive small-panel results, not a precise efficiency estimate.

The clean correct-to-correct transplant was exactly inert in all ten units: per-replan action SHA-256 sequences, simulator outcomes, step counts, contact records, and path lengths matched, and every recorded donor-to-host edit ratio was zero. There were no duplicate `(task, initial state, condition)` keys. The independently recomputed counts match the stored summary exactly.

For live repair versus clean behavior, each binary endpoint had at most one discordant pair, so the exact two-sided McNemar value is `p=1`. This does not establish equivalence. The direct observations—a wrong-first-contact episode and a separate failure—are evidence that the broad edit is not guaranteed to reproduce the clean trajectory.

## Interpretation

The broad late site carries donor-specific behavioral information and is substantially more useful than the equally broad early site, but the transplant is not a clean policy switch. It usually restored the intended task in this panel, while occasionally changing contact order or failing and generally taking a somewhat longer route. That is consistent with a broad, entangled control state that contains more than a single object-choice variable.

The result does **not** show broad safety harm: the live repair caused no non-target grasps in these ten episodes, and the sample covers only two tasks. Conversely, `9/10` success does not prove preservation. The wrong-donor arm is a destructive specificity control, not a measurement of incidental side effects.

## Provenance

- Preregistration: `docs/PREREG-pi05-state-repair-side-effects-2026-09-04.md`
- Runner: `scripts/vla/pi05_state_repair_side_effects.py`
- Configuration: `configs/pi05_state_repair_side_effects.json`
- Episodes: `50`; SHA-256 `b85348d10557076306c111589070516df3bb0609149dde5cc2dbfb94d6340af2`
- Stored summary SHA-256: `8dd520b48971b223f80c075695b8ef6fc1b971ba4582c69b6b731006584741c0`
- Manifest SHA-256: `703e59625c16f42ba2a34b11026739eb5a73bb262affac8bb639d535ea909683`
- Runtime log SHA-256: `90aa66bfef8894db1778a23234dc6c1f489a79ba4eae00ba91c2a800b3f0974a`
- Independent audit: `scripts/audit_research_numbers.py` → `artifacts/numbers-audit-derived.json`


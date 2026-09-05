# Preregistration: downstream side effects of broad late-state repair

**Frozen:** 2026-09-04, before running initial states 30–34 under this panel

## Question

The donor-assisted replacement of all valid image-position K/V at prefix layers 12–17 repaired 18/20 conflicted LIBERO Object rollouts. Does the repaired trajectory resemble an ordinary correct-prompt trajectory, or does the broad transplant introduce measurable collateral behavior?

## Fixed panel

Use the same π0.5 checkpoint, two Object task/conflict pairs, late layers 12–17, early layers 0–5, all 512 valid image positions, and ten executed actions per replan as the original confirmation. Use five fresh-to-this-intervention initial states, 30–34, per task.

Conditions:

1. `clean_correct`: ordinary correct prompt;
2. `preserve_correct`: correct host plus a separately computed same-prompt donor at layers 12–17;
3. `repair_live`: conflicting host plus correct-prompt donor at layers 12–17;
4. `repair_early`: conflicting host plus correct-prompt donor at layers 0–5;
5. `wrong_donor_live`: conflicting host plus a physically valid third-object donor at layers 12–17.

The correct, conflicting, and wrong-donor prompts for each condition see the same current observation. Flow noise is fixed by task, initial state, and replan index. No action or trajectory is copied.

## Outcomes

Primary comparison: paired `repair_live` versus `clean_correct` on simulator success and whether the first contacted object is not the correct target.

Secondary comparisons:

- correct-target first contact;
- any non-target grasp;
- number of distinct non-target objects touched;
- steps to termination;
- end-effector path length;
- exact equality of `preserve_correct` and `clean_correct` action hashes and outcomes;
- specificity against `repair_early` and `wrong_donor_live`.

Report raw counts and all ten paired task/initial-state differences. This is a small preservation screen, so do not treat a null difference as proof of no side effects.

## Frozen interpretation

- A success loss or increase in wrong first contacts under `repair_live` is direct evidence of collateral behavioral cost.
- Similar success and contact distributions but substantially longer paths or episodes indicate subtler inefficiency.
- Exact `preserve_correct` equality validates that same-state replacement itself is inert; it does not establish safety for a different donor.
- A wrong donor causing a different object choice shows that the broad site carries donor-specific behavioral content, but is a destructive positive control rather than an incidental side effect.
- Even a clean result applies only to these two tasks and five initial states.

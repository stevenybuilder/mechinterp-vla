# Preregistration: resolving the π0.5 attention writer and reader

**Frozen:** 2026-09-04, before any attention activation or action was evaluated on the new holdout pairs  
**Parent experiment:** `docs/FINDINGS-pi05-attention-pathway-block-rescue-2026-09-04.md`  
**Scope:** MATS model-biology project only

## Why this follow-up exists

The parent experiment established two facts but left two interpretations unresolved.

First, replacing instruction-to-image attention messages with the counterfactual instruction's messages across prefix layers 0–17 moved all tested actions toward that counterfactual. Restoring the B message at block 0 did not rescue B. That does **not** yet prove repeated writing: a single decisive later write, later overwriting, or hybrid-state incompatibility remain possible.

Second, a behavior-blind set of 13 late image-to-action layer/head channels rescued B under reader block. The set was selected to capture 90% of attention-output difference energy. That does **not** show that all 13 are behaviorally needed or that no smaller reader works.

This experiment distinguishes those alternatives. It is not another residual direction, rank, dose, or generic layer-importance search.

## Fixed panels

Development uses the existing 12 directed LIBERO Goal contrasts and initial states 25–27. Those states have appeared in older studies but were not used to select the attention writer or reader.

Holdout uses eight directed LIBERO Object contrasts:

- task 0 ↔ task 2: alphabet soup versus salad dressing;
- task 2 ↔ task 4: salad dressing versus ketchup;
- task 3 ↔ task 4: barbecue sauce versus ketchup;
- task 3 ↔ task 8: barbecue sauce versus chocolate pudding.

These pair relations were not used in attention-pathway selection. In both directions, the receiver scene contains the alternative task object. Object names have matched tokenizer length within a pair. Initial states 25–29 are later than the historical Object localization run's 0–24 range. Before freezing this document, only task text, tokenizer length, and simulator object membership were checked; no model activation or action was evaluated for these pairs.

All comparisons hold pixels, robot state, token positions, and action noise fixed. Primary metrics remain normalized distance to both clean action endpoints over the first ten environment-unit actions.

## Writer experiment

Run B while substituting clean-A instruction-source K/V only inside attention messages to valid image receiver queries. All query vectors, non-source K/V, masks, scales, and untargeted outputs remain from the live B run.

For every boundary `L=0..17`, evaluate two complementary curves:

1. **Block prefix through L:** substitute A messages at layers `0..L`, leaving later layers native B.
2. **Block suffix from L:** leave earlier layers native B and substitute A messages at layers `L..17`.

The full image-route block and clean endpoints anchor the curves. This design distinguishes:

- **localized/decisive writing:** both complementary curves change sharply at the same layer or narrow band;
- **distributed integration:** both change gradually and monotonically across at least six layers;
- **ambiguous/hybrid behavior:** curves are nonmonotonic, disagree on location, or have too little range.

The exact classification thresholds are frozen in `configs/pi05_attention_resolution.json`. Development selects one interpretation and its critical layer or transition window. The corresponding holdout test then adds direct controls:

- substitute A at only the selected layer/band in an otherwise clean B run (necessity);
- restore B only at that layer/band under the full A-message block (sufficiency);
- use an equal-width nonoverlapping layer band as the matched control;
- reproduce the complete complementary curves and compare them with development.

An ambiguous development curve fails and writer holdout remains unopened.

## Reader experiment

Use the frozen reader ranking from the parent experiment. Under the full A-image reader block at action-expert layers 12–17, restore the first `k ∈ {1,2,4,8,13}` ranked B layer/head channels. Compare each with the frozen equal-size control in the config. Every channel still covers the 512 valid image source keys, all 50 action queries, and all ten denoising steps; “small” refers only to layer/head count.

Rescue alone is not evidence that the clean model normally depends on those channels. For every `k`, run the complementary **clean-B removal**: leave the clean-B reader intact except for substituting A image K/V at exactly the selected `k` channels. Run the same removal for the equal-size control set. Thus the identical selected set must pass both tests:

- **necessity:** removing it from clean B makes the action A-like;
- **sufficiency:** restoring it under the full reader block makes the action B-like.

The random controls are nested prefixes of one seed-`20260904` permutation drawn entirely outside the full selected top-13 set. They are therefore disjoint from every selected top-`k` set.

Select the smallest `k≤8` satisfying all development gates:

- rescue median `D_B≤0.25` and at least 75% of runs B-like;
- removal median `D_A≤0.35` and at least 75% of runs A-like;
- both endpoint changes occur in at least 8/12 directed cells;
- selected removal and rescue jointly beat their matched controls in at least 8/12 directed cells.

If no `k≤8` passes, the compact-reader hypothesis fails and reader holdout remains unopened. The `k=13` condition is retained to replicate the parent result, not to qualify as compact.

If a compact `k` passes, freeze it and test only that set and its already frozen matched control on the eight new Object directions.

## What could change the research assessment

An accept-level outcome is not guaranteed. The strongest outcome would be a narrow writer layer/band that is both necessary and sufficient on the new pairs, together with a top-`k≤8` reader that is both necessary under clean-B removal and sufficient under rescue. A cleanly replicated distributed writer curve could still be mechanistically interesting if both complementary curves agree, but it would support a weaker application claim and does not qualify for the joint compact-mechanism gate.

If writer curves are ambiguous, direct block/rescue fails, the reader still needs 13 channels, or effects do not transfer to the new prompt pairs, the project remains borderline. Negative results will not be reframed as accept-level merely because the experiment was rigorous.

## Integrity and stopping

The runner, intervention primitive, hooks, config, pair file, and preregistration are hashed before execution. Development and holdout are separate commands. The holdout commands require a completed frozen selection file with a passing gate for their corresponding component. Results are append-only and confirmation does not silently resume partially completed cells without checking all required conditions.

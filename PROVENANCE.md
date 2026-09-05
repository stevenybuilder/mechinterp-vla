# Provenance audit

**Audit date:** 2026-09-04  
**MATS archive root:** `/Users/stevenyang/Documents/mats-mech-interp`  
**Historical source root:** `/Users/stevenyang/Documents/mechinterp-vla`

## Status

The earlier local-evidence gap is closed for the records needed to substantiate the MATS model-biology narrative. The MATS archive now contains the previously missing 60,000 raw Stage-2 localization rows, early negative-result artifacts, raw Stage-1 records, historical runtime logs, and snapshots of the exact 4090 and OpenVLA-OFT runtime code trees that accompanied those runs.

The 2026-09-04 MATS-only follow-ups are also local: attention writer/reader development, the fixed layers-6–8 writer-band confirmation, the lean midpoint-curvature diagnostic, the curvature action follow-up, the 420-row matched layer-6→8 transformation experiment, its 36-episode closed-loop breadth screen, and the 50-episode broad-repair side-effect screen. The transformation transfers across initial scenes within known prompt pairs and the full attention-plus-MLP update is much stronger than the attention-only version. In the simulator it produced `3/12` B-first contacts but `0/12` task-B successes. The broad late repair separately produced `9/10` successes against clean `10/10`, with one wrong-first contact and one separate failure. `numbers audit.md` and `artifacts/numbers-audit-derived.json` consolidate the checked claims and correct earlier overstatements about repeated writing, reader confirmation, universal scene specificity, action-vector movement as behavioral repair, and task success as clean preservation.

The copy was additive and non-destructive. Existing MATS files were never replaced. The source robotics project was not modified.

## Source state and limits

The source Git repository reported commit:

`179845a12ec76ffa7fbbddfe64a84e32bce2094d`

It also had tracked and untracked changes on 2026-09-04, overwhelmingly in the separate safety/COAST/Sonar robotics work. The commit is therefore a reference point, not a claim that every imported historical file came from a clean checkout. For the early 4090 and OpenVLA-OFT runs, the archived runtime snapshots and logs—not the current source checkout—are the primary execution provenance.

## Imported historical evidence

| Evidence group | Local location | Verification |
|---|---|---|
| LIBERO Object Stage-2 discovery | `artifacts/vla_stage2/20260830-094027/libero_object_discovery/rows.jsonl` | 24,000 valid raw rows |
| LIBERO Goal Stage-2 discovery | `artifacts/vla_stage2/20260830-094027/libero_goal_discovery/rows.jsonl` | 18,000 valid raw rows |
| LIBERO Goal Stage-2 confirmation | `artifacts/vla_stage2/20260830-094027/libero_goal_confirm/rows.jsonl` | 18,000 valid raw rows |
| Early Stage-1 raw runs | `artifacts/vla_stage1/runs/` | two JSONL condition records plus existing summaries/features |
| Early region-direction null | `artifacts/region_dir2/` | result JSON and underlying NumPy record |
| Arbitration behavior runs | `artifacts/vla_arbitration/20260830-192300/` | per-episode rows, results, summaries, manifest |
| OpenVLA-OFT Stage 2 | `artifacts/vla_oft_stage2/20260830-112339/` | attention, patching, clean chunks, result, summary, manifest |
| 4090 runtime snapshot | `provenance/runtime-snapshots/4090/` | 271 files: scripts, top-level diagnostics, configs, tokenizer, stage data, logs |
| OpenVLA-OFT 3090 runtime snapshot | `provenance/runtime-snapshots/oft-3090/` | 42 files: code, logs, environment freeze |
| Historical runtime logs | `logs/vla_4090/`, `logs/vla_oft/`, and merged `logs/` | retained verbatim |

The three restored Stage-2 files contain exactly 60,000 lines in total. They are the raw localization records that were absent from the first curated MATS copy.

## New MATS-only attention-pathway experiment

The 2026-09-04 writer/reader experiment was implemented and executed only under this MATS project. Its primary provenance is:

- preregistration: `docs/PREREG-pi05-attention-pathway-block-rescue-2026-09-04.md` and adjacent SHA-256 file;
- frozen numerical contract: `configs/pi05_attention_pathway.json` and adjacent SHA-256 file;
- implementation: `scripts/vla/attention_pathway.py`, `scripts/vla/pi05_attention_pathway.py`, and the attention-hook additions in `scripts/vla/hooks.py`;
- synthetic tests: `tests/test_attention_pathway.py` (`7/7` passed in the remote runtime);
- raw results: `artifacts/pi05_attention_pathway_2026-09-04_v1/`;
- canonical interpretation: `docs/FINDINGS-pi05-attention-pathway-block-rescue-2026-09-04.md`.
- execution environment and exact commands: `provenance/runtime-snapshots/mats-4090-2026-09-04/`.

The result directory's sealed manifest records exact hashes of the runner, attention primitive, hooks, config, and prompt-pair file. Those hashes still match the local files. All four raw JSONL tables are parseable and have unique primary keys: 576 writer-selection rows, 2,304 reader-selection rows, 480 writer-screen rows, and 240 reader-screen rows.

The writer gate failed and the reader gate passed. The runner rejected the confirmation command before model loading; no confirmation rows or closed-loop pathway results exist.

## New writer-resolution and nonlinear follow-ups

| Evidence group | Local location | Verification |
|---|---|---|
| Writer/reader development | `artifacts/pi05_attention_resolution_2026-09-04_v1/` | 1,440 writer rows and 828 reader rows; both match manifests |
| Fixed layers-6–8 writer band | `artifacts/pi05_writer_band_6_8_confirmation_2026-09-04_v3/` | 280 valid rows; frozen gate failed with 0/8 joint endpoints |
| Midpoint representation curvature | `artifacts/pi05_lean_midpoint_curvature_2026-09-04_v1/` | 60 valid rows; 12/12 direction medians ≥0.1; endpoint identities exact |
| Curvature action follow-up | `artifacts/pi05_curvature_action_2026-09-04_v2/` | 60 valid rows; median 0.09729 and 5/12 directions ≥0.1; frozen gate failed |
| Matched layer-6→8 transformation | `artifacts/pi05_matched_band_transform_2026-09-04_v1/` | 420 valid rows; unique cells; raw action arrays independently reproduce the stored metrics |

The corrected Object prompt-pair configuration is retained, but no valid held-out prompt-pair reader confirmation was run. The reader's positive result is development evidence on new initial states within known Goal prompt pairs.

## Final matched layer-6→8 transformation experiment

The final experiment was preregistered and executed only under this MATS project. Its primary provenance is:

- preregistration: `docs/PREREG-pi05-matched-layer6-8-transform-2026-09-04.md`;
- frozen configuration: `configs/pi05_matched_band_transform.json`;
- implementation and tests: `scripts/vla/pi05_matched_band_transform.py` and `tests/test_pi05_matched_band_transform.py`;
- raw results: `artifacts/pi05_matched_band_transform_2026-09-04_v1/`;
- runtime log: `logs/pi05_matched_band_transform_2026-09-04_v1.log`;
- canonical interpretation: `docs/FINDINGS-pi05-matched-layer6-8-transform-2026-09-04.md`.

The panel contains 12 directed LIBERO Goal prompt-pair cells, five new initial states per cell, and seven conditions: 420 rows total. The audit reconstructs the action-axis metric from the stored first-ten-action arrays with maximum absolute error `1.26e-7`; other-scene message norms match their targets within maximum relative error `2.99e-6`. Other-scene full progress was `0.21078` versus `0.00578` for the equal-norm random control, with the paired difference positive in `12/12` cells (exact sign `p=0.000488`). Matching full progress was `0.33999` versus `0.04022` for the MLP-clamped attention-only transformation, again positive in `12/12` cells (`p=0.000488`). The local and remote artifact hashes matched after transfer.

## Matched-transform closed-loop breadth screen

The original 300-episode confirmation design is frozen in `docs/PREREG-pi05-matched-layer6-8-closed-loop-2026-09-04.md` and `configs/pi05_matched_band_rollout.json`. It was stopped after 12 preserved rows because it was unnecessarily expensive for the first behavioral question. No partial row was deleted or relabeled.

Before examining the remaining prompt pairs, `docs/ADDENDUM-pi05-matched-layer6-8-rollout-screen-2026-09-04.md` froze a 36-episode breadth screen: 12 directed Goal pairs × one initial state × clean A, clean B, and matching full intervention. Its provenance is:

- implementation: `scripts/vla/pi05_matched_band_rollout.py`;
- test: `tests/test_pi05_matched_band_rollout.py`;
- screen config: `configs/pi05_matched_band_rollout_screen.json`;
- complete raw screen: `artifacts/pi05_matched_band_rollout_screen_2026-09-04_v1/`;
- runtime log: `logs/pi05_matched_band_rollout_screen_2026-09-04_v1.log`;
- canonical interpretation: `docs/FINDINGS-pi05-matched-layer6-8-closed-loop-2026-09-04.md`.

The complete table has 36 unique rows. The audit reproduces all stored counts and median episode lengths exactly and verifies nonzero message norms in every edited replan. The edit produced B-first contact in `3/12` pairs and task-B success in `0/12`; clean B produced `9/12` and `12/12`. The 12-row stopped panel and three-row smoke test are retained in their separately named artifact directories and excluded from the complete-screen count.

## Broad-repair side-effect screen

This 50-episode panel was preregistered and executed only under the MATS project. Its primary provenance is:

- preregistration: `docs/PREREG-pi05-state-repair-side-effects-2026-09-04.md`;
- frozen configuration: `configs/pi05_state_repair_side_effects.json`;
- implementation and test: `scripts/vla/pi05_state_repair_side_effects.py` and `tests/test_pi05_state_repair_side_effects.py`;
- raw results: `artifacts/pi05_state_repair_side_effects_2026-09-04_v1/`;
- runtime log: `logs/pi05_state_repair_side_effects_2026-09-04_v1.log`;
- canonical interpretation: `docs/FINDINGS-pi05-state-repair-side-effects-2026-09-04.md`.

The table contains 50 unique rows: two Object tasks × five new initial states × five conditions. Remote and local hashes matched after transfer. The independent audit reproduced all stored counts and medians, verified the absence of duplicate keys, and confirmed exact action hashes and outcomes for all ten clean correct→correct preservation pairs. Live late repair succeeded `9/10`; clean succeeded `10/10`; early and wrong-donor controls succeeded `0/10`.

## Derived audit, synthesis, and figures

`scripts/audit_research_numbers.py` independently parses the primary raw tables, checks row counts and hashes, and recomputes the headline values in `artifacts/numbers-audit-derived.json`. The human-readable audit is `numbers audit.md`; `Research Direction.md` is the consolidated narrative based on that ledger.

`scripts/make_research_figures.py` reads the preserved raw JSONL records, canonical Stage-2 `results.json`, the derived audit JSON, and verified video metadata. It creates ten current claim-first figures under `figures/`, with 19 adjacent current source CSVs and both PNG and PDF output. The tenth figure uses real frames from a canonical state-confirmation recapture whose four outcomes match the archived records. The earlier seven-figure set remains present but is explicitly marked superseded in `figures/README.md`. The current figures were visually inspected after generation. These files summarize existing evidence and do not create new experimental claims.

## Deliberately excluded

The MATS archive does not include:

- approximately 2.3 GB of per-example Stage-2 activation `.npz` intermediates whose aggregate/raw row evidence is already present;
- approximately 8.4 GB of safety experiment artifacts;
- current `scripts/libero_safety/` development;
- COAST, Cosmos, current Sonar safety, JEPA, or MetaWorld experiments;
- `/Users/stevenyang/Documents/mechinterp-vla/cross model plan.md` or the performance-steering benchmark plans.

Those remain part of the separate robotics-performance project and must not be used to imply MATS evidence or chronology. The omissions save space but mean this folder is a claim-complete curated archive, not a byte-for-byte mirror of every historical intermediate.

## Integrity inventory

`PROVENANCE-SHA256SUMS` is the unified archive inventory. The 2026-09-04 release inventory contains 1,011 entries. It hashes every regular file in the curated project except itself, temporary files, macOS metadata, Python bytecode, cache directories, and `tmp/`. It supplements rather than rewrites the older frozen checksum receipts, some of which use historical absolute or source-root paths.

To verify from the archive root:

```bash
sha256sum -c PROVENANCE-SHA256SUMS
```

Any intentional future edit requires regenerating the inventory and recording a new audit date. Frozen preregistration/config checksum files should never be silently changed.

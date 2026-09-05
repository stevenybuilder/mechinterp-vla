# Source manifest

Created on 2026-09-02 from:

`/Users/stevenyang/Documents/mechinterp-vla`

This is a non-destructive, curated copy for MATS application writing. It became the independent home of
subsequent MATS-only experiments on 2026-09-04.

The initial curated copy was provenance-audited and expanded on 2026-09-04. See `PROVENANCE.md` and
`PROVENANCE-SHA256SUMS` for the current exact boundary and integrity inventory.

## Copied evidence groups

### pi0.5 closed-loop instruction repair

- `artifacts/pi05_instruction_repair_2026-08-31/`
- the associated preregistrations, implementation addendum, findings, and three analysis/runner scripts

### pi0.5 prefill mediation

- `artifacts/pi05_prefill_mediation_2026-08-31/`
- the associated preregistration, findings, runner, and analyzer

### pi0.5 source-mediator refinement (2026-09-04)

- `docs/FINDINGS-pi05-sonar-lite-source-mediator-2026-09-04.md`
- `docs/manifests/pi05-sonar-lite-source-mediator-v1.json`
- `artifacts/pi05_sonar_lite_source_mediator_v1/`
- `scripts/vla/sonar_lite_geometry.py`
- `scripts/vla/pi05_sonar_lite_mediator.py`
- `scripts/vla/analyze_pi05_sonar_lite_mediator.py`
- `scripts/vla/audit_pi05_sonar_lite_partial_effect.py`
- `tests/test_pi05_sonar_lite_geometry.py`
- `logs/sonar_lite_fit.log`, `logs/sonar_lite_select.log`, and `logs/sonar_lite_screen.log`

### Frozen donor-free conditional repair

- `docs/PREREG-pi05-donor-free-low-rank-repair-2026-09-02.md`
- `docs/ADDENDUM-pi05-donor-free-early-control-2026-09-02.md`
- `docs/DONOR-FREE-REPAIR-RUNBOOK.md`
- `configs/pi05_donor_free_repair.json`
- `artifacts/pi05_donor_free_repair_2026-09-02/`
- the associated builder, runner, analyzer, tensor utilities, and tests under `scripts/vla/` and `tests/`

### Compact supporting archives

- `artifacts/pi05_mediation_2026-08-31/`
- `artifacts/pi05_monitor/`
- `artifacts/mediation_pairs.json`

### Earlier behavioral and localization evidence

- `artifacts/vla_stage0/`
- `artifacts/vla_stage0_libero_base/`
- `artifacts/vla_stage1/`
- `artifacts/vla_stage2/`
- `artifacts/vla_control_pilot/`

The audit restored the raw files that were missing from the first curated copy:

- 60,000 Stage-2 JSONL rows under `artifacts/vla_stage2/20260830-094027/`;
- raw Stage-1 condition runs under `artifacts/vla_stage1/runs/`;
- `artifacts/region_dir2/` and `artifacts/vla_arbitration/20260830-192300/`;
- historical 4090 scripts, configs, tokenizer material, and logs under `provenance/runtime-snapshots/4090/`.

### Qualified cross-architecture comparison

- `artifacts/vla_oft_stage0/`
- `artifacts/oft_downstream_kv/v2_20260831/`
- the associated OpenVLA-OFT preregistrations, addendum, findings, runner, and analyzer
- the early OpenVLA-OFT Stage-2 raw tables under `artifacts/vla_oft_stage2/20260830-112339/`
- the historical runtime code, logs, and environment freeze under `provenance/runtime-snapshots/oft-3090/`

### MATS-only attention-pathway experiment (2026-09-04)

- `docs/PREREG-pi05-attention-pathway-block-rescue-2026-09-04.md`
- `configs/pi05_attention_pathway.json`
- `scripts/vla/attention_pathway.py`
- `scripts/vla/pi05_attention_pathway.py`
- `tests/test_attention_pathway.py`
- `artifacts/pi05_attention_pathway_2026-09-04_v1/`
- `provenance/runtime-snapshots/mats-4090-2026-09-04/`
- `docs/FINDINGS-pi05-attention-pathway-block-rescue-2026-09-04.md`

The pathway runner enforced the failed writer gate; confirmation states and closed-loop pathway rollouts were not run.

### MATS-only writer-resolution and nonlinear follow-ups (2026-09-04)

- `docs/PREREG-pi05-attention-pathway-resolution-2026-09-04.md`
- `docs/PREREG-pi05-writer-band-6-8-confirmation-2026-09-04.md`
- `docs/PREREG-pi05-lean-nonlinear-writer-diagnostic-2026-09-04.md`
- `docs/PREREG-pi05-curvature-removal-rescue-action-2026-09-04.md`
- the associated frozen configs under `configs/`
- the associated runners under `scripts/vla/` and tests under `tests/`
- `artifacts/pi05_attention_resolution_2026-09-04_v1/`
- `artifacts/pi05_writer_band_6_8_confirmation_2026-09-04_v3/`
- `artifacts/pi05_lean_midpoint_curvature_2026-09-04_v1/`
- `artifacts/pi05_curvature_action_2026-09-04_v2/`
- `logs/pi05_writer_band_6_8_confirmation_v3.log`
- `logs/pi05_lean_midpoint_curvature_v1.log`
- `logs/pi05_curvature_action_v2.log`
- `docs/FINDINGS-pi05-writer-band-and-nonlinear-followups-2026-09-04.md`

The fixed layers-6–8 writer band failed. The midpoint representation-curvature diagnostic passed, but its preregistered action-sensitivity follow-up failed (`0.09729 < 0.10`; `5/12 < 8/12`). No additional search was opened.

### MATS-only matched layer-6→8 transformation (2026-09-04)

- `docs/PREREG-pi05-matched-layer6-8-transform-2026-09-04.md`
- `configs/pi05_matched_band_transform.json`
- `scripts/vla/pi05_matched_band_transform.py`
- `tests/test_pi05_matched_band_transform.py`
- `artifacts/pi05_matched_band_transform_2026-09-04_v1/`
- `logs/pi05_matched_band_transform_2026-09-04_v1.log`
- `docs/FINDINGS-pi05-matched-layer6-8-transform-2026-09-04.md`

This final panel contains 420 rows and compares the full natural attention-plus-MLP update with matching-scene, other-scene, attention-only, and equal-norm random controls. It supports cross-initial-state transfer within known prompt pairs, not held-out instruction semantics or a compact donor-free intervention.

### MATS-only matched-transform rollout screen (2026-09-04)

- `docs/PREREG-pi05-matched-layer6-8-closed-loop-2026-09-04.md`
- `docs/ADDENDUM-pi05-matched-layer6-8-rollout-screen-2026-09-04.md`
- `configs/pi05_matched_band_rollout.json`
- `configs/pi05_matched_band_rollout_screen.json`
- `scripts/vla/pi05_matched_band_rollout.py`
- `tests/test_pi05_matched_band_rollout.py`
- `artifacts/pi05_matched_band_rollout_screen_2026-09-04_v1/`
- `logs/pi05_matched_band_rollout_screen_2026-09-04_v1.log`
- `docs/FINDINGS-pi05-matched-layer6-8-closed-loop-2026-09-04.md`

The complete exploratory screen contains 36 unique episode rows. The edit produced `3/12` B-first contacts but `0/12` task-B successes. The stopped 12-row large panel and three-row smoke test remain separately preserved and must not be pooled into the breadth-screen estimate.

### MATS-only broad-repair side-effect screen (2026-09-04)

- `docs/PREREG-pi05-state-repair-side-effects-2026-09-04.md`
- `configs/pi05_state_repair_side_effects.json`
- `scripts/vla/pi05_state_repair_side_effects.py`
- `tests/test_pi05_state_repair_side_effects.py`
- `artifacts/pi05_state_repair_side_effects_2026-09-04_v1/`
- `logs/pi05_state_repair_side_effects_2026-09-04_v1.log`
- `docs/FINDINGS-pi05-state-repair-side-effects-2026-09-04.md`

This complete preregistered panel contains 50 unique rows: two Object tasks, five new initial states, and five conditions. It compares the successful late repair against paired clean behavior, an exact correct→correct preservation arm, the early site, and a valid wrong-object donor while recording contacts, grasps, steps, and end-effector path length.

### Consolidated audit and research map (2026-09-04)

- `numbers audit.md`
- `Research Direction.md`
- `scripts/audit_research_numbers.py`
- `artifacts/numbers-audit-derived.json`
- `scripts/make_research_figures.py`
- `figures/` (ten current claim-first PNG/PDF figures, 19 current source CSVs, the superseded earlier figure set, and a figure README)
- `media/videos/state-confirmation/` (canonical four-way simulator recapture; MP4, WebM, GIF, poster, and metadata; every recaptured outcome matches the archived episode record)
- `VLA Model Biology - Plain English Research Report.docx` and its reproducible builder `scripts/build_plain_english_report.py`

These are derived synthesis artifacts, not new model evidence. The audit independently parses the preserved raw rows and records material corrections to earlier prose.

### Writing and evaluation references

- the application scope/assessment
- relevant project preregistrations, findings, audits, protocol, and draft
- Neel Nanda's admissions FAQ, application evaluation, and research-interests documents
- the prior MATS applications used for calibration

## Boundary

No `scripts/libero_safety/`, `artifacts/libero_safety*`, COAST, current Sonar safety, JEPA, Cosmos, or MetaWorld
experiment directories were copied. Multi-GB per-example activation intermediates were also excluded where their
raw tabular records and analyzed outputs were preserved. The safety and robotics-performance work—including
`../mechinterp-vla/cross model plan.md`—remains an ongoing, inferentially separate research project and must not be
updated from this workspace.

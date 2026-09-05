# Numbers audit

**Audit date:** 2026-09-04  
**Scope:** the evidence used for the MATS VLA model-biology story in this folder. The separate safety/COAST/robotics-performance archive in `../mechinterp-vla` is not primary evidence here.  
**Machine-readable audit:** `artifacts/numbers-audit-derived.json`  
**Recomputation code:** `scripts/audit_research_numbers.py`

## Bottom line

The core raw evidence is present and internally consistent. The strongest results survive recomputation: π0.5's instruction is used during prefill; after prefill, swapping the tested instruction-token state has almost no immediate action effect; broad late image-position state can transfer the action and closed-loop behavior; and every tested attempt to compress that effect into a small portable intervention failed its frozen causal gate.

The final matched-transformation tests add a qualified result. A full 512-position instruction-induced update across layers 6–8 transferred between initial scenes within the same directed prompt pair: other-scene progress toward B was `0.21078` versus `0.00578` for random, and it beat random in all `12/12` cells (`p=0.000488`). Clamping the image-position MLP outputs reduced matching-scene progress from `0.33999` to `0.04022`; full beat attention-only in `12/12` cells (`p=0.000488`). But the 36-episode simulator screen found only `3/12` B-first contacts and `0/12` task-B successes under the edit, versus `0/12` and `0/12` for clean A and `9/12` and `12/12` for clean B. This is evidence for local causal leverage by a broad multi-block transformation, not policy transfer or a compact, donor-free, semantically held-out mechanism.

The preregistered broad-repair preservation screen adds a second qualification. On ten new paired Object task/state units, the late L12–17 repair succeeded `9/10` versus clean `10/10`; one repaired success contacted the wrong object first, a different repair failed, and repaired episodes took a median `15` extra steps. There were no non-target grasps under live repair, and the correct→correct transplant reproduced all ten action-hash sequences and outcomes exactly. This is evidence that the broad causal site is specific and usually effective, but not a guarantee of an ordinary clean trajectory.

## What was checked

I independently parsed the primary JSON/JSONL records, checked expected row counts, recomputed the headline medians and endpoint tests, and recorded SHA-256 hashes. I also reran the original analyzers where the local environment supported them:

- the prefill analyzer reproduced its canonical JSON exactly after normalizing the input path;
- the state-repair analyzer reproduced its canonical JSON exactly;
- the OpenVLA-OFT analyzer reproduced its canonical `results.json` exactly in a temporary environment with NumPy;
- the Sonar-lite partial-effect analyzer reproduced its canonical output exactly, while an independent raw-table recomputation reproduced the full headline numbers;
- the matched layer-6→8 audit reconstructed every stored action-axis metric from the raw first-ten-action vectors, reproduced all condition/contrast medians and sign-test probabilities exactly, checked norm matching, and found no duplicate keys;
- the rollout-screen audit parsed all 36 episodes, found no duplicate cell/init/condition keys, reproduced every stored count and median episode length exactly, and verified nonzero intervention-message norms only in the edited condition;
- the side-effect audit parsed all 50 episodes, found no duplicate task/init/condition keys, reproduced every stored count and median exactly, and independently verified all ten correct→correct action-hash and outcome identities;
- the full Sonar-lite analyzer was not rerun because the local Python environment lacks PyTorch. This is an environment limitation, not a positive validation claim.

The audit does not pretend to re-create training or GPU inference from scratch. It validates preserved inference records against the analysis code and canonical reports. Sealed preregistrations and configs were not edited.

## Verified headline results

| Question | Verified result | Raw evidence | Honest interpretation |
|---|---:|---|---|
| Does the fine-tuned π0.5 obey conflicting instructions uniformly? | 190/200 episodes in the identified 10-cell region obeyed (0.950), versus 131/800 outside it (0.16375); difference 0.78625 | `artifacts/vla_arbitration/20260830-192300/libero_object/per_episode.jsonl`; `artifacts/two_stage_cells.json` | There is a strong, scene-dependent behavioral regime. The archived task-stratified permutation result is **p < 5e-5**, not an exact p-value. A stricter wrong-object-only subset was much weaker (p = 0.065). |
| Is the instruction readable throughout the prefix? | Residual-stream instruction probe accuracy was 1.000 at layers 0–16 and 0.9987 at layer 17 | `artifacts/vla_stage2/20260830-094027/libero_goal_confirm/rows.jsonl` | The text remains decodable. Decodability does not show that the readable text state is still being used. |
| When does prompt information appear at image positions? | Image-position probe accuracy rose from 0.1133 at layer 0 to 0.9927 at layer 1 and 1.000 at layer 4 | same 18,000-row confirmation table | Prompt-dependent information enters non-text/image-associated state very early. A probe alone does not identify the message or prove causality. |
| What happens if post-prefill instruction state is swapped? | Cell-median action-repair score: `KV[INSTR]@all = 0.01055`; `KO[INSTR] = 0.000073` | same confirmation table | The tested cached text state and instruction key output were almost behaviorally redundant after prefill, despite being readable and attended. |
| What happens if image-position cache state is swapped? | `KV[IMG]@12–17 = 0.83210`; `KV[IMG]@all = 0.96252` | same confirmation table | Broad late image-associated state strongly controls the tested immediate action. This is a state-transplant result, not a compact feature. |
| Is text ignored by attention? | Late instruction attention per token was 0.005710 versus 0.000500 for an image token: 11.41× higher per token. Because there are many image tokens, total image attention was 4.99× total instruction attention | same confirmation table and analyzer | Attention magnitude and causal use disagree. Neither per-token nor total attention alone identifies the mechanism. |
| Is the handoff made during prefill? | Across 150 units, an instruction-input swap exactly produced the source action (`D_dst=1`, `D_src=0`); restoring destination instruction K/V left it source-like (`D_dst=0.98786`), while restoring all non-instruction state returned it destination-like (`D_dst=0.05058`) | `artifacts/pi05_prefill_mediation_2026-08-31/rows.jsonl` | The instruction matters while the prefix is constructed. After that construction, its causal consequence mainly lives outside the tested instruction-token cache. |
| Are image positions specifically involved during prefill? | Restoring all image-position state moved the action near destination (`D_dst=0.10498`), while a matched small random image subset did not (`D_dst=0.99843`). The random-minus-all-image advantage was positive in 150/150 units; median advantage 0.89192 | same 1,350-row prefill table | The broad image-position field is an important carrier. This does not imply each image token is visual in a semantic sense. |
| Does broad late-state repair affect real rollouts? | Conflict 0/20 success and 0/20 correct-first; clean correct 19/20 and 19/20; live late L12–17 repair 18/20 and 18/20; early L0–5 repair 0/20 and 0/20 | `artifacts/pi05_instruction_repair_2026-08-31/state_confirm/episodes.jsonl` | The late state transplant changes complete closed-loop object choice and task success, not merely an action-distance metric. |
| Does the successful broad repair preserve ordinary behavior? | New paired panel: clean `10/10` success, `0/10` non-target-first; late repair `9/10` success, `1/10` non-target-first, `0/10` non-target grasps. Median paired step increase `15` (ratio `1.1324`); longer in `8/10`. Correct→correct preservation was exact in `10/10`; early and wrong-donor controls succeeded `0/10` | `artifacts/pi05_state_repair_side_effects_2026-09-04_v1/episodes.jsonl` | Usually, but not perfectly. One repair contacted the wrong object before succeeding and another failed. The small panel exposes collateral behavior but cannot estimate a rare-event rate. |
| Can a static vector replace that state? | The static-vector pilot mostly failed. The isolated task-2, α=1 condition reached 5/5 correct-first but 0/5 task success; task 1 reached 0/5 correct-first and 1/5 success | state-repair pilot records and findings | A momentary steering effect can harm downstream execution. Correct first motion is not equivalent to repairing the policy. |
| Does the broad carrier survive mediation controls? | Whole-image carrier: median `D_B=0.23487`, B-like in 10/12 directions. Resetting the proposed mediator: `D_B=0.80728`, B-like in 0/12. Dead-site and matched-random carriers were B-like in 0/12; unrelated carrier in 1/12 | `artifacts/pi05_mediation_2026-08-31/run2/rows.jsonl` | The transferred effect depends on the proposed intermediate state. The carrier remains broad and donor-conditioned. |
| Is it a small localized patch? | Local carrier was B-like in 0/8 eligible directions (`D_B=0.99450`). For 4–256 selected image positions, `D_B` remained 0.99745–0.92599; only all 512 positions moved strongly (`D_B=0.25636`) | run2 and `artifacts/pi05_mediation_2026-08-31/dose1/rows.jsonl` | No small spatial/token subset carried the effect under this test. The signal behaves like a field over the image-position state. |
| Is there a compact linear Sonar-style subspace? | Layer 13, rank 16 fit was more selective than wrong-pair controls in 12/12 cells (median fit cosine 0.5272 vs 0.2769), but insertion/removal landed on the target endpoint in 0/12 directions and joint success was 0/12 | `artifacts/pi05_sonar_lite_source_mediator_v1/selection_rows.jsonl`; `screen_rows.jsonl` | The geometry was non-random yet not causally sufficient. This is the cleanest example of association without portable control. |
| Did donor-free low-rank repair work? | On the three completed held-out edges: correct and preserve controls succeeded 30/30; conflict, rank-8 repair, early, random, orthogonal, and wrong-instruction conditions all succeeded 0/30 | `artifacts/pi05_donor_free_repair_2026-09-02/confirm/episodes.jsonl` | The learned operator failed on the completed edges. The run stopped at 250/400 rows, so it is a paused negative result, not a completed five-edge confirmation. |
| Was a compact attention writer isolated? | Full instruction→non-instruction blocking made the action A-like (preference 0.92190), and instruction→image-only blocking was also A-like (0.76052). Restoring the selected block-0 messages failed (`B` preference −0.92206). A later fixed L6–8 band also failed: block preference for A −0.43394, rescue preference for B −0.24914, 0/8 joint endpoints | `artifacts/pi05_attention_pathway_2026-09-04_v1/writer_screen_rows.jsonl`; `artifacts/pi05_writer_band_6_8_confirmation_2026-09-04_v3/rows.jsonl` | Instruction-to-image communication somewhere in the prefix is necessary, but neither a block-0 head set nor the fixed 6–8 route was sufficient to restore it. Repeated direct writing is **not established**. |
| Was a compact reader isolated? | On development initial states, an 8-edge reader set passed: removal median `D_A=0.28560`, A-like fraction 0.833; rescue median `D_B=0.09669`, B-like fraction 1.0; 10/12 joint endpoint directions; beat random in 12/12 | `artifacts/pi05_attention_resolution_2026-09-04_v1/reader_development_rows.jsonl` | This is promising development evidence, not held-out prompt-pair confirmation. The original pair panel was invalid and the corrected pair panel was not run. |
| Are layers 6–8 simply affine along the tested instruction chord? | Midpoint-curvature/chord median 0.18797 (range 0.11070–0.26520); all 12 direction-level medians ≥0.1; endpoint identity exact | `artifacts/pi05_lean_midpoint_curvature_2026-09-04_v1/rows.jsonl` | The tested transformation is non-affine along this synthetic interpolation. This does not prove a dense nonlinear manifold or identify what computation occurs. |
| Does that curvature materially control action? | Median normalized action change 0.09729, just below the frozen ≥0.1 gate; only 5/12 direction medians met ≥0.1 versus the required 8/12; range 0.00844–0.52012; rescue identities exact | `artifacts/pi05_curvature_action_2026-09-04_v2/rows.jsonl` | The follow-up **failed**. The representation diagnostic did not earn a causal-action claim. No post-hoc threshold change is justified. |
| Does the full layer-6→8 instruction transformation transfer across initial scenes? | Matching-scene full progress `0.33999` `[0.13946, 0.47487]`; norm-matched other-scene progress `0.21078` `[0.11241, 0.26699]`; random `0.00578` `[-0.00404, 0.01687]`. Other-scene minus random median `0.19991`, positive in `12/12`, exact sign `p=0.000488` | `artifacts/pi05_matched_band_transform_2026-09-04_v1/rows.jsonl` | A broad instruction-induced update has a reusable component across initial states within known prompt pairs. This is not held-out instruction semantics or a compact representation. |
| Does attention alone produce the useful layer-6→8 update? | Matching attention-only progress `0.04022` versus full `0.33999`; full-minus-attention median `0.22356` `[0.13165, 0.44080]`, positive in `12/12`, exact sign `p=0.000488` | same 420-row table | No. Token-wise MLP processing inside the three blocks materially contributes after attention mixes the state. This does not localize a particular MLP feature. |
| Does the matched layer-6→8 edit change closed-loop behavior? | Clean A: B-first `0/12`, success `0/12`; edit: B-first `3/12`, success `0/12`; clean B: B-first `9/12`, success `12/12`. B-first matched-versus-clean-A exact sign `p=0.25`; success `p=1` | `artifacts/pi05_matched_band_rollout_screen_2026-09-04_v1/episodes.jsonl` | The edit sometimes redirects the initial target but does not transfer the task policy. This was a one-initial-state-per-pair exploratory screen, not confirmation. |
| Does the result transfer to OpenVLA-OFT? | Persistent image K/V patch L8–31 had median `D_src=0.96875`, only 0.03073 better than a single-residual control; same sign in 10/10 tasks (two-sided sign probability 0.001953) but 0/10 beat all controls by 0.2. Equal-token panel: image 0.95743, instruction 0.01005, both 0.00698 | `artifacts/oft_downstream_kv/v2_20260831/rows.jsonl` | It did **not** replicate the π0.5 image-carrier result. In this checkpoint the direct text route dominates; whole-image patching resembles a weak state transplant. |
| Does a simple activation monitor generalize? | F1: development 0.9093; held-out instruction 0.6810; held-out scene 0.7454. An always-positive baseline has F1 0.8006 | `artifacts/pi05_monitor/monitor_cost.json` | A seemingly good in-distribution monitor failed both held-out splits. This supports the scene-bound/generalization warning. |

Distances use the archived convention: `D_A=0` means the intervention matches clean A, and `D_B=0` means it matches clean B. The repair score `R` is useful only together with distances to both endpoints; a damaged or off-manifold state can otherwise be mistaken for a successful move.

## Raw-record integrity

| Evidence | Rows | SHA-256 |
|---|---:|---|
| Stage-2 Object discovery | 24,000 | `6a331b7173d09866efd3a277101fe7324ee9df5dec369d7bc336f7dfcbdd49ff` |
| Stage-2 Goal discovery | 18,000 | `4782590cc07d4f543cc0b09a3af71b513581197cb257e6787a070975f3c2039c` |
| Stage-2 Goal confirmation | 18,000 | `8eab8eebfaa4b0f9b66e2e0c512dbd1f789ee0db5494a4136e514a9824c3aa13` |
| Prefill mediation | 1,350 | `7ee09361d8f15c6aaa0f8c854ed59eec35916ea5ba3f430ad75c6b4eba75c57b` |
| Closed-loop state repair | 80 | `b902988e6193c39a91fc7988e3a1d305807e18e7f331d8b9dc9b13c77684e30e` |
| Mediation/control screen | 696 | `2a907c6f13f420cb0e06a4c6e86467f39888debdd72938017777e849cc94c673` |
| Mediation dose curve | 384 | `4980714dee3a1cdf2c416f46b0a22cae073aecf602234a8119efc9bc974e38f4` |
| Sonar-lite selection | 864 | `2be0f18bcfa5979eb55c23ee7a25698b2aada5a364690ce01cfc777a4e3daa35` |
| Sonar-lite causal screen | 288 | `08d59eaaa4dcbd5e1bd394d97960ad62f09149cbb71388e40687065f201ba5c9` |
| Donor-free repair | 250 | `3bddc2ba336dcefdd14d064157651f458aef15e440b085c87767ba136885f15a` |
| Attention writer screen | 480 | `4364a60f3f525d4f0ccda31a0f6b8fc4dd0ff10b52b49f98b65f820c976b0e92` |
| Attention reader screen | 240 | `44820c9b4d9c0149fdd4fa22e1a789f96d05293912b7ab2588817fc82d459487` |
| Writer development | 1,440 | `1de9ea33df9e1bedbb3f25c65a0cb5c2fd24a56178b64d91fd16dc87ea77f31b` |
| Reader development | 828 | `c9711fe49edf03a9d7fab820b691eb1b85813afa3596c414b3f697381843009b` |
| Fixed L6–8 writer band | 280 | `21da8f53476337a5f68ada3c2dd5988fbaefb20fcd34902f39275afc75b33d58` |
| Midpoint curvature | 60 | `d40d84603ac0c8f5149f29d6525902f7b5efc23ada8a95d2d27aa565379a861c` |
| Curvature action follow-up | 60 | `dd9eb961cc23a306930d0b742a291b8eacaed862b0d467e7e578afdf14d193e3` |
| Matched layer-6→8 transformation | 420 | `c3f38266fe998593b05ad0a6b36ecbaa31cc63f474c232c49872fee673467993` |
| Matched-transform rollout breadth screen | 36 | `8f7022a0b5db6bad967faf570ee76f9e86e40dfa6b2e4af93da04a1e8233f549` |
| Broad-repair side-effect screen | 50 | `b85348d10557076306c111589070516df3bb0609149dde5cc2dbfb94d6340af2` |
| OpenVLA-OFT downstream comparison | 3,440 | `82a3e51036b79e1231d80fd31a7192cfd91272ad91643c7cb5bec8d50ae7de26` |

All listed JSONL files parse, have the expected row count, and passed the duplicate-cell checks implemented by their experiment harnesses or this audit where applicable. The full archive checksum boundary is in `PROVENANCE-SHA256SUMS`.

## Corrections to older notes

| Older shorthand or overclaim | Corrected statement |
|---|---|
| “`KO[INSTR] = −0.006`” as the aggregate result | −0.00597 is one prompt-pair direction. The median over the six confirmed direction-level values is 0.000073, effectively zero. |
| “Direct attention writes the effect repeatedly” | Blocking instruction→image communication across the prefix removes the B effect, but block-0 rescue and fixed L6–8 rescue fail. Later rewriting, nonlinear transformation, overwriting, relays, and hybrid mismatch remain live explanations. |
| “An eight-edge reader is confirmed” | It passes on development initial states across the six known prompt pairs. It has not passed a corrected held-out prompt-pair panel. |
| “The low-rank donor-free test failed on five held-out edges” | Only three edges completed. They failed strongly, but 250/400 planned rows were collected and the last two edges were incomplete/not begun. |
| “Layers 6–8 contain a nonlinear causal mechanism” | They show non-affinity along one tested synthetic chord. The preregistered action-sensitivity gate failed (median 0.0973; 5/12 cells). |
| “Everything is scene-local” | Too strong. A full layer-6→8 message measured in another initial state for the same prompt pair retained substantial causal effect (`0.21078`) and beat random in `12/12` cells. Matching-scene messages were somewhat stronger, but the exact sign test for matched minus mismatched was `p=0.146`. |
| “The matched layer-6→8 action shift changes robot behavior” | In a limited sense: B became the first target in `3/12` edited rollouts versus `0/12` clean-A rollouts. But task-B success remained `0/12`; the result is initial redirection, not task repair or policy transfer. |
| “The `18/20` broad repair reproduces clean behavior” | It establishes strong causal control, not trajectory equivalence. In a new preservation panel, repair was `9/10` versus clean `10/10`, with one wrong-first contact, one separate failure, and longer episodes in `8/10`. |
| “π0.5 has an image pathway; OpenVLA confirms it” | OpenVLA-OFT is a boundary condition, not confirmation: its direct instruction K/V route dominates under the equal-token intervention. |
| “Attention is on language, therefore language drives action” | Language receives 11.4× more attention per token, while post-prefill instruction-state swaps barely affect action. Attention weight is not causal attribution. |
| “A compact direction was found because it beat random controls” | Sonar-lite beat matched controls but never reached the donor endpoint. Relative selectivity is not causal sufficiency. |
| “p = 5e-5” for behavioral concentration | The stored Monte Carlo result licenses `p < 5e-5`. The narrower wrong-object-only audit gives p = 0.065. |
| “Wayve is integrating JEPA into autonomous driving” | This audit found support for Wayve's GAIA latent-diffusion world models, and separate recent JEPA-driving papers, but no primary source tying Wayve to JEPA. The original sentence conflated two trends. |
| “We applied CAFT to a VLA” | We tested a DiD-derived candidate direction (null across 36 cells) and verified the training-hook mechanics. The substantive CAFT fine-tune was cancelled; no CAFT intervention result exists. |
| “We did Othello residual-stream intervention” | We were inspired by Othello-GPT's internal-state/intervention logic. We did not run an Othello model. A separate, excluded safety archive contains an Othello-style VLA probe/steering null and a successful residual-stream instrumentation check, not a causal residual result. |

## Conceptual mistakes and failure modes found

1. **Token labels were treated too literally.** “Image positions” are slots in a multimodal prefix, not guaranteed pure visual concepts. Once self-attention mixes modalities, an image-position state may carry language, scene, state, and motor information together.
2. **Decodability was repeatedly allowed to sound like use.** A perfect linear probe says information is recoverable, not that the downstream policy reads it.
3. **State transplantation was sometimes described as feature transplantation.** Replacing 512 positions across six layers imports a large, donor- and scene-specific computation. It can establish causal leverage without isolating a portable concept.
4. **Beating controls was confused with reaching the causal endpoint.** Several compact interventions were more selective than random or wrong controls yet remained near the original action.
5. **Synthetic geometry was at risk of being overinterpreted.** Midpoint curvature on an A–B chord is evidence against an affine map on that path, not proof of a globally dense nonlinear manifold, dimension expansion, or manifold flattening. CKA would not by itself distinguish those stories either.
6. **Development generalization was overstated.** New initial states within known prompt pairs are not new prompt-pair semantics. The reader result still needs genuine prompt-pair holdout.
7. **Immediate action and task success were sometimes conflated.** The static-vector pilot demonstrates why both are required: it could improve first touch while leaving success at zero.
8. **Cross-model comparison was framed too strongly.** The OFT result is evidence of architecture/checkpoint dependence, not a replication.
9. **“Scene transfer” was allowed to sound like semantic transfer.** The matched-transform donor changes the initial scene but preserves the directed instruction pair. It does not establish new-task or new-language generalization.
10. **A predicted-action shift was allowed to sound like behavioral control.** The simulator screen showed why both levels matter: the edit produced three B-first contacts but zero completed B tasks.
11. **Task success was allowed to sound like clean preservation.** A rollout can eventually succeed after an unintended contact or by taking a less direct path. Contact order, grasps, path length, and paired clean behavior must be reported separately.

## What this audit may still have missed

- It does not validate every historical brainstorming sentence, literature claim, or rejected branch in every Markdown file. It validates the numerical claims needed for the consolidated research direction and flags the material contradictions found during that process.
- It does not regenerate raw model outputs. Re-running all π0.5 and OpenVLA inference would require the original GPU environments and substantial compute.
- Some historical receipts use absolute source-machine paths. The curated runtime snapshots and raw hashes are the authoritative local provenance.
- Large per-example activation `.npz` intermediates were intentionally excluded; their tabular outputs, analyzers, manifests, and runtime snapshots are retained.
- The donor-free experiment is incomplete, and the reader has no valid held-out prompt-pair confirmation. Neither should be described as a completed confirmation.
- The matched-transform rollout screen has only one initial state per prompt pair and omits attention-only and random rollout controls. Its `3/12` first-contact result is exploratory; its `0/12` success result is a clear screen failure but not a precise zero-effect estimate.
- The broad-repair preservation screen covers only two Object tasks and ten paired task/state units. It can demonstrate concrete deviations from clean behavior but cannot bound rare side effects or establish unrelated-task preservation.
- Safety/Othello, COAST, Cosmos, and robotics-performance steering are not part of this archive's primary claim set. They can be discussed as historical context only when clearly labeled as source-only evidence.

## Reproduce the audit

From this folder:

```bash
python3 scripts/audit_research_numbers.py
sha256sum artifacts/numbers-audit-derived.json
```

The current derived audit SHA-256 is:

`8eec292ee59d299d60329b7f6ead4482bef3d757328b02e3c51f0556a357e170`

The derived JSON is deliberately not a frozen preregistration. If the script or evidence boundary changes, rerun it, review the diff, and regenerate the archive checksum inventory.

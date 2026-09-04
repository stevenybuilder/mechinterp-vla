# π0.5 writer-band and nonlinear follow-ups

**Date:** 2026-09-04  
**Model:** `lerobot/pi05_libero_finetuned_v044`, revision `8e174154ef5f6c60a8da12ae99c303d8963138c1`, float32  
**Decision:** the fixed writer-band test failed; the representation-curvature diagnostic passed; the preregistered action-sensitivity follow-up failed

## Question

The attention-pathway screen showed that instruction→image communication across the prefix is necessary, while an all-head block-0 rescue is insufficient. Follow-ups asked whether the important update could be localized to a small later layer band and, if a linear band transplant failed, whether non-affine transformation across that band was causally important to action.

## Layer-transition development

Prefix and suffix cumulative replacement curves changed most sharply around layers 6–7:

- prefix critical layer 7, maximum-step fraction `0.43496`, monotonic ρ `0.99587`;
- suffix critical layer 6, maximum-step fraction `0.45081`, monotonic ρ `0.99794`;
- both transition bands were `[6, 7]`.

The frozen classifier labeled the profile **ambiguous**, because neither maximum-step fraction reached the `0.50` localized threshold and both exceeded the `0.40` distributed ceiling. The profile localized an area for a direct test; it did not identify a mechanism.

## Fixed layers-6–8 block-and-rescue

The selected band was layers 6–8 and the equal-width control band was layers 14–16. The run contained 280 rows.

| Frozen metric | Result | Gate |
|---|---:|---:|
| selected block median preference for A | `−0.43394` | `≥ +0.50` |
| selected rescue median preference for B | `−0.24914` | `≥ +0.50` |
| joint endpoint directions | `0/8` | `≥ 6/8` |
| selected beats control jointly | `8/8` | `≥ 6/8` |

The band was more specific than the control but did not perform either required causal operation. **Status: fail.** This is another example of beating a control without reaching the target endpoint.

## Lean midpoint-curvature diagnostic

For the same 12 directed Goal contrasts on five initial states each, the test compared the actual layer-8 midpoint output to the midpoint predicted by an affine layer-6→8 map. All endpoint identity rescues were bitwise exact.

- 60 rows;
- median curvature/chord `0.18797`;
- range `0.11070–0.26520`;
- `12/12` direction-level medians at least `0.10`.

This establishes non-affinity along the tested synthetic A/B chords. It does not establish a globally dense nonlinear manifold, dimension expansion, manifold flattening, or behavioral relevance.

## Curvature removal/rescue action follow-up

The action test used the same 12 directions on five later initial states each. All rescue identities were bitwise exact.

| Frozen metric | Result | Gate |
|---|---:|---:|
| median action change / clean endpoint distance | `0.09729` | `≥ 0.10` |
| direction medians at least `0.10` | `5/12` | `≥ 8/12` |
| range over rows | `0.00844–0.52012` | descriptive |

**Status: fail.** The miss is numerically close on the pooled median but clear on the replication criterion. The threshold must not be loosened after seeing the result. No additional search was authorized.

## What may be claimed

- Cumulative replacement curves identify a sharp layers-6–7 transition area.
- A fixed layers-6–8 attention-route intervention does not block and rescue the instruction effect.
- The layer-6→8 map is non-affine along the tested A/B interpolation chords.
- The measured curvature is not a confirmed action bottleneck under the preregistered test.

## What may not be claimed

- a decisive writer at layers 6–8;
- a dense nonlinear manifold;
- that CKA or curvature identifies dimension expansion or manifold flattening;
- that the non-affine component causally controls action;
- that the near-threshold pooled median is a positive result.

## Artifacts

- Preregistrations: `docs/PREREG-pi05-attention-pathway-resolution-2026-09-04.md`, `docs/PREREG-pi05-writer-band-6-8-confirmation-2026-09-04.md`, `docs/PREREG-pi05-lean-nonlinear-writer-diagnostic-2026-09-04.md`, `docs/PREREG-pi05-curvature-removal-rescue-action-2026-09-04.md`
- Development: `artifacts/pi05_attention_resolution_2026-09-04_v1/`
- Writer band: `artifacts/pi05_writer_band_6_8_confirmation_2026-09-04_v3/`
- Representation curvature: `artifacts/pi05_lean_midpoint_curvature_2026-09-04_v1/`
- Action follow-up: `artifacts/pi05_curvature_action_2026-09-04_v2/`
- Logs: `logs/pi05_writer_band_6_8_confirmation_v3.log`, `logs/pi05_lean_midpoint_curvature_v1.log`, `logs/pi05_curvature_action_v2.log`
- Implementations: `scripts/vla/pi05_attention_resolution.py`, `scripts/vla/pi05_writer_band_confirmation.py`, `scripts/vla/pi05_lean_nonlinear_writer.py`, `scripts/vla/pi05_curvature_action.py`
- Tests: `tests/test_pi05_attention_resolution.py`, `tests/test_pi05_lean_nonlinear_writer.py`

The checked numerical source is `artifacts/numbers-audit-derived.json`; see `numbers audit.md` for hashes and cross-experiment corrections.

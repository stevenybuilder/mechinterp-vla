# Pause receipt: π0.5 donor-free rank-8 repair

**Paused:** 2026-09-04 at the user's request  
**Status:** 250/400 confirmation rows complete; runner stopped; GPU idle  
**Project:** `/Users/stevenyang/Documents/mats-mech-interp`  
**Remote runtime:** `/root/mats-mech-interp`

## Frozen identities

- Configuration SHA-256: `bcc55b5678f5ebd3dd1546f3b60201f723372688e080489e9e13e2c218c2b251`
- Model SHA-256: `a074341b644810a4e79222e1cde61962e911f8df4534b5217e18c662108be483`
- Partial `episodes.jsonl` SHA-256 at pause: `3bddc2ba336dcefdd14d064157651f458aef15e440b085c87767ba136885f15a`
- Partial runtime log SHA-256 at pause: `48e1812d1cd1a7bcce020a06a5325f9330117822e435b0d4f723d9019de6aaaa`

## Completed evidence

- `250` valid rows and `250` unique `(pair, init, condition)` cells.
- Complete edges: `1←5`, `2←4`, and `3←4` (`80` rows each).
- Partial edge: `5←6` (`10` rows); next missing cell is init 41, `repair`.
- Unopened edge: `8←9` (`80` rows).
- Remaining: `150` rows.
- Complete-edge outcome: correct `30/30`, conflict `0/30`, repair `0/30`, each matched control `0/30`, and preserve-correct `30/30`.
- Available-row outcome: correct `32/32`, conflict `0/32`, repair `0/31`, and preserve-correct `31/31`.
- Zero correct-prompt leakage flags.
- Zero duplicate rows.
- Maximum matched-control relative norm error: `2.171715126137015e-7` against the frozen `1e-5` maximum.
- Correct and `preserve_correct` action hashes and outcomes match in all `31/31` completed comparisons.

The preregistered overall pass is already impossible because the first three held-out edges each achieved
repair `0/10` against the required `7/10`, while their correct-prompt ceilings achieved `10/10`. The partial
run is therefore already a decisive failure of the strong claim, although it is not a completed five-edge
confirmation.

## Local artifacts

- `artifacts/pi05_donor_free_repair_2026-09-02/sentinel/`
- `artifacts/pi05_donor_free_repair_2026-09-02/confirm/episodes.jsonl`
- `logs/donor_free_sentinel.log`
- `logs/donor_free_confirm.log`

No final `confirm/manifest.json` or `analysis.json` exists because the runner writes the manifest only after all
400 rows finish.

## Resume contract

The runner is append-only and skips the 250 completed cells after verifying their model hash. If the user later
chooses to complete failure breadth, resume the exact frozen command from `/root/mats-mech-interp`:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=/root/mats-mech-interp/scripts/vla:/root/vla/src \
/opt/conda/bin/python scripts/vla/run_donor_free_repair.py \
  --config configs/pi05_donor_free_repair.json \
  --model artifacts/pi05_donor_free_repair_2026-09-02/model.pt \
  --out artifacts/pi05_donor_free_repair_2026-09-02/confirm \
  --phase confirm
```

Do not refit, retune rank/dose/layers, edit the existing rows, or present the paused run as a completed
confirmation.

# Donor-free repair runbook

The protocol is frozen in
`docs/PREREG-pi05-donor-free-low-rank-repair-2026-09-02.md`; the machine-readable design is
`configs/pi05_donor_free_repair.json`. Read
`docs/ADDENDUM-pi05-donor-free-early-control-2026-09-02.md` alongside it: before evaluation, the early
control's impossible per-tensor match was replaced by a global 0–5-band Frobenius-norm match.

Run from a checkout with the π0.5/LIBERO environment used by the existing VLA scripts:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python scripts/vla/build_donor_free_repair.py \
  --config configs/pi05_donor_free_repair.json \
  --out-model artifacts/pi05_donor_free_repair_2026-09-02/model.pt \
  --out-dir artifacts/pi05_donor_free_repair_2026-09-02/calibration

python scripts/vla/run_donor_free_repair.py \
  --config configs/pi05_donor_free_repair.json \
  --model artifacts/pi05_donor_free_repair_2026-09-02/model.pt \
  --out artifacts/pi05_donor_free_repair_2026-09-02/sentinel \
  --phase calibration_sentinel

python scripts/vla/run_donor_free_repair.py \
  --config configs/pi05_donor_free_repair.json \
  --model artifacts/pi05_donor_free_repair_2026-09-02/model.pt \
  --out artifacts/pi05_donor_free_repair_2026-09-02/confirm \
  --phase confirm

python scripts/vla/analyze_donor_free_repair.py \
  --episodes artifacts/pi05_donor_free_repair_2026-09-02/confirm/episodes.jsonl \
  --manifest artifacts/pi05_donor_free_repair_2026-09-02/confirm/manifest.json \
  --out artifacts/pi05_donor_free_repair_2026-09-02/analysis.json
```

The calibration builder refuses to overwrite an existing model or calibration log. The confirmation runner
is resumable by episode cell and rejects rows produced with another model hash. The analysis is deliberately
pair-level: a pooled success rate cannot compensate for a failed held-out pair.

The offline flags force use of the tokenizer and processor artifacts already cached by the verified π0.5
runs. Omit them only in an authenticated environment where the pinned Hugging Face artifacts can be fetched
without changing revisions.

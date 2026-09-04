#!/usr/bin/env bash
# Stage 1: scene twins for libero_object (built by build_stage1_twins.py). For each pair:
#   ambiguous (shipped scene): prompts A, B, except_A, notA_B (negation set)   -> 20 init x 4
#   swap (A<->B positions):    prompts A, B                                  -> 20 init x 2
#   determinate (never-target distractors): prompts A, B                     -> 20 init x 2
# Usage: bash scripts/vla/run_stage1.sh /root/vla/stage1 /root/vla/runs/stage1 [pairs...]
set -euo pipefail
S1=${1:-/root/vla/stage1}; OUT=${2:-/root/vla/runs/stage1}; shift 2 || true
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl
mkdir -p "$OUT"
PAIRS=${*:-$(ls -d $S1/t*/ | xargs -n1 basename)}
REV=8e174154ef5f6c60a8da12ae99c303d8963138c1
for pair in $PAIRS; do
  tid=$(echo "$pair" | sed -E 's/^t([0-9]+)_.*/\1/')
  P=$S1/$pair
  python scripts/vla/run_libero_prompt_conditions.py --suite libero_object --task_ids "$tid" --init_ids 0-19 \
    --conditions custom:A,custom:B,custom:except_A,custom:notA_B --custom_prompts $P/prompts.json \
    --bddl_file $P/ambiguous.bddl --init_file $P/ambiguous_init.pt --revision $REV --tag "stage1:$pair:ambiguous" \
    --out $OUT/${pair}__ambiguous.jsonl
  for twin in swap determinate; do
    python scripts/vla/run_libero_prompt_conditions.py --suite libero_object --task_ids "$tid" --init_ids 0-19 \
      --conditions custom:A,custom:B --custom_prompts $P/prompts.json \
      --bddl_file $P/$twin.bddl --init_file $P/${twin}_init.pt --revision $REV --tag "stage1:$pair:$twin" \
      --out $OUT/${pair}__${twin}.jsonl
  done
done
echo "[stage1] DONE"

#!/usr/bin/env bash
# Cross-distractor arbitration grid: every (task x valid wrong-instruction) cell x init states.
# Run in INIT-STATE PASSES so that a stop at any point leaves ALL 10 tasks covered at equal n per
# cell (a complete task x distractor matrix), instead of a prefix of tasks at full n.
# The box is shared with two other agents' jobs, so episodes cost ~5-8x their Stage-0 wall time;
# the pass sizes are chosen so pass 1 (n=3/cell) completes inside the GPU-hour budget.
set -u
cd /root/vla
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
REV=8e174154ef5f6c60a8da12ae99c303d8963138c1
mkdir -p runs/arbitration logs

run_pass () {  # suite cbt out log inits
  echo "[chain] $1 inits=$5 start $(date -u +%FT%TZ)"
  /opt/conda/bin/python scripts/vla/run_libero_prompt_conditions.py \
    --suite "$1" --task_ids all --init_ids "$5" \
    --conditions_by_task "$2" --out "$3" \
    --revision $REV --tag arbitration >> "$4" 2>&1
  echo "[chain] $1 inits=$5 exit=$? $(date -u +%FT%TZ)"
}

for INITS in 0-2 3-4 5-9 10-19; do
  run_pass libero_object configs/cbt_libero_object.json \
    runs/arbitration/libero_object.jsonl logs/vla_arbitration_object.log "$INITS"
done
echo "[chain] OBJECT DONE $(date -u +%FT%TZ)"

for INITS in 0-2 3-4 5-9 10-19; do
  run_pass libero_goal configs/cbt_libero_goal_reduced.json \
    runs/arbitration/libero_goal.jsonl logs/vla_arbitration_goal.log "$INITS"
done
echo "[chain] DONE $(date -u +%FT%TZ)"

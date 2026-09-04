#!/usr/bin/env bash
# Stage 2 CONFIRMATION split: libero_goal, init states 25-49 (disjoint from discovery 0-24), same pairs/conditions.
set -uo pipefail
cd /root/vla
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/vla/stage2_discovery.py --suite libero_goal --pairs "[[2,5,3],[0,6,9],[9,6,0]]" --init_ids 25-49 --tps 0,10 --features --dtype ${DTYPE:-bfloat16} --out runs/stage2_confirm/libero_goal > logs/vla_stage2_confirm_goal.log 2>&1
echo "[stage2 confirm] DONE" >> logs/vla_stage2_confirm_goal.log

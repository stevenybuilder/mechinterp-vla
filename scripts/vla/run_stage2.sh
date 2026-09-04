#!/usr/bin/env bash
# Stage 2 discovery chain: libero_goal then libero_object (25 init states x tps {0,10}, features on).
set -uo pipefail
cd /root/vla
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DTYPE=${DTYPE:-float32}
python scripts/vla/stage2_discovery.py --suite libero_goal --pairs "[[2,5,3],[0,6,9],[9,6,0]]" --init_ids 0-24 --tps 0,10 --features --dtype $DTYPE --out runs/stage2/libero_goal > logs/vla_stage2_goal.log 2>&1
python scripts/vla/stage2_discovery.py --suite libero_object --pairs "[[0,1,3],[9,2,1],[1,5,3],[8,9,5]]" --init_ids 0-24 --tps 0,10 --features --dtype $DTYPE --out runs/stage2/libero_object > logs/vla_stage2_object.log 2>&1
echo "[stage2 chain] DONE" >> logs/vla_stage2_object.log

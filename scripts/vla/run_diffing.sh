#!/usr/bin/env bash
# Model diffing (protocol §9.2): lerobot/pi05_base vs pi05_libero_finetuned_v044, same harness / observations / noise.
# (a) Stage 0 conditions on the base checkpoint, libero_object, 10 init states x {correct,null,wrong_object}
#     (v044 pre/post-processors = LIBERO MEAN_STD stats, because pi05_base ships no normaliser stats).
# (b) Stage 2 discovery core + features on the base checkpoint, goal pairs, 25 init states, tp 0 (bands skipped).
set -uo pipefail
cd /root/vla
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=lerobot/pi05_base; BREV=b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba; PROC=lerobot/pi05_libero_finetuned_v044
mkdir -p runs/diffing
# (a) is being run by another agent on this box (run_libero_prompt_conditions_stats.py); skipped here. To run it:
#   python scripts/vla/run_libero_prompt_conditions.py --suite libero_object --task_ids all --init_ids 0-9 \
#     --conditions correct,null,wrong_object --policy $BASE --revision $BREV --processor_path $PROC --tag diffing_base \
#     --out runs/diffing/base_libero_object.jsonl
python scripts/vla/stage2_discovery.py --suite libero_goal --pairs "[[2,5,3],[0,6,9],[9,6,0]]" --init_ids 0-24 --tps 0 \
  --features --skip_bands --dtype ${DTYPE:-bfloat16} --policy $BASE --revision $BREV --processor_path $PROC \
  --out runs/diffing/base_stage2_libero_goal > logs/vla_diffing_stage2.log 2>&1
echo "[diffing] DONE" >> logs/vla_diffing_stage2.log

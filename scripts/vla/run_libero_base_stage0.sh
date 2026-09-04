#!/bin/bash
# Checkpoint-diffing Stage 0: lerobot/pi05_libero_base (rev a217bfd3) — identical protocol to v044 Stage 0, init 0-9.
# The checkpoint ships NO normalizer stats (features={} -> no-op), so we inject openpi's native pi05_libero norm_stats
# (QUANTILES, q01/q99) via run_libero_prompt_conditions_stats.py; bf16 to fit beside the other agent's jobs.
# Robust to GPU contention: waits for >=9.5GB free, retries a stage on OOM (up to 15x, 3-min backoff); runner resumes.
cd /root/vla
PY=/opt/conda/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
REV=a217bfd3b14673cf2ce597e69997ab21866438dd
BASE_REV=b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba
STATS=assets/openpi_pi05_libero_norm_stats.json
COMMON="--stats_json $STATS --norm_mode QUANTILES --dtype bfloat16"
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT=artifacts/vla_stage0_libero_base/$TS
mkdir -p $OUT runs/stage0_libero_base
echo "ts=$TS out=$OUT"
wait_gpu() { for i in $(seq 1 120); do F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); [ "$F" -ge 9500 ] && { echo "$(date -u +%T) free=${F}MiB go"; return 0; }; sleep 30; done; echo "TIMEOUT waiting for GPU"; return 2; }
run_retry() {
  LOG=$1; shift
  for a in $(seq 1 15); do
    wait_gpu || return 2
    "$@" > $LOG 2>&1 && { echo "ok attempt=$a $(date -u +%T)"; return 0; }
    if grep -q "OutOfMemoryError" $LOG; then echo "OOM attempt=$a $(date -u +%T), backoff"; cp $LOG $LOG.oom$a; sleep 180; else echo "FAIL non-OOM attempt=$a"; return 1; fi
  done; return 1
}
echo "=== START $(date -u)"
for SUITE in libero_object libero_goal; do
  run_retry logs/vla_stage0_libero_base_$SUITE.log \
    $PY scripts/vla/run_libero_prompt_conditions_stats.py --suite $SUITE --task_ids all --init_ids 0-9 \
    --conditions correct,null,wrong_object --policy lerobot/pi05_libero_base --revision $REV $COMMON \
    --tag stage0_libero_base --out runs/stage0_libero_base/$SUITE.jsonl
  echo "=== $SUITE exit=$? $(date -u)"
done
$PY scripts/vla/analyze_stage0.py --inputs runs/stage0_libero_base/libero_object.jsonl runs/stage0_libero_base/libero_goal.jsonl --out $OUT > logs/vla_stage0_libero_base_analyze.log 2>&1
echo "=== analyze exit=$?"
cp runs/stage0_libero_base/*.jsonl $OUT/; cp $STATS $OUT/; cp run_libero_base_stage0.sh $OUT/
# pi05_base (pretrained, not LIBERO-tuned): 3 episodes, libero_object t0, correct prompt; same LIBERO stats injected
run_retry logs/vla_pi05_base_probe.log \
  $PY scripts/vla/run_libero_prompt_conditions_stats.py --suite libero_object --task_ids 0 --init_ids 0-2 \
  --conditions correct --policy lerobot/pi05_base --revision $BASE_REV $COMMON --tag pi05_base_probe \
  --out $OUT/pi05_base_libero_object_t0_correct.jsonl
echo "=== pi05_base exit=$? $(date -u)"
echo "=== DONE"

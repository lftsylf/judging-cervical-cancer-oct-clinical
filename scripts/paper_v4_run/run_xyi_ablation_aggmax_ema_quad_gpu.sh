#!/usr/bin/env bash
# 四卡并行：消融① mean→max（保留 EMA），5 seeds 分到 4 GPU
#   GPU0: 42, 114514 | GPU1: 123 | GPU2: 2024 | GPU3: 3407
#
#   ./run_experiment.sh --detach ./tsy_loho ./scripts/paper_v4_run/run_xyi_ablation_aggmax_ema_quad_gpu.sh
#   tail -f logs/detached_latest.log
#   tail -f outputs/paper_v4/baseline2和消融/logs_aggmax_ema/gpu*.log
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="/home/amax/anaconda3/bin:${PATH:-}"
export TMPDIR="${TMPDIR:-$ROOT/.tmp_run}"
mkdir -p "$TMPDIR"

BASE2="$ROOT/outputs/paper_v4/baseline2和消融"
OUT_ROOT="$BASE2/1[消融mean->max] xyi_sitebag_n2_uw_mil_aggmax_ema099"
LOGDIR="$BASE2/logs_aggmax_ema"
mkdir -p "$LOGDIR" "$OUT_ROOT"
STATUS="$LOGDIR/quad_status.md"
ONE="$ROOT/scripts/paper_v4_run/run_xyi_ablation_aggmax_ema_one.sh"

{
  echo "# aggmax+EMA quad $(date '+%F %T')"
  echo "out=$OUT_ROOT"
} >"$STATUS"

echo "======================================"
echo "① mean→max + EMA0.99  四卡"
echo "OUT=$OUT_ROOT"
echo "======================================"

if [[ ! -f "$ROOT/dataset/train_xyi.csv" ]]; then
  /home/amax/anaconda3/bin/python3 data/prepare_xyi_sitebag_splits.py --write --split-seed 20260731 --val-ratio 0.2
fi

PIDS=()
NAMES=()

launch() {
  local gpu="$1"
  local seeds="$2"
  local tag="$3"
  local log="$LOGDIR/gpu${gpu}_${tag}.log"
  echo "▶ GPU$gpu seeds=[$seeds] → $log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export SEEDS="$seeds"
    export SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
    export OPTIGENESIS_OUTPUT_DIR="$OUT_ROOT"
    bash "$ONE" >"$log" 2>&1
  ) &
  PIDS+=($!)
  NAMES+=("GPU${gpu}_${tag}")
  echo $! >"$LOGDIR/gpu${gpu}_${tag}.pid"
}

# 5 seeds → 4 GPUs（GPU0 多扛一个）
launch 0 "42 114514" "s42_114514"
launch 1 "123" "s123"
launch 2 "2024" "s2024"
launch 3 "3407" "s3407"

fail=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "✅ ${NAMES[$i]} done" | tee -a "$STATUS"
  else
    echo "❌ ${NAMES[$i]} failed" | tee -a "$STATUS"
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "结束：有失败，见 $LOGDIR" | tee -a "$STATUS"
  exit 1
fi
echo "结束：① 五 seed 完成" | tee -a "$STATUS"
exit 0

#!/usr/bin/env bash
# 四卡并行对比实验 → outputs/paper_v4/对比实验/
#   GPU0 ABMIL | GPU1 UBIX | GPU2 DSMIL | GPU3 Ours+ConvNeXt-Tiny
#
# 断线续跑（推荐）：
#   ./run_experiment.sh --detach ./tsy_loho ./scripts/paper_v4_run/run_cmp_quad_gpu.sh
#   tail -f logs/detached_latest.log
#   tail -f outputs/paper_v4/对比实验/logs/gpu0_abmil.log
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="/home/amax/anaconda3/bin:${PATH:-}"
export TMPDIR="${TMPDIR:-$ROOT/.tmp_run}"
mkdir -p "$TMPDIR"

CMP_ROOT="$ROOT/outputs/paper_v4/对比实验"
LOGDIR="$CMP_ROOT/logs"
mkdir -p "$LOGDIR"
STATUS="$LOGDIR/quad_status.md"
SEEDS="${SEEDS:-42 123 2024 3407 114514}"

{
  echo "# cmp quad $(date '+%F %T')"
  echo "seeds=$SEEDS"
  echo "out=$CMP_ROOT"
} >"$STATUS"

echo "======================================"
echo "CMP QUAD → $CMP_ROOT"
echo "GPU0 ABMIL | GPU1 UBIX | GPU2 DSMIL | GPU3 ConvNeXt"
echo "======================================"

if [[ ! -f "$ROOT/dataset/train_xyi.csv" ]]; then
  /home/amax/anaconda3/bin/python3 data/prepare_xyi_sitebag_splits.py --write --split-seed 20260731 --val-ratio 0.2
fi

PIDS=()
NAMES=()

launch() {
  local gpu="$1"
  local name="$2"
  local script="$3"
  local log="$LOGDIR/gpu${gpu}_${name}.log"
  echo "▶ launch GPU$gpu $name | log=$log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export SEEDS="$SEEDS"
    export SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
    bash "$script" >"$log" 2>&1
  ) &
  PIDS+=($!)
  NAMES+=("GPU${gpu}_${name}")
  echo $! >"$LOGDIR/gpu${gpu}_${name}.pid"
}

launch 0 abmil    "$ROOT/scripts/paper_v4_run/run_cmp_abmil_one.sh"
launch 1 ubix     "$ROOT/scripts/paper_v4_run/run_cmp_ubix_one.sh"
launch 2 dsmil    "$ROOT/scripts/paper_v4_run/run_cmp_dsmil_one.sh"
launch 3 convnext "$ROOT/scripts/paper_v4_run/run_cmp_ours_convnext_one.sh"

fail=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  name="${NAMES[$i]}"
  if wait "$pid"; then
    echo "✅ $name done" | tee -a "$STATUS"
  else
    echo "❌ $name failed (pid=$pid)" | tee -a "$STATUS"
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "结束：有失败，见 $LOGDIR/gpu*.log" | tee -a "$STATUS"
  exit 1
fi
echo "结束：四路完成" | tee -a "$STATUS"
exit 0

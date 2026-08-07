#!/usr/bin/env bash
# 四卡跑满相对④的消融/对比（5 组：GPU3 串行两条）
#   GPU0: agg_equal   （FRAME_AGG=equal，证 UW）
#   GPU1: noaux       （关 FrameAux）
#   GPU2: broadcast   （MIL→broadcast）
#   GPU3: eval_first → page1（只 1 窗；再 sitebag×仅首页）
#
#   ./scripts/paper_v4_run/run_xyi_final4_ablation_quad_gpu.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="/home/amax/anaconda3/bin:${PATH:-}"
export TMPDIR="${TMPDIR:-$ROOT/.tmp_run}"
mkdir -p "$TMPDIR"
PYTHON="${OPTIGENESIS_PYTHON:-/home/amax/anaconda3/bin/python3}"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOGDIR/xyi_final4_ablation_quad_${STAMP}.log"
ln -sfn "$MASTER_LOG" "$LOGDIR/xyi_final4_ablation_quad_latest.log"

{
  echo "======================================"
  echo "④ ablation/cmp quad-GPU @ $STAMP"
  echo "  GPU0 agg_equal | GPU1 noaux | GPU2 broadcast"
  echo "  GPU3 eval_first → page1（串行）"
  echo "  底座=④；SEED=42 123 2024 3407 114514"
  echo "======================================"
} | tee -a "$MASTER_LOG"

if [[ ! -L "$ROOT/dataset" ]] || [[ "$(readlink -f "$ROOT/dataset")" != "$(readlink -f "$ROOT/tsy_loho")" ]]; then
  "$ROOT/run_experiment.sh" "$ROOT/tsy_loho"
fi
"$PYTHON" data/prepare_xyi_sitebag_splits.py --write --split-seed 20260731 --val-ratio 0.2 | tee -a "$MASTER_LOG"

WORKER="$ROOT/scripts/paper_v4_run/run_xyi_final4_ablation_one.sh"
chmod +x "$WORKER"
SEEDS_ALL="42 123 2024 3407 114514"

launch() {
  local gpu="$1" method="$2"
  local wlog="$LOGDIR/xyi_final4_${method}_gpu${gpu}_${STAMP}.log"
  echo "launch GPU=$gpu METHOD=$method → $wlog" | tee -a "$MASTER_LOG"
  (
    export TMPDIR
    export CUDA_VISIBLE_DEVICES="$gpu"
    export METHOD="$method"
    export SEEDS="$SEEDS_ALL"
    export SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
    "$WORKER"
  ) >"$wlog" 2>&1 &
  echo $! >"$LOGDIR/xyi_final4_${method}_gpu${gpu}.pid"
}

launch_chain() {
  local gpu="$1"
  shift
  local methods=("$@")
  local chain_name
  chain_name=$(IFS=_; echo "${methods[*]}")
  local wlog="$LOGDIR/xyi_final4_${chain_name}_gpu${gpu}_${STAMP}.log"
  echo "launch GPU=$gpu CHAIN=${methods[*]} → $wlog" | tee -a "$MASTER_LOG"
  (
    export TMPDIR
    export CUDA_VISIBLE_DEVICES="$gpu"
    export SEEDS="$SEEDS_ALL"
    export SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
    for method in "${methods[@]}"; do
      export METHOD="$method"
      echo "==== CHAIN start METHOD=$method @ $(date) ===="
      "$WORKER"
      echo "==== CHAIN done METHOD=$method @ $(date) ===="
    done
  ) >"$wlog" 2>&1 &
  echo $! >"$LOGDIR/xyi_final4_${chain_name}_gpu${gpu}.pid"
}

launch 0 agg_equal
launch 1 noaux
launch 2 broadcast
launch_chain 3 eval_first page1

echo "已启动。主日志: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "  tail -f $LOGDIR/xyi_final4_ablation_quad_latest.log" | tee -a "$MASTER_LOG"
echo "  tail -f $LOGDIR/xyi_final4_agg_equal_gpu0_${STAMP}.log" | tee -a "$MASTER_LOG"

wait
echo "四卡全部结束 @ $(date)" | tee -a "$MASTER_LOG"

#!/usr/bin/env bash
# 四卡：②（UW+mean）上稳定性四组，每卡满 5 seeds
#   GPU0: ema_aux（EMA0.99+Aux0.1）
#   GPU1: ema
#   GPU2: aux
#   GPU3: wma C=0.1
#
#   ./scripts/paper_v4_run/run_xyi_mean_stab_quad_gpu.sh
# 或：
#   TMPDIR=.../ .tmp_run ./run_experiment.sh --detach ./tsy_loho ./scripts/paper_v4_run/run_xyi_mean_stab_quad_gpu.sh
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
MASTER_LOG="$LOGDIR/xyi_mean_stab_quad_${STAMP}.log"
ln -sfn "$MASTER_LOG" "$LOGDIR/xyi_mean_stab_quad_latest.log"

{
  echo "======================================"
  echo "② UW+mean + stab quad-GPU @ $STAMP"
  echo "  GPU0 ema_aux | GPU1 ema | GPU2 aux | GPU3 wmaC01"
  echo "  底座无 Attn；SEED=42 123 2024 3407 114514"
  echo "======================================"
} | tee -a "$MASTER_LOG"

if [[ ! -L "$ROOT/dataset" ]] || [[ "$(readlink -f "$ROOT/dataset")" != "$(readlink -f "$ROOT/tsy_loho")" ]]; then
  "$ROOT/run_experiment.sh" "$ROOT/tsy_loho"
fi
"$PYTHON" data/prepare_xyi_sitebag_splits.py --write --split-seed 20260731 --val-ratio 0.2 | tee -a "$MASTER_LOG"

WORKER="$ROOT/scripts/paper_v4_run/run_xyi_mean_stab_one.sh"
chmod +x "$WORKER"
SEEDS_ALL="42 123 2024 3407 114514"

launch() {
  local gpu="$1" method="$2"
  local wlog="$LOGDIR/xyi_mean_stab_${method}_gpu${gpu}_${STAMP}.log"
  echo "launch GPU=$gpu METHOD=$method → $wlog" | tee -a "$MASTER_LOG"
  (
    export TMPDIR
    export CUDA_VISIBLE_DEVICES="$gpu"
    export METHOD="$method"
    export SEEDS="$SEEDS_ALL"
    export SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
    "$WORKER"
  ) >"$wlog" 2>&1 &
  echo $! >"$LOGDIR/xyi_mean_stab_${method}_gpu${gpu}.pid"
}

launch 0 ema_aux
launch 1 ema
launch 2 aux
launch 3 wma

echo "已启动。主日志: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "  tail -f $LOGDIR/xyi_mean_stab_quad_latest.log" | tee -a "$MASTER_LOG"
echo "  tail -f $LOGDIR/xyi_mean_stab_ema_aux_gpu0_${STAMP}.log" | tee -a "$MASTER_LOG"

wait
echo "四卡全部结束 @ $(date)" | tee -a "$MASTER_LOG"

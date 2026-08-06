#!/usr/bin/env bash
# 四卡并行：xyi 消融
#   GPU0: sitebag n=2 + agg=mean   seeds 42 123
#   GPU1: sitebag n=2 + agg=mean   seeds 2024 3407 114514
#   GPU2: 同切分 12×首页           seeds 42 123
#   GPU3: 同切分 12×首页           seeds 2024 3407 114514
#
# 只绑一次 dataset，避免多进程抢软链。
#
#   ./scripts/paper_v4_run/run_xyi_ablation_quad_gpu.sh
# 或：
#   ./run_experiment.sh --detach ./tsy_loho ./scripts/paper_v4_run/run_xyi_ablation_quad_gpu.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="/home/amax/anaconda3/bin:${PATH:-}"
PYTHON="${OPTIGENESIS_PYTHON:-/home/amax/anaconda3/bin/python3}"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOGDIR/xyi_ablation_quad_${STAMP}.log"
ln -sfn "$MASTER_LOG" "$LOGDIR/xyi_ablation_quad_latest.log"

echo "======================================" | tee -a "$MASTER_LOG"
echo "xyi ablation quad-GPU @ $STAMP" | tee -a "$MASTER_LOG"
echo "  A) sitebag n=2 agg=mean（弱化 OR）" | tee -a "$MASTER_LOG"
echo "  B) 同切分 12 点 × 首页（无 sitebag）" | tee -a "$MASTER_LOG"
echo "======================================" | tee -a "$MASTER_LOG"

# 绑定数据（若尚未指向 tsy_loho）
if [[ ! -L "$ROOT/dataset" ]] || [[ "$(readlink -f "$ROOT/dataset")" != "$(readlink -f "$ROOT/tsy_loho")" ]]; then
  "$ROOT/run_experiment.sh" "$ROOT/tsy_loho"
fi
"$PYTHON" data/prepare_xyi_sitebag_splits.py --write --split-seed 20260731 --val-ratio 0.2 | tee -a "$MASTER_LOG"

WORKER="$ROOT/scripts/paper_v4_run/run_xyi_ablation_one.sh"
chmod +x "$WORKER"

launch() {
  local gpu="$1" method="$2" seeds="$3"
  local wlog="$LOGDIR/xyi_ablation_${method}_gpu${gpu}_${STAMP}.log"
  echo "launch GPU=$gpu METHOD=$method SEEDS=$seeds → $wlog" | tee -a "$MASTER_LOG"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export METHOD="$method"
    export SEEDS="$seeds"
    export SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
    "$WORKER"
  ) >"$wlog" 2>&1 &
  echo $! >"$LOGDIR/xyi_ablation_${method}_gpu${gpu}.pid"
}

launch 0 mean "42 123"
launch 1 mean "2024 3407 114514"
launch 2 n12 "42 123"
launch 3 n12 "2024 3407 114514"

echo "已后台启动 4 个 worker。主日志: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "看进度:" | tee -a "$MASTER_LOG"
echo "  tail -f $LOGDIR/xyi_ablation_quad_latest.log" | tee -a "$MASTER_LOG"
echo "  tail -f $LOGDIR/xyi_ablation_mean_gpu0_${STAMP}.log" | tee -a "$MASTER_LOG"
echo "  tail -f $LOGDIR/xyi_ablation_n12_gpu2_${STAMP}.log" | tee -a "$MASTER_LOG"
echo "  tail -f outputs/paper_v4/baseline/xyi_sitebag_n2_uw_mil_aggmean_t0/seed_42/logs/train_console.log" | tee -a "$MASTER_LOG"
echo "  tail -f outputs/paper_v4/baseline/xyi_n12_page1_uw_mil_t0/seed_42/logs/train_console.log" | tee -a "$MASTER_LOG"

wait
echo "四卡全部结束 @ $(date)" | tee -a "$MASTER_LOG"

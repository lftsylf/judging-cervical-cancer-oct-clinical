#!/usr/bin/env bash
# Paper v4 · 全时序 TIFF + edl@0.3 + 放大 u 差的加权（edl_u_amp）
# score = (u_base − u) · scale，再 softmax(/τ)
#   u_base=0.5（相对观测 u∈0.2–0.3 居中偏上，使 (base−u) 多为正）
#   scale=10：Δu=0.05 → Δscore=0.5；再 /τ=0.5 → Δlogit≈1，权重比约 e:1，拉开但不至于硬 top-1
#
# 辽宁 ≈60 帧；华西/湘雅 ≈120；batch pad+mask，padding 不进聚合
#
# 用法:
#   ./run_experiment.sh --detach ./tsy_loho \
#     ./scripts/paper_v4_run/run_expand_pages_edl03_uamp_t2_oct_only.sh
#
# 看日志（两处，内容同步）:
#   tail -f logs/detached_latest.log
#   tail -f outputs/.../huaxi/seed_42/logs/train_console.log
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="/home/amax/anaconda3/bin:${PATH:-}"
PYTHON="${PYTHON:-python}"

export OPTIGENESIS_EXPAND_TIFF_PAGES=1
export OPTIGENESIS_BATCH_SIZE="${OPTIGENESIS_BATCH_SIZE:-1}"
export OPTIGENESIS_MAX_PAGES_PER_TIFF="${OPTIGENESIS_MAX_PAGES_PER_TIFF:-0}"
# 骨干分块+checkpoint，避免 N=120 在 8GB 卡 OOM
export OPTIGENESIS_FRAME_ENCODE_CHUNK="${OPTIGENESIS_FRAME_ENCODE_CHUNK:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export OPTIGENESIS_FRAME_AGG=uncertainty_weighted
export OPTIGENESIS_FRAME_AGG_TEMP="${OPTIGENESIS_FRAME_AGG_TEMP:-0.5}"
export OPTIGENESIS_FRAME_WEIGHT_SIGNAL=edl_u_amp
export OPTIGENESIS_FRAME_U_SCORE_BASE="${OPTIGENESIS_FRAME_U_SCORE_BASE:-0.5}"
export OPTIGENESIS_FRAME_U_SCORE_SCALE="${OPTIGENESIS_FRAME_U_SCORE_SCALE:-10}"

export OPTIGENESIS_ENABLE_FRAME_AUX=1
export OPTIGENESIS_FRAME_AUX_WEIGHT=0.3
export OPTIGENESIS_FRAME_AUX_TYPE=edl
export OPTIGENESIS_USE_WMA=0
export OPTIGENESIS_ENABLE_EMA=0
export OPTIGENESIS_ENABLE_AUX=0
export OPTIGENESIS_USE_CLINICAL=0

OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3_expand_uamp10}"
# 避免脚本里已带 hospital 路径时再被 main 拼一层 hospital
unset OPTIGENESIS_OUTPUT_DIR 2>/dev/null || true
mkdir -p "$OUT_ROOT"

SEEDS=(42)
HOSPITALS=(huaxi liaoning xiangya)

echo "======================================"
echo "全时序 + edl@0.3 + edl_u_amp"
echo "  EXPAND=$OPTIGENESIS_EXPAND_TIFF_PAGES  BATCH=$OPTIGENESIS_BATCH_SIZE"
echo "  signal=$OPTIGENESIS_FRAME_WEIGHT_SIGNAL  u_base=$OPTIGENESIS_FRAME_U_SCORE_BASE  scale=$OPTIGENESIS_FRAME_U_SCORE_SCALE  τ=$OPTIGENESIS_FRAME_AGG_TEMP"
echo "  OUT=$OUT_ROOT"
echo "======================================"

for h in "${HOSPITALS[@]}"; do
  for s in "${SEEDS[@]}"; do
    export HOSPITAL_NAME="$h"
    export OPTIGENESIS_SEED="$s"
    # 与其它 T2 脚本一致：OUTPUT_DIR=.../hospital ，RUN_NAME=seed_XX
    # → 最终目录 .../hospital/seed_XX/{checkpoints,logs}/ ，不会多嵌一层 hospital
    export OPTIGENESIS_OUTPUT_DIR="${OUT_ROOT}/$h"
    export OPTIGENESIS_OUTPUT_RUN_NAME="seed_${s}"

    LOG_DIR="${OPTIGENESIS_OUTPUT_DIR}/${OPTIGENESIS_OUTPUT_RUN_NAME}/logs"
    mkdir -p "$LOG_DIR"
    RUN_LOG="${LOG_DIR}/train_console.log"

    echo ">>> $h seed=$s"
    echo "    console log → $RUN_LOG"
    "$PYTHON" main.py 2>&1 | tee "$RUN_LOG"
  done
done

echo "完成 → $OUT_ROOT"

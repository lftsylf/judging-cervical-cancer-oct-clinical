#!/usr/bin/env bash
# Paper v4 · TIFF 全时序展开探路（每 TIFF 读满多页）
# 辽宁 ≈ 12×5=60 帧/人；华西/湘雅 ≈ 12×10=120 帧/人
#
# 用法（建议 detach）:
#   ./run_experiment.sh --detach ./tsy_loho ./scripts/paper_v4_run/run_expand_tiff_pages_t2_oct_only.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PATH="${PATH:-}"
export OPTIGENESIS_EXPAND_TIFF_PAGES=1
# 显存：展开后 N 大，默认 batch=1；机器够可改 2
export OPTIGENESIS_BATCH_SIZE="${OPTIGENESIS_BATCH_SIZE:-1}"
# 0=不截断；若只要「每 TIFF 前 5 页」可设 OPTIGENESIS_MAX_PAGES_PER_TIFF=5
export OPTIGENESIS_MAX_PAGES_PER_TIFF="${OPTIGENESIS_MAX_PAGES_PER_TIFF:-0}"

# 与当前最好设置对齐：UW + 帧弱监督 edl@0.3
export OPTIGENESIS_FRAME_AGG=uncertainty_weighted
export OPTIGENESIS_FRAME_AGG_TEMP="${OPTIGENESIS_FRAME_AGG_TEMP:-0.5}"
export OPTIGENESIS_FRAME_WEIGHT_SIGNAL="${OPTIGENESIS_FRAME_WEIGHT_SIGNAL:-edl_u}"
export OPTIGENESIS_ENABLE_FRAME_AUX=1
export OPTIGENESIS_FRAME_AUX_WEIGHT="${OPTIGENESIS_FRAME_AUX_WEIGHT:-0.3}"
export OPTIGENESIS_FRAME_AUX_TYPE=edl
export OPTIGENESIS_USE_WMA=0
export OPTIGENESIS_ENABLE_EMA=0
export OPTIGENESIS_ENABLE_AUX=0
export OPTIGENESIS_USE_CLINICAL=0

OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3_expand_pages}"
mkdir -p "$OUT_ROOT"

SEEDS=(42)
HOSPITALS=(huaxi liaoning xiangya)

echo "======================================"
echo "TIFF 全时序展开 T2"
echo "  EXPAND_TIFF_PAGES=$OPTIGENESIS_EXPAND_TIFF_PAGES"
echo "  BATCH_SIZE=$OPTIGENESIS_BATCH_SIZE"
echo "  MAX_PAGES_PER_TIFF=$OPTIGENESIS_MAX_PAGES_PER_TIFF"
echo "  OUT=$OUT_ROOT"
echo "======================================"

for h in "${HOSPITALS[@]}"; do
  for s in "${SEEDS[@]}"; do
    export HOSPITAL_NAME="$h"
    export OPTIGENESIS_SEED="$s"
    export OPTIGENESIS_OUTPUT_DIR="$OUT_ROOT/$h/seed_$s"
    mkdir -p "$OPTIGENESIS_OUTPUT_DIR"
    echo ">>> $h seed=$s"
    python main.py
  done
done

echo "完成 → $OUT_ROOT"

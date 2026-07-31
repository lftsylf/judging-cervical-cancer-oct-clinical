#!/usr/bin/env bash
# 单卡串行：旧 Attn（broadcast@0.3）+ WMA + EMA + Aux（T0：多 seed × 三折）
#
#   CUDA_VISIBLE_DEVICES=0 SEEDS="42 123" ./scripts/paper_v4_run/run_n12_attn_wea_one_serial.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="/home/amax/anaconda3/bin:${PATH:-}"
PYTHON="${PYTHON:-python}"

GPU="${CUDA_VISIBLE_DEVICES:-0}"

export OPTIGENESIS_EXPAND_TIFF_PAGES=0
unset OPTIGENESIS_MAX_PAGES_PER_TIFF 2>/dev/null || true
export OPTIGENESIS_BATCH_SIZE="${OPTIGENESIS_BATCH_SIZE:-4}"
export OPTIGENESIS_EPOCHS="${OPTIGENESIS_EPOCHS:-30}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# 旧 Attn 定稿设定
export OPTIGENESIS_FRAME_AGG=attention
export OPTIGENESIS_FRAME_ATTN_QUERY=mean
export OPTIGENESIS_FRAME_AGG_TEMP=0.5
export OPTIGENESIS_FRAME_WEIGHT_SIGNAL=edl_u
unset OPTIGENESIS_FRAME_U_SCORE_BASE OPTIGENESIS_FRAME_U_SCORE_SCALE 2>/dev/null || true
export OPTIGENESIS_ENABLE_FRAME_AUX=1
export OPTIGENESIS_FRAME_AUX_WEIGHT=0.3
export OPTIGENESIS_FRAME_AUX_TYPE=edl
export OPTIGENESIS_FRAME_AUX_MODE=broadcast
export OPTIGENESIS_USE_CLINICAL=0
export OPTIGENESIS_LABEL_SMOOTHING=0

# 三稳定性模块
export OPTIGENESIS_USE_WMA=1
export OPTIGENESIS_ENABLE_EMA=1
export OPTIGENESIS_ENABLE_AUX=1
# WMA 默认 C/warmup/τ 用 config；显式写出便于日志核对
export OPTIGENESIS_WMA_C="${OPTIGENESIS_WMA_C:-0.2}"
export OPTIGENESIS_WMA_WARMUP="${OPTIGENESIS_WMA_WARMUP:-10}"
export OPTIGENESIS_WMA_TEMP="${OPTIGENESIS_WMA_TEMP:-1.0}"

OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-outputs/paper_v4/baseline/ours_uw_frameaux_t0_edl_w0.3_n12_attn_wea}"
unset OPTIGENESIS_OUTPUT_DIR 2>/dev/null || true
mkdir -p "$OUT_ROOT"

HOSPITALS=(huaxi liaoning xiangya)
# shellcheck disable=SC2206
SEEDS=(${SEEDS:-42 123 2024 3407 114514})

echo "======================================"
echo "Attn + WMA/EMA/Aux | GPU=$GPU"
echo "  AGG=attention query=mean τ=$OPTIGENESIS_FRAME_AGG_TEMP"
echo "  frame_aux=broadcast@${OPTIGENESIS_FRAME_AUX_WEIGHT}"
echo "  WMA=$OPTIGENESIS_USE_WMA EMA=$OPTIGENESIS_ENABLE_EMA AUX=$OPTIGENESIS_ENABLE_AUX"
echo "  SEEDS=${SEEDS[*]}  HOSPITALS=${HOSPITALS[*]}"
echo "  OUT=$OUT_ROOT"
echo "======================================"

need_split=0
for h in "${HOSPITALS[@]}"; do
  if [[ ! -f "$ROOT/dataset/train_${h}.csv" || ! -f "$ROOT/dataset/val_${h}.csv" ]]; then
    need_split=1
  fi
done
if [[ "$need_split" == "1" ]]; then
  "$PYTHON" data/prepare_paper_v4_splits.py --write --split-seed 20260720 --val-ratio 0.2
fi

FAIL=0
for s in "${SEEDS[@]}"; do
  for h in "${HOSPITALS[@]}"; do
    export CUDA_VISIBLE_DEVICES="$GPU"
    export HOSPITAL_NAME="$h"
    export OPTIGENESIS_SEED="$s"
    export OPTIGENESIS_OUTPUT_DIR="${OUT_ROOT}/$h"
    export OPTIGENESIS_OUTPUT_RUN_NAME="seed_${s}"
    LOG_DIR="${OPTIGENESIS_OUTPUT_DIR}/${OPTIGENESIS_OUTPUT_RUN_NAME}/logs"
    mkdir -p "$LOG_DIR"
    RUN_LOG="${LOG_DIR}/train_console.log"
    if [[ "${SKIP_COMPLETED:-1}" == "1" && -f "$RUN_LOG" ]] && grep -q "训练完成！" "$RUN_LOG"; then
      echo "⏭️  跳过 attn_wea $h seed=$s"
      continue
    fi
    echo ">>> [attn_wea] $h seed=$s GPU=$GPU → $RUN_LOG"
    if ! "$PYTHON" main.py 2>&1 | tee "$RUN_LOG"; then
      echo "❌ 失败 attn_wea $h seed=$s"
      FAIL=1
    else
      echo "<<< 完成 attn_wea $h seed=$s"
    fi
  done
done

if [[ "$FAIL" -ne 0 ]]; then
  echo "有失败 → $OUT_ROOT"
  exit 1
fi
echo "全部完成 attn_wea → $OUT_ROOT"

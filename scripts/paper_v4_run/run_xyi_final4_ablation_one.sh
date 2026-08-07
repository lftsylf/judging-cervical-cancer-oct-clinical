#!/usr/bin/env bash
# 单卡：相对最终④（sitebag n=2×全页 + UW + MIL@0.3 + 多窗 mean + EMA0.99）的消融/对比
#
# METHOD ∈ {
#   agg_equal   — 消融：FRAME_AGG=equal（有帧 EDL、无不确定加权）
#   noaux       — 消融：关 FrameAux
#   broadcast   — 消融：FrameAux MIL → broadcast
#   eval_first  — 对比：val/test 只取第 1 窗（不扫全圈 6 窗 mean）
#   page1       — 对比：sitebag 同设定，每点只取首页（证全页有用）
# }
#
#   CUDA_VISIBLE_DEVICES=0 METHOD=agg_equal SEEDS="42 123 2024 3407 114514" \
#     ./scripts/paper_v4_run/run_xyi_final4_ablation_one.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="/home/amax/anaconda3/bin:${PATH:-}"
export TMPDIR="${TMPDIR:-$ROOT/.tmp_run}"
mkdir -p "$TMPDIR"
PYTHON="${OPTIGENESIS_PYTHON:-/home/amax/anaconda3/bin/python3}"

METHOD="${METHOD:?设 METHOD=agg_equal|noaux|broadcast|eval_first|page1}"
# shellcheck disable=SC2206
SEEDS=(${SEEDS:-42 123 2024 3407 114514})
GPU="${CUDA_VISIBLE_DEVICES:-0}"
MAX_EPOCHS="${OPTIGENESIS_EPOCHS:-30}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
BASE2="$ROOT/outputs/paper_v4/baseline2"

case "$METHOD" in
  agg_equal)
    OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-$BASE2/xyi_sitebag_n2_equal_mil_aggmean_ema099}"
    ;;
  noaux)
    OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-$BASE2/xyi_sitebag_n2_uw_noaux_aggmean_ema099}"
    ;;
  broadcast)
    OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-$BASE2/xyi_sitebag_n2_uw_bcast_aggmean_ema099}"
    ;;
  eval_first)
    OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-$BASE2/xyi_sitebag_n2_uw_mil_aggfirst_ema099}"
    ;;
  page1)
    OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-$BASE2/xyi_sitebag_n2_uw_mil_page1_aggmean_ema099}"
    ;;
  *)
    echo "未知 METHOD=$METHOD"; exit 1
    ;;
esac

is_run_complete() {
  local log_file="$1"
  [[ -f "$log_file" ]] && grep -q "训练完成！" "$log_file"
}

mkdir -p "$OUT_ROOT"
echo "======================================"
echo "④ ablation/cmp | METHOD=$METHOD | GPU=$GPU | seeds=${SEEDS[*]}"
echo "底座: sitebag n=2 + UW + MIL@0.3 + mean + EMA0.99（无 Attn/WMA/Aux）"
echo "OUT=$OUT_ROOT"
echo "======================================"

if [[ ! -f "$ROOT/dataset/train_xyi.csv" ]]; then
  "$PYTHON" data/prepare_xyi_sitebag_splits.py --write --split-seed 20260731 --val-ratio 0.2
fi

for SEED in "${SEEDS[@]}"; do
  RUN_DIR="${OUT_ROOT}/seed_${SEED}"
  LOG="${RUN_DIR}/logs/train_console.log"
  mkdir -p "${RUN_DIR}/logs"
  if [[ "${FORCE_RERUN:-0}" != "1" && "${SKIP_COMPLETED}" == "1" ]] && is_run_complete "$LOG"; then
    echo "✅ skip seed=$SEED"
    continue
  fi
  echo "▶ METHOD=$METHOD seed=$SEED"

  export CUDA_VISIBLE_DEVICES="$GPU"
  export HOSPITAL_NAME=xyi
  export OPTIGENESIS_OUTPUT_DIR="$OUT_ROOT"
  export OPTIGENESIS_OUTPUT_RUN_NAME="seed_${SEED}"
  export OPTIGENESIS_SEED="$SEED"
  export OPTIGENESIS_EPOCHS="$MAX_EPOCHS"
  export OPTIGENESIS_LR="${OPTIGENESIS_LR:-5e-5}"
  export OPTIGENESIS_POS_WEIGHT="${OPTIGENESIS_POS_WEIGHT:-1.25}"
  export OPTIGENESIS_BACKBONE="${OPTIGENESIS_BACKBONE:-resnet50}"
  export OPTIGENESIS_USE_CLINICAL=0
  export OPTIGENESIS_LABEL_SMOOTHING=0

  # ④ 底座
  export OPTIGENESIS_FRAME_AGG=uncertainty_weighted
  export OPTIGENESIS_FRAME_AGG_TEMP=0.5
  export OPTIGENESIS_FRAME_WEIGHT_SIGNAL=edl_u
  export OPTIGENESIS_ENABLE_FRAME_AUX=1
  export OPTIGENESIS_FRAME_AUX_WEIGHT=0.3
  export OPTIGENESIS_FRAME_AUX_TYPE=edl
  export OPTIGENESIS_FRAME_AUX_MODE=mil
  export OPTIGENESIS_FRAME_AUX_MIL_POS_THR=0.5
  export OPTIGENESIS_SITEBAG=1
  export OPTIGENESIS_SITEBAG_N=2
  export OPTIGENESIS_SITEBAG_EVAL_OR=1
  export OPTIGENESIS_SITEBAG_EVAL_AGG=mean
  export OPTIGENESIS_EXPAND_TIFF_PAGES=0
  export OPTIGENESIS_MAX_PAGES_PER_TIFF=0
  export OPTIGENESIS_BATCH_SIZE="${OPTIGENESIS_BATCH_SIZE:-2}"
  export OPTIGENESIS_FRAME_ENCODE_CHUNK="${OPTIGENESIS_FRAME_ENCODE_CHUNK:-16}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export OPTIGENESIS_ENABLE_CORAL=0
  export OPTIGENESIS_USE_WMA=0
  export OPTIGENESIS_ENABLE_AUX=0
  export OPTIGENESIS_ENABLE_EMA=1
  export OPTIGENESIS_EMA_DECAY="${OPTIGENESIS_EMA_DECAY:-0.99}"
  export OPTIGENESIS_FRAME_AUX_USE_WMA=0

  case "$METHOD" in
    agg_equal)
      export OPTIGENESIS_FRAME_AGG=equal
      ;;
    noaux)
      export OPTIGENESIS_ENABLE_FRAME_AUX=0
      ;;
    broadcast)
      export OPTIGENESIS_FRAME_AUX_MODE=broadcast
      ;;
    eval_first)
      export OPTIGENESIS_SITEBAG_EVAL_AGG=first
      ;;
    page1)
      # sitebag 仍 expand，但截断为每 TIFF 仅 1 页 → 证「全页有用」
      export OPTIGENESIS_MAX_PAGES_PER_TIFF=1
      ;;
  esac

  set +e
  "$PYTHON" -u main.py 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ "$rc" -ne 0 ]]; then
    echo "❌ METHOD=$METHOD seed=$SEED rc=$rc"
    exit "$rc"
  fi
  echo "✅ METHOD=$METHOD seed=$SEED done"
done

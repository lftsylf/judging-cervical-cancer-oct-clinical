#!/usr/bin/env bash
# 单卡：② 底座（UW + sitebag n=2×全页 + 多窗 mean + MIL@0.3）上叠稳定性模块
# METHOD ∈ {ema_aux, ema, aux, wma}
#
#   CUDA_VISIBLE_DEVICES=0 METHOD=ema_aux SEEDS="42 123 2024 3407 114514" \
#     ./scripts/paper_v4_run/run_xyi_mean_stab_one.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="/home/amax/anaconda3/bin:${PATH:-}"
# 系统盘满时避免写 /tmp
export TMPDIR="${TMPDIR:-$ROOT/.tmp_run}"
mkdir -p "$TMPDIR"
PYTHON="${OPTIGENESIS_PYTHON:-/home/amax/anaconda3/bin/python3}"

METHOD="${METHOD:?设 METHOD=ema_aux|ema|aux|wma}"
# shellcheck disable=SC2206
SEEDS=(${SEEDS:-42 123 2024 3407 114514})
GPU="${CUDA_VISIBLE_DEVICES:-0}"
MAX_EPOCHS="${OPTIGENESIS_EPOCHS:-30}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

case "$METHOD" in
  ema_aux)
    OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-$ROOT/outputs/paper_v4/baseline2/xyi_sitebag_n2_uw_mil_aggmean_ema099_aux01}"
    ;;
  ema)
    OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-$ROOT/outputs/paper_v4/baseline2/xyi_sitebag_n2_uw_mil_aggmean_ema099}"
    ;;
  aux)
    OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-$ROOT/outputs/paper_v4/baseline2/xyi_sitebag_n2_uw_mil_aggmean_aux01}"
    ;;
  wma)
    OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-$ROOT/outputs/paper_v4/baseline2/xyi_sitebag_n2_uw_mil_aggmean_wmaC01}"
    ;;
  *)
    echo "未知 METHOD=$METHOD（期望 ema_aux|ema|aux|wma）"; exit 1
    ;;
esac

is_run_complete() {
  local log_file="$1"
  [[ -f "$log_file" ]] && grep -q "训练完成！" "$log_file"
}

mkdir -p "$OUT_ROOT"
echo "======================================"
echo "②+stab | METHOD=$METHOD | GPU=$GPU | seeds=${SEEDS[*]}"
echo "底座: UW + MIL@0.3 + sitebag n=2 + eval_agg=mean（无 Attn）"
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

  # ② 底座（UW，不是 Attn）
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
  export OPTIGENESIS_BATCH_SIZE="${OPTIGENESIS_BATCH_SIZE:-2}"
  export OPTIGENESIS_FRAME_ENCODE_CHUNK="${OPTIGENESIS_FRAME_ENCODE_CHUNK:-16}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export OPTIGENESIS_ENABLE_CORAL=0

  # 默认全关，再按 METHOD 打开
  export OPTIGENESIS_USE_WMA=0
  export OPTIGENESIS_ENABLE_EMA=0
  export OPTIGENESIS_ENABLE_AUX=0
  export OPTIGENESIS_FRAME_AUX_USE_WMA=1
  export OPTIGENESIS_WMA_C="${OPTIGENESIS_WMA_C:-0.1}"
  export OPTIGENESIS_WMA_WARMUP="${OPTIGENESIS_WMA_WARMUP:-10}"
  export OPTIGENESIS_WMA_TEMP="${OPTIGENESIS_WMA_TEMP:-1.0}"
  export OPTIGENESIS_EMA_DECAY="${OPTIGENESIS_EMA_DECAY:-0.99}"
  export OPTIGENESIS_AUX_W_VISION="${OPTIGENESIS_AUX_W_VISION:-0.1}"
  export OPTIGENESIS_AUX_W_CLINICAL="${OPTIGENESIS_AUX_W_CLINICAL:-0.1}"

  case "$METHOD" in
    ema_aux)
      export OPTIGENESIS_ENABLE_EMA=1
      export OPTIGENESIS_ENABLE_AUX=1
      ;;
    ema)
      export OPTIGENESIS_ENABLE_EMA=1
      ;;
    aux)
      export OPTIGENESIS_ENABLE_AUX=1
      ;;
    wma)
      export OPTIGENESIS_USE_WMA=1
      export OPTIGENESIS_FRAME_AUX_USE_WMA=1
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

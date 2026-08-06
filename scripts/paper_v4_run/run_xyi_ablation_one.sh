#!/usr/bin/env bash
# 单卡 worker：xyi 消融
#   METHOD=mean  → sitebag n=2，多窗聚合改为 mean（弱化 OR）
#   METHOD=n12   → 同切分，全 12 点 × 仅首页（无 sitebag）
#   METHOD=first → sitebag n=2，只取第一窗（关多窗 OR）
#
#   CUDA_VISIBLE_DEVICES=0 METHOD=mean SEEDS="42 123" \
#     ./scripts/paper_v4_run/run_xyi_ablation_one.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="/home/amax/anaconda3/bin:${PATH:-}"
PYTHON="${OPTIGENESIS_PYTHON:-/home/amax/anaconda3/bin/python3}"

METHOD="${METHOD:?设 METHOD=mean|n12|first}"
# shellcheck disable=SC2206
SEEDS=(${SEEDS:-42 123})
GPU="${CUDA_VISIBLE_DEVICES:-0}"
MAX_EPOCHS="${OPTIGENESIS_EPOCHS:-30}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

case "$METHOD" in
  mean)
    OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-$ROOT/outputs/paper_v4/baseline/xyi_sitebag_n2_uw_mil_aggmean_t0}"
    ;;
  first)
    OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-$ROOT/outputs/paper_v4/baseline/xyi_sitebag_n2_uw_mil_aggfirst_t0}"
    ;;
  n12)
    OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-$ROOT/outputs/paper_v4/baseline/xyi_n12_page1_uw_mil_t0}"
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
echo "xyi ablation | METHOD=$METHOD | GPU=$GPU | seeds=${SEEDS[*]}"
echo "OUT=$OUT_ROOT"
echo "======================================"

# 确保切分存在（不抢软链；由 orchestrator 绑定 dataset）
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
  export OPTIGENESIS_FRAME_AGG=uncertainty_weighted
  export OPTIGENESIS_FRAME_AGG_TEMP=0.5
  export OPTIGENESIS_FRAME_WEIGHT_SIGNAL=edl_u
  export OPTIGENESIS_ENABLE_FRAME_AUX=1
  export OPTIGENESIS_FRAME_AUX_WEIGHT=0.3
  export OPTIGENESIS_FRAME_AUX_TYPE=edl
  export OPTIGENESIS_FRAME_AUX_MODE=mil
  export OPTIGENESIS_FRAME_AUX_MIL_POS_THR=0.5
  export OPTIGENESIS_USE_WMA=0
  export OPTIGENESIS_ENABLE_EMA=0
  export OPTIGENESIS_ENABLE_AUX=0
  export OPTIGENESIS_ENABLE_CORAL=0
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

  if [[ "$METHOD" == "n12" ]]; then
    export OPTIGENESIS_SITEBAG=0
    unset OPTIGENESIS_SITEBAG_N OPTIGENESIS_SITEBAG_EVAL_OR OPTIGENESIS_SITEBAG_EVAL_AGG 2>/dev/null || true
    export OPTIGENESIS_EXPAND_TIFF_PAGES=0
    export OPTIGENESIS_BATCH_SIZE="${OPTIGENESIS_BATCH_SIZE:-4}"
  else
    export OPTIGENESIS_SITEBAG=1
    export OPTIGENESIS_SITEBAG_N=2
    export OPTIGENESIS_SITEBAG_EVAL_OR=1
    export OPTIGENESIS_SITEBAG_EVAL_AGG="$METHOD"  # mean | first
    export OPTIGENESIS_EXPAND_TIFF_PAGES=0
    export OPTIGENESIS_BATCH_SIZE="${OPTIGENESIS_BATCH_SIZE:-2}"
    export OPTIGENESIS_FRAME_ENCODE_CHUNK="${OPTIGENESIS_FRAME_ENCODE_CHUNK:-16}"
  fi

  unset OPTIGENESIS_OUTPUT_DIR 2>/dev/null || true
  # 上面又 unset 了，需要重新 export 给 main（main 读环境变量在 import Config 时）
  export OPTIGENESIS_OUTPUT_DIR="$OUT_ROOT"

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

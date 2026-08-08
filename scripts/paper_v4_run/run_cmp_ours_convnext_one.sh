#!/usr/bin/env bash
# 变体 · Ours(4[final]) + ConvNeXt-Tiny 骨干
# 机位建议：GPU3 / 服务器 #4
#
# 状态：可直接跑（只改 BACKBONE；流程同 4[final]）
# 显存紧：OPTIGENESIS_BATCH_SIZE=1 或 FRAME_ENCODE_CHUNK=8
#
#   CUDA_VISIBLE_DEVICES=3 SEEDS="42 123 2024 3407 114514" \
#     ./scripts/paper_v4_run/run_cmp_ours_convnext_one.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="/home/amax/anaconda3/bin:${PATH:-}"
PYTHON="${OPTIGENESIS_PYTHON:-/home/amax/anaconda3/bin/python3}"

# shellcheck disable=SC2206
SEEDS=(${SEEDS:-42 123 2024 3407 114514})
GPU="${CUDA_VISIBLE_DEVICES:-0}"
MAX_EPOCHS="${OPTIGENESIS_EPOCHS:-30}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CMP_ROOT="$ROOT/outputs/paper_v4/对比实验"
# 可用 OPTIGENESIS_BACKBONE=convnext_small 覆盖
BB="${OPTIGENESIS_BACKBONE:-convnext_tiny}"
OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-$CMP_ROOT/13[变体] final_uw_mil_aggmean_ema099_${BB}}"

is_run_complete() {
  local log_file="$1"
  [[ -f "$log_file" ]] && grep -q "训练完成！" "$log_file"
}

mkdir -p "$OUT_ROOT"
echo "======================================"
echo "OURS+${BB} | GPU=$GPU | seeds=${SEEDS[*]}"
echo "底座 = 4[final]：UW + MIL@0.3 + mean + EMA0.99"
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
  echo "▶ Ours+${BB} seed=$SEED"

  # shellcheck source=/dev/null
  source "$ROOT/scripts/paper_v4_run/cmp_env_common.inc.sh"

  export OPTIGENESIS_BACKBONE="$BB"
  # ConvNeXt 通常比 ResNet50 吃显存；默认 batch=2，OOM 再降
  export OPTIGENESIS_BATCH_SIZE="${OPTIGENESIS_BATCH_SIZE:-2}"
  export OPTIGENESIS_FRAME_ENCODE_CHUNK="${OPTIGENESIS_FRAME_ENCODE_CHUNK:-8}"

  # —— 完整 4[final] ——
  export OPTIGENESIS_FRAME_AGG=uncertainty_weighted
  export OPTIGENESIS_FRAME_AGG_TEMP=0.5
  export OPTIGENESIS_FRAME_WEIGHT_SIGNAL=edl_u
  export OPTIGENESIS_ENABLE_FRAME_AUX=1
  export OPTIGENESIS_FRAME_AUX_WEIGHT=0.3
  export OPTIGENESIS_FRAME_AUX_TYPE=edl
  export OPTIGENESIS_FRAME_AUX_MODE=mil
  export OPTIGENESIS_FRAME_AUX_MIL_POS_THR=0.5
  export OPTIGENESIS_ENABLE_EMA=1
  export OPTIGENESIS_EMA_DECAY=0.99
  export OPTIGENESIS_CMP_METHOD=ours_convnext

  set +e
  "$PYTHON" -u main.py 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ "$rc" -ne 0 ]]; then
    echo "❌ Ours+${BB} seed=$SEED rc=$rc"
    exit "$rc"
  fi
  echo "✅ Ours+${BB} seed=$SEED done"
done

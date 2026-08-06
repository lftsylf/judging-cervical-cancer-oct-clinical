#!/usr/bin/env bash
# =============================================================================
# Paper v4 · 湘雅内部 site-bag baseline T0（无 EMA / Aux / WMA / Attn）
#
# 协议：
#   - 内部：湘雅 97 → 8:2 train/val（HOSPITAL_NAME=xyi）
#   - 外部终评：华西、辽宁分开（另报 pooled external）
#   - SITEBAG_N=2：每次 2 点位 × 全页；训优先病理阳点；val/test 6 窗 OR
#   - FRAME_AGG=uncertainty_weighted + FrameAux MIL@0.3
#   - T0：单切分 × 5 seeds
#
# 前台：
#   ./scripts/paper_v4_run/run_xyi_sitebag_n2_uw_mil_t0.sh
#
# 后台：
#   ./run_experiment.sh --detach ./tsy_loho ./scripts/paper_v4_run/run_xyi_sitebag_n2_uw_mil_t0.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export PATH="/home/amax/anaconda3/bin:${PATH:-}"
PYTHON="${OPTIGENESIS_PYTHON:-/home/amax/anaconda3/bin/python3}"

SEEDS=(${SEEDS:-42 123 2024 3407 114514})
MAX_EPOCHS="${OPTIGENESIS_EPOCHS:-30}"
OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-$PROJECT_ROOT/outputs/paper_v4/baseline/xyi_sitebag_n2_uw_mil_t0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

echo "======================================"
echo "xyi site-bag T0 | n=2 | UW + MIL@0.3 | 无稳定模块"
echo "内部=湘雅8:2 | 外部=华西/辽宁分开 | Seeds: ${SEEDS[*]}"
echo "输出: ${OUT_ROOT}"
echo "======================================"

"$PROJECT_ROOT/run_experiment.sh" "$PROJECT_ROOT/tsy_loho"

# 生成 / 刷新切分
"$PYTHON" data/prepare_xyi_sitebag_splits.py --write --split-seed 20260731 --val-ratio 0.2

is_run_complete() {
  local log_file="$1"
  [[ -f "$log_file" ]] && grep -q "训练完成！" "$log_file"
}

mkdir -p "$OUT_ROOT"
unset OPTIGENESIS_OUTPUT_DIR 2>/dev/null || true

for SEED in "${SEEDS[@]}"; do
  RUN_DIR="${OUT_ROOT}/seed_${SEED}"
  LOG="${RUN_DIR}/logs/train_console.log"
  mkdir -p "${RUN_DIR}/logs"

  if [[ "${FORCE_RERUN:-0}" != "1" && "${SKIP_COMPLETED}" == "1" ]] && is_run_complete "$LOG"; then
    echo "✅ skip completed seed=${SEED}"
    continue
  fi

  echo "--------------------------------------"
  echo "▶ seed=${SEED} → ${RUN_DIR}"

  export HOSPITAL_NAME=xyi
  export OPTIGENESIS_OUTPUT_DIR="$OUT_ROOT"
  export OPTIGENESIS_OUTPUT_RUN_NAME="seed_${SEED}"
  export OPTIGENESIS_SEED="$SEED"
  export OPTIGENESIS_EPOCHS="$MAX_EPOCHS"
  export OPTIGENESIS_BATCH_SIZE="${OPTIGENESIS_BATCH_SIZE:-2}"
  export OPTIGENESIS_LR="${OPTIGENESIS_LR:-5e-5}"
  export OPTIGENESIS_POS_WEIGHT="${OPTIGENESIS_POS_WEIGHT:-1.25}"
  export OPTIGENESIS_BACKBONE="${OPTIGENESIS_BACKBONE:-resnet50}"
  export OPTIGENESIS_USE_CLINICAL=0
  export OPTIGENESIS_LABEL_SMOOTHING=0

  # site-bag
  export OPTIGENESIS_SITEBAG=1
  export OPTIGENESIS_SITEBAG_N=2
  export OPTIGENESIS_SITEBAG_EVAL_OR=1
  # 页展开由 sitebag 强制；全局 expand 关以免误读 12 点全页
  export OPTIGENESIS_EXPAND_TIFF_PAGES=0
  unset OPTIGENESIS_MAX_PAGES_PER_TIFF 2>/dev/null || true
  export OPTIGENESIS_FRAME_ENCODE_CHUNK="${OPTIGENESIS_FRAME_ENCODE_CHUNK:-16}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

  # UW + MIL，无稳定模块
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

  set +e
  "$PYTHON" -u main.py 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ "$rc" -ne 0 ]]; then
    echo "❌ seed=${SEED} failed rc=$rc"
    exit "$rc"
  fi
  echo "✅ seed=${SEED} done"
done

echo "全部完成 → ${OUT_ROOT}"

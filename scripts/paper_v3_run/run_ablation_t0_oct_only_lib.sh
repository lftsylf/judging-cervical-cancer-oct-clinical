#!/usr/bin/env bash
# v2 单模态 T0 消融 — 共享训练循环（由 run_*_t0_oct_only.sh source，勿直接执行）
#
# 调用方必须设置：
#   ABLATION_TAG     — full | no_wma | no_ema | no_aux | only_wma | only_ema | only_aux
#   ABLATION_USE_WMA — 0 | 1
#   ABLATION_ENABLE_EMA — 0 | 1
#   ABLATION_ENABLE_AUX — 0 | 1
#
# 协议与 run_baseline_t0_oct_only.sh 一致：
#   resnet50, USE_CLINICAL=0, LR=5e-5, POS_WEIGHT=1.25, BATCH=4, CORAL=0
#   3 折 × 5 seeds = 15 次
set -euo pipefail

if [[ -z "${ABLATION_TAG:-}" ]]; then
  echo "❌ ABLATION_TAG 未设置（full | no_wma | no_ema | no_aux | only_wma | only_ema | only_aux）"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${OPTIGENESIS_PYTHON:-/home/amax/anaconda3/bin/python3}"
CAPTURE_BACKBONE="${OPTIGENESIS_BACKBONE:-resnet50}"
CAPTURE_POS_WEIGHT="${OPTIGENESIS_POS_WEIGHT:-1.25}"
CAPTURE_LR="${OPTIGENESIS_LR:-5e-5}"
CAPTURE_BATCH_SIZE="${ABLATION_BATCH_SIZE:-4}"
_safe_bb="${CAPTURE_BACKBONE//[^a-zA-Z0-9_]/_}"

HOSPITALS=(huaxi liaoning xiangya)
SEEDS=(42 123 2024 3407 114514)
MAX_EPOCHS=30

ABLATION_PARENT="${ABLATION_PARENT:-$PROJECT_ROOT/outputs/第三版/消融实验v2单模态}"
ABLATION_OUT_ROOT="${ABLATION_OUT_ROOT:-${ABLATION_PARENT}/outputs_ablation_v2_oct_${ABLATION_TAG}}"

SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
TOTAL_RUNS=$((${#HOSPITALS[@]} * ${#SEEDS[@]}))

is_run_complete() {
  local log_file="$1"
  [[ -f "$log_file" ]] && grep -q "训练完成！" "$log_file"
}

echo "======================================"
echo "T0 Ablation | OCT-only | ${CAPTURE_BACKBONE} | tag=${ABLATION_TAG}"
echo "USE_CLINICAL=0 | LR=${CAPTURE_LR} | POS_WEIGHT=${CAPTURE_POS_WEIGHT}"
echo "BATCH_SIZE=${CAPTURE_BATCH_SIZE} | EPOCHS<=${MAX_EPOCHS} | early stop patience=10"
echo "WMA=${ABLATION_USE_WMA} | EMA=${ABLATION_ENABLE_EMA} | AUX=${ABLATION_ENABLE_AUX} | CORAL=0"
echo "折: ${HOSPITALS[*]}"
echo "Seeds: ${SEEDS[*]} | 共 ${TOTAL_RUNS} 次"
echo "Python: ${PYTHON}"
echo "输出根: ${ABLATION_OUT_ROOT}"
echo "跳过已完成: SKIP_COMPLETED=${SKIP_COMPLETED} | FORCE_RERUN=${FORCE_RERUN:-0}"
echo "定稿 baseline 对照: outputs_baseline_t0_v2_resnet50_oct_only/"
echo "======================================"

"$PROJECT_ROOT/run_experiment.sh" "$PROJECT_ROOT/tsy_loho"
"$PYTHON" data/prepare_loho_data.py

RUN_IDX=0
for HOSP in "${HOSPITALS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    RUN_IDX=$((RUN_IDX + 1))
    echo "--------------------------------------"
    echo "T0 ablation [${ABLATION_TAG}] [${RUN_IDX}/${TOTAL_RUNS}] | ${HOSP} | seed=${SEED}"
    echo "--------------------------------------"

    unset OPTIGENESIS_LR OPTIGENESIS_BACKBONE_LR OPTIGENESIS_HEAD_LR
    unset OPTIGENESIS_FREEZE_BACKBONE_EPOCHS OPTIGENESIS_FREEZE_HEAD_LR
    unset OPTIGENESIS_BACKBONE
    unset OPTIGENESIS_ENABLE_AUX
    unset OPTIGENESIS_ENABLE_EMA
    unset OPTIGENESIS_ENABLE_CORAL
    unset OPTIGENESIS_OUTPUT_DIR
    unset OPTIGENESIS_OUTPUT_RUN_NAME
    unset OPTIGENESIS_BATCH_SIZE
    unset OPTIGENESIS_CORAL_LAMBDA
    unset OPTIGENESIS_CORAL_WARMUP
    unset OPTIGENESIS_EPOCHS
    unset OPTIGENESIS_SEED
    unset OPTIGENESIS_USE_WMA
    unset OPTIGENESIS_USE_CLINICAL
    unset OPTIGENESIS_POS_WEIGHT
    unset OPTIGENESIS_WMA_C
    unset OPTIGENESIS_WMA_WARMUP
    unset OPTIGENESIS_WMA_TEMP
    unset HOSPITAL_NAME

    export OPTIGENESIS_BACKBONE="$CAPTURE_BACKBONE"
    export OPTIGENESIS_USE_CLINICAL=0
    export OPTIGENESIS_LR="$CAPTURE_LR"
    export OPTIGENESIS_POS_WEIGHT="$CAPTURE_POS_WEIGHT"
    export OPTIGENESIS_ENABLE_AUX="$ABLATION_ENABLE_AUX"
    export OPTIGENESIS_ENABLE_EMA="$ABLATION_ENABLE_EMA"
    export OPTIGENESIS_ENABLE_CORAL=0
    export OPTIGENESIS_BATCH_SIZE="$CAPTURE_BATCH_SIZE"
    export OPTIGENESIS_USE_WMA="$ABLATION_USE_WMA"
    export HOSPITAL_NAME="$HOSP"
    export OPTIGENESIS_SEED="$SEED"
    export OPTIGENESIS_EPOCHS="$MAX_EPOCHS"
    export OPTIGENESIS_OUTPUT_DIR="${ABLATION_OUT_ROOT}/$HOSP"
    export OPTIGENESIS_OUTPUT_RUN_NAME="seed_${SEED}"

    LOG_DIR="${OPTIGENESIS_OUTPUT_DIR}/${OPTIGENESIS_OUTPUT_RUN_NAME}/logs"
    mkdir -p "$LOG_DIR"
    RUN_LOG="${LOG_DIR}/train_console.log"

    if [[ "$SKIP_COMPLETED" == "1" && "${FORCE_RERUN:-0}" != "1" ]] && is_run_complete "$RUN_LOG"; then
      echo "⏭️  跳过：${RUN_LOG} 已含「训练完成！」"
      continue
    fi

    "$PYTHON" main.py 2>&1 | tee "$RUN_LOG"
  done
done

echo "======================================"
echo "T0 ablation [${ABLATION_TAG}] 全部完成。"
echo "结果根目录: ${ABLATION_OUT_ROOT}"
echo "======================================"

#!/usr/bin/env bash
# T1 baseline（v2 定稿组 1）：ResNet50 + 全网 LR=5e-5 + POS_WEIGHT=1.25（config 默认）
# 规格：3 折 × 3 seeds（42, 123, 2024）；早停 patience=10；无 aux/EMA/WMA/CORAL
#
# 与旧 ResNet50 T1（outputs_baseline_t1_v2_resnet50/）区分，本次默认输出到新目录。
#
# 前台跑：
#   ./run_baseline_t1.sh
#
# 断线可续跑（推荐）：
#   ./run_experiment.sh --detach /path/to/tsy_loho ./run_baseline_t1.sh
#
# 续跑时跳过 log 已含「训练完成！」的折：
#   FORCE_RERUN=1 ./run_baseline_t1.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${OPTIGENESIS_PYTHON:-/home/amax/anaconda3/bin/python3}"
CAPTURE_BACKBONE="${OPTIGENESIS_BACKBONE:-resnet50}"
_safe_bb="${CAPTURE_BACKBONE//[^a-zA-Z0-9_]/_}"

HOSPITALS=(huaxi liaoning xiangya)
SEEDS=(42 123 2024)
MAX_EPOCHS=30
# 新 T1（T2 探路确认组 1 后）；旧结果仍在 outputs_baseline_t1_v2_resnet50/
BASELINE_OUT_ROOT="${BASELINE_OUT_ROOT:-$PROJECT_ROOT/outputs_baseline_t1_v2_${_safe_bb}_g1_lr5e5}"

SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

is_run_complete() {
  local log_file="$1"
  [[ -f "$log_file" ]] && grep -q "训练完成！" "$log_file"
}

echo "======================================"
echo "T1 Baseline（组1 定稿）| ${CAPTURE_BACKBONE}"
echo "LR=5e-5（全网）| 无 BACKBONE_LR/HEAD_LR/FREEZE"
echo "折: ${HOSPITALS[*]} | Seeds: ${SEEDS[*]} | 共 $(( ${#HOSPITALS[@]} * ${#SEEDS[@]} )) 次"
echo "Python: ${PYTHON}"
echo "输出根: ${BASELINE_OUT_ROOT}"
echo "跳过已完成: SKIP_COMPLETED=${SKIP_COMPLETED} | FORCE_RERUN=${FORCE_RERUN:-0}"
echo "======================================"

./run_experiment.sh "$PROJECT_ROOT/tsy_loho"
"$PYTHON" data/prepare_loho_data.py

for HOSP in "${HOSPITALS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    echo "--------------------------------------"
    echo "T1 | ${HOSP} | seed=${SEED}"
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
    unset HOSPITAL_NAME

    # 组 1：只设全网学习率
    export OPTIGENESIS_LR=5e-5
    export OPTIGENESIS_BACKBONE="$CAPTURE_BACKBONE"
    export OPTIGENESIS_ENABLE_AUX=0
    export OPTIGENESIS_ENABLE_EMA=0
    export OPTIGENESIS_ENABLE_CORAL=0
    export OPTIGENESIS_BATCH_SIZE=4
    export OPTIGENESIS_USE_WMA=0
    export HOSPITAL_NAME="$HOSP"
    export OPTIGENESIS_SEED="$SEED"
    export OPTIGENESIS_EPOCHS="$MAX_EPOCHS"
    export OPTIGENESIS_OUTPUT_DIR="${BASELINE_OUT_ROOT}/$HOSP"
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
echo "T1 全部完成。结果根目录: ${BASELINE_OUT_ROOT}"
echo "对比旧 ResNet50 T1: outputs_baseline_t1_v2_resnet50/"
echo "======================================"

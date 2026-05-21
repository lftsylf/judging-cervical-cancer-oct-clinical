#!/usr/bin/env bash
# T0 Baseline（v2 定稿 · OCT-only / 单模态）
#
# 协议（与 T1 clinical_off 一致，仅 seed 数为 T0）：
#   - 骨干: resnet50
#   - OPTIGENESIS_USE_CLINICAL=0（仅 OCT，不用 Age/HPV/TCT）
#   - LR=5e-5（全网，无分层 LR / 无 freeze）
#   - POS_WEIGHT=1.25
#   - BATCH_SIZE=4
#   - 3 折 × 5 seeds（42, 123, 2024, 3407, 114514）= 15 次
#   - 最多 30 epoch，main.py 验证集 AUC early stopping patience=10
#   - 关闭: WMA / aux / EMA / CORAL
#
# 前台：
#   ./run_baseline_t0_oct_only.sh
#
# SSH 断线仍继续（推荐）：
#   export PATH="/home/amax/anaconda3/bin:$PATH"
#   cd /ssd_data/tsy_study_venv/OptiGenesis_Lancet
#   ./run_experiment.sh --detach \
#     /ssd_data/tsy_study_venv/OptiGenesis_Lancet/tsy_loho \
#     ./run_baseline_t0_oct_only.sh
#
# 断线后看总日志：
#   tail -f logs/detached_latest.log
# 看某一折：
#   tail -f outputs_baseline_t0_v2_resnet50_oct_only/huaxi/seed_42/logs/train_console.log
#
# 续跑（跳过已完成）：
#   SKIP_COMPLETED=1 ./run_baseline_t0_oct_only.sh
# 强制重跑某一折：
#   FORCE_RERUN=1 ./run_baseline_t0_oct_only.sh
#
# 可选环境变量：
#   BASELINE_OUT_ROOT — 输出根目录
#   OPTIGENESIS_PYTHON — Python 解释器（默认 /home/amax/anaconda3/bin/python3）
#   OPTIGENESIS_BACKBONE — 默认 resnet50
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${OPTIGENESIS_PYTHON:-/home/amax/anaconda3/bin/python3}"
CAPTURE_BACKBONE="${OPTIGENESIS_BACKBONE:-resnet50}"
CAPTURE_POS_WEIGHT="${OPTIGENESIS_POS_WEIGHT:-1.25}"
CAPTURE_LR="${OPTIGENESIS_LR:-5e-5}"
_safe_bb="${CAPTURE_BACKBONE//[^a-zA-Z0-9_]/_}"

HOSPITALS=(huaxi liaoning xiangya)
SEEDS=(42 123 2024 3407 114514)
MAX_EPOCHS=30
BASELINE_OUT_ROOT="${BASELINE_OUT_ROOT:-$PROJECT_ROOT/outputs_baseline_t0_v2_${_safe_bb}_oct_only}"

SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
TOTAL_RUNS=$((${#HOSPITALS[@]} * ${#SEEDS[@]}))

is_run_complete() {
  local log_file="$1"
  [[ -f "$log_file" ]] && grep -q "训练完成！" "$log_file"
}

echo "======================================"
echo "T0 Baseline | OCT-only | ${CAPTURE_BACKBONE}"
echo "USE_CLINICAL=0 | LR=${CAPTURE_LR} | POS_WEIGHT=${CAPTURE_POS_WEIGHT}"
echo "BATCH_SIZE=4 | EPOCHS<=${MAX_EPOCHS} | early stop patience=10"
echo "WMA=0 | AUX=0 | EMA=0 | CORAL=0"
echo "折: ${HOSPITALS[*]}"
echo "Seeds: ${SEEDS[*]} | 共 ${TOTAL_RUNS} 次"
echo "Python: ${PYTHON}"
echo "输出根: ${BASELINE_OUT_ROOT}"
echo "跳过已完成: SKIP_COMPLETED=${SKIP_COMPLETED} | FORCE_RERUN=${FORCE_RERUN:-0}"
echo "对照 T1 单模态: outputs_baseline_t1_v2_resnet50_clinical_off/"
echo "======================================"

./run_experiment.sh "$PROJECT_ROOT/tsy_loho"
"$PYTHON" data/prepare_loho_data.py

RUN_IDX=0
for HOSP in "${HOSPITALS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    RUN_IDX=$((RUN_IDX + 1))
    echo "--------------------------------------"
    echo "T0 OCT-only [${RUN_IDX}/${TOTAL_RUNS}] | ${HOSP} | seed=${SEED}"
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
    unset HOSPITAL_NAME

    export OPTIGENESIS_BACKBONE="$CAPTURE_BACKBONE"
    export OPTIGENESIS_USE_CLINICAL=0
    export OPTIGENESIS_LR="$CAPTURE_LR"
    export OPTIGENESIS_POS_WEIGHT="$CAPTURE_POS_WEIGHT"
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
echo "T0 OCT-only baseline 全部完成。"
echo "结果根目录: ${BASELINE_OUT_ROOT}"
echo "======================================"

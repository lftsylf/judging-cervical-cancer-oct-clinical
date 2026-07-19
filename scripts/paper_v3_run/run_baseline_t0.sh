#!/usr/bin/env bash
# T0 Baseline（v2）：默认 ResNet50 骨干 + 3 折 × 5 seeds，最大 30 epoch（含 main.py early stopping）
# 旧稿 Swin-Tiny 复现：OPTIGENESIS_BACKBONE=swin_tiny_patch4_window7_224 ./run_baseline_t0.sh（输出目录名会随骨干变化）
# 轻量 CNN：OPTIGENESIS_BACKBONE=resnet18 ./run_baseline_t0.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# 在 unset 之前捕获骨干与输出根（便于：OPTIGENESIS_BACKBONE=resnet18 ./run_baseline_t0.sh）
CAPTURE_BACKBONE="${OPTIGENESIS_BACKBONE:-resnet50}"
_safe_bb="${CAPTURE_BACKBONE//[^a-zA-Z0-9_]/_}"

# ========== 可按需调整 ==========
HOSPITALS=(huaxi liaoning xiangya)
SEEDS=(42 123 2024 3407 114514)
MAX_EPOCHS=30
BASELINE_OUT_ROOT="${BASELINE_OUT_ROOT:-$PROJECT_ROOT/outputs_baseline_v2_${_safe_bb}}"
# ==============================

# 防止环境残留污染 baseline
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

# 显式锁定 baseline 关键开关（默认 CNN；覆盖方式见文件头注释）
export OPTIGENESIS_BACKBONE="$CAPTURE_BACKBONE"
export OPTIGENESIS_ENABLE_AUX=0
export OPTIGENESIS_ENABLE_EMA=0
export OPTIGENESIS_ENABLE_CORAL=0
export OPTIGENESIS_BATCH_SIZE=4
export OPTIGENESIS_USE_WMA=0

echo "======================================"
echo "T0 Baseline (v2 / ${OPTIGENESIS_BACKBONE}) 自动执行开始"
echo "Hospitals: ${HOSPITALS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Max epochs: ${MAX_EPOCHS} (含 Early Stopping)"
echo "输出根: ${BASELINE_OUT_ROOT}"
echo "======================================"

echo "1) 绑定 dataset -> tsy_loho"
"$PROJECT_ROOT/run_experiment.sh" "$PROJECT_ROOT/tsy_loho"

echo "2) 生成 LOHO development/external CSV"
python data/prepare_loho_data.py

for HOSP in "${HOSPITALS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    echo "--------------------------------------"
    echo "Starting ${HOSP} - Seed ${SEED} ..."
    echo "--------------------------------------"

    export HOSPITAL_NAME="$HOSP"
    export OPTIGENESIS_SEED="$SEED"
    export OPTIGENESIS_EPOCHS="$MAX_EPOCHS"
    export OPTIGENESIS_OUTPUT_DIR="${BASELINE_OUT_ROOT}/$HOSP"
    export OPTIGENESIS_OUTPUT_RUN_NAME="seed_${SEED}"

    LOG_DIR="$OPTIGENESIS_OUTPUT_DIR/$OPTIGENESIS_OUTPUT_RUN_NAME/logs"
    mkdir -p "$LOG_DIR"
    RUN_LOG="$LOG_DIR/train_console.log"

    # 每个组合单独日志，便于排查与续跑
    python main.py 2>&1 | tee "$RUN_LOG"
  done
done

echo "======================================"
echo "全部完成。结果根目录: ${BASELINE_OUT_ROOT}"
echo "======================================"

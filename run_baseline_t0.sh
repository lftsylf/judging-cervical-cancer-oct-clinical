#!/usr/bin/env bash
# T0 Baseline 自动化：3 折医院 × 5 seeds，最大 30 epoch（保留 main.py 里的 early stopping）
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# ========== 可按需调整 ==========
HOSPITALS=(huaxi liaoning xiangya)
SEEDS=(42 123 2024 3407 114514)
MAX_EPOCHS=30
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

# 显式锁定 baseline 关键开关
export OPTIGENESIS_BACKBONE="swin_tiny_patch4_window7_224"
export OPTIGENESIS_ENABLE_AUX=0
export OPTIGENESIS_ENABLE_EMA=0
export OPTIGENESIS_ENABLE_CORAL=0
export OPTIGENESIS_BATCH_SIZE=4
export OPTIGENESIS_USE_WMA=0

echo "======================================"
echo "T0 Baseline 自动执行开始"
echo "Hospitals: ${HOSPITALS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Max epochs: ${MAX_EPOCHS} (含 Early Stopping)"
echo "======================================"

echo "1) 绑定 dataset -> tsy_loho"
./run_experiment.sh "$PROJECT_ROOT/tsy_loho"

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
    export OPTIGENESIS_OUTPUT_DIR="$PROJECT_ROOT/outputs/$HOSP"
    export OPTIGENESIS_OUTPUT_RUN_NAME="seed_${SEED}"

    LOG_DIR="$OPTIGENESIS_OUTPUT_DIR/$OPTIGENESIS_OUTPUT_RUN_NAME/logs"
    mkdir -p "$LOG_DIR"
    RUN_LOG="$LOG_DIR/train_console.log"

    # 每个组合单独日志，便于排查与续跑
    python main.py 2>&1 | tee "$RUN_LOG"
  done
done

echo "======================================"
echo "全部完成。结果根目录: $PROJECT_ROOT/outputs"
echo "======================================"

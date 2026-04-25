#!/usr/bin/env bash
# Swin-Small T0 对比：标准多模态（Focal+EDL），关闭 WMA / EMA / AUX / CORAL。
# 3 折 × 5 seeds，30 epoch，batch=2。输出根目录：outputs_comparison_swin_small/
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

HOSPITALS=(huaxi liaoning xiangya)
SEEDS=(42 123 2024 3407 114514)
MAX_EPOCHS=30

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

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
unset OPTIGENESIS_BACKBONE
unset HOSPITAL_NAME

export OPTIGENESIS_BACKBONE="swin_small_patch4_window7_224"
export OPTIGENESIS_ENABLE_AUX=0
export OPTIGENESIS_ENABLE_EMA=0
export OPTIGENESIS_ENABLE_CORAL=0
export OPTIGENESIS_BATCH_SIZE=2
export OPTIGENESIS_USE_WMA=0

echo "======================================"
echo "T0 Swin-Small 对比 | backbone=${OPTIGENESIS_BACKBONE}"
echo "Hospitals: ${HOSPITALS[*]} | Seeds: ${SEEDS[*]} | epochs=${MAX_EPOCHS} | batch=${OPTIGENESIS_BATCH_SIZE}"
echo "WMA=0 EMA=0 AUX=0 CORAL=0"
echo "输出根: ${PROJECT_ROOT}/outputs_comparison_swin_small"
echo "======================================"

echo "1) 绑定 dataset -> tsy_loho"
./run_experiment.sh "$PROJECT_ROOT/tsy_loho"

echo "2) 生成 LOHO development/external CSV"
python data/prepare_loho_data.py

for HOSP in "${HOSPITALS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    echo "--------------------------------------"
    echo "Swin-Small | ${HOSP} | seed=${SEED}"
    echo "--------------------------------------"

    export HOSPITAL_NAME="$HOSP"
    export OPTIGENESIS_SEED="$SEED"
    export OPTIGENESIS_EPOCHS="$MAX_EPOCHS"
    export OPTIGENESIS_OUTPUT_DIR="$PROJECT_ROOT/outputs_comparison_swin_small/$HOSP"
    export OPTIGENESIS_OUTPUT_RUN_NAME="seed_${SEED}"

    LOG_DIR="$OPTIGENESIS_OUTPUT_DIR/$OPTIGENESIS_OUTPUT_RUN_NAME/logs"
    mkdir -p "$LOG_DIR"
    RUN_LOG="$LOG_DIR/train_console.log"

    python main.py 2>&1 | tee "$RUN_LOG"
  done
done

echo "======================================"
echo "Swin-Small 对比全部完成: ${PROJECT_ROOT}/outputs_comparison_swin_small"
echo "======================================"

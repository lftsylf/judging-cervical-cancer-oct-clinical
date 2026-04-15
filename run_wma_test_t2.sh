#!/usr/bin/env bash
# WMA Loss 单折单 Seed 探路：验证中心=liaoning，seed=42，最多 30 epoch（保留 main.py 早停）
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

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
unset OPTIGENESIS_WMA_C
unset OPTIGENESIS_WMA_WARMUP
unset OPTIGENESIS_WMA_TEMP
unset HOSPITAL_NAME

export OPTIGENESIS_BACKBONE="swin_tiny_patch4_window7_224"
export OPTIGENESIS_ENABLE_AUX=0
export OPTIGENESIS_ENABLE_EMA=0
export OPTIGENESIS_ENABLE_CORAL=0
export OPTIGENESIS_BATCH_SIZE=4

# 显式开启 WMA（与 configs 默认一致时可省略，此处便于复制运行）
export OPTIGENESIS_USE_WMA=1

export HOSPITAL_NAME="liaoning"
export OPTIGENESIS_SEED=42
export OPTIGENESIS_EPOCHS=30
export OPTIGENESIS_OUTPUT_DIR="$PROJECT_ROOT/outputs_wma_test/liaoning"
export OPTIGENESIS_OUTPUT_RUN_NAME="seed_42"

echo "======================================"
echo "WMA 探路 (t2): liaoning | seed=42 | max_epochs=30"
echo "输出: ${OPTIGENESIS_OUTPUT_DIR}/${OPTIGENESIS_OUTPUT_RUN_NAME}"
echo "======================================"

echo "1) 绑定 dataset -> tsy_loho"
./run_experiment.sh "$PROJECT_ROOT/tsy_loho"

echo "2) 生成 LOHO development/external CSV"
python data/prepare_loho_data.py

LOG_DIR="${OPTIGENESIS_OUTPUT_DIR}/${OPTIGENESIS_OUTPUT_RUN_NAME}/logs"
mkdir -p "$LOG_DIR"
RUN_LOG="${LOG_DIR}/train_console.log"

python main.py 2>&1 | tee "$RUN_LOG"

echo "======================================"
echo "完成。日志: $RUN_LOG"
echo "======================================"

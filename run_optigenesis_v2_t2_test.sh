#!/usr/bin/env bash
# 完全体 V2 单折探路：WMA + Clinical + EMA + Aux，关闭 CORAL（防 OOM）
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
unset OPTIGENESIS_USE_CLINICAL
unset OPTIGENESIS_WMA_C
unset OPTIGENESIS_WMA_WARMUP
unset OPTIGENESIS_WMA_TEMP
unset HOSPITAL_NAME

export OPTIGENESIS_BACKBONE="swin_tiny_patch4_window7_224"
export OPTIGENESIS_ENABLE_CORAL=0
export OPTIGENESIS_ENABLE_EMA=1
export OPTIGENESIS_ENABLE_AUX=1
export OPTIGENESIS_USE_WMA=1
export OPTIGENESIS_USE_CLINICAL=1
export OPTIGENESIS_BATCH_SIZE=2
export OPTIGENESIS_EPOCHS=30

export HOSPITAL_NAME="liaoning"
export OPTIGENESIS_SEED=42
export OPTIGENESIS_OUTPUT_DIR="$PROJECT_ROOT/outputs_v2_t2_test/liaoning"
export OPTIGENESIS_OUTPUT_RUN_NAME="seed_42"

echo "======================================"
echo "V2 探路 (t2): liaoning | seed=42 | max_epochs=30"
echo "# ── 探路模式: WMA=1 | Clinical=1 | EMA=1 | AUX=1 | CORAL=0 (防OOM终极版) ──"
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

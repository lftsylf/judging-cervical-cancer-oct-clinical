#!/usr/bin/env bash
# 华西单折 baseline 复现（与 run_loho_baseline_repro_newckpt.sh 同超参），
# 终端 stdout/stderr 同时写入 OPTIGENESIS_OUTPUT_DIR/<RUN_NAME>/logs/train_console.log
#
# 用法:
#   ./run_huaxi_baseline_repro_with_log.sh                    # 前台 + 默认 huaxi3
#   RUN_NAME=huaxi4 ./run_huaxi_baseline_repro_with_log.sh
#   BACKGROUND=1 ./run_huaxi_baseline_repro_with_log.sh     # 后台，仅日志文件持续追加
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

RUN_NAME="${RUN_NAME:-huaxi3}"

unset OPTIGENESIS_BACKBONE
unset OPTIGENESIS_ENABLE_AUX
unset OPTIGENESIS_ENABLE_EMA
unset OPTIGENESIS_ENABLE_CORAL
unset OPTIGENESIS_OUTPUT_DIR
unset OPTIGENESIS_OUTPUT_RUN_NAME
unset OPTIGENESIS_BATCH_SIZE
unset OPTIGENESIS_CORAL_LAMBDA
unset OPTIGENESIS_CORAL_WARMUP
unset HOSPITAL_NAME

export OPTIGENESIS_BACKBONE="swin_tiny_patch4_window7_224"
export OPTIGENESIS_ENABLE_AUX=0
export OPTIGENESIS_ENABLE_EMA=0
export OPTIGENESIS_ENABLE_CORAL=0
export OPTIGENESIS_BATCH_SIZE=4
export OPTIGENESIS_OUTPUT_DIR="$PROJECT_ROOT/baseline_repro_newckpt_outputs"
export OPTIGENESIS_OUTPUT_RUN_NAME="$RUN_NAME"
export HOSPITAL_NAME="huaxi"

LOG_DIR="$OPTIGENESIS_OUTPUT_DIR/$RUN_NAME/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_console.log"

echo "======================================"
echo "华西单折 baseline | 输出子目录: $RUN_NAME"
echo "控制台日志: $LOG_FILE"
echo "======================================"

run_tee() {
  python main.py 2>&1 | tee "$LOG_FILE"
}

if [[ "${BACKGROUND:-0}" == "1" ]]; then
  run_tee &
  echo "已在后台启动，PID=$!；查看进度: tail -f $LOG_FILE"
else
  run_tee
fi

#!/usr/bin/env bash
# WMA 正式全程：3 中心 × 5 seeds，最多 30 epoch（main.py 验证集 AUC Early Stopping，patience=10 不变）
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# ========== 可按需调整 ==========
HOSPITALS=(huaxi liaoning xiangya)
SEEDS=(42 123 2024 3407 114514)
MAX_EPOCHS=30
# ==============================

TOTAL_RUNS=$((${#HOSPITALS[@]} * ${#SEEDS[@]}))
RUN_IDX=0

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
export OPTIGENESIS_USE_WMA=1
export OPTIGENESIS_EPOCHS="$MAX_EPOCHS"

echo "======================================"
echo "WMA 正式全程 (T0) 开始"
echo "  医院: ${HOSPITALS[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo "  总运行数: ${TOTAL_RUNS} (${#HOSPITALS[@]} × ${#SEEDS[@]})"
echo "  OPTIGENESIS_EPOCHS=${MAX_EPOCHS}（Early Stopping patience=10，见 main.py）"
echo "  OPTIGENESIS_USE_WMA=1"
echo "  输出根目录: ${PROJECT_ROOT}/outputs_wma_full/<hospital>/seed_<seed>/"
echo "======================================"

echo ""
echo "[预处理1/2] 绑定 dataset -> tsy_loho"
./run_experiment.sh "$PROJECT_ROOT/tsy_loho"

echo ""
echo "[预处理 2/2] 生成 LOHO development/external CSV"
python data/prepare_loho_data.py

HOSP_IDX=0
for HOSP in "${HOSPITALS[@]}"; do
  HOSP_IDX=$((HOSP_IDX + 1))
  SEED_IDX=0
  for SEED in "${SEEDS[@]}"; do
    SEED_IDX=$((SEED_IDX + 1))
    RUN_IDX=$((RUN_IDX + 1))

    echo ""
    echo "##############################################"
    echo "# 进度: 全局 ${RUN_IDX}/${TOTAL_RUNS}"
    echo "# 医院 ${HOSP} (${HOSP_IDX}/${#HOSPITALS[@]})"
    echo "#       Seed ${SEED}  (${SEED_IDX}/${#SEEDS[@]})"
    echo "#即将启动: python main.py"
    echo "# 输出: outputs_wma_full/${HOSP}/seed_${SEED}/"
    echo "##############################################"

    export HOSPITAL_NAME="$HOSP"
    export OPTIGENESIS_SEED="$SEED"
    export OPTIGENESIS_OUTPUT_DIR="$PROJECT_ROOT/outputs_wma_full/${HOSP}"
    export OPTIGENESIS_OUTPUT_RUN_NAME="seed_${SEED}"

    LOG_DIR="${OPTIGENESIS_OUTPUT_DIR}/${OPTIGENESIS_OUTPUT_RUN_NAME}/logs"
    mkdir -p "$LOG_DIR"
    RUN_LOG="${LOG_DIR}/train_console.log"

    python main.py 2>&1 | tee "$RUN_LOG"

    echo ""
    echo ">>> 完成 ${HOSP} / seed ${SEED}，日志: ${RUN_LOG}"
  done
done

echo ""
echo "======================================"
echo "WMA 正式全程全部结束。"
echo "结果根目录: ${PROJECT_ROOT}/outputs_wma_full"
echo "======================================"

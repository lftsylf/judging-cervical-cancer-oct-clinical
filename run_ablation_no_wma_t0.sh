#!/usr/bin/env bash
# Ablation：关闭 WMA，保留 EMA + Aux；Clinical=1，CORAL=0
# 3 中心 × 5 seeds，30 epoch，batch=2
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

HOSPITALS=(huaxi liaoning xiangya)
SEEDS=(42 123 2024 3407 114514)
MAX_EPOCHS=30

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
unset OPTIGENESIS_USE_CLINICAL
unset OPTIGENESIS_WMA_C
unset OPTIGENESIS_WMA_WARMUP
unset OPTIGENESIS_WMA_TEMP
unset HOSPITAL_NAME

export OPTIGENESIS_BACKBONE="swin_tiny_patch4_window7_224"
export OPTIGENESIS_ENABLE_CORAL=0
export OPTIGENESIS_USE_WMA=0
export OPTIGENESIS_ENABLE_EMA=1
export OPTIGENESIS_ENABLE_AUX=1
export OPTIGENESIS_USE_CLINICAL=1
export OPTIGENESIS_BATCH_SIZE=2
export OPTIGENESIS_EPOCHS="$MAX_EPOCHS"

echo "======================================"
echo "Ablation: NO WMA (EMA+Aux) | CORAL=0 | Clinical=1"
echo "  医院: ${HOSPITALS[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo "  总运行数: ${TOTAL_RUNS}"
echo "  USE_WMA=0 | ENABLE_EMA=1 | ENABLE_AUX=1 | BATCH=${OPTIGENESIS_BATCH_SIZE} | EPOCHS=${MAX_EPOCHS}"
echo "  输出根: ${PROJECT_ROOT}/outputs_ablation_no_wma/"
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
    echo "# 医院 ${HOSP} (${HOSP_IDX}/${#HOSPITALS[@]}) | Seed ${SEED} (${SEED_IDX}/${#SEEDS[@]})"
    echo "# 模式: WMA=0 | EMA=1 | AUX=1 | Clinical=1 | CORAL=0"
    echo "# 输出: outputs_ablation_no_wma/${HOSP}/seed_${SEED}/"
    echo "##############################################"

    export HOSPITAL_NAME="$HOSP"
    export OPTIGENESIS_SEED="$SEED"
    export OPTIGENESIS_OUTPUT_DIR="$PROJECT_ROOT/outputs_ablation_no_wma/${HOSP}"
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
echo "Ablation NO WMA 全部结束: ${PROJECT_ROOT}/outputs_ablation_no_wma"
echo "======================================"

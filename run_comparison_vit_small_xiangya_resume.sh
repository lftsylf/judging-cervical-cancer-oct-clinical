#!/usr/bin/env bash
# 续跑 ViT-Small T0 对比：仅湘雅折 × 三 seed（2024 / 3407 / 114514）。
# 与 run_comparison_vit_small_t0.sh 同配置；用于主流程中断后补跑。
#
# 建议：若 seed_2024 曾中断，可先删除不完整目录再跑，避免与旧 checkpoint 混淆：
#   rm -rf /path/to/OptiGenesis_Lancet/outputs_comparison_vit_small/xiangya/seed_2024
#
# 用法（项目根目录）：
#   chmod +x run_comparison_vit_small_xiangya_resume.sh
#   ./run_comparison_vit_small_xiangya_resume.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

HOSP="xiangya"
SEEDS=(2024 3407 114514)
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

export OPTIGENESIS_BACKBONE="vit_small_patch16_224"
export OPTIGENESIS_ENABLE_AUX=0
export OPTIGENESIS_ENABLE_EMA=0
export OPTIGENESIS_ENABLE_CORAL=0
export OPTIGENESIS_BATCH_SIZE=2
export OPTIGENESIS_USE_WMA=0

echo "======================================"
echo "ViT-Small 续跑 | 医院=${HOSP} | seeds=${SEEDS[*]} | epochs=${MAX_EPOCHS} | batch=${OPTIGENESIS_BATCH_SIZE}"
echo "输出: ${PROJECT_ROOT}/outputs_comparison_vit_small/${HOSP}/seed_<SEED>/"
echo "======================================"

echo "1) 绑定 dataset -> tsy_loho"
./run_experiment.sh "$PROJECT_ROOT/tsy_loho"

echo "2) 生成 LOHO development/external CSV"
python data/prepare_loho_data.py

for SEED in "${SEEDS[@]}"; do
  echo "--------------------------------------"
  echo "ViT-Small | ${HOSP} | seed=${SEED}"
  echo "--------------------------------------"

  export HOSPITAL_NAME="$HOSP"
  export OPTIGENESIS_SEED="$SEED"
  export OPTIGENESIS_EPOCHS="$MAX_EPOCHS"
  export OPTIGENESIS_OUTPUT_DIR="$PROJECT_ROOT/outputs_comparison_vit_small/$HOSP"
  export OPTIGENESIS_OUTPUT_RUN_NAME="seed_${SEED}"

  LOG_DIR="$OPTIGENESIS_OUTPUT_DIR/$OPTIGENESIS_OUTPUT_RUN_NAME/logs"
  mkdir -p "$LOG_DIR"
  RUN_LOG="$LOG_DIR/train_console.log"

  python main.py 2>&1 | tee "$RUN_LOG"
done

echo "======================================"
echo "湘雅三折 seed 续跑结束。"
echo "======================================"

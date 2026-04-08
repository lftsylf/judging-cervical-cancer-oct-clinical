#!/bin/bash
# ViT-Base 经典骨干 LOHO 三中心对比（与 ViT-Small 脚本相同设定：EDL+Focal、无 aux、无 EMA）。
# 输出：vit_loho_outputs/vit_base_patch16_224/{liaoning,huaxi,xiangya}/
#
# 显存紧张时可在下方取消注释，将 batch 降为 2：
#   export OPTIGENESIS_BATCH_SIZE=2
#
# 用法：在项目根目录执行 ./run_loho_vit_base.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

export OPTIGENESIS_ENABLE_AUX=0
export OPTIGENESIS_ENABLE_EMA=0
# export OPTIGENESIS_BATCH_SIZE=2

echo "1) 绑定 dataset -> tsy_loho"
./run_experiment.sh "$PROJECT_ROOT/tsy_loho"

echo "2) 生成 LOHO development/external CSV"
python data/prepare_loho_data.py

export OPTIGENESIS_BACKBONE="vit_base_patch16_224"
export OPTIGENESIS_OUTPUT_DIR="$PROJECT_ROOT/vit_loho_outputs/${OPTIGENESIS_BACKBONE}"
echo "======================================"
echo "骨干: ${OPTIGENESIS_BACKBONE}  |  输出: ${OPTIGENESIS_OUTPUT_DIR}"
echo "BATCH_SIZE=${OPTIGENESIS_BATCH_SIZE:-4} (可用 OPTIGENESIS_BATCH_SIZE 覆盖)"
echo "======================================"
for HOSP in liaoning huaxi xiangya; do
  echo "---------- LOHO fold: ${HOSP} ----------"
  HOSPITAL_NAME="$HOSP" python main.py
done

echo "全部完成。结果目录: $PROJECT_ROOT/vit_loho_outputs/vit_base_patch16_224/"

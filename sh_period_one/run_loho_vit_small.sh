#!/bin/bash
# ViT 经典骨干 LOHO 三中心对比：与 baseline 相同训练设定（EDL+Focal、无 aux、无 EMA）。
# 用法：在项目根目录执行 ./run_loho_vit.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

export OPTIGENESIS_ENABLE_AUX=0
export OPTIGENESIS_ENABLE_EMA=0

echo "1) 绑定 dataset -> tsy_loho"
./run_experiment.sh "$PROJECT_ROOT/tsy_loho"

echo "2) 生成 LOHO development/external CSV"
python data/prepare_loho_data.py

for BACKBONE in vit_small_patch16_224; do
  export OPTIGENESIS_BACKBONE="$BACKBONE"
  export OPTIGENESIS_OUTPUT_DIR="$PROJECT_ROOT/vit_loho_outputs/${BACKBONE}"
  echo "======================================"
  echo "骨干: ${BACKBONE}  |  输出: ${OPTIGENESIS_OUTPUT_DIR}"
  echo "======================================"
  for HOSP in liaoning huaxi xiangya; do
    echo "---------- LOHO fold: ${HOSP} ----------"
    HOSPITAL_NAME="$HOSP" python main.py
  done
done

echo "全部完成。结果目录: $PROJECT_ROOT/vit_loho_outputs/vit_small_patch16_224/"

#!/bin/bash
# 仅重跑 baseline 的 HuaXi / XiangYa 两折（AUC-only best 保存规则）
# 目的：与已跑好的第二版 Liaoning（AUC-only）保持选模口径一致。
#
# 用法：在项目根目录执行 ./run_loho_baseline_auc_hx_xy.sh
# 可选：显存不足时设置 OPTIGENESIS_BATCH_SIZE=2
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# baseline 设定：Swin + 无 aux + 无 EMA
export OPTIGENESIS_BACKBONE="swin_tiny_patch4_window7_224"
export OPTIGENESIS_ENABLE_AUX=0
export OPTIGENESIS_ENABLE_EMA=0

# 输出到独立目录，避免覆盖历史 baseline
export OPTIGENESIS_OUTPUT_DIR="$PROJECT_ROOT/baseline_auc_only_hx_xy_outputs"

echo "1) 绑定 dataset -> tsy_loho"
./run_experiment.sh "$PROJECT_ROOT/tsy_loho"

echo "2) 生成 LOHO development/external CSV"
python data/prepare_loho_data.py

echo "======================================"
echo "输出目录: ${OPTIGENESIS_OUTPUT_DIR}"
echo "骨干: ${OPTIGENESIS_BACKBONE}"
echo "AUX=${OPTIGENESIS_ENABLE_AUX}, EMA=${OPTIGENESIS_ENABLE_EMA}, BATCH=${OPTIGENESIS_BATCH_SIZE:-4}"
echo "======================================"

for HOSP in huaxi xiangya; do
  echo "---------- baseline AUC-only fold: ${HOSP} ----------"
  HOSPITAL_NAME="$HOSP" python main.py
done

echo "全部完成。结果目录: $PROJECT_ROOT/baseline_auc_only_hx_xy_outputs/{huaxi,xiangya}/"

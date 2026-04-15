#!/bin/bash
# 多模态辅助监督（aux）三中心 LOHO，50 epoch，与 baseline 完整消融同预算；EMA 关闭。
# 输出：$PROJECT_ROOT/aux_50_outputs/{liaoning,huaxi,xiangya}/

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

export OPTIGENESIS_ENABLE_AUX=1
export OPTIGENESIS_ENABLE_EMA=0
export OPTIGENESIS_ENABLE_CORAL=0
export OPTIGENESIS_OUTPUT_DIR="${PROJECT_ROOT}/aux_50_outputs"
# 与 Swin baseline 消融对齐，避免继承终端里残留的 ViT / 小 batch
export OPTIGENESIS_BACKBONE="swin_tiny_patch4_window7_224"
# 若 GPU OOM，请在运行本脚本前：export OPTIGENESIS_BATCH_SIZE=2（此处勿 unset，以免覆盖你的设置）

echo "【aux 50 轮 LOHO】ENABLE_AUX=1, ENABLE_EMA=0, CORAL=0, OUTPUT_DIR=${OPTIGENESIS_OUTPUT_DIR}"

if [[ "${SKIP_LOHO_PREP:-0}" != "1" ]]; then
  echo "1) 绑定 dataset -> tsy_loho"
  ./run_experiment.sh "$PROJECT_ROOT/tsy_loho"
  echo "2) 生成 LOHO development/external CSV"
  python data/prepare_loho_data.py
else
  echo "跳过数据准备 (SKIP_LOHO_PREP=1)"
fi

for HOSP in liaoning huaxi xiangya; do
  echo "======================================"
  echo "开始训练 LOHO fold (aux): ${HOSP}"
  echo "======================================"
  HOSPITAL_NAME="$HOSP" python main.py
done

echo "全部完成。输出目录: ${OPTIGENESIS_OUTPUT_DIR}/{liaoning,huaxi,xiangya}/"

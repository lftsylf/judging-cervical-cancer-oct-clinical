#!/bin/bash
# CORAL 无监督域对齐（最小 UDA）三中心 LOHO，50 epoch；aux/EMA 关闭，与 Swin baseline 同预算。
# 输出：$PROJECT_ROOT/uda_50_outputs/{liaoning,huaxi,xiangya}/

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

export OPTIGENESIS_ENABLE_AUX=0
export OPTIGENESIS_ENABLE_EMA=0
export OPTIGENESIS_ENABLE_CORAL=1
export OPTIGENESIS_OUTPUT_DIR="${PROJECT_ROOT}/uda_50_outputs"
# 默认：λ_max=0.02，warmup=8（见 configs/lancet_config.py）；更保守可调低：OPTIGENESIS_CORAL_LAMBDA=0.01
export OPTIGENESIS_CORAL_LAMBDA="${OPTIGENESIS_CORAL_LAMBDA:-0.02}"
export OPTIGENESIS_CORAL_WARMUP="${OPTIGENESIS_CORAL_WARMUP:-8}"
export OPTIGENESIS_BACKBONE="swin_tiny_patch4_window7_224"
# 若 GPU OOM，请在运行前：export OPTIGENESIS_BATCH_SIZE=2

echo "【CORAL/UDA 50 轮】λ_max=${OPTIGENESIS_CORAL_LAMBDA}, warmup=${OPTIGENESIS_CORAL_WARMUP}, OUTPUT_DIR=${OPTIGENESIS_OUTPUT_DIR}"

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
  echo "开始训练 LOHO fold (CORAL): ${HOSP}"
  echo "======================================"
  HOSPITAL_NAME="$HOSP" python main.py
done

echo "全部完成。输出目录: ${OPTIGENESIS_OUTPUT_DIR}/{liaoning,huaxi,xiangya}/"

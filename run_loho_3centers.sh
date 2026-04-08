#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo "1) 绑定 dataset -> tsy_loho"
./run_experiment.sh "$PROJECT_ROOT/tsy_loho"

echo "2) 生成 LOHO development/external CSV"
python data/prepare_loho_data.py

for HOSP in liaoning huaxi xiangya; do
  echo "======================================"
  echo "开始训练 LOHO fold: ${HOSP}"
  echo "======================================"
  HOSPITAL_NAME="$HOSP" python main.py
done

echo "全部完成。输出目录: $PROJECT_ROOT/outputs/{liaoning,huaxi,xiangya}/"

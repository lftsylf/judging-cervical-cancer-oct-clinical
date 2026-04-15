#!/bin/bash
# 过夜串行：先 aux 三中心 50 轮，再 CORAL/UDA 三中心 50 轮（共 6 次 main.py）。
# 仅第一次做数据绑定与 CSV 生成，第二次复用（SKIP_LOHO_PREP=1）。
#
# 用法：
#   cd /ssd_data/tsy_study_venv/OptiGenesis_Lancet
#   nohup bash run_loho_aux_then_uda_overnight.sh > overnight_aux_uda.log 2>&1 &

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

export SKIP_LOHO_PREP=0
bash "${PROJECT_ROOT}/run_loho_aux_50.sh"

export SKIP_LOHO_PREP=1
bash "${PROJECT_ROOT}/run_loho_uda_50.sh"

echo "过夜任务全部结束：aux_50_outputs + uda_50_outputs"

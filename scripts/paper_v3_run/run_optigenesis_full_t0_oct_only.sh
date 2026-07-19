#!/usr/bin/env bash
# T0 完全体（v2 单模态 · OCT-only）：WMA + EMA + Aux，CORAL=0
#
# 与定稿 baseline 相同骨干/LR/POS/BATCH/折×seed；仅开启 WMA+EMA+Aux。
#
# 前台：
#   ./run_optigenesis_full_t0_oct_only.sh
#
# SSH 断线仍继续（推荐）：
#   export PATH="/home/amax/anaconda3/bin:$PATH"
#   cd /ssd_data/tsy_study_venv/OptiGenesis_Lancet
#   "$PROJECT_ROOT/run_experiment.sh" --detach \
#     /ssd_data/tsy_study_venv/OptiGenesis_Lancet/tsy_loho \
#     ./run_optigenesis_full_t0_oct_only.sh
#
# 续跑：
#   SKIP_COMPLETED=1 ./run_optigenesis_full_t0_oct_only.sh
# 强制重跑：
#   FORCE_RERUN=1 ./run_optigenesis_full_t0_oct_only.sh
#
# 输出：outputs/第三版/消融实验v2单模态/outputs_ablation_v2_oct_full/
set -euo pipefail

ABLATION_TAG=full
ABLATION_USE_WMA=1
ABLATION_ENABLE_EMA=1
ABLATION_ENABLE_AUX=1

# shellcheck source=run_ablation_t0_oct_only_lib.sh
source "$(cd "$(dirname "$0")" && pwd)/run_ablation_t0_oct_only_lib.sh"

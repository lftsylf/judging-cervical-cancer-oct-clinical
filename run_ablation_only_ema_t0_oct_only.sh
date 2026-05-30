#!/usr/bin/env bash
# T0 单模块消融 · 仅 EMA（v2 单模态 · OCT-only）：WMA=0，EMA=1，Aux=0，CORAL=0
#
# 前台：./run_ablation_only_ema_t0_oct_only.sh
# Detach（SSH 断线仍继续）：
#   export PATH="/home/amax/anaconda3/bin:$PATH"
#   cd /ssd_data/tsy_study_venv/OptiGenesis_Lancet
#   ./run_experiment.sh --detach \
#     /ssd_data/tsy_study_venv/OptiGenesis_Lancet/tsy_loho \
#     ./run_ablation_only_ema_t0_oct_only.sh
#
# 续跑：SKIP_COMPLETED=1 ./run_ablation_only_ema_t0_oct_only.sh
# 强制重跑：FORCE_RERUN=1 ./run_ablation_only_ema_t0_oct_only.sh
#
# 输出：outputs/消融实验v2单模态/outputs_ablation_v2_oct_only_ema/
set -euo pipefail

ABLATION_TAG=only_ema
ABLATION_USE_WMA=0
ABLATION_ENABLE_EMA=1
ABLATION_ENABLE_AUX=0

source "$(cd "$(dirname "$0")" && pwd)/run_ablation_t0_oct_only_lib.sh"

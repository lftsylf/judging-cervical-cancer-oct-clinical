#!/usr/bin/env bash
# T0 消融 -EMA（v2 单模态 · OCT-only）：WMA + Aux，EMA=0，CORAL=0
#
# 前台：./run_ablation_no_ema_t0_oct_only.sh
# Detach：
#   ./run_experiment.sh --detach /path/to/tsy_loho ./run_ablation_no_ema_t0_oct_only.sh
# 续跑：SKIP_COMPLETED=1 ./run_ablation_no_ema_t0_oct_only.sh
#
# 输出：outputs/消融实验v2单模态/outputs_ablation_v2_oct_no_ema/
set -euo pipefail

ABLATION_TAG=no_ema
ABLATION_USE_WMA=1
ABLATION_ENABLE_EMA=0
ABLATION_ENABLE_AUX=1

source "$(cd "$(dirname "$0")" && pwd)/run_ablation_t0_oct_only_lib.sh"

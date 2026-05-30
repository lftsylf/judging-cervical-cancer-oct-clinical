#!/usr/bin/env bash
# T0 ResNet18 对比（v2 单模态 · OCT-only）：plain Focal+EDL，无 WMA/EMA/Aux
# 轻量 CNN 对照：小样本下容量更匹配，替代 ViT-Base（过大、易过拟合）。
#
# 前台：
#   ./run_comparison_resnet18_t0_oct_only.sh
#
# SSH 断线仍继续（推荐）：
#   export PATH="/home/amax/anaconda3/bin:$PATH"
#   cd /ssd_data/tsy_study_venv/OptiGenesis_Lancet
#   ./run_experiment.sh --detach \
#     /ssd_data/tsy_study_venv/OptiGenesis_Lancet/tsy_loho \
#     ./run_comparison_resnet18_t0_oct_only.sh
#
# 断线后看总日志：
#   tail -f logs/detached_latest.log
#
# 续跑：
#   SKIP_COMPLETED=1 ./run_comparison_resnet18_t0_oct_only.sh
#
# 输出：outputs/对比试验v2单模态/outputs_comparison_v2_oct_resnet18/
set -euo pipefail

COMPARISON_TAG=resnet18
COMPARISON_BACKBONE=resnet18
COMPARISON_BATCH_SIZE=4

# shellcheck source=run_comparison_t0_oct_only_lib.sh
source "$(cd "$(dirname "$0")" && pwd)/run_comparison_t0_oct_only_lib.sh"

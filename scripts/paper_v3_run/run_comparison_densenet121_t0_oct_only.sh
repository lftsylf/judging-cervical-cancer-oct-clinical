#!/usr/bin/env bash
# T0 DenseNet121 对比（v2 单模态 · OCT-only）：plain Focal+EDL，无 WMA/EMA/Aux
# 经典 CNN 对照（2017，医学影像常用），与 ResNet18/50 形成老一代 CNN 谱系。
#
# 前台：
#   ./run_comparison_densenet121_t0_oct_only.sh
#
# SSH 断线仍继续（推荐）：
#   export PATH="/home/amax/anaconda3/bin:$PATH"
#   cd /ssd_data/tsy_study_venv/OptiGenesis_Lancet
#   "$PROJECT_ROOT/run_experiment.sh" --detach \
#     /ssd_data/tsy_study_venv/OptiGenesis_Lancet/tsy_loho \
#     ./run_comparison_densenet121_t0_oct_only.sh
#
# 续跑：
#   SKIP_COMPLETED=1 ./run_comparison_densenet121_t0_oct_only.sh
#
# batch=4（与 ResNet18/50 baseline 一致，8GB 实测无 OOM）
# 输出：outputs/第三版/对比试验v2单模态/outputs_comparison_v2_oct_densenet121/
set -euo pipefail

COMPARISON_TAG=densenet121
COMPARISON_BACKBONE=densenet121
COMPARISON_BATCH_SIZE=4

# shellcheck source=run_comparison_t0_oct_only_lib.sh
source "$(cd "$(dirname "$0")" && pwd)/run_comparison_t0_oct_only_lib.sh"

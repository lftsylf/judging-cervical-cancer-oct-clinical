#!/usr/bin/env bash
# T0 ConvNeXt-Small 对比（v2 单模态 · OCT-only）：plain Focal+EDL，无 WMA/EMA/Aux
#
# 前台：
#   ./run_comparison_convnext_small_t0_oct_only.sh
#
# SSH 断线仍继续（推荐）：
#   export PATH="/home/amax/anaconda3/bin:$PATH"
#   cd /ssd_data/tsy_study_venv/OptiGenesis_Lancet
#   "$PROJECT_ROOT/run_experiment.sh" --detach \
#     /ssd_data/tsy_study_venv/OptiGenesis_Lancet/tsy_loho \
#     ./run_comparison_convnext_small_t0_oct_only.sh
#
# 断线后看总日志：
#   tail -f logs/detached_latest.log
#
# 续跑：
#   SKIP_COMPLETED=1 ./run_comparison_convnext_small_t0_oct_only.sh
#
# OOM：8GB 卡 batch=4 易 OOM，与 Swin 对齐为 batch=2
# 输出：outputs/第三版/对比试验v2单模态/outputs_comparison_v2_oct_convnext_small/
set -euo pipefail

COMPARISON_TAG=convnext_small
COMPARISON_BACKBONE=convnext_small
COMPARISON_BATCH_SIZE=2
COMPARISON_CUDA_ALLOC="max_split_size_mb:128"

# shellcheck source=run_comparison_t0_oct_only_lib.sh
source "$(cd "$(dirname "$0")" && pwd)/run_comparison_t0_oct_only_lib.sh"

#!/usr/bin/env bash
# T0 Swin-Small 对比（v2 单模态 · OCT-only）：plain Focal+EDL，无 WMA/EMA/Aux
# 旧多模态对比试验中的 Swin 骨干，v2 OCT-only 重跑。
#
# 前台：
#   ./run_comparison_swin_small_t0_oct_only.sh
#
# SSH 断线仍继续（推荐）：
#   export PATH="/home/amax/anaconda3/bin:$PATH"
#   cd /ssd_data/tsy_study_venv/OptiGenesis_Lancet
#   ./run_experiment.sh --detach \
#     /ssd_data/tsy_study_venv/OptiGenesis_Lancet/tsy_loho \
#     ./run_comparison_swin_small_t0_oct_only.sh
#
# 断线后看总日志：
#   tail -f logs/detached_latest.log
#
# 续跑：
#   SKIP_COMPLETED=1 ./run_comparison_swin_small_t0_oct_only.sh
#
# OOM：Swin 使用 batch=2 + CUDA 分配优化
# 输出：outputs/对比试验v2单模态/outputs_comparison_v2_oct_swin_small/
set -euo pipefail

COMPARISON_TAG=swin_small
COMPARISON_BACKBONE=swin_small_patch4_window7_224
COMPARISON_BATCH_SIZE=2
COMPARISON_CUDA_ALLOC="max_split_size_mb:128"

# shellcheck source=run_comparison_t0_oct_only_lib.sh
source "$(cd "$(dirname "$0")" && pwd)/run_comparison_t0_oct_only_lib.sh"

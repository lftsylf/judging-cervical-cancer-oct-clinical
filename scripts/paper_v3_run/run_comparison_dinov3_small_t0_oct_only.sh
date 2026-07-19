#!/usr/bin/env bash
# T0 DINOv3-Small 对比（v2 单模态 · OCT-only）：plain Focal+EDL，无 WMA/EMA/Aux
# 通用视觉基础模型对照（Meta DINOv3，timm 预训练）。
#
# 前台：
#   ./run_comparison_dinov3_small_t0_oct_only.sh
#
# SSH 断线仍继续（推荐）：
#   export PATH="/home/amax/anaconda3/bin:$PATH"
#   cd /ssd_data/tsy_study_venv/OptiGenesis_Lancet
#   "$PROJECT_ROOT/run_experiment.sh" --detach \
#     /ssd_data/tsy_study_venv/OptiGenesis_Lancet/tsy_loho \
#     ./run_comparison_dinov3_small_t0_oct_only.sh
#
# 续跑：
#   SKIP_COMPLETED=1 ./run_comparison_dinov3_small_t0_oct_only.sh
#
# OOM：8GB 卡与 ViT-S 对齐 batch=2
# 输出：outputs/第三版/对比试验v2单模态/outputs_comparison_v2_oct_dinov3_small/
set -euo pipefail

COMPARISON_TAG=dinov3_small
COMPARISON_BACKBONE=vit_small_patch16_dinov3
COMPARISON_BATCH_SIZE=2
COMPARISON_CUDA_ALLOC="max_split_size_mb:128"

# shellcheck source=run_comparison_t0_oct_only_lib.sh
source "$(cd "$(dirname "$0")" && pwd)/run_comparison_t0_oct_only_lib.sh"

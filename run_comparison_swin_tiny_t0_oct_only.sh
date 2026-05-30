#!/usr/bin/env bash
# T0 Swin-Tiny 对比（v2 单模态 · OCT-only）：plain Focal+EDL，无 WMA/EMA/Aux
# v1 历史主骨干（paper-v1-swin-baseline），与定稿 ResNet50 baseline 同协议，仅改 backbone。
#
# 前台：
#   ./run_comparison_swin_tiny_t0_oct_only.sh
#
# SSH 断线仍继续（推荐）：
#   export PATH="/home/amax/anaconda3/bin:$PATH"
#   cd /ssd_data/tsy_study_venv/OptiGenesis_Lancet
#   ./run_experiment.sh --detach \
#     /ssd_data/tsy_study_venv/OptiGenesis_Lancet/tsy_loho \
#     ./run_comparison_swin_tiny_t0_oct_only.sh
#
# 断线后看总日志：
#   tail -f logs/detached_latest.log
#
# 续跑（跳过已完成）：
#   SKIP_COMPLETED=1 ./run_comparison_swin_tiny_t0_oct_only.sh
#
# OOM：Swin 使用 batch=2 + CUDA 分配优化（其余超参与 baseline 一致）
# 输出：outputs/对比试验v2单模态/outputs_comparison_v2_oct_swin_tiny/
set -euo pipefail

COMPARISON_TAG=swin_tiny
COMPARISON_BACKBONE=swin_tiny_patch4_window7_224
COMPARISON_BATCH_SIZE=2
COMPARISON_CUDA_ALLOC="max_split_size_mb:128"

# shellcheck source=run_comparison_t0_oct_only_lib.sh
source "$(cd "$(dirname "$0")" && pwd)/run_comparison_t0_oct_only_lib.sh"

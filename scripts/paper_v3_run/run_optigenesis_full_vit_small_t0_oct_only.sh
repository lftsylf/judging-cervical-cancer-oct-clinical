#!/usr/bin/env bash
# T0 OptiGenesis Full（v2 单模态 · OCT-only · ViT-Small）
#
# 与 ViT-S plain 对比试验同骨干/batch/LR/POS/折×seed；开启 WMA+EMA+Aux。
# 用于验证 Full 模块是否能在更强 plain backbone 上继续增益。
#
# 前台：
#   ./run_optigenesis_full_vit_small_t0_oct_only.sh
#
# SSH 断线仍继续（推荐）：
#   export PATH="/home/amax/anaconda3/bin:$PATH"
#   cd /ssd_data/tsy_study_venv/OptiGenesis_Lancet
#   "$PROJECT_ROOT/run_experiment.sh" --detach \
#     /ssd_data/tsy_study_venv/OptiGenesis_Lancet/tsy_loho \
#     ./run_optigenesis_full_vit_small_t0_oct_only.sh
#
# 续跑：
#   SKIP_COMPLETED=1 ./run_optigenesis_full_vit_small_t0_oct_only.sh
# 强制重跑：
#   FORCE_RERUN=1 ./run_optigenesis_full_vit_small_t0_oct_only.sh
#
# 输出：outputs/第三版/对比试验v2单模态/outputs_full_v2_oct_vit_small/
set -euo pipefail

export OPTIGENESIS_BACKBONE=vit_small_patch16_224
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

ABLATION_TAG=full_vit_small
ABLATION_BATCH_SIZE=2
ABLATION_USE_WMA=1
ABLATION_ENABLE_EMA=1
ABLATION_ENABLE_AUX=1
ABLATION_OUT_ROOT="${ABLATION_OUT_ROOT:-$PROJECT_ROOT/outputs/第三版/对比试验v2单模态/outputs_full_v2_oct_vit_small}"

# shellcheck source=run_ablation_t0_oct_only_lib.sh
source "$(cd "$(dirname "$0")" && pwd)/run_ablation_t0_oct_only_lib.sh"

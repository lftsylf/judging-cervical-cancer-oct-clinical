#!/usr/bin/env bash
# 依次跑 v2 单模态 T0 SOTA 对比五组 backbone（各 15 次）
#
# 顺序：Swin-Tiny → Swin-Small → ConvNeXt-Small → ViT-Small → ResNet18
# ResNet18 替代 ViT-Base：小样本更合理（轻量 CNN 容量对照，见 README）
#
# Detach 整包五组：
#   export PATH="/home/amax/anaconda3/bin:$PATH"
#   cd /ssd_data/tsy_study_venv/OptiGenesis_Lancet
#   "$PROJECT_ROOT/run_experiment.sh" --detach \
#     /ssd_data/tsy_study_venv/OptiGenesis_Lancet/tsy_loho \
#     ./run_comparison_t0_oct_only_all.sh
#
# 断线后：
#   tail -f logs/detached_latest.log
#
# 续跑（默认跳过已完成）：
#   SKIP_COMPLETED=1 ./run_comparison_t0_oct_only_all.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

SCRIPTS=(
  run_comparison_swin_tiny_t0_oct_only.sh
  run_comparison_swin_small_t0_oct_only.sh
  run_comparison_convnext_small_t0_oct_only.sh
  run_comparison_vit_small_t0_oct_only.sh
  run_comparison_resnet18_t0_oct_only.sh
)

echo "======================================"
echo "v2 OCT-only T0 SOTA 对比 · 串行 ${#SCRIPTS[@]} 组 × 15 runs"
echo "SKIP_COMPLETED=${SKIP_COMPLETED:-1}"
echo "======================================"

for s in "${SCRIPTS[@]}"; do
  echo ""
  echo ">>>>>>>>>> 启动: ${s} <<<<<<<<<<"
  bash "$PROJECT_ROOT/$s"
done

echo ""
echo "======================================"
echo "SOTA 对比全部结束。结果目录: outputs/第三版/对比试验v2单模态/"
echo "======================================"

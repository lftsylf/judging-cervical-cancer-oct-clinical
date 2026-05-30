#!/usr/bin/env bash
# 续跑未完成的 v2 对比三组（Swin 两组已完成时用这个）
#
# 顺序：ResNet18（batch=4，先跑）→ ConvNeXt-Small（batch=2）→ ViT-Small（batch=2）
#
# Detach：
#   ./run_experiment.sh --detach /path/to/tsy_loho ./run_comparison_t0_oct_only_remaining.sh
#
# 续跑：
#   SKIP_COMPLETED=1 ./run_comparison_t0_oct_only_remaining.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

SCRIPTS=(
  run_comparison_resnet18_t0_oct_only.sh
  run_comparison_convnext_small_t0_oct_only.sh
  run_comparison_vit_small_t0_oct_only.sh
)

echo "======================================"
echo "v2 OCT-only 对比 · 剩余 ${#SCRIPTS[@]} 组（ResNet18→ConvNeXt→ViT-S）"
echo "SKIP_COMPLETED=${SKIP_COMPLETED:-1}"
echo "======================================"

for s in "${SCRIPTS[@]}"; do
  echo ""
  echo ">>>>>>>>>> 启动: ${s} <<<<<<<<<<"
  bash "$PROJECT_ROOT/$s"
done

echo ""
echo "======================================"
echo "剩余三组对比结束。结果目录: outputs/对比试验v2单模态/"
echo "======================================"

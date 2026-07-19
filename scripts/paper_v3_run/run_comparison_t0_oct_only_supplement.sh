#!/usr/bin/env bash
# v2 对比补充两组串行：DenseNet121（经典 CNN）→ DINOv3-Small（视觉基础模型）
#
# 各 3 折 × 5 seeds = 15 runs；共 30 runs。
#
# Detach（推荐）：
#   export PATH="/home/amax/anaconda3/bin:$PATH"
#   cd /ssd_data/tsy_study_venv/OptiGenesis_Lancet
#   "$PROJECT_ROOT/run_experiment.sh" --detach \
#     /ssd_data/tsy_study_venv/OptiGenesis_Lancet/tsy_loho \
#     ./run_comparison_t0_oct_only_supplement.sh
#
# 续跑：
#   SKIP_COMPLETED=1 ./run_comparison_t0_oct_only_supplement.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

SCRIPTS=(
  run_comparison_densenet121_t0_oct_only.sh
  run_comparison_dinov3_small_t0_oct_only.sh
)

echo "======================================"
echo "v2 OCT-only 对比 · 补充 ${#SCRIPTS[@]} 组（DenseNet121 → DINOv3-S）"
echo "SKIP_COMPLETED=${SKIP_COMPLETED:-1}"
echo "======================================"

for s in "${SCRIPTS[@]}"; do
  echo ""
  echo ">>>>>>>>>> 启动: ${s} <<<<<<<<<<"
  bash "$PROJECT_ROOT/$s"
done

echo ""
echo "======================================"
echo "补充两组对比结束。"
echo "  outputs/第三版/对比试验v2单模态/outputs_comparison_v2_oct_densenet121/"
echo "  outputs/第三版/对比试验v2单模态/outputs_comparison_v2_oct_dinov3_small/"
echo "======================================"

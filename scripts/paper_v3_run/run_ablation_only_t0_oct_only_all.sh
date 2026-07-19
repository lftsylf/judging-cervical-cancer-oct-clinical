#!/usr/bin/env bash
# 依次跑 v2 单模态 T0「单模块」三组：仅 WMA → 仅 EMA → 仅 Aux（各 15 次）
#
# Detach 整包三组（推荐 overnight）：
#   export PATH="/home/amax/anaconda3/bin:$PATH"
#   cd /ssd_data/tsy_study_venv/OptiGenesis_Lancet
#   "$PROJECT_ROOT/run_experiment.sh" --detach \
#     /ssd_data/tsy_study_venv/OptiGenesis_Lancet/tsy_loho \
#     ./run_ablation_only_t0_oct_only_all.sh
#
# 只跑其中一组：直接调用 run_ablation_only_{wma,ema,aux}_t0_oct_only.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

SCRIPTS=(
  run_ablation_only_wma_t0_oct_only.sh
  run_ablation_only_ema_t0_oct_only.sh
  run_ablation_only_aux_t0_oct_only.sh
)

echo "======================================"
echo "v2 OCT-only T0 单模块消融 · 串行 ${#SCRIPTS[@]} 组 × 15 runs"
echo "SKIP_COMPLETED=${SKIP_COMPLETED:-1}"
echo "输出: outputs/第三版/消融实验v2单模态/outputs_ablation_v2_oct_only_{wma,ema,aux}/"
echo "======================================"

for s in "${SCRIPTS[@]}"; do
  echo ""
  echo ">>>>>>>>>> 启动: ${s} <<<<<<<<<<"
  bash "$PROJECT_ROOT/$s"
done

echo ""
echo "======================================"
echo "单模块三组消融全部结束。"
echo "完整 2³ 矩阵见 outputs/第三版/消融实验v2单模态/README.md"
echo "======================================"

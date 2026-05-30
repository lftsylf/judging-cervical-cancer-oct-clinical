#!/usr/bin/env bash
# 依次跑 v2 单模态 T0 消融四组：Full → -WMA → -EMA → -Aux（各 15 次）
#
# 建议先单独跑 Full，确认 AUC 后再跑其余三组；本脚本适合 overnight 串行。
#
# Detach 整包四组：
#   ./run_experiment.sh --detach /path/to/tsy_loho ./run_ablation_t0_oct_only_all.sh
#
# 只跑其中一组：直接调用对应脚本（见 run_optigenesis_full_t0_oct_only.sh 等）
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

SCRIPTS=(
  run_optigenesis_full_t0_oct_only.sh
  run_ablation_no_wma_t0_oct_only.sh
  run_ablation_no_ema_t0_oct_only.sh
  run_ablation_no_aux_t0_oct_only.sh
)

echo "======================================"
echo "v2 OCT-only T0 消融 · 串行 ${#SCRIPTS[@]} 组 × 15 runs"
echo "SKIP_COMPLETED=${SKIP_COMPLETED:-1}"
echo "======================================"

for s in "${SCRIPTS[@]}"; do
  echo ""
  echo ">>>>>>>>>> 启动: ${s} <<<<<<<<<<"
  bash "$PROJECT_ROOT/$s"
done

echo ""
echo "======================================"
echo "四组消融全部结束。结果目录: outputs/消融实验v2单模态/"
echo "======================================"

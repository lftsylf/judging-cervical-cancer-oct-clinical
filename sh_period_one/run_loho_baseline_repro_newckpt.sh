#!/bin/bash
# Reproduce baseline under current AUC-only checkpoint rule.
# Usage:
#   ./run_loho_baseline_repro_newckpt.sh --dry-run
#   ./run_loho_baseline_repro_newckpt.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

# 1) Prevent env leakage from previous experiments
unset OPTIGENESIS_BACKBONE
unset OPTIGENESIS_ENABLE_AUX
unset OPTIGENESIS_ENABLE_EMA
unset OPTIGENESIS_ENABLE_CORAL
unset OPTIGENESIS_OUTPUT_DIR
unset OPTIGENESIS_BATCH_SIZE
unset OPTIGENESIS_CORAL_LAMBDA
unset OPTIGENESIS_CORAL_WARMUP
unset HOSPITAL_NAME

# 2) Explicitly lock baseline-repro knobs
export OPTIGENESIS_BACKBONE="swin_tiny_patch4_window7_224"
export OPTIGENESIS_ENABLE_AUX=0
export OPTIGENESIS_ENABLE_EMA=0
export OPTIGENESIS_ENABLE_CORAL=0
export OPTIGENESIS_BATCH_SIZE=4
export OPTIGENESIS_OUTPUT_DIR="$PROJECT_ROOT/baseline_repro_newckpt_outputs"

RUN_TAG="$(date '+%Y%m%d_%H%M%S')"
LOCK_FILE="$OPTIGENESIS_OUTPUT_DIR/repro_lock_${RUN_TAG}.txt"

mkdir -p "$OPTIGENESIS_OUTPUT_DIR"
cat > "$LOCK_FILE" <<EOF
run_tag=${RUN_TAG}
dry_run=${DRY_RUN}
project_root=${PROJECT_ROOT}
dataset_link_expected=${PROJECT_ROOT}/dataset -> ${PROJECT_ROOT}/tsy_loho
backbone=${OPTIGENESIS_BACKBONE}
enable_aux=${OPTIGENESIS_ENABLE_AUX}
enable_ema=${OPTIGENESIS_ENABLE_EMA}
enable_coral=${OPTIGENESIS_ENABLE_CORAL}
batch_size=${OPTIGENESIS_BATCH_SIZE}
output_dir=${OPTIGENESIS_OUTPUT_DIR}
notes=checkpoint_rule_is_AUC_only_in_current_main_py
EOF

echo "======================================"
echo "LOHO Baseline Repro (new checkpoint rule)"
echo "lock_file: ${LOCK_FILE}"
echo "======================================"
cat "$LOCK_FILE"
echo "======================================"

echo "1) 绑定 dataset -> tsy_loho"
./run_experiment.sh "$PROJECT_ROOT/tsy_loho"

echo "2) 生成 LOHO development/external CSV"
python data/prepare_loho_data.py

echo "3) 三折顺序: xiangya -> huaxi -> liaoning"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[DRY-RUN] 配置检查完成，不启动训练。"
  exit 0
fi

for HOSP in xiangya huaxi liaoning; do
  echo "---------- baseline repro fold: ${HOSP} ----------"
  HOSPITAL_NAME="$HOSP" python main.py
done

echo "全部完成。结果目录: $OPTIGENESIS_OUTPUT_DIR/{liaoning,huaxi,xiangya}/"

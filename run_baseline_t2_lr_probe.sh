#!/usr/bin/env bash
# T2 学习率探路：ResNet50 baseline × 3 折 × 1 seed，依次跑 4 组互斥 LR 配置
# 早停：验证集 ROC-AUC 连续 10 epoch 未提升（main.py 内固定 patience=10）
#
# 说明：第 2、3 组是两次独立实验（backbone LR 不同），不是同时设两个 backbone LR。
#
# 用法：
#   ./run_baseline_t2_lr_probe.sh
#   RUN_ONLY=2 ./run_baseline_t2_lr_probe.sh
#   HOSPITALS=(huaxi) ./run_baseline_t2_lr_probe.sh   # 只跑指定折
#   FORCE_RERUN=1 ./run_baseline_t2_lr_probe.sh       # 忽略已完成，全部重跑
#   SKIP_COMPLETED=0 ./run_baseline_t2_lr_probe.sh    # 关闭跳过（默认开启）
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${OPTIGENESIS_PYTHON:-/home/amax/anaconda3/bin/python3}"
SEED="${OPTIGENESIS_SEED:-42}"
BACKBONE="${OPTIGENESIS_BACKBONE:-resnet50}"
_safe_bb="${BACKBONE//[^a-zA-Z0-9_]/_}"
OUT_ROOT="${T2_LR_OUT_ROOT:-$PROJECT_ROOT/outputs_baseline_t2_lr_${_safe_bb}}"

# 3 折 LOHO；可通过环境变量覆盖，例如 HOSPITALS=(huaxi)
if [[ -n "${HOSPITALS+x}" && ${#HOSPITALS[@]} -gt 0 ]]; then
  :
else
  HOSPITALS=(huaxi liaoning xiangya)
fi

# 默认跳过 log 中已有「训练完成！」的折；FORCE_RERUN=1 时强制重跑
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

is_run_complete() {
  local log_file="$1"
  [[ -f "$log_file" ]] && grep -q "训练完成！" "$log_file"
}

run_one() {
  local id="$1"
  local run_name="$2"
  shift 2
  local lr_exports=("$@")

  if [[ -n "${RUN_ONLY:-}" && "$RUN_ONLY" != "$id" ]]; then
    return 0
  fi

  echo ""
  echo "######################################################################"
  echo "# T2 LR 探路 组 ${id}: ${run_name}  |  seeds=${SEED}  |  折: ${HOSPITALS[*]}"
  echo "######################################################################"

  for HOSP in "${HOSPITALS[@]}"; do
    echo ""
    echo "--------------------------------------"
    echo "组 ${id} | ${run_name} | ${HOSP} | seed=${SEED}"
    echo "--------------------------------------"

    unset OPTIGENESIS_LR OPTIGENESIS_BACKBONE_LR OPTIGENESIS_HEAD_LR
    unset OPTIGENESIS_FREEZE_BACKBONE_EPOCHS OPTIGENESIS_FREEZE_HEAD_LR

    export OPTIGENESIS_BACKBONE="$BACKBONE"
    export OPTIGENESIS_ENABLE_AUX=0
    export OPTIGENESIS_ENABLE_EMA=0
    export OPTIGENESIS_ENABLE_CORAL=0
    export OPTIGENESIS_USE_WMA=0
    export OPTIGENESIS_BATCH_SIZE=4
    export OPTIGENESIS_EPOCHS=30
    export HOSPITAL_NAME="$HOSP"
    export OPTIGENESIS_SEED="$SEED"
    export OPTIGENESIS_OUTPUT_DIR="${OUT_ROOT}/${run_name}/${HOSP}"
    export OPTIGENESIS_OUTPUT_RUN_NAME="seed_${SEED}"

    # shellcheck disable=SC2068
    export "${lr_exports[@]}"

    LOG_DIR="${OPTIGENESIS_OUTPUT_DIR}/${OPTIGENESIS_OUTPUT_RUN_NAME}/logs"
    mkdir -p "$LOG_DIR"
    RUN_LOG="${LOG_DIR}/train_console.log"

    if [[ "$SKIP_COMPLETED" == "1" && "${FORCE_RERUN:-0}" != "1" ]] && is_run_complete "$RUN_LOG"; then
      echo "⏭️  跳过：${RUN_LOG} 已含「训练完成！」"
      continue
    fi

    "$PYTHON" main.py 2>&1 | tee "$RUN_LOG"
  done
}

echo "======================================"
echo "T2 LR 探路 | ${BACKBONE} | 3折×1seed | seed=${SEED}"
echo "折: ${HOSPITALS[*]}"
echo "Python: ${PYTHON}"
echo "输出根: ${OUT_ROOT}/<g1..g4>/<hospital>/seed_${SEED}/"
echo "早停: patience=10（main.py）| POS_WEIGHT 默认 1.25（lancet_config）"
echo "跳过已完成: SKIP_COMPLETED=${SKIP_COMPLETED}（log 含「训练完成！」）| FORCE_RERUN=${FORCE_RERUN:-0}"
echo "计划槽位: 4 组 × ${#HOSPITALS[@]} 折 = $((4 * ${#HOSPITALS[@]})) 次（已完成的会自动跳过）"
echo "======================================"

./run_experiment.sh "$PROJECT_ROOT/tsy_loho"
"$PYTHON" data/prepare_loho_data.py

# 1) 基线：全网 5e-5
run_one 1 "g1_uniform_5e5" OPTIGENESIS_LR=5e-5

# 2) 分层：backbone 1e-5，head 1e-4
run_one 2 "g2_disc_bb1e5_head1e4" \
  OPTIGENESIS_BACKBONE_LR=1e-5 \
  OPTIGENESIS_HEAD_LR=1e-4

# 3) 分层：backbone 5e-6，head 1e-4
run_one 3 "g3_disc_bb5e6_head1e4" \
  OPTIGENESIS_BACKBONE_LR=5e-6 \
  OPTIGENESIS_HEAD_LR=1e-4

# 4) 冻结 5 epoch 后分层
run_one 4 "g4_freeze5_disc_bb1e5_head1e4" \
  OPTIGENESIS_FREEZE_BACKBONE_EPOCHS=5 \
  OPTIGENESIS_FREEZE_HEAD_LR=1e-4 \
  OPTIGENESIS_BACKBONE_LR=1e-5 \
  OPTIGENESIS_HEAD_LR=1e-4

echo ""
echo "======================================"
echo "T2 LR 探路结束。对比各组各折:"
echo "  ${OUT_ROOT}/g*/<huaxi|liaoning|xiangya>/seed_${SEED}/logs/train_console.log"
echo "======================================"

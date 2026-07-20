#!/usr/bin/env bash
# =============================================================================
# Paper v4 · B1 T0（等权 mean pooling + 患者级 EDL · OCT-only）
#
# 对齐 v3 baseline 精神，但使用 v4 诚实协议：
#   - 训练: train_{hospital}.csv
#   - 早停/选模: val_{hospital}.csv（内部验证；patience=10，看 ROC-AUC）
#   - 终评: external_{hospital}.csv（训练过程完全不用）
#   - FRAME_AGG=mean（B1，不是 Ours）
#   - ResNet50，OCT-only，LR=5e-5，POS_WEIGHT=1.25，batch=4
#   - 关闭 WMA / Aux / EMA / CORAL
#   - T0：3 折 × 5 seeds = 15 次，≤30 epoch
#
# 前台：
#   ./scripts/paper_v4_run/run_b1_mean_edl_t0_oct_only.sh
#
# 后台（推荐）：
#   export PATH="/home/amax/anaconda3/bin:$PATH"
#   cd /ssd_data/tsy_study_venv/OptiGenesis_Lancet
#   ./run_experiment.sh --detach ./tsy_loho ./scripts/paper_v4_run/run_b1_mean_edl_t0_oct_only.sh
#
# 看日志：
#   tail -f logs/detached_latest.log
#   tail -f outputs/paper_v4/baseline/b1_mean_edl_t0/huaxi/seed_42/logs/train_console.log
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${OPTIGENESIS_PYTHON:-/home/amax/anaconda3/bin/python3}"
CAPTURE_BACKBONE="${OPTIGENESIS_BACKBONE:-resnet50}"
CAPTURE_POS_WEIGHT="${OPTIGENESIS_POS_WEIGHT:-1.25}"
CAPTURE_LR="${OPTIGENESIS_LR:-5e-5}"
_safe_bb="${CAPTURE_BACKBONE//[^a-zA-Z0-9_]/_}"

HOSPITALS=(huaxi liaoning xiangya)
SEEDS=(42 123 2024 3407 114514)
MAX_EPOCHS=30
BASELINE_OUT_ROOT="${BASELINE_OUT_ROOT:-$PROJECT_ROOT/outputs/paper_v4/baseline/b1_mean_edl_t0}"

SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
TOTAL_RUNS=$((${#HOSPITALS[@]} * ${#SEEDS[@]}))

is_run_complete() {
  local log_file="$1"
  [[ -f "$log_file" ]] && grep -q "训练完成！" "$log_file"
}

echo "======================================"
echo "Paper v4 | B1 T0 | mean pooling + 患者级 EDL | OCT-only"
echo "骨干=${CAPTURE_BACKBONE} | LR=${CAPTURE_LR} | POS_WEIGHT=${CAPTURE_POS_WEIGHT}"
echo "FRAME_AGG=mean | USE_CLINICAL=0 | BATCH=4 | EPOCHS<=${MAX_EPOCHS}"
echo "早停: 内部 val ROC-AUC | patience=10 | external 只终评"
echo "WMA=0 | AUX=0 | EMA=0 | CORAL=0"
echo "折: ${HOSPITALS[*]} | Seeds: ${SEEDS[*]} | 共 ${TOTAL_RUNS} 次"
echo "Python: ${PYTHON}"
echo "输出根: ${BASELINE_OUT_ROOT}"
echo "SKIP_COMPLETED=${SKIP_COMPLETED} | FORCE_RERUN=${FORCE_RERUN:-0}"
echo "======================================"

# 绑定数据软链（不重跑 prepare_loho，以免打乱已校验的 train/val）
"$PROJECT_ROOT/run_experiment.sh" "$PROJECT_ROOT/tsy_loho"

# 确认 v4 划分 CSV 存在；缺失则从 development 重新生成
need_split=0
for HOSP in "${HOSPITALS[@]}"; do
  if [[ ! -f "$PROJECT_ROOT/dataset/train_${HOSP}.csv" || ! -f "$PROJECT_ROOT/dataset/val_${HOSP}.csv" ]]; then
    need_split=1
  fi
done
if [[ "$need_split" == "1" ]]; then
  echo "⚠️ 缺少 train/val CSV，正在生成（prepare_paper_v4_splits.py --write）…"
  "$PYTHON" data/prepare_paper_v4_splits.py --write --split-seed 20260720 --val-ratio 0.2
else
  echo "✅ 已找到 train_*.csv / val_*.csv，跳过重新划分"
fi

RUN_IDX=0
for HOSP in "${HOSPITALS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    RUN_IDX=$((RUN_IDX + 1))
    echo "--------------------------------------"
    echo "B1 T0 [${RUN_IDX}/${TOTAL_RUNS}] | ${HOSP} | seed=${SEED}"
    echo "--------------------------------------"

    unset OPTIGENESIS_LR OPTIGENESIS_BACKBONE_LR OPTIGENESIS_HEAD_LR
    unset OPTIGENESIS_FREEZE_BACKBONE_EPOCHS OPTIGENESIS_FREEZE_HEAD_LR
    unset OPTIGENESIS_BACKBONE
    unset OPTIGENESIS_ENABLE_AUX
    unset OPTIGENESIS_ENABLE_EMA
    unset OPTIGENESIS_ENABLE_CORAL
    unset OPTIGENESIS_OUTPUT_DIR
    unset OPTIGENESIS_OUTPUT_RUN_NAME
    unset OPTIGENESIS_BATCH_SIZE
    unset OPTIGENESIS_CORAL_LAMBDA
    unset OPTIGENESIS_CORAL_WARMUP
    unset OPTIGENESIS_EPOCHS
    unset OPTIGENESIS_SEED
    unset OPTIGENESIS_USE_WMA
    unset OPTIGENESIS_USE_CLINICAL
    unset OPTIGENESIS_POS_WEIGHT
    unset OPTIGENESIS_FRAME_AGG
    unset OPTIGENESIS_FRAME_AGG_TEMP
    unset HOSPITAL_NAME

    export OPTIGENESIS_BACKBONE="$CAPTURE_BACKBONE"
    export OPTIGENESIS_USE_CLINICAL=0
    export OPTIGENESIS_LR="$CAPTURE_LR"
    export OPTIGENESIS_POS_WEIGHT="$CAPTURE_POS_WEIGHT"
    export OPTIGENESIS_ENABLE_AUX=0
    export OPTIGENESIS_ENABLE_EMA=0
    export OPTIGENESIS_ENABLE_CORAL=0
    export OPTIGENESIS_BATCH_SIZE=4
    export OPTIGENESIS_USE_WMA=0
    # B1：等权 mean → 患者级 EDL（对齐 v3 baseline 结构）
    export OPTIGENESIS_FRAME_AGG=mean
    export HOSPITAL_NAME="$HOSP"
    export OPTIGENESIS_SEED="$SEED"
    export OPTIGENESIS_EPOCHS="$MAX_EPOCHS"
    export OPTIGENESIS_OUTPUT_DIR="${BASELINE_OUT_ROOT}/$HOSP"
    export OPTIGENESIS_OUTPUT_RUN_NAME="seed_${SEED}"

    LOG_DIR="${OPTIGENESIS_OUTPUT_DIR}/${OPTIGENESIS_OUTPUT_RUN_NAME}/logs"
    mkdir -p "$LOG_DIR"
    RUN_LOG="${LOG_DIR}/train_console.log"

    if [[ "$SKIP_COMPLETED" == "1" && "${FORCE_RERUN:-0}" != "1" ]] && is_run_complete "$RUN_LOG"; then
      echo "⏭️  跳过：${RUN_LOG} 已含「训练完成！」"
      continue
    fi

    "$PYTHON" main.py 2>&1 | tee "$RUN_LOG"
  done
done

echo "======================================"
echo "Paper v4 B1 T0 全部完成。"
echo "结果根目录: ${BASELINE_OUT_ROOT}"
echo "主指标请看各 run 日志中的「外部终评 external」ROC-AUC"
echo "======================================"

#!/usr/bin/env bash
# T1 临床模态消融：ResNet50 baseline，clinical ON（只核对/跳过）+ clinical OFF（自动跑）
#
# 规格：3 折 × 3 seeds（42, 123, 2024）；LR=5e-5；POS_WEIGHT 见 lancet_config 默认 1.25
#       无 aux / EMA / WMA / CORAL
#
# Clinical ON：默认不训练，仅检查 CLINICAL_ON_ROOT 下 9 次是否已有「训练完成！」
# Clinical OFF：写入 CLINICAL_OFF_ROOT，OPTIGENESIS_USE_CLINICAL=0
#
# 前台：
#   ./run_baseline_t1_clinical_ablation.sh
#
# SSH 断线仍继续（推荐）：
#   export PATH="/home/amax/anaconda3/bin:$PATH"
#   ./run_experiment.sh --detach \
#     /ssd_data/tsy_study_venv/OptiGenesis_Lancet/tsy_loho \
#     ./run_baseline_t1_clinical_ablation.sh
#
# 看总进度：
#   tail -f logs/detached_latest.log
# 看某一折 OFF：
#   tail -f outputs_baseline_t1_v2_resnet50_clinical_off/huaxi/seed_42/logs/train_console.log
#
# 环境变量：
#   CLINICAL_ON_ROOT   — 已有 clinical ON 结果根目录（默认 outputs_baseline_t1_v2_resnet50）
#   CLINICAL_OFF_ROOT  — OFF 输出根目录
#   OPTIGENESIS_POS_WEIGHT — 可选覆盖；未设置时用 lancet_config 默认 1.25
#   RUN_CLINICAL_ON=1  — 罕见：在 ON_ROOT 重跑缺失的 ON（仍用 clinical=1）
#   FORCE_RERUN=1      — 忽略 OFF 已完成标记，强制重跑 OFF
#   SKIP_COMPLETED=1   — 默认跳过已完成 OFF
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${OPTIGENESIS_PYTHON:-/home/amax/anaconda3/bin/python3}"
CAPTURE_BACKBONE="${OPTIGENESIS_BACKBONE:-resnet50}"
CAPTURE_POS_WEIGHT="${OPTIGENESIS_POS_WEIGHT:-}"
_safe_bb="${CAPTURE_BACKBONE//[^a-zA-Z0-9_]/_}"

HOSPITALS=(huaxi liaoning xiangya)
SEEDS=(42 123 2024)
MAX_EPOCHS=30

CLINICAL_ON_ROOT="${CLINICAL_ON_ROOT:-$PROJECT_ROOT/outputs_baseline_t1_v2_${_safe_bb}}"
CLINICAL_OFF_ROOT="${CLINICAL_OFF_ROOT:-$PROJECT_ROOT/outputs_baseline_t1_v2_${_safe_bb}_clinical_off}"
ABLATION_LR="${ABLATION_LR:-5e-5}"

SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RUN_CLINICAL_ON="${RUN_CLINICAL_ON:-0}"

is_run_complete() {
  local log_file="$1"
  [[ -f "$log_file" ]] && grep -q "训练完成！" "$log_file"
}

count_completed() {
  local root="$1"
  local n=0
  local hosp seed logf
  for hosp in "${HOSPITALS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      logf="${root}/${hosp}/seed_${seed}/logs/train_console.log"
      if is_run_complete "$logf"; then
        n=$((n + 1))
      fi
    done
  done
  echo "$n"
}

run_one_baseline() {
  local use_clinical="$1"
  local out_root="$2"
  local label="$3"

  for HOSP in "${HOSPITALS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      echo "--------------------------------------"
      echo "${label} | ${HOSP} | seed=${SEED} | USE_CLINICAL=${use_clinical}"
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
      unset HOSPITAL_NAME

      export OPTIGENESIS_LR="$ABLATION_LR"
      export OPTIGENESIS_BACKBONE="$CAPTURE_BACKBONE"
      if [[ -n "$CAPTURE_POS_WEIGHT" ]]; then
        export OPTIGENESIS_POS_WEIGHT="$CAPTURE_POS_WEIGHT"
      fi
      export OPTIGENESIS_USE_CLINICAL="$use_clinical"
      export OPTIGENESIS_ENABLE_AUX=0
      export OPTIGENESIS_ENABLE_EMA=0
      export OPTIGENESIS_ENABLE_CORAL=0
      export OPTIGENESIS_BATCH_SIZE=4
      export OPTIGENESIS_USE_WMA=0
      export HOSPITAL_NAME="$HOSP"
      export OPTIGENESIS_SEED="$SEED"
      export OPTIGENESIS_EPOCHS="$MAX_EPOCHS"
      export OPTIGENESIS_OUTPUT_DIR="${out_root}/$HOSP"
      export OPTIGENESIS_OUTPUT_RUN_NAME="seed_${SEED}"

      LOG_DIR="${OPTIGENESIS_OUTPUT_DIR}/${OPTIGENESIS_OUTPUT_RUN_NAME}/logs"
      mkdir -p "$LOG_DIR"
      RUN_LOG="${LOG_DIR}/train_console.log"

      if [[ "$use_clinical" == "0" ]] && [[ "$SKIP_COMPLETED" == "1" && "${FORCE_RERUN:-0}" != "1" ]] && is_run_complete "$RUN_LOG"; then
        echo "⏭️  跳过 OFF：${RUN_LOG} 已含「训练完成！」"
        continue
      fi
      if [[ "$use_clinical" == "1" ]] && [[ "$SKIP_COMPLETED" == "1" && "${FORCE_RERUN_ON:-0}" != "1" ]] && is_run_complete "$RUN_LOG"; then
        echo "⏭️  跳过 ON：${RUN_LOG} 已含「训练完成！」"
        continue
      fi

      "$PYTHON" main.py 2>&1 | tee "$RUN_LOG"
    done
  done
}

TOTAL_EXPECTED=$((${#HOSPITALS[@]} * ${#SEEDS[@]}))
ON_DONE="$(count_completed "$CLINICAL_ON_ROOT")"
OFF_DONE="$(count_completed "$CLINICAL_OFF_ROOT")"

echo "======================================"
echo "T1 Clinical Ablation | ${CAPTURE_BACKBONE}"
echo "LR=${ABLATION_LR} | POS_WEIGHT=${CAPTURE_POS_WEIGHT:-1.25（lancet_config 默认）}"
echo "折: ${HOSPITALS[*]} | Seeds: ${SEEDS[*]} | 每组 ${TOTAL_EXPECTED} 次"
echo "Python: ${PYTHON}"
echo "Clinical ON  （只核对，默认不训）: ${CLINICAL_ON_ROOT}"
echo "  已完成: ${ON_DONE}/${TOTAL_EXPECTED}"
echo "Clinical OFF（本脚本主任务）    : ${CLINICAL_OFF_ROOT}"
echo "  已完成: ${OFF_DONE}/${TOTAL_EXPECTED}"
echo "RUN_CLINICAL_ON=${RUN_CLINICAL_ON} | SKIP_COMPLETED=${SKIP_COMPLETED}"
echo "======================================"

if [[ "$ON_DONE" -lt "$TOTAL_EXPECTED" ]]; then
  echo "⚠️  Clinical ON 目录未满 ${TOTAL_EXPECTED}/${TOTAL_EXPECTED}。"
  echo "   若旧结果在别的路径，请: export CLINICAL_ON_ROOT=/path/to/your/on_root"
  if [[ "$RUN_CLINICAL_ON" != "1" ]]; then
    echo "   仅跑 OFF 仍可继续；对比 ON 时请确保 ON_ROOT 指向正确目录。"
    echo "   若要补跑缺失 ON: RUN_CLINICAL_ON=1 ./run_baseline_t1_clinical_ablation.sh"
  fi
fi

./run_experiment.sh "$PROJECT_ROOT/tsy_loho"
"$PYTHON" data/prepare_loho_data.py

if [[ "$RUN_CLINICAL_ON" == "1" ]]; then
  echo ""
  echo "======== 补跑 Clinical ON（缺失项）========"
  run_one_baseline 1 "$CLINICAL_ON_ROOT" "Clinical ON"
fi

echo ""
echo "======== 训练 Clinical OFF（OCT-only）========"
run_one_baseline 0 "$CLINICAL_OFF_ROOT" "Clinical OFF"

OFF_DONE_AFTER="$(count_completed "$CLINICAL_OFF_ROOT")"
echo "======================================"
echo "Clinical OFF 完成: ${OFF_DONE_AFTER}/${TOTAL_EXPECTED}"
echo "  OFF 结果: ${CLINICAL_OFF_ROOT}"
echo "  ON  参考: ${CLINICAL_ON_ROOT} （${ON_DONE}/${TOTAL_EXPECTED} 已存在）"
echo "对比外部 AUC 后决定主设定（多模态 vs OCT-only）。"
echo "======================================"

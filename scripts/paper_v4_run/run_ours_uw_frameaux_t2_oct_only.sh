#!/usr/bin/env bash
# =============================================================================
# Paper v4 · Ours+帧级弱监督 T2（3 折 × 1 seed 探路）
#
# 建议叙事（若 T2/T0 有效）：
#   - 最终方法 Full = FRAME_AGG=uncertainty_weighted + 帧级弱监督
#   - 消融 w/o frame aux = 已跑完的 ours_uw_agg_t0（无帧损）
#   - B1 = mean + 患者 EDL（结构对照）
#
# T2：huaxi/liaoning/xiangya × seed=42，其余协议同 T0
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${OPTIGENESIS_PYTHON:-/home/amax/anaconda3/bin/python3}"
CAPTURE_BACKBONE="${OPTIGENESIS_BACKBONE:-resnet50}"
CAPTURE_POS_WEIGHT="${OPTIGENESIS_POS_WEIGHT:-1.25}"
CAPTURE_LR="${OPTIGENESIS_LR:-5e-5}"
CAPTURE_AGG_TEMP="${OPTIGENESIS_FRAME_AGG_TEMP:-0.5}"
CAPTURE_WEIGHT_SIGNAL="${OPTIGENESIS_FRAME_WEIGHT_SIGNAL:-edl_u}"
CAPTURE_TOPK_K="${OPTIGENESIS_FRAME_TOPK_K:-5}"
CAPTURE_FRAME_AUX_W="${OPTIGENESIS_FRAME_AUX_WEIGHT:-0.2}"
CAPTURE_FRAME_AUX_TYPE="${OPTIGENESIS_FRAME_AUX_TYPE:-edl}"
CAPTURE_EXPAND="${OPTIGENESIS_EXPAND_TIFF_PAGES:-0}"
CAPTURE_MAX_PAGES="${OPTIGENESIS_MAX_PAGES_PER_TIFF:-0}"
CAPTURE_BATCH="${OPTIGENESIS_BATCH_SIZE:-4}"

HOSPITALS=(huaxi liaoning xiangya)
SEEDS=(42)
MAX_EPOCHS=30
OUT_ROOT="${BASELINE_OUT_ROOT:-$PROJECT_ROOT/outputs/paper_v4/baseline/ours_uw_frameaux_t2}"

SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
TOTAL_RUNS=$((${#HOSPITALS[@]} * ${#SEEDS[@]}))

is_run_complete() {
  local log_file="$1"
  [[ -f "$log_file" ]] && grep -q "训练完成！" "$log_file"
}

echo "======================================"
echo "Paper v4 | Ours+帧级弱监督 T2 | 3折×1seed"
echo "FRAME_AGG=uncertainty_weighted | signal=${CAPTURE_WEIGHT_SIGNAL} | τ=${CAPTURE_AGG_TEMP} | topk_k=${CAPTURE_TOPK_K}"
echo "FRAME_AUX=1 weight=${CAPTURE_FRAME_AUX_W} type=${CAPTURE_FRAME_AUX_TYPE}"
echo "EXPAND=${CAPTURE_EXPAND} MAX_PAGES=${CAPTURE_MAX_PAGES} BATCH=${CAPTURE_BATCH}"
echo "骨干=${CAPTURE_BACKBONE} | LR=${CAPTURE_LR} | POS_WEIGHT=${CAPTURE_POS_WEIGHT} | OCT-only"
echo "WMA=0 | multimodal AUX=0 | EMA=0 | CORAL=0"
echo "折: ${HOSPITALS[*]} | Seeds: ${SEEDS[*]} | 共 ${TOTAL_RUNS} 次"
echo "输出根: ${OUT_ROOT}"
echo "======================================"

if [[ "${SKIP_DATASET_BIND:-0}" != "1" ]]; then
  "$PROJECT_ROOT/run_experiment.sh" "$PROJECT_ROOT/tsy_loho"
else
  echo "⏭️  跳过 dataset 软链绑定（SKIP_DATASET_BIND=1）"
fi

need_split=0
for HOSP in "${HOSPITALS[@]}"; do
  if [[ ! -f "$PROJECT_ROOT/dataset/train_${HOSP}.csv" || ! -f "$PROJECT_ROOT/dataset/val_${HOSP}.csv" ]]; then
    need_split=1
  fi
done
if [[ "$need_split" == "1" ]]; then
  "$PYTHON" data/prepare_paper_v4_splits.py --write --split-seed 20260720 --val-ratio 0.2
else
  echo "✅ 已找到 train/val CSV"
fi

RUN_IDX=0
for HOSP in "${HOSPITALS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    RUN_IDX=$((RUN_IDX + 1))
    echo "--------------------------------------"
    echo "Ours+FrameAux T2 [${RUN_IDX}/${TOTAL_RUNS}] | ${HOSP} | seed=${SEED}"
    echo "--------------------------------------"

    unset OPTIGENESIS_LR OPTIGENESIS_BACKBONE_LR OPTIGENESIS_HEAD_LR
    unset OPTIGENESIS_FREEZE_BACKBONE_EPOCHS OPTIGENESIS_FREEZE_HEAD_LR
    unset OPTIGENESIS_BACKBONE OPTIGENESIS_ENABLE_AUX OPTIGENESIS_ENABLE_EMA
    unset OPTIGENESIS_ENABLE_CORAL OPTIGENESIS_OUTPUT_DIR OPTIGENESIS_OUTPUT_RUN_NAME
    unset OPTIGENESIS_BATCH_SIZE OPTIGENESIS_EPOCHS OPTIGENESIS_SEED
    unset OPTIGENESIS_USE_WMA OPTIGENESIS_USE_CLINICAL OPTIGENESIS_POS_WEIGHT
    unset OPTIGENESIS_FRAME_AGG OPTIGENESIS_FRAME_AGG_TEMP OPTIGENESIS_FRAME_WEIGHT_SIGNAL
    unset OPTIGENESIS_FRAME_TOPK_K OPTIGENESIS_EXPAND_TIFF_PAGES OPTIGENESIS_MAX_PAGES_PER_TIFF
    unset OPTIGENESIS_ENABLE_FRAME_AUX OPTIGENESIS_FRAME_AUX_WEIGHT OPTIGENESIS_FRAME_AUX_TYPE
    unset HOSPITAL_NAME

    export OPTIGENESIS_BACKBONE="$CAPTURE_BACKBONE"
    export OPTIGENESIS_USE_CLINICAL=0
    export OPTIGENESIS_LR="$CAPTURE_LR"
    export OPTIGENESIS_POS_WEIGHT="$CAPTURE_POS_WEIGHT"
    export OPTIGENESIS_ENABLE_AUX=0
    export OPTIGENESIS_ENABLE_EMA=0
    export OPTIGENESIS_ENABLE_CORAL=0
    export OPTIGENESIS_BATCH_SIZE="$CAPTURE_BATCH"
    export OPTIGENESIS_USE_WMA=0
    export OPTIGENESIS_FRAME_AGG=uncertainty_weighted
    export OPTIGENESIS_FRAME_AGG_TEMP="$CAPTURE_AGG_TEMP"
    export OPTIGENESIS_FRAME_WEIGHT_SIGNAL="$CAPTURE_WEIGHT_SIGNAL"
    export OPTIGENESIS_FRAME_TOPK_K="$CAPTURE_TOPK_K"
    export OPTIGENESIS_EXPAND_TIFF_PAGES="$CAPTURE_EXPAND"
    export OPTIGENESIS_MAX_PAGES_PER_TIFF="$CAPTURE_MAX_PAGES"
    export OPTIGENESIS_ENABLE_FRAME_AUX=1
    export OPTIGENESIS_FRAME_AUX_WEIGHT="$CAPTURE_FRAME_AUX_W"
    export OPTIGENESIS_FRAME_AUX_TYPE="$CAPTURE_FRAME_AUX_TYPE"
    export HOSPITAL_NAME="$HOSP"
    export OPTIGENESIS_SEED="$SEED"
    export OPTIGENESIS_EPOCHS="$MAX_EPOCHS"
    export OPTIGENESIS_OUTPUT_DIR="${OUT_ROOT}/$HOSP"
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
echo "T2 完成。结果: ${OUT_ROOT}"
echo "请对照同 seed 的:"
echo "  B1:   outputs/paper_v4/baseline/b1_mean_edl_t0/*/seed_42/"
echo "  Ours: outputs/paper_v4/baseline/ours_uw_agg_t0/*/seed_42/"
echo "======================================"

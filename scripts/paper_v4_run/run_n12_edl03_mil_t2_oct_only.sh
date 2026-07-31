#!/usr/bin/env bash
# Paper v4 · N=12 + FrameAux edl@0.3 + MIL 帧辅损（探路 seed=42）
#
# MIL（相对旧 broadcast）：
#   - 阴性袋：各有效点压阴
#   - 阳性袋：若已有点 p_pos≥thr → 不再做帧辅损（袋级主损失仍在）
#             否则只对最阳的那一点往阳推
#
# 聚合仍用主候选 edl_u（τ=0.5），只改监督方式，便于对照。
# 默认占 GPU 1/2/3（GPU0 可能仍在跑 amp）；可用 OPTIGENESIS_GPUS 覆盖。
#
# 用法:
#   ./run_experiment.sh --detach ./tsy_loho \
#     ./scripts/paper_v4_run/run_n12_edl03_mil_t2_oct_only.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="/home/amax/anaconda3/bin:${PATH:-}"
PYTHON="${PYTHON:-python}"

export OPTIGENESIS_EXPAND_TIFF_PAGES=0
unset OPTIGENESIS_MAX_PAGES_PER_TIFF 2>/dev/null || true
export OPTIGENESIS_BATCH_SIZE="${OPTIGENESIS_BATCH_SIZE:-4}"
export OPTIGENESIS_EPOCHS="${OPTIGENESIS_EPOCHS:-30}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export OPTIGENESIS_FRAME_AGG=uncertainty_weighted
export OPTIGENESIS_FRAME_AGG_TEMP="${OPTIGENESIS_FRAME_AGG_TEMP:-0.5}"
export OPTIGENESIS_FRAME_WEIGHT_SIGNAL=edl_u
unset OPTIGENESIS_FRAME_U_SCORE_BASE OPTIGENESIS_FRAME_U_SCORE_SCALE 2>/dev/null || true

export OPTIGENESIS_ENABLE_FRAME_AUX=1
export OPTIGENESIS_FRAME_AUX_WEIGHT=0.3
export OPTIGENESIS_FRAME_AUX_TYPE=edl
export OPTIGENESIS_FRAME_AUX_MODE=mil
export OPTIGENESIS_FRAME_AUX_MIL_POS_THR="${OPTIGENESIS_FRAME_AUX_MIL_POS_THR:-0.5}"

export OPTIGENESIS_USE_WMA=0
export OPTIGENESIS_ENABLE_EMA=0
export OPTIGENESIS_ENABLE_AUX=0
export OPTIGENESIS_USE_CLINICAL=0

OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3_n12_mil}"
unset OPTIGENESIS_OUTPUT_DIR 2>/dev/null || true
mkdir -p "$OUT_ROOT"

SEEDS=(42)
HOSPITALS=(huaxi liaoning xiangya)
PARALLEL="${OPTIGENESIS_PARALLEL:-1}"
# 默认避开可能仍占用的 GPU0
read -r -a GPUS <<< "${OPTIGENESIS_GPUS:-1 2 3}"

echo "======================================"
echo "N=12 + edl@0.3 + MIL 帧辅损（探路）"
echo "  signal=edl_u  τ=$OPTIGENESIS_FRAME_AGG_TEMP"
echo "  FRAME_AUX mode=$OPTIGENESIS_FRAME_AUX_MODE weight=$OPTIGENESIS_FRAME_AUX_WEIGHT type=$OPTIGENESIS_FRAME_AUX_TYPE thr=$OPTIGENESIS_FRAME_AUX_MIL_POS_THR"
echo "  PARALLEL=$PARALLEL  GPUS=${GPUS[*]}"
echo "  OUT=$OUT_ROOT"
echo "======================================"

need_split=0
for h in "${HOSPITALS[@]}"; do
  if [[ ! -f "$ROOT/dataset/train_${h}.csv" || ! -f "$ROOT/dataset/val_${h}.csv" ]]; then
    need_split=1
  fi
done
if [[ "$need_split" == "1" ]]; then
  "$PYTHON" data/prepare_paper_v4_splits.py --write --split-seed 20260720 --val-ratio 0.2
else
  echo "✅ 已找到 train/val CSV"
fi

run_one() {
  local h="$1" s="$2" gpu="$3"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export HOSPITAL_NAME="$h"
    export OPTIGENESIS_SEED="$s"
    export OPTIGENESIS_OUTPUT_DIR="${OUT_ROOT}/$h"
    export OPTIGENESIS_OUTPUT_RUN_NAME="seed_${s}"

    local LOG_DIR="${OPTIGENESIS_OUTPUT_DIR}/${OPTIGENESIS_OUTPUT_RUN_NAME}/logs"
    mkdir -p "$LOG_DIR"
    local RUN_LOG="${LOG_DIR}/train_console.log"

    if [[ "${SKIP_COMPLETED:-1}" == "1" && -f "$RUN_LOG" ]] && grep -q "训练完成！" "$RUN_LOG"; then
      echo "⏭️  跳过 $h seed=$s"
      exit 0
    fi

    echo ">>> $h seed=$s  GPU=$gpu → $RUN_LOG"
    "$PYTHON" main.py 2>&1 | tee "$RUN_LOG"
    echo "<<< 完成 $h seed=$s  GPU=$gpu"
  )
}

JOBS=()
GPU_IDX=0
FAIL=0

for h in "${HOSPITALS[@]}"; do
  for s in "${SEEDS[@]}"; do
    gpu="${GPUS[$((GPU_IDX % ${#GPUS[@]}))]}"
    GPU_IDX=$((GPU_IDX + 1))
    if [[ "$PARALLEL" == "1" ]]; then
      if (( ${#JOBS[@]} >= ${#GPUS[@]} )); then
        if ! wait "${JOBS[0]}"; then FAIL=1; fi
        JOBS=("${JOBS[@]:1}")
      fi
      run_one "$h" "$s" "$gpu" &
      JOBS+=($!)
      echo "  launched pid=${JOBS[-1]}  $h seed=$s on GPU $gpu"
    else
      run_one "$h" "$s" "$gpu" || FAIL=1
    fi
  done
done

if [[ "$PARALLEL" == "1" ]]; then
  for pid in "${JOBS[@]+"${JOBS[@]}"}"; do
    if ! wait "$pid"; then FAIL=1; fi
  done
fi

if [[ "$FAIL" -ne 0 ]]; then
  echo "有任务失败 → $OUT_ROOT"
  exit 1
fi
echo "完成 → $OUT_ROOT"
echo "对照主候选 broadcast: outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3/"

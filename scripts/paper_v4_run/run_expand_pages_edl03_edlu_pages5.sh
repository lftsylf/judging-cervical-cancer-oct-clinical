#!/usr/bin/env bash
# Paper v4 · pages≤5（N≈60）+ edl@0.3 + 普通 edl_u（无 amp）对照
# 默认 3 折并行占 GPU 0/1/2
#
# 用法:
#   OPTIGENESIS_GPUS="0 1 2" ./scripts/paper_v4_run/run_expand_pages_edl03_edlu_pages5.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="/home/amax/anaconda3/bin:${PATH:-}"
PYTHON="${PYTHON:-python}"

export OPTIGENESIS_EXPAND_TIFF_PAGES=1
export OPTIGENESIS_BATCH_SIZE="${OPTIGENESIS_BATCH_SIZE:-8}"
export OPTIGENESIS_MAX_PAGES_PER_TIFF="${OPTIGENESIS_MAX_PAGES_PER_TIFF:-5}"
export OPTIGENESIS_FRAME_ENCODE_CHUNK="${OPTIGENESIS_FRAME_ENCODE_CHUNK:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export OPTIGENESIS_FRAME_AGG=uncertainty_weighted
export OPTIGENESIS_FRAME_AGG_TEMP="${OPTIGENESIS_FRAME_AGG_TEMP:-0.5}"
export OPTIGENESIS_FRAME_WEIGHT_SIGNAL=edl_u
# 明确关掉 amp 相关（即使残留 env）
unset OPTIGENESIS_FRAME_U_SCORE_BASE OPTIGENESIS_FRAME_U_SCORE_SCALE 2>/dev/null || true

export OPTIGENESIS_ENABLE_FRAME_AUX=1
export OPTIGENESIS_FRAME_AUX_WEIGHT=0.3
export OPTIGENESIS_FRAME_AUX_TYPE=edl
export OPTIGENESIS_USE_WMA=0
export OPTIGENESIS_ENABLE_EMA=0
export OPTIGENESIS_ENABLE_AUX=0
export OPTIGENESIS_USE_CLINICAL=0

OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3_expand_edlu_pages5}"
unset OPTIGENESIS_OUTPUT_DIR 2>/dev/null || true
mkdir -p "$OUT_ROOT"

SEEDS=(42)
HOSPITALS=(huaxi liaoning xiangya)
PARALLEL="${OPTIGENESIS_PARALLEL:-1}"
read -r -a GPUS <<< "${OPTIGENESIS_GPUS:-0 1 2}"

echo "======================================"
echo "pages≤5 + edl@0.3 + edl_u（无 amp）"
echo "  EXPAND=1 MAX_PAGES=$OPTIGENESIS_MAX_PAGES_PER_TIFF BATCH=$OPTIGENESIS_BATCH_SIZE"
echo "  signal=edl_u PARALLEL=$PARALLEL GPUS=${GPUS[*]}"
echo "  OUT=$OUT_ROOT"
echo "======================================"

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
    echo ">>> $h seed=$s GPU=$gpu → $RUN_LOG"
    "$PYTHON" main.py 2>&1 | tee "$RUN_LOG"
    echo "<<< 完成 $h seed=$s"
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
      echo "  launched pid=${JOBS[-1]} $h on GPU $gpu"
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

#!/usr/bin/env bash
# Paper v4 · 定稿规模：N=12 + UW(edl_u) + FrameAux edl@0.3
# 只跑剩余 4 seeds（123/2024/3407/114514）；seed=42 复用已有探路产物。
#
# 输出写入同一目录树（与探路对齐）：
#   outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3/{hospital}/seed_{SEED}/
#
# 默认 3 折并行占 GPU 0/1/2；每个 seed 一波（三折同时），共 4 波。
#
# 用法:
#   ./run_experiment.sh --detach ./tsy_loho \
#     ./scripts/paper_v4_run/run_n12_edl03_uw_frameaux_final4seeds.sh
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
# 与已有 seed_42 一致：旧广播辅损（默认）；显式写出防 env 残留 mil
export OPTIGENESIS_FRAME_AUX_MODE=broadcast

export OPTIGENESIS_USE_WMA=0
export OPTIGENESIS_ENABLE_EMA=0
export OPTIGENESIS_ENABLE_AUX=0
export OPTIGENESIS_USE_CLINICAL=0

OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3}"
unset OPTIGENESIS_OUTPUT_DIR 2>/dev/null || true
mkdir -p "$OUT_ROOT"

# 定稿五种子：42 已完成；此处只跑后四个
SEEDS=(123 2024 3407 114514)
HOSPITALS=(huaxi liaoning xiangya)
PARALLEL="${OPTIGENESIS_PARALLEL:-1}"
read -r -a GPUS <<< "${OPTIGENESIS_GPUS:-0 1 2}"

echo "======================================"
echo "定稿补跑 · N=12 + edl_u + FrameAux edl@0.3"
echo "  仅 seeds: ${SEEDS[*]}  （seed=42 复用已有）"
echo "  signal=edl_u  τ=$OPTIGENESIS_FRAME_AGG_TEMP  aux_mode=$OPTIGENESIS_FRAME_AUX_MODE w=0.3"
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

FAIL=0
# 按 seed 分波：每波三折占满 3 卡，便于跟踪
for s in "${SEEDS[@]}"; do
  echo "-------- seed=$s 三折并行 --------"
  JOBS=()
  GPU_IDX=0
  for h in "${HOSPITALS[@]}"; do
    gpu="${GPUS[$((GPU_IDX % ${#GPUS[@]}))]}"
    GPU_IDX=$((GPU_IDX + 1))
    if [[ "$PARALLEL" == "1" ]]; then
      run_one "$h" "$s" "$gpu" &
      JOBS+=($!)
      echo "  launched pid=${JOBS[-1]}  $h seed=$s on GPU $gpu"
    else
      run_one "$h" "$s" "$gpu" || FAIL=1
    fi
  done
  if [[ "$PARALLEL" == "1" ]]; then
    for pid in "${JOBS[@]+"${JOBS[@]}"}"; do
      if ! wait "$pid"; then FAIL=1; fi
    done
  fi
  echo "-------- seed=$s 本波结束 --------"
done

if [[ "$FAIL" -ne 0 ]]; then
  echo "有任务失败 → $OUT_ROOT"
  exit 1
fi
echo "完成四种子补跑 → $OUT_ROOT"
echo "提数时合并 seed=42 + ${SEEDS[*]}（共 5 seeds × 3 折）"

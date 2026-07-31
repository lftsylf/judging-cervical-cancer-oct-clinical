#!/usr/bin/env bash
# Paper v4 · N=12（不扩页）+ FrameAux edl@0.3 + edl_u_amp（探路 1 seed）
# score = (u_base − u) · scale，再 softmax(/τ)
#   固定：u_base=0.5；scale=10；τ=0.5；λ=0.3（不扫）
#
# 与主候选 ours_uw_frameaux_t2_edl_w0.3 的唯一差别：加权信号 edl_u → edl_u_amp
# 默认：3 折并行占 GPU 0/1/2；GPU 3 备用
#
# 用法:
#   ./run_experiment.sh --detach ./tsy_loho \
#     ./scripts/paper_v4_run/run_n12_edl03_uamp_t2_oct_only.sh
#
# 看日志:
#   tail -f logs/detached_latest.log
#   tail -f outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3_n12_uamp10/huaxi/seed_42/logs/train_console.log
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="/home/amax/anaconda3/bin:${PATH:-}"
PYTHON="${PYTHON:-python}"

# N=12：不扩 TIFF 页
export OPTIGENESIS_EXPAND_TIFF_PAGES=0
unset OPTIGENESIS_MAX_PAGES_PER_TIFF 2>/dev/null || true
export OPTIGENESIS_BATCH_SIZE="${OPTIGENESIS_BATCH_SIZE:-4}"
export OPTIGENESIS_EPOCHS="${OPTIGENESIS_EPOCHS:-30}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export OPTIGENESIS_FRAME_AGG=uncertainty_weighted
export OPTIGENESIS_FRAME_AGG_TEMP="${OPTIGENESIS_FRAME_AGG_TEMP:-0.5}"
export OPTIGENESIS_FRAME_WEIGHT_SIGNAL=edl_u_amp
export OPTIGENESIS_FRAME_U_SCORE_BASE="${OPTIGENESIS_FRAME_U_SCORE_BASE:-0.5}"
export OPTIGENESIS_FRAME_U_SCORE_SCALE="${OPTIGENESIS_FRAME_U_SCORE_SCALE:-10}"

export OPTIGENESIS_ENABLE_FRAME_AUX=1
export OPTIGENESIS_FRAME_AUX_WEIGHT=0.3
export OPTIGENESIS_FRAME_AUX_TYPE=edl
export OPTIGENESIS_USE_WMA=0
export OPTIGENESIS_ENABLE_EMA=0
export OPTIGENESIS_ENABLE_AUX=0
export OPTIGENESIS_USE_CLINICAL=0

OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3_n12_uamp10}"
unset OPTIGENESIS_OUTPUT_DIR 2>/dev/null || true
mkdir -p "$OUT_ROOT"

SEEDS=(42)
HOSPITALS=(huaxi liaoning xiangya)
PARALLEL="${OPTIGENESIS_PARALLEL:-1}"
read -r -a GPUS <<< "${OPTIGENESIS_GPUS:-0 1 2}"

echo "======================================"
echo "N=12 + edl@0.3 + edl_u_amp（不扫 scale/τ/λ）"
echo "  EXPAND=0  BATCH=$OPTIGENESIS_BATCH_SIZE  EPOCHS=$OPTIGENESIS_EPOCHS"
echo "  signal=$OPTIGENESIS_FRAME_WEIGHT_SIGNAL  u_base=$OPTIGENESIS_FRAME_U_SCORE_BASE  scale=$OPTIGENESIS_FRAME_U_SCORE_SCALE  τ=$OPTIGENESIS_FRAME_AGG_TEMP"
echo "  FRAME_AUX weight=$OPTIGENESIS_FRAME_AUX_WEIGHT type=$OPTIGENESIS_FRAME_AUX_TYPE"
echo "  PARALLEL=$PARALLEL  GPUS=${GPUS[*]}"
echo "  OUT=$OUT_ROOT"
echo "======================================"

# 确保 train/val 切分存在
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

    echo ">>> $h seed=$s  GPU=$gpu (visible=0)  → $RUN_LOG"
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
echo "提数对照主候选: outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3/"

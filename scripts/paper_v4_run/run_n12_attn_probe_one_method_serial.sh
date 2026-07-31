#!/usr/bin/env bash
# 单卡串行：attn 改进探路（3 seeds × 三折）
# METHOD ∈ {eq, tau02, noaux, ls05}
# Seeds 默认：42 123 3407（attn 三 seed 均值最高且略优于同子集 B1）
#
#   CUDA_VISIBLE_DEVICES=0 METHOD=eq ./scripts/paper_v4_run/run_n12_attn_probe_one_method_serial.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="/home/amax/anaconda3/bin:${PATH:-}"
PYTHON="${PYTHON:-python}"

METHOD="${METHOD:?请设 METHOD=eq|tau02|noaux|ls05}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

export OPTIGENESIS_EXPAND_TIFF_PAGES=0
unset OPTIGENESIS_MAX_PAGES_PER_TIFF 2>/dev/null || true
export OPTIGENESIS_BATCH_SIZE="${OPTIGENESIS_BATCH_SIZE:-4}"
export OPTIGENESIS_EPOCHS="${OPTIGENESIS_EPOCHS:-30}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OPTIGENESIS_FRAME_AGG=attention
export OPTIGENESIS_FRAME_WEIGHT_SIGNAL=edl_u
unset OPTIGENESIS_FRAME_U_SCORE_BASE OPTIGENESIS_FRAME_U_SCORE_SCALE 2>/dev/null || true
export OPTIGENESIS_USE_WMA=0
export OPTIGENESIS_ENABLE_EMA=0
export OPTIGENESIS_ENABLE_AUX=0
export OPTIGENESIS_USE_CLINICAL=0
# 默认：broadcast@0.3 + τ=0.5 + mean query + 无 LS；各 METHOD 覆盖其中一处
export OPTIGENESIS_ENABLE_FRAME_AUX=1
export OPTIGENESIS_FRAME_AUX_WEIGHT=0.3
export OPTIGENESIS_FRAME_AUX_TYPE=edl
export OPTIGENESIS_FRAME_AUX_MODE=broadcast
export OPTIGENESIS_FRAME_AGG_TEMP=0.5
export OPTIGENESIS_FRAME_ATTN_QUERY=mean
export OPTIGENESIS_LABEL_SMOOTHING=0

case "$METHOD" in
  eq)
    # A: evidence query（p+ 加权特征作 query）
    export OPTIGENESIS_FRAME_ATTN_QUERY=evidence
    OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3_n12_attn_eq}"
    ;;
  tau02)
    # B: 降温 τ=0.2
    export OPTIGENESIS_FRAME_AGG_TEMP=0.2
    OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3_n12_attn_tau02}"
    ;;
  noaux)
    # C: 卸掉 FrameAux broadcast
    export OPTIGENESIS_ENABLE_FRAME_AUX=0
    OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3_n12_attn_noaux}"
    ;;
  ls05)
    # D: 患者级 label smoothing ε=0.05
    export OPTIGENESIS_LABEL_SMOOTHING=0.05
    OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3_n12_attn_ls05}"
    ;;
  *)
    echo "未知 METHOD=$METHOD（期望 eq|tau02|noaux|ls05）"; exit 1
    ;;
esac

unset OPTIGENESIS_OUTPUT_DIR 2>/dev/null || true
mkdir -p "$OUT_ROOT"

HOSPITALS=(huaxi liaoning xiangya)
# shellcheck disable=SC2206
SEEDS=(${SEEDS:-42 123 3407})

echo "======================================"
echo "attn 探路单卡 | METHOD=$METHOD | GPU=$GPU"
echo "  query=$OPTIGENESIS_FRAME_ATTN_QUERY τ=$OPTIGENESIS_FRAME_AGG_TEMP"
echo "  frame_aux=${OPTIGENESIS_ENABLE_FRAME_AUX} mode=${OPTIGENESIS_FRAME_AUX_MODE:--} w=${OPTIGENESIS_FRAME_AUX_WEIGHT:-0}"
echo "  label_smooth=$OPTIGENESIS_LABEL_SMOOTHING"
echo "  SEEDS=${SEEDS[*]}  HOSPITALS=${HOSPITALS[*]}"
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
fi

FAIL=0
for s in "${SEEDS[@]}"; do
  for h in "${HOSPITALS[@]}"; do
    export CUDA_VISIBLE_DEVICES="$GPU"
    export HOSPITAL_NAME="$h"
    export OPTIGENESIS_SEED="$s"
    export OPTIGENESIS_OUTPUT_DIR="${OUT_ROOT}/$h"
    export OPTIGENESIS_OUTPUT_RUN_NAME="seed_${s}"
    LOG_DIR="${OPTIGENESIS_OUTPUT_DIR}/${OPTIGENESIS_OUTPUT_RUN_NAME}/logs"
    mkdir -p "$LOG_DIR"
    RUN_LOG="${LOG_DIR}/train_console.log"
    if [[ "${SKIP_COMPLETED:-1}" == "1" && -f "$RUN_LOG" ]] && grep -q "训练完成！" "$RUN_LOG"; then
      echo "⏭️  跳过 $METHOD $h seed=$s"
      continue
    fi
    echo ">>> [$METHOD] $h seed=$s GPU=$GPU → $RUN_LOG"
    if ! "$PYTHON" main.py 2>&1 | tee "$RUN_LOG"; then
      echo "❌ 失败 $METHOD $h seed=$s"
      FAIL=1
    else
      echo "<<< 完成 $METHOD $h seed=$s"
    fi
  done
done

if [[ "$FAIL" -ne 0 ]]; then
  echo "有失败 → $OUT_ROOT"
  exit 1
fi
echo "全部完成 METHOD=$METHOD → $OUT_ROOT"

#!/usr/bin/env bash
# 单卡串行：N=12 + FrameAux edl@0.3 定稿补跑（多 seed × 三折）
# 由四卡总队列调用；也可单独：
#   CUDA_VISIBLE_DEVICES=0 METHOD=amp ./scripts/paper_v4_run/run_n12_final_one_method_serial.sh
#
# METHOD ∈ {amp, maxp, mil, attn}
# 默认 seeds：若 seed_42 已完成则只跑 123/2024/3407/114514；attn 无旧产物则跑满 5 seeds。
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="/home/amax/anaconda3/bin:${PATH:-}"
PYTHON="${PYTHON:-python}"

METHOD="${METHOD:?请设 METHOD=amp|maxp|mil|attn}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

export OPTIGENESIS_EXPAND_TIFF_PAGES=0
unset OPTIGENESIS_MAX_PAGES_PER_TIFF 2>/dev/null || true
export OPTIGENESIS_BATCH_SIZE="${OPTIGENESIS_BATCH_SIZE:-4}"
export OPTIGENESIS_EPOCHS="${OPTIGENESIS_EPOCHS:-30}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OPTIGENESIS_ENABLE_FRAME_AUX=1
export OPTIGENESIS_FRAME_AUX_WEIGHT=0.3
export OPTIGENESIS_FRAME_AUX_TYPE=edl
export OPTIGENESIS_FRAME_AGG_TEMP="${OPTIGENESIS_FRAME_AGG_TEMP:-0.5}"
export OPTIGENESIS_USE_WMA=0
export OPTIGENESIS_ENABLE_EMA=0
export OPTIGENESIS_ENABLE_AUX=0
export OPTIGENESIS_USE_CLINICAL=0

case "$METHOD" in
  amp)
    export OPTIGENESIS_FRAME_AGG=uncertainty_weighted
    export OPTIGENESIS_FRAME_WEIGHT_SIGNAL=edl_u_amp
    export OPTIGENESIS_FRAME_U_SCORE_BASE=0.5
    export OPTIGENESIS_FRAME_U_SCORE_SCALE=10
    export OPTIGENESIS_FRAME_AUX_MODE=broadcast
    OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3_n12_uamp10}"
    ;;
  maxp)
    export OPTIGENESIS_FRAME_AGG=uncertainty_weighted
    export OPTIGENESIS_FRAME_WEIGHT_SIGNAL=max_p_pool
    unset OPTIGENESIS_FRAME_U_SCORE_BASE OPTIGENESIS_FRAME_U_SCORE_SCALE 2>/dev/null || true
    export OPTIGENESIS_FRAME_AUX_MODE=broadcast
    OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3_n12_maxp}"
    ;;
  mil)
    export OPTIGENESIS_FRAME_AGG=uncertainty_weighted
    export OPTIGENESIS_FRAME_WEIGHT_SIGNAL=edl_u
    unset OPTIGENESIS_FRAME_U_SCORE_BASE OPTIGENESIS_FRAME_U_SCORE_SCALE 2>/dev/null || true
    export OPTIGENESIS_FRAME_AUX_MODE=mil
    export OPTIGENESIS_FRAME_AUX_MIL_POS_THR="${OPTIGENESIS_FRAME_AUX_MIL_POS_THR:-0.5}"
    OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3_n12_mil}"
    ;;
  attn)
    export OPTIGENESIS_FRAME_AGG=attention
    export OPTIGENESIS_FRAME_WEIGHT_SIGNAL=edl_u  # 忽略；加权由 attention 学
    unset OPTIGENESIS_FRAME_U_SCORE_BASE OPTIGENESIS_FRAME_U_SCORE_SCALE 2>/dev/null || true
    export OPTIGENESIS_FRAME_AUX_MODE=broadcast
    OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3_n12_attn}"
    ;;
  *)
    echo "未知 METHOD=$METHOD"; exit 1
    ;;
esac

unset OPTIGENESIS_OUTPUT_DIR 2>/dev/null || true
mkdir -p "$OUT_ROOT"

HOSPITALS=(huaxi liaoning xiangya)
ALL_SEEDS=(42 123 2024 3407 114514)
# 自动：若三折 seed_42 都完成则跳过 42；否则包含 42
SEEDS=()
seed42_done=1
for h in "${HOSPITALS[@]}"; do
  f="$OUT_ROOT/$h/seed_42/logs/train_console.log"
  if [[ ! -f "$f" ]] || ! grep -q "训练完成！" "$f"; then
    seed42_done=0
  fi
done
if [[ "${FORCE_ALL_SEEDS:-0}" == "1" ]]; then
  SEEDS=("${ALL_SEEDS[@]}")
elif [[ "$seed42_done" == "1" ]]; then
  SEEDS=(123 2024 3407 114514)
  echo "✅ seed_42 三折已齐 → 只跑剩余 4 seeds"
else
  SEEDS=("${ALL_SEEDS[@]}")
  echo "ℹ️  seed_42 未齐 → 跑满 5 seeds"
fi

echo "======================================"
echo "定稿单卡串行 | METHOD=$METHOD | GPU=$GPU"
echo "  AGG=$OPTIGENESIS_FRAME_AGG signal=${OPTIGENESIS_FRAME_WEIGHT_SIGNAL:--} aux_mode=$OPTIGENESIS_FRAME_AUX_MODE"
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

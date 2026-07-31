#!/usr/bin/env bash
# 单卡串行：旧 Attn + 稳定性模块（降强度）
# METHOD ∈ {wma, ema, aux, wma_pat, ema_aux, wea_tuned}
#   wma       : 仅 WMA，C=0.1；帧辅仍跟随 WMA
#   ema       : 仅 EMA，decay=0.99
#   aux       : 仅 Aux，vision=0.1
#   wma_pat   : 仅患者级 WMA（C=0.1），帧辅不跟 WMA
#   ema_aux   : EMA(0.99) + Aux(0.1) 搭配
#   wea_tuned : EMA(0.99) + Aux(0.1) + WMA(C=0.1，帧辅跟)——相对旧 WEA 全家桶降强度
#
#   CUDA_VISIBLE_DEVICES=0 METHOD=wma SEEDS="42 123" ./scripts/paper_v4_run/run_n12_attn_stab_probe_one.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="/home/amax/anaconda3/bin:${PATH:-}"
PYTHON="${PYTHON:-python}"

METHOD="${METHOD:?请设 METHOD=wma|ema|aux|wma_pat|ema_aux|wea_tuned}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

export OPTIGENESIS_EXPAND_TIFF_PAGES=0
unset OPTIGENESIS_MAX_PAGES_PER_TIFF 2>/dev/null || true
export OPTIGENESIS_BATCH_SIZE="${OPTIGENESIS_BATCH_SIZE:-4}"
export OPTIGENESIS_EPOCHS="${OPTIGENESIS_EPOCHS:-30}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# 旧 Attn 基座
export OPTIGENESIS_FRAME_AGG=attention
export OPTIGENESIS_FRAME_ATTN_QUERY=mean
export OPTIGENESIS_FRAME_AGG_TEMP=0.5
export OPTIGENESIS_FRAME_WEIGHT_SIGNAL=edl_u
unset OPTIGENESIS_FRAME_U_SCORE_BASE OPTIGENESIS_FRAME_U_SCORE_SCALE 2>/dev/null || true
export OPTIGENESIS_ENABLE_FRAME_AUX=1
export OPTIGENESIS_FRAME_AUX_WEIGHT=0.3
export OPTIGENESIS_FRAME_AUX_TYPE=edl
export OPTIGENESIS_FRAME_AUX_MODE=broadcast
export OPTIGENESIS_USE_CLINICAL=0
export OPTIGENESIS_LABEL_SMOOTHING=0

# 默认全关，再按 METHOD 打开；强度可用环境变量覆盖
export OPTIGENESIS_USE_WMA=0
export OPTIGENESIS_ENABLE_EMA=0
export OPTIGENESIS_ENABLE_AUX=0
export OPTIGENESIS_FRAME_AUX_USE_WMA=1
export OPTIGENESIS_WMA_C="${OPTIGENESIS_WMA_C:-0.1}"
export OPTIGENESIS_WMA_WARMUP="${OPTIGENESIS_WMA_WARMUP:-10}"
export OPTIGENESIS_WMA_TEMP="${OPTIGENESIS_WMA_TEMP:-1.0}"
export OPTIGENESIS_EMA_DECAY="${OPTIGENESIS_EMA_DECAY:-0.99}"
export OPTIGENESIS_AUX_W_VISION="${OPTIGENESIS_AUX_W_VISION:-0.1}"
export OPTIGENESIS_AUX_W_CLINICAL="${OPTIGENESIS_AUX_W_CLINICAL:-0.1}"

case "$METHOD" in
  wma)
    export OPTIGENESIS_USE_WMA=1
    export OPTIGENESIS_FRAME_AUX_USE_WMA=1
    OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3_n12_attn_wmaC01}"
    ;;
  wma_pat)
    export OPTIGENESIS_USE_WMA=1
    export OPTIGENESIS_FRAME_AUX_USE_WMA=0
    OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3_n12_attn_wmaPatC01}"
    ;;
  ema)
    export OPTIGENESIS_ENABLE_EMA=1
    OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3_n12_attn_ema099}"
    ;;
  aux)
    export OPTIGENESIS_ENABLE_AUX=1
    OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3_n12_attn_aux01}"
    ;;
  ema_aux)
    # 常见搭配：EMA + 视觉 Aux（降强度：decay=0.99, aux=0.1）
    export OPTIGENESIS_ENABLE_EMA=1
    export OPTIGENESIS_ENABLE_AUX=1
    OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-outputs/paper_v4/baseline/ours_uw_frameaux_t0_edl_w0.3_n12_attn_ema099_aux01}"
    ;;
  wea_tuned)
    # 降强度全家桶：相对旧 WEA（C≈0.2 + 默认 Aux）→ C=0.1 + EMA0.99 + Aux0.1
    export OPTIGENESIS_USE_WMA=1
    export OPTIGENESIS_FRAME_AUX_USE_WMA=1
    export OPTIGENESIS_ENABLE_EMA=1
    export OPTIGENESIS_ENABLE_AUX=1
    OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-outputs/paper_v4/baseline/ours_uw_frameaux_t0_edl_w0.3_n12_attn_wea_tuned}"
    ;;
  *)
    echo "未知 METHOD=$METHOD（期望 wma|ema|aux|wma_pat|ema_aux|wea_tuned）"; exit 1
    ;;
esac

unset OPTIGENESIS_OUTPUT_DIR 2>/dev/null || true
mkdir -p "$OUT_ROOT"

HOSPITALS=(huaxi liaoning xiangya)
# shellcheck disable=SC2206
SEEDS=(${SEEDS:-42 123})

echo "======================================"
echo "Attn 稳定性单模块探路 | METHOD=$METHOD | GPU=$GPU"
echo "  WMA=$OPTIGENESIS_USE_WMA (C=$OPTIGENESIS_WMA_C, frame_follow=$OPTIGENESIS_FRAME_AUX_USE_WMA)"
echo "  EMA=$OPTIGENESIS_ENABLE_EMA (decay=$OPTIGENESIS_EMA_DECAY)"
echo "  AUX=$OPTIGENESIS_ENABLE_AUX (vis=$OPTIGENESIS_AUX_W_VISION)"
echo "  SEEDS=${SEEDS[*]}  OUT=$OUT_ROOT"
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

#!/usr/bin/env bash
# 对比 · DSMIL（Li et al., CVPR 2021）
# 机位建议：GPU2 / 服务器 #3
#
# 状态：骨架；需接入双流 MIL（max 关键实例 + 注意力相关流）
# 参考：https://github.com/binli123/dsmil-wsi 或 mahmoodlab/MIL-Lab
#
#   CUDA_VISIBLE_DEVICES=2 SEEDS="42 123 2024 3407 114514" \
#     ./scripts/paper_v4_run/run_cmp_dsmil_one.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="/home/amax/anaconda3/bin:${PATH:-}"
PYTHON="${OPTIGENESIS_PYTHON:-/home/amax/anaconda3/bin/python3}"

# shellcheck disable=SC2206
SEEDS=(${SEEDS:-42 123 2024 3407 114514})
GPU="${CUDA_VISIBLE_DEVICES:-0}"
MAX_EPOCHS="${OPTIGENESIS_EPOCHS:-30}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CMP_ROOT="$ROOT/outputs/paper_v4/对比实验"
OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-$CMP_ROOT/12[对比] dsmil_sitebag_n2_aggmean_r50}"

is_run_complete() {
  local log_file="$1"
  [[ -f "$log_file" ]] && grep -q "训练完成！" "$log_file"
}

require_dsmil_hook() {
  export OPTIGENESIS_FRAME_AGG=dsmil
}

mkdir -p "$OUT_ROOT"
echo "======================================"
echo "CMP DSMIL | GPU=$GPU | seeds=${SEEDS[*]}"
echo "OUT=$OUT_ROOT"
echo "======================================"

if [[ ! -f "$ROOT/dataset/train_xyi.csv" ]]; then
  "$PYTHON" data/prepare_xyi_sitebag_splits.py --write --split-seed 20260731 --val-ratio 0.2
fi

require_dsmil_hook

for SEED in "${SEEDS[@]}"; do
  RUN_DIR="${OUT_ROOT}/seed_${SEED}"
  LOG="${RUN_DIR}/logs/train_console.log"
  mkdir -p "${RUN_DIR}/logs"
  if [[ "${FORCE_RERUN:-0}" != "1" && "${SKIP_COMPLETED}" == "1" ]] && is_run_complete "$LOG"; then
    echo "✅ skip seed=$SEED"
    continue
  fi
  echo "▶ DSMIL seed=$SEED"

  # shellcheck source=/dev/null
  source "$ROOT/scripts/paper_v4_run/cmp_env_common.inc.sh"

  export OPTIGENESIS_BACKBONE="${OPTIGENESIS_BACKBONE:-resnet50}"

  # 双流替换 UWA；关 FrameAux / EMA（默认）
  export OPTIGENESIS_FRAME_AGG=dsmil
  export OPTIGENESIS_FRAME_AGG_TEMP=1.0
  export OPTIGENESIS_ENABLE_FRAME_AUX=0
  export OPTIGENESIS_ENABLE_EMA=0
  export OPTIGENESIS_EMA_DECAY=0.99

  # —— DSMIL 专用 ——
  export OPTIGENESIS_CMP_METHOD=dsmil
  export OPTIGENESIS_DSMIL_ATTN_DIM="${OPTIGENESIS_DSMIL_ATTN_DIM:-128}"
  export OPTIGENESIS_DSMIL_DROPOUT="${OPTIGENESIS_DSMIL_DROPOUT:-0.25}"
  # 患者分：双流融合（实现时：0.5*(max_stream + attn_stream) 或论文默认）
  export OPTIGENESIS_DSMIL_FUSE="${OPTIGENESIS_DSMIL_FUSE:-mean}"

  set +e
  "$PYTHON" -u main.py 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ "$rc" -ne 0 ]]; then
    echo "❌ DSMIL seed=$SEED rc=$rc"
    exit "$rc"
  fi
  echo "✅ DSMIL seed=$SEED done"
done

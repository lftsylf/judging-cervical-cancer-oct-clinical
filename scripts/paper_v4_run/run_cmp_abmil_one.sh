#!/usr/bin/env bash
# 对比 · ABMIL（Ilse et al., ICML 2018）
# 机位建议：GPU0 / 服务器 #1
#
# 状态：骨架可启；真正 ABMIL 需代码接入（见 CMP_BASELINE2_LAUNCH.md §ABMIL）
# 临时冒烟（非论文口径）：OPTIGENESIS_CMP_ABMIL_INTERIM=1 → 用现有 FRAME_AGG=attention
#
#   CUDA_VISIBLE_DEVICES=0 SEEDS="42 123 2024 3407 114514" \
#     ./scripts/paper_v4_run/run_cmp_abmil_one.sh
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
OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-$CMP_ROOT/10[对比] abmil_sitebag_n2_aggmean_r50}"

is_run_complete() {
  local log_file="$1"
  [[ -f "$log_file" ]] && grep -q "训练完成！" "$log_file"
}

require_hook_or_interim() {
  if [[ "${OPTIGENESIS_CMP_ABMIL_INTERIM:-0}" == "1" ]]; then
    echo "⚠️  INTERIM：FRAME_AGG=attention（勿当正式 ABMIL）"
    export OPTIGENESIS_FRAME_AGG=attention
    export OPTIGENESIS_FRAME_ATTN_QUERY="${OPTIGENESIS_FRAME_ATTN_QUERY:-mean}"
    return 0
  fi
  export OPTIGENESIS_FRAME_AGG=abmil
}

mkdir -p "$OUT_ROOT"
echo "======================================"
echo "CMP ABMIL | GPU=$GPU | seeds=${SEEDS[*]}"
echo "OUT=$OUT_ROOT"
echo "======================================"

if [[ ! -f "$ROOT/dataset/train_xyi.csv" ]]; then
  "$PYTHON" data/prepare_xyi_sitebag_splits.py --write --split-seed 20260731 --val-ratio 0.2
fi

require_hook_or_interim

for SEED in "${SEEDS[@]}"; do
  RUN_DIR="${OUT_ROOT}/seed_${SEED}"
  LOG="${RUN_DIR}/logs/train_console.log"
  mkdir -p "${RUN_DIR}/logs"
  if [[ "${FORCE_RERUN:-0}" != "1" && "${SKIP_COMPLETED}" == "1" ]] && is_run_complete "$LOG"; then
    echo "✅ skip seed=$SEED"
    continue
  fi
  echo "▶ ABMIL seed=$SEED"

  # shellcheck source=/dev/null
  source "$ROOT/scripts/paper_v4_run/cmp_env_common.inc.sh"

  export OPTIGENESIS_BACKBONE="${OPTIGENESIS_BACKBONE:-resnet50}"

  # 聚合：ABMIL 替换 UWA；关帧 EDL-UWA 信号
  # （正式 abmil / interim attention 已在 require_* 里设好 FRAME_AGG）
  export OPTIGENESIS_FRAME_AGG_TEMP="${OPTIGENESIS_FRAME_AGG_TEMP:-1.0}"
  export OPTIGENESIS_FRAME_WEIGHT_SIGNAL=edl_u   # abmil 路径应忽略；保留仅为不炸配置

  # 损失：患者级主损；默认关 FrameAux（经典 ABMIL 无实例辅损）
  # 若要对齐「也有弱监督」可 export OPTIGENESIS_ENABLE_FRAME_AUX=1
  export OPTIGENESIS_ENABLE_FRAME_AUX="${OPTIGENESIS_ENABLE_FRAME_AUX:-0}"
  export OPTIGENESIS_FRAME_AUX_WEIGHT=0.3
  export OPTIGENESIS_FRAME_AUX_TYPE=edl
  export OPTIGENESIS_FRAME_AUX_MODE=mil
  export OPTIGENESIS_FRAME_AUX_MIL_POS_THR=0.5

  # 公平：对比方法默认不开 EMA（与 WMA 对比(5) 同类；若要与 4[final] 对齐可开）
  export OPTIGENESIS_ENABLE_EMA="${OPTIGENESIS_ENABLE_EMA:-0}"
  export OPTIGENESIS_EMA_DECAY=0.99

  # —— 接入点标记（实现时读这些）——
  export OPTIGENESIS_CMP_METHOD=abmil
  export OPTIGENESIS_ABMIL_GATE="${OPTIGENESIS_ABMIL_GATE:-1}"   # 1=gated tanh(V)·sig(U)（Ilse）
  export OPTIGENESIS_ABMIL_ATTN_DIM="${OPTIGENESIS_ABMIL_ATTN_DIM:-128}"

  set +e
  "$PYTHON" -u main.py 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ "$rc" -ne 0 ]]; then
    echo "❌ ABMIL seed=$SEED rc=$rc"
    exit "$rc"
  fi
  echo "✅ ABMIL seed=$SEED done"
done

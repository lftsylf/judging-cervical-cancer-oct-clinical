#!/usr/bin/env bash
# 对比 · UBIX（de Vente et al., MedIA 2024）
# 机位建议：GPU1 / 服务器 #2
#
# 状态：骨架；需接入推理期 Uncertainty-Based Instance eXclusion
# 参考仓：https://github.com/qurAI-amsterdam/ubix-for-reliable-classification
#
# 公平设定（本骨架默认）：
#   训练：sitebag + FRAME_AGG=equal（帧级分类头，等权池化；无 EDL-UWA、无你们的 edl_u）
#   推理：在池化前按 MC-Dropout 不确定度 soft/hard 降权（UBIX）
#   禁止：把 OPTIGENESIS_FRAME_WEIGHT_SIGNAL=edl_u 接到 UBIX（否则半消融）
#
#   CUDA_VISIBLE_DEVICES=1 SEEDS="42 123 2024 3407 114514" \
#     ./scripts/paper_v4_run/run_cmp_ubix_one.sh
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
OUT_ROOT="${OPTIGENESIS_OUTPUT_DIR:-$CMP_ROOT/11[对比] ubix_sitebag_n2_equal_mcdrop_r50}"

is_run_complete() {
  local log_file="$1"
  [[ -f "$log_file" ]] && grep -q "训练完成！" "$log_file"
}

require_ubix_hook() {
  if "$PYTHON" - <<'PY'
import importlib.util
spec = importlib.util.find_spec("models.cmp_ubix")
raise SystemExit(0 if spec is not None else 1)
PY
  then
    return 0
  fi
  echo "❌ 缺少 models.cmp_ubix"; exit 2
}

mkdir -p "$OUT_ROOT"
echo "======================================"
echo "CMP UBIX | GPU=$GPU | seeds=${SEEDS[*]}"
echo "OUT=$OUT_ROOT"
echo "======================================"

if [[ ! -f "$ROOT/dataset/train_xyi.csv" ]]; then
  "$PYTHON" data/prepare_xyi_sitebag_splits.py --write --split-seed 20260731 --val-ratio 0.2
fi

require_ubix_hook

for SEED in "${SEEDS[@]}"; do
  RUN_DIR="${OUT_ROOT}/seed_${SEED}"
  LOG="${RUN_DIR}/logs/train_console.log"
  mkdir -p "${RUN_DIR}/logs"
  if [[ "${FORCE_RERUN:-0}" != "1" && "${SKIP_COMPLETED}" == "1" ]] && is_run_complete "$LOG"; then
    echo "✅ skip seed=$SEED"
    continue
  fi
  echo "▶ UBIX seed=$SEED"

  # shellcheck source=/dev/null
  source "$ROOT/scripts/paper_v4_run/cmp_env_common.inc.sh"

  export OPTIGENESIS_BACKBONE="${OPTIGENESIS_BACKBONE:-resnet50}"

  # 训练底座：等权帧聚合（非 UWA）；可开轻量 FrameAux 或不开——默认关，突出 UBIX
  export OPTIGENESIS_FRAME_AGG=equal
  export OPTIGENESIS_FRAME_AGG_TEMP=0.5
  export OPTIGENESIS_FRAME_WEIGHT_SIGNAL=edl_u   # equal 路径忽略；切勿给 UBIX 用
  export OPTIGENESIS_ENABLE_FRAME_AUX=0
  export OPTIGENESIS_ENABLE_EMA=0
  export OPTIGENESIS_EMA_DECAY=0.99

  # —— UBIX 专用（实现时读）——
  export OPTIGENESIS_CMP_METHOD=ubix
  export OPTIGENESIS_UBIX_ENABLE=1
  export OPTIGENESIS_UBIX_MODE="${OPTIGENESIS_UBIX_MODE:-soft}"       # soft | hard
  export OPTIGENESIS_UBIX_U_SOURCE="${OPTIGENESIS_UBIX_U_SOURCE:-mc_dropout}"  # 禁止 edl_u
  export OPTIGENESIS_UBIX_MC_T="${OPTIGENESIS_UBIX_MC_T:-16}"         # MC 前向次数
  export OPTIGENESIS_UBIX_DROPOUT_P="${OPTIGENESIS_UBIX_DROPOUT_P:-0.2}"
  export OPTIGENESIS_UBIX_THRESH="${OPTIGENESIS_UBIX_THRESH:-}"       # hard 阈值；空=用 val 选
  # 推理仍走六窗 mean（与 Ours 对齐）；若改为整袋一次：SITEBAG_EVAL_AGG 需另设计
  export OPTIGENESIS_SITEBAG_EVAL_AGG=mean

  set +e
  "$PYTHON" -u main.py 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ "$rc" -ne 0 ]]; then
    echo "❌ UBIX seed=$SEED rc=$rc"
    exit "$rc"
  fi
  echo "✅ UBIX seed=$SEED done"
done

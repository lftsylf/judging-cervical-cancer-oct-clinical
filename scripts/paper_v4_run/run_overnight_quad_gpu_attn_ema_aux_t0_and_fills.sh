#!/usr/bin/env bash
# Paper v4 · 四卡并行：
#   GPU0–1  EMA+Aux（T0 满 5 seeds）——用户指定两卡
#   GPU2    补齐 WMA C=0.1 单独 的 seed 2024/3407/114514
#   GPU3    补齐 Aux 0.1 单独 的 seed 2024/3407/114514
#
# 配置（相对全家桶 WEA 已降强度）：
#   EMA decay=0.99  + Aux vision=0.1
#   WMA C=0.1（帧辅跟随）；Aux alone vision=0.1
#
# 用法:
#   ./run_experiment.sh --detach ./tsy_loho \
#     ./scripts/paper_v4_run/run_overnight_quad_gpu_attn_ema_aux_t0_and_fills.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="/home/amax/anaconda3/bin:${PATH:-}"

ONE="$ROOT/scripts/paper_v4_run/run_n12_attn_stab_probe_one.sh"
STATUS="$ROOT/outputs/paper_v4/baseline/OVERNIGHT_ATTN_EMA_AUX_T0_FILLS_STATUS.md"
mkdir -p "$(dirname "$STATUS")" "$ROOT/logs"
chmod +x "$ONE"

write_status() {
  {
    echo "# EMA+Aux T0 + WMA/Aux 补 seed 状态"
    echo ""
    echo "- 更新: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "- $1"
    echo ""
    echo "| GPU | 任务 | seeds | 输出 |"
    echo "|----:|------|-------|------|"
    echo "| 0 | EMA+Aux T0 | 42 123 | \`..._attn_ema099_aux01/\` |"
    echo "| 1 | EMA+Aux T0 | 2024 3407 114514 | 同上 |"
    echo "| 2 | WMA alone 补齐 | 2024 3407 114514 | \`..._attn_wmaC01/\` |"
    echo "| 3 | Aux alone 补齐 | 2024 3407 114514 | \`..._attn_aux01/\` |"
    echo ""
    echo "- 对照门槛：Attn 5-seed **0.544±0.022**；B1 **0.537±0.034**"
    echo "- 2-seed 探路：WMA≈0.560（持平）、Aux≈0.553、EMA alone 差；现试 EMA∥Aux 搭配"
  } >"$STATUS"
  echo "[STATUS] $1"
}

export SKIP_DATASET_BIND=1
export SKIP_COMPLETED=1

write_status "四卡启动中…"

PIDS=()
FAIL=0

run_one() {
  local gpu="$1" method="$2" seeds="$3" tag="$4"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export METHOD="$method"
    export SEEDS="$seeds"
    # 显式降强度（可被外层覆盖）
    export OPTIGENESIS_EMA_DECAY="${OPTIGENESIS_EMA_DECAY:-0.99}"
    export OPTIGENESIS_AUX_W_VISION="${OPTIGENESIS_AUX_W_VISION:-0.1}"
    export OPTIGENESIS_AUX_W_CLINICAL="${OPTIGENESIS_AUX_W_CLINICAL:-0.1}"
    export OPTIGENESIS_WMA_C="${OPTIGENESIS_WMA_C:-0.1}"
    bash "$ONE"
  ) >"$ROOT/logs/overnight_attn_${tag}.log" 2>&1 &
  PIDS+=($!)
  echo "GPU${gpu} ${tag} method=$method seeds=[$seeds] pid=${PIDS[-1]}  log=logs/overnight_attn_${tag}.log"
}

# 两卡跑 EMA+Aux T0
run_one 0 ema_aux "42 123" ema_aux_s42_123
run_one 1 ema_aux "2024 3407 114514" ema_aux_s2024_3407_114514
# 两卡补齐单模块剩余 3 seeds
run_one 2 wma "2024 3407 114514" wma_fill3
run_one 3 aux "2024 3407 114514" aux_fill3

write_status "四卡已启动 PIDs=${PIDS[*]}；断线看 logs/overnight_attn_ema_aux_*.log / *_fill3.log"

for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then FAIL=1; fi
done

if [[ "$FAIL" -ne 0 ]]; then
  write_status "有任务失败，请检查对应 overnight_attn_*.log"
  exit 1
fi
write_status "全部完成：EMA+Aux T0 + WMA/Aux 三 seed 补齐。请提数。"
echo "全部完成。"

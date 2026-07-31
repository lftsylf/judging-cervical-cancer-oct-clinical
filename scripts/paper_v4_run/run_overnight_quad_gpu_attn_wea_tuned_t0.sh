#!/usr/bin/env bash
# Paper v4 · 四卡并行 T0：Attn + EMA0.99 + Aux0.1 + WMA C=0.1（降强度全家桶）
#
# 动机：EMA+Aux 已定稿 0.556±0.014 > Attn 0.544；WMA alone ≈ Attn。
#       再叠调试后的 WMA，看三者能否再涨；若不能超过 EMA+Aux，则弃 WMA。
# 对照门槛：EMA+Aux 0.556±0.014；Attn 0.544±0.022；旧 WEA（强）0.527±0.058
#
# 用法:
#   ./run_experiment.sh --detach ./tsy_loho \
#     ./scripts/paper_v4_run/run_overnight_quad_gpu_attn_wea_tuned_t0.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="/home/amax/anaconda3/bin:${PATH:-}"

ONE="$ROOT/scripts/paper_v4_run/run_n12_attn_stab_probe_one.sh"
STATUS="$ROOT/outputs/paper_v4/baseline/OVERNIGHT_ATTN_WEA_TUNED_T0_STATUS.md"
mkdir -p "$(dirname "$STATUS")" "$ROOT/logs"
chmod +x "$ONE"

write_status() {
  {
    echo "# Attn + WEA-tuned（EMA0.99+Aux0.1+WMA C=0.1）T0 状态"
    echo ""
    echo "- 更新: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "- $1"
    echo ""
    echo "| GPU | seeds | 输出 |"
    echo "|----:|-------|------|"
    echo "| 0 | 42 | \`..._attn_wea_tuned/\` |"
    echo "| 1 | 123 | 同上 |"
    echo "| 2 | 2024 3407 | 同上 |"
    echo "| 3 | 114514 | 同上 |"
    echo ""
    echo "- 配置：EMA decay=0.99，Aux vision=0.1，WMA C=0.1（帧辅跟随）"
    echo "- 门槛：EMA+Aux **0.556±0.014**；Attn **0.544±0.022**；旧 WEA **0.527±0.058**"
    echo "- 判定：若 ≤ EMA+Aux（Δ≲0），弃 WMA，主方法锁 EMA+Aux"
  } >"$STATUS"
  echo "[STATUS] $1"
}

export SKIP_DATASET_BIND=1
export SKIP_COMPLETED=1

write_status "四卡启动中…"

PIDS=()
FAIL=0

run_one() {
  local gpu="$1" seeds="$2" tag="$3"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export METHOD=wea_tuned
    export SEEDS="$seeds"
    export OPTIGENESIS_EMA_DECAY="${OPTIGENESIS_EMA_DECAY:-0.99}"
    export OPTIGENESIS_AUX_W_VISION="${OPTIGENESIS_AUX_W_VISION:-0.1}"
    export OPTIGENESIS_AUX_W_CLINICAL="${OPTIGENESIS_AUX_W_CLINICAL:-0.1}"
    export OPTIGENESIS_WMA_C="${OPTIGENESIS_WMA_C:-0.1}"
    bash "$ONE"
  ) >"$ROOT/logs/overnight_attn_wea_tuned_${tag}.log" 2>&1 &
  PIDS+=($!)
  echo "GPU${gpu} wea_tuned seeds=[$seeds] pid=${PIDS[-1]}  log=logs/overnight_attn_wea_tuned_${tag}.log"
}

# 四卡按 seed 切分（15 折）
run_one 0 "42" s42
run_one 1 "123" s123
run_one 2 "2024 3407" s2024_3407
run_one 3 "114514" s114514

write_status "四卡已启动 PIDs=${PIDS[*]}；断线看 logs/overnight_attn_wea_tuned_*.log"

for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then FAIL=1; fi
done

if [[ "$FAIL" -ne 0 ]]; then
  write_status "有任务失败，请检查 logs/overnight_attn_wea_tuned_*.log"
  exit 1
fi
write_status "全部完成：wea_tuned T0（5 seeds × 3 折）。请提数 vs EMA+Aux。"
echo "全部完成。"

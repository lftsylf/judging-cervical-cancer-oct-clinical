#!/usr/bin/env bash
# Paper v4 · 四卡 T0：旧 Attn（broadcast@0.3）+ WMA + EMA + Aux
#
# GPU0  seed 42
# GPU1  seed 123
# GPU2  seed 2024
# GPU3  seed 3407 + 114514（串行）
#
# 输出：outputs/paper_v4/baseline/ours_uw_frameaux_t0_edl_w0.3_n12_attn_wea/
#
# 用法:
#   ./run_experiment.sh --detach ./tsy_loho \
#     ./scripts/paper_v4_run/run_overnight_quad_gpu_attn_wea_t0.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="/home/amax/anaconda3/bin:${PATH:-}"

ONE="$ROOT/scripts/paper_v4_run/run_n12_attn_wea_one_serial.sh"
STATUS="$ROOT/outputs/paper_v4/baseline/OVERNIGHT_ATTN_WEA_T0_STATUS.md"
OUT="outputs/paper_v4/baseline/ours_uw_frameaux_t0_edl_w0.3_n12_attn_wea"
mkdir -p "$(dirname "$STATUS")" "$ROOT/logs"
chmod +x "$ONE"

write_status() {
  {
    echo "# 四卡 Attn+WMA/EMA/Aux T0 状态"
    echo ""
    echo "- 更新: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "- $1"
    echo "- 方法：旧 Attn（mean query, τ=0.5, broadcast@0.3）+ **WMA + EMA + Aux**"
    echo "- Seeds：42, 123, 2024, 3407, 114514（T0）"
    echo "- 输出：\`$OUT/\`"
    echo "- 对照：旧 Attn 无三模块 0.544±0.022；B1 0.537±0.034"
    echo ""
    echo "| GPU | seeds |"
    echo "|----:|-------|"
    echo "| 0 | 42 |"
    echo "| 1 | 123 |"
    echo "| 2 | 2024 |"
    echo "| 3 | 3407, 114514 |"
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
    export SEEDS="$seeds"
    bash "$ONE"
  ) >"$ROOT/logs/overnight_attn_wea_${tag}.log" 2>&1 &
  PIDS+=($!)
  echo "GPU${gpu} seeds=[$seeds] pid=${PIDS[-1]}  log=logs/overnight_attn_wea_${tag}.log"
}

run_one 0 "42" s42
run_one 1 "123" s123
run_one 2 "2024" s2024
run_one 3 "3407 114514" s3407_114514

write_status "四卡已启动 PIDs=${PIDS[*]}；断线看 logs/overnight_attn_wea_*.log"

for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then FAIL=1; fi
done

if [[ "$FAIL" -ne 0 ]]; then
  write_status "有任务失败，请检查 logs/overnight_attn_wea_*.log"
  exit 1
fi
write_status "Attn+WEA T0 全部完成。请提数对比旧 Attn / B1。"
echo "全部完成。"

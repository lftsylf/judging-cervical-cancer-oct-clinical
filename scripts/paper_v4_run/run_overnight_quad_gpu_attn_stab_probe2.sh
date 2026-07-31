#!/usr/bin/env bash
# Paper v4 · 四卡：Attn 稳定性模块分开探路（降强度 · 2 seeds）
#
# Seeds：42, 123
# GPU0  wma     （WMA C=0.1，帧辅跟随 WMA）
# GPU1  ema     （EMA decay=0.99）
# GPU2  aux     （Aux vision=0.1）
# GPU3  wma_pat （WMA C=0.1，仅患者级；帧辅不跟 WMA）
#
# 用法:
#   ./run_experiment.sh --detach ./tsy_loho \
#     ./scripts/paper_v4_run/run_overnight_quad_gpu_attn_stab_probe2.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="/home/amax/anaconda3/bin:${PATH:-}"

ONE="$ROOT/scripts/paper_v4_run/run_n12_attn_stab_probe_one.sh"
STATUS="$ROOT/outputs/paper_v4/baseline/OVERNIGHT_ATTN_STAB_PROBE2_STATUS.md"
mkdir -p "$(dirname "$STATUS")" "$ROOT/logs"
chmod +x "$ONE"

write_status() {
  {
    echo "# 四卡 Attn 稳定性分开探路状态"
    echo ""
    echo "- 更新: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "- $1"
    echo "- Seeds：**42, 123**（理由见 README）"
    echo "- 对照同子集：Attn 折均值 42=0.574、123=0.542；均值≈0.558"
    echo ""
    echo "| GPU | METHOD | 配置要点 | 输出 |"
    echo "|----:|--------|----------|------|"
    echo "| 0 | wma | C=0.1，帧辅跟 WMA | \`..._attn_wmaC01/\` |"
    echo "| 1 | ema | decay=0.99 | \`..._attn_ema099/\` |"
    echo "| 2 | aux | vision=0.1 | \`..._attn_aux01/\` |"
    echo "| 3 | wma_pat | C=0.1，帧辅不跟 WMA | \`..._attn_wmaPatC01/\` |"
  } >"$STATUS"
  echo "[STATUS] $1"
}

export SKIP_DATASET_BIND=1
export SKIP_COMPLETED=1
export SEEDS="42 123"

write_status "四卡启动中…"

PIDS=()
FAIL=0

run_one() {
  local gpu="$1" method="$2"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export METHOD="$method"
    export SEEDS="42 123"
    bash "$ONE"
  ) >"$ROOT/logs/overnight_attn_stab_${method}.log" 2>&1 &
  PIDS+=($!)
  echo "GPU${gpu} ${method} pid=${PIDS[-1]}  log=logs/overnight_attn_stab_${method}.log"
}

run_one 0 wma
run_one 1 ema
run_one 2 aux
run_one 3 wma_pat

write_status "四卡已启动 PIDs=${PIDS[*]}；断线看 logs/overnight_attn_stab_*.log"

for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then FAIL=1; fi
done

if [[ "$FAIL" -ne 0 ]]; then
  write_status "有任务失败，请检查 logs/overnight_attn_stab_*.log"
  exit 1
fi
write_status "四模块探路完成。请提数对比同子集旧 Attn。"
echo "全部完成。"

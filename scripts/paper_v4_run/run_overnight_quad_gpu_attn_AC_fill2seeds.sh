#!/usr/bin/env bash
# Paper v4 · 四卡补齐 A(eq) / C(noaux) 的 seed 2024 & 114514 → 满 5-seed 定稿
#
# GPU0  A eq      seed=2024
# GPU1  A eq      seed=114514
# GPU2  C noaux   seed=2024
# GPU3  C noaux   seed=114514
#
# 已有 42/123/3407 会因 SKIP_COMPLETED=1 自动跳过（本脚本只传缺的 seed）。
#
# 用法:
#   ./run_experiment.sh --detach ./tsy_loho \
#     ./scripts/paper_v4_run/run_overnight_quad_gpu_attn_AC_fill2seeds.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="/home/amax/anaconda3/bin:${PATH:-}"

ONE="$ROOT/scripts/paper_v4_run/run_n12_attn_probe_one_method_serial.sh"
STATUS="$ROOT/outputs/paper_v4/baseline/OVERNIGHT_ATTN_AC_FILL_STATUS.md"
mkdir -p "$(dirname "$STATUS")" "$ROOT/logs"
chmod +x "$ONE"

write_status() {
  {
    echo "# 四卡补齐 A/C 两 seed 状态"
    echo ""
    echo "- 更新: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "- $1"
    echo "- 目标：A(eq) + C(noaux) 补 2024 / 114514 → 满 5 seeds（42,123,2024,3407,114514）"
    echo ""
    echo "| GPU | 方法 | seed | 输出 |"
    echo "|----:|------|-----:|------|"
    echo "| 0 | A eq | 2024 | \`ours_uw_frameaux_t2_edl_w0.3_n12_attn_eq/\` |"
    echo "| 1 | A eq | 114514 | \`ours_uw_frameaux_t2_edl_w0.3_n12_attn_eq/\` |"
    echo "| 2 | C noaux | 2024 | \`ours_uw_frameaux_t2_edl_w0.3_n12_attn_noaux/\` |"
    echo "| 3 | C noaux | 114514 | \`ours_uw_frameaux_t2_edl_w0.3_n12_attn_noaux/\` |"
  } >"$STATUS"
  echo "[STATUS] $1"
}

export SKIP_DATASET_BIND=1
export SKIP_COMPLETED=1

write_status "四卡启动中…"

PIDS=()
FAIL=0

run_one() {
  local gpu="$1" method="$2" seed="$3" tag="$4"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export METHOD="$method"
    export SEEDS="$seed"
    bash "$ONE"
  ) >"$ROOT/logs/overnight_attn_fill_${tag}.log" 2>&1 &
  PIDS+=($!)
  echo "GPU${gpu} ${tag} pid=${PIDS[-1]}  log=logs/overnight_attn_fill_${tag}.log"
}

run_one 0 eq 2024 eq_s2024
run_one 1 eq 114514 eq_s114514
run_one 2 noaux 2024 noaux_s2024
run_one 3 noaux 114514 noaux_s114514

write_status "四卡已启动 PIDs=${PIDS[*]}；断线看 logs/overnight_attn_fill_*.log"

for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then FAIL=1; fi
done

if [[ "$FAIL" -ne 0 ]]; then
  write_status "有任务失败，请检查 logs/overnight_attn_fill_*.log"
  exit 1
fi
write_status "A/C 两 seed 补齐完成。请提数 5-seed 定稿对比旧 Attn / B1。"
echo "全部完成。"

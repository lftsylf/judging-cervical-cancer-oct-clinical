#!/usr/bin/env bash
# =============================================================================
# 四卡过夜总队列（一条命令；nohup/detach 后断线继续）
#
# GPU 0/1/2：pages5 + edl@0.3 + edl_u（无 amp）三折并行 ≈10h
# GPU 3    ：N=12 串行 T2×2 = topk5 → max_p_pool ≈6–8h
#
# 启动:
#   export PATH="/home/amax/anaconda3/bin:$PATH"
#   cd /ssd_data/tsy_study_venv/OptiGenesis_Lancet
#   ./run_experiment.sh --detach ./tsy_loho \
#     ./scripts/paper_v4_run/run_overnight_quad_gpu_pages5_edlu_and_n12_rules.sh
#
# 进度:
#   tail -f logs/detached_latest.log
#   cat outputs/paper_v4/baseline/QUAD_GPU_QUEUE_STATUS.md
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="/home/amax/anaconda3/bin:${PATH:-}"

PAGES5="$ROOT/scripts/paper_v4_run/run_expand_pages_edl03_edlu_pages5.sh"
N12="$ROOT/scripts/paper_v4_run/run_n12_topk5_then_maxp_t2_serial.sh"
STATUS="$ROOT/outputs/paper_v4/baseline/QUAD_GPU_QUEUE_STATUS.md"
mkdir -p "$(dirname "$STATUS")"

write_status() {
  {
    echo "# 四卡过夜队列状态"
    echo ""
    echo "- 更新: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "- $1"
    echo ""
    echo "| 支路 | GPU | 内容 | 输出 |"
    echo "|------|-----|------|------|"
    echo "| A | 0,1,2 | pages5 + edl_u（无 amp） | \`ours_uw_frameaux_t2_edl_w0.3_expand_edlu_pages5/\` |"
    echo "| B | 3 | N=12 topk5 → max_p_pool | \`..._n12_topk5/\` → \`..._n12_maxp/\` |"
    echo ""
    echo "日志: \`logs/detached_latest.log\`"
  } >"$STATUS"
  echo "[STATUS] $1"
}

# 绑定数据一次
"$ROOT/run_experiment.sh" "$ROOT/tsy_loho"

write_status "已启动：A(pages5 无 amp 并行) 与 B(N12 规则串行) 同时跑"

# A: GPU 0/1/2
(
  export OPTIGENESIS_PARALLEL=1
  export OPTIGENESIS_GPUS="0 1 2"
  bash "$PAGES5"
  echo "[A] pages5 无 amp 全部完成"
) &
PID_A=$!

# B: GPU 3 串行两次 T2
(
  export CUDA_VISIBLE_DEVICES=3
  bash "$N12"
  echo "[B] N12 topk5+max_p 全部完成"
) &
PID_B=$!

echo "PID_A(pages5)=$PID_A  PID_B(n12)=$PID_B"
FAIL=0
if ! wait "$PID_A"; then FAIL=1; echo "[A] 失败"; fi
if ! wait "$PID_B"; then FAIL=1; echo "[B] 失败"; fi

if [[ "$FAIL" -ne 0 ]]; then
  write_status "结束：有支路失败（见 detached 日志）"
  exit 1
fi
write_status "全部完成。请提数对比：pages5±amp、N12 UW / topk5 / max_p。"
echo "全部完成。"

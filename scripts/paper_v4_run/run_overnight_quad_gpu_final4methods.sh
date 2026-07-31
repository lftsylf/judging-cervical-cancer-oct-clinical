#!/usr/bin/env bash
# Paper v4 · 四卡定稿过夜：探路曾否定的四条线各占 1 卡，补齐 5 seeds
#
# GPU0 ① N=12 + edl_u_amp×10 + FrameAux broadcast@0.3
# GPU1 ② N=12 + max_p_pool + FrameAux broadcast@0.3
# GPU2 ③ N=12 + edl_u + FrameAux MIL@0.3
# GPU3 ④ N=12 + 可学习 attention + FrameAux broadcast@0.3
#
# seed=42：①②③ 若已完成则自动跳过，只跑剩余 4 seeds；④ 通常跑满 5 seeds。
# 断线可续：SKIP_COMPLETED=1
#
# 用法:
#   ./run_experiment.sh --detach ./tsy_loho \
#     ./scripts/paper_v4_run/run_overnight_quad_gpu_final4methods.sh
#
# 看进度:
#   tail -f logs/detached_latest.log
#   cat outputs/paper_v4/baseline/OVERNIGHT_FINAL4_STATUS.md
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="/home/amax/anaconda3/bin:${PATH:-}"

ONE="$ROOT/scripts/paper_v4_run/run_n12_final_one_method_serial.sh"
STATUS="$ROOT/outputs/paper_v4/baseline/OVERNIGHT_FINAL4_STATUS.md"
mkdir -p "$(dirname "$STATUS")"

write_status() {
  {
    echo "# 四卡定稿过夜状态"
    echo ""
    echo "- 更新: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "- $1"
    echo ""
    echo "| GPU | 方法 | 输出目录 |"
    echo "|----:|------|----------|"
    echo "| 0 | amp | \`ours_uw_frameaux_t2_edl_w0.3_n12_uamp10/\` |"
    echo "| 1 | max_p | \`ours_uw_frameaux_t2_edl_w0.3_n12_maxp/\` |"
    echo "| 2 | mil | \`ours_uw_frameaux_t2_edl_w0.3_n12_mil/\` |"
    echo "| 3 | attn | \`ours_uw_frameaux_t2_edl_w0.3_n12_attn/\` |"
  } >"$STATUS"
  echo "[STATUS] $1"
}

chmod +x "$ONE"

# 绑定 dataset 已由 run_experiment.sh 完成；子脚本不再绑
export SKIP_DATASET_BIND=1
export SKIP_COMPLETED=1

write_status "四卡并行启动中…"

PIDS=()
FAIL=0

(
  export CUDA_VISIBLE_DEVICES=0
  export METHOD=amp
  bash "$ONE"
) >"$ROOT/logs/overnight_final_amp.log" 2>&1 &
PIDS+=($!)
echo "GPU0 amp pid=${PIDS[-1]}  log=logs/overnight_final_amp.log"

(
  export CUDA_VISIBLE_DEVICES=1
  export METHOD=maxp
  bash "$ONE"
) >"$ROOT/logs/overnight_final_maxp.log" 2>&1 &
PIDS+=($!)
echo "GPU1 maxp pid=${PIDS[-1]}  log=logs/overnight_final_maxp.log"

(
  export CUDA_VISIBLE_DEVICES=2
  export METHOD=mil
  bash "$ONE"
) >"$ROOT/logs/overnight_final_mil.log" 2>&1 &
PIDS+=($!)
echo "GPU2 mil pid=${PIDS[-1]}  log=logs/overnight_final_mil.log"

(
  export CUDA_VISIBLE_DEVICES=3
  export METHOD=attn
  bash "$ONE"
) >"$ROOT/logs/overnight_final_attn.log" 2>&1 &
PIDS+=($!)
echo "GPU3 attn pid=${PIDS[-1]}  log=logs/overnight_final_attn.log"

write_status "四卡已启动 PIDs=${PIDS[*]}；断线后看各 logs/overnight_final_*.log"

for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then FAIL=1; fi
done

if [[ "$FAIL" -ne 0 ]]; then
  write_status "有任务失败，请检查 logs/overnight_final_*.log"
  exit 1
fi
write_status "四方法定稿全部完成。请提数对比 Full/B1 的 5-seed 外部均值。"
echo "全部完成。"

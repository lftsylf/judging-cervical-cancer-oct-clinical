#!/usr/bin/env bash
# Paper v4 · 四卡 attn 改进探路（可断联）
#
# GPU0  A  evidence query          → attn_eq
# GPU1  B  τ=0.2                   → attn_tau02
# GPU2  C  无 FrameAux             → attn_noaux
# GPU3  D  患者 LS ε=0.05          → attn_ls05
#
# Seeds：42 123 3407（相对旧 attn 三 seed 均值最高，且略优于同子集 B1）
# 断线可续：SKIP_COMPLETED=1
#
# 用法:
#   ./run_experiment.sh --detach ./tsy_loho \
#     ./scripts/paper_v4_run/run_overnight_quad_gpu_attn_probe4.sh
#
# 看进度:
#   tail -f logs/detached_latest.log
#   cat outputs/paper_v4/baseline/OVERNIGHT_ATTN_PROBE4_STATUS.md
#   tail -f logs/overnight_attn_probe_*.log
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="/home/amax/anaconda3/bin:${PATH:-}"

ONE="$ROOT/scripts/paper_v4_run/run_n12_attn_probe_one_method_serial.sh"
STATUS="$ROOT/outputs/paper_v4/baseline/OVERNIGHT_ATTN_PROBE4_STATUS.md"
mkdir -p "$(dirname "$STATUS")" "$ROOT/logs"

write_status() {
  {
    echo "# 四卡 attn 探路状态"
    echo ""
    echo "- 更新: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "- $1"
    echo "- Seeds: 42, 123, 3407"
    echo "- 同子集对照：旧 attn≈0.559，B1≈0.554"
    echo ""
    echo "| GPU | 方法 | 输出目录 |"
    echo "|----:|------|----------|"
    echo "| 0 | A evidence query | \`ours_uw_frameaux_t2_edl_w0.3_n12_attn_eq/\` |"
    echo "| 1 | B τ=0.2 | \`ours_uw_frameaux_t2_edl_w0.3_n12_attn_tau02/\` |"
    echo "| 2 | C no FrameAux | \`ours_uw_frameaux_t2_edl_w0.3_n12_attn_noaux/\` |"
    echo "| 3 | D LS=0.05 | \`ours_uw_frameaux_t2_edl_w0.3_n12_attn_ls05/\` |"
  } >"$STATUS"
  echo "[STATUS] $1"
}

chmod +x "$ONE"

export SKIP_DATASET_BIND=1
export SKIP_COMPLETED=1
export SEEDS="42 123 3407"

write_status "四卡并行启动中…"

PIDS=()
FAIL=0

(
  export CUDA_VISIBLE_DEVICES=0
  export METHOD=eq
  bash "$ONE"
) >"$ROOT/logs/overnight_attn_probe_eq.log" 2>&1 &
PIDS+=($!)
echo "GPU0 eq pid=${PIDS[-1]}  log=logs/overnight_attn_probe_eq.log"

(
  export CUDA_VISIBLE_DEVICES=1
  export METHOD=tau02
  bash "$ONE"
) >"$ROOT/logs/overnight_attn_probe_tau02.log" 2>&1 &
PIDS+=($!)
echo "GPU1 tau02 pid=${PIDS[-1]}  log=logs/overnight_attn_probe_tau02.log"

(
  export CUDA_VISIBLE_DEVICES=2
  export METHOD=noaux
  bash "$ONE"
) >"$ROOT/logs/overnight_attn_probe_noaux.log" 2>&1 &
PIDS+=($!)
echo "GPU2 noaux pid=${PIDS[-1]}  log=logs/overnight_attn_probe_noaux.log"

(
  export CUDA_VISIBLE_DEVICES=3
  export METHOD=ls05
  bash "$ONE"
) >"$ROOT/logs/overnight_attn_probe_ls05.log" 2>&1 &
PIDS+=($!)
echo "GPU3 ls05 pid=${PIDS[-1]}  log=logs/overnight_attn_probe_ls05.log"

write_status "四卡已启动 PIDs=${PIDS[*]}；断线后看 logs/overnight_attn_probe_*.log"

for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then FAIL=1; fi
done

if [[ "$FAIL" -ne 0 ]]; then
  write_status "有任务失败，请检查 logs/overnight_attn_probe_*.log"
  exit 1
fi
write_status "四方法探路全部完成。请提数对比同子集旧 attn / B1。"
echo "全部完成。"

#!/usr/bin/env bash
# =============================================================================
# 过夜串行队列：4 个权重信号 T2（断线继续）
#
# 顺序（每个 = 3 折 × seed42，FrameAux=edl@0.3）：
#   1) maxprob τ=0.5
#   2) maxprob τ=0.1
#   3) negent  τ=0.1
#   4) negent  τ=0.01
#
# 启动：
#   export PATH="/home/amax/anaconda3/bin:$PATH"
#   cd /ssd_data/tsy_study_venv/OptiGenesis_Lancet
#   ./run_experiment.sh --detach ./tsy_loho \
#     ./scripts/paper_v4_run/run_overnight_weight_signal_t2_queue.sh
#
# 进度：
#   tail -f logs/detached_latest.log
#   cat outputs/paper_v4/baseline/WEIGHT_SIGNAL_QUEUE_STATUS.md
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

T2_SCRIPT="${SCRIPT_DIR}/run_ours_uw_frameaux_t2_oct_only.sh"
STATUS_MD="${PROJECT_ROOT}/outputs/paper_v4/baseline/WEIGHT_SIGNAL_QUEUE_STATUS.md"
BASE="${PROJECT_ROOT}/outputs/paper_v4/baseline"

write_status() {
  local msg="$1"
  mkdir -p "$(dirname "$STATUS_MD")"
  {
    echo "# 权重信号 T2 过夜队列状态"
    echo ""
    echo "- 更新时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "- ${msg}"
    echo ""
    echo "日志: \`logs/detached_latest.log\`"
    echo ""
    echo "## 计划"
    echo ""
    echo "| # | signal | τ | 输出目录 |"
    echo "|---|--------|--:|----------|"
    echo "| 1 | maxprob | 0.5 | \`ours_uw_frameaux_t2_maxprob_t0.5/\` |"
    echo "| 2 | maxprob | 0.1 | \`ours_uw_frameaux_t2_maxprob_t0.1/\` |"
    echo "| 3 | negent | 0.1 | \`ours_uw_frameaux_t2_negent_t0.1/\` |"
    echo "| 4 | negent | 0.01 | \`ours_uw_frameaux_t2_negent_t0.01/\` |"
    echo ""
    echo "对照: 已有 Full edl@0.3 + edl_u τ=0.5 → \`ours_uw_frameaux_t2_edl_w0.3/\`"
  } >"$STATUS_MD"
  echo "[STATUS] $msg"
}

run_one() {
  local idx="$1"
  local total="$2"
  local signal="$3"
  local tau="$4"
  local out_name="$5"
  local out_root="${BASE}/${out_name}"

  write_status "开始 [${idx}/${total}] signal=${signal} τ=${tau} → ${out_root}"
  echo "======================================"
  echo "[${idx}/${total}] WEIGHT_SIGNAL=${signal} τ=${tau}"
  echo "OUT=${out_root}"
  echo "======================================"

  export BASELINE_OUT_ROOT="$out_root"
  export OPTIGENESIS_FRAME_WEIGHT_SIGNAL="$signal"
  export OPTIGENESIS_FRAME_AGG_TEMP="$tau"
  # 与当前最好 Full 对齐：帧辅损 edl@0.3
  export OPTIGENESIS_FRAME_AUX_TYPE=edl
  export OPTIGENESIS_FRAME_AUX_WEIGHT=0.3
  export SKIP_COMPLETED=1

  bash "$T2_SCRIPT"
  write_status "完成 [${idx}/${total}] signal=${signal} τ=${tau} → ${out_root}"
}

echo "======================================"
echo "过夜队列：maxprob×2 + negent×2（串行，断线继续）"
echo "FrameAux=edl@0.3 | OCT-only | 3折×seed42"
echo "======================================"

"$PROJECT_ROOT/run_experiment.sh" "$PROJECT_ROOT/tsy_loho"

TOTAL=4
run_one 1 "$TOTAL" maxprob 0.5  "ours_uw_frameaux_t2_maxprob_t0.5"
run_one 2 "$TOTAL" maxprob 0.1  "ours_uw_frameaux_t2_maxprob_t0.1"
run_one 3 "$TOTAL" negent  0.1  "ours_uw_frameaux_t2_negent_t0.1"
run_one 4 "$TOTAL" negent  0.01 "ours_uw_frameaux_t2_negent_t0.01"

write_status "全部 4 个 T2 已完成。请用 val 选模后看 external；对照 edl_u edl@0.3。"
echo "======================================"
echo "队列全部完成。"
echo "状态: ${STATUS_MD}"
echo "======================================"

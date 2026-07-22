#!/usr/bin/env bash
# =============================================================================
# 无人值守探路（断线可继续）：
#   1) 跑 T2：ce @ weight=0.2
#   2) 自动提数，对比已有 edl@0.2
#   3) 若仍明显更差（默认 gap≥0.02），自动再跑 T2：edl @ weight=0.3
#
# 启动（推荐）：
#   export PATH="/home/amax/anaconda3/bin:$PATH"
#   cd /ssd_data/tsy_study_venv/OptiGenesis_Lancet
#   ./run_experiment.sh --detach ./tsy_loho ./scripts/paper_v4_run/run_auto_frameaux_probe_ce_then_edl03.sh
#
# 进度 / 结论：
#   tail -f logs/detached_latest.log
#   cat outputs/paper_v4/baseline/AUTO_PROBE_STATUS.md
#   cat scripts/paper_v4_run/AUTO_PROBE_ce_w0.2_report.md
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${OPTIGENESIS_PYTHON:-/home/amax/anaconda3/bin/python3}"
STATUS_MD="${PROJECT_ROOT}/outputs/paper_v4/baseline/AUTO_PROBE_STATUS.md"
GAP_THRESHOLD="${GAP_THRESHOLD:-0.02}"

CE_OUT="${PROJECT_ROOT}/outputs/paper_v4/baseline/ours_uw_frameaux_t2_ce_w0.2"
EDL03_OUT="${PROJECT_ROOT}/outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3"
REF_EDL02="${PROJECT_ROOT}/outputs/paper_v4/baseline/ours_uw_frameaux_t2"
T2_SCRIPT="${SCRIPT_DIR}/run_ours_uw_frameaux_t2_oct_only.sh"

write_status() {
  local msg="$1"
  mkdir -p "$(dirname "$STATUS_MD")"
  {
    echo "# 自动探路状态"
    echo ""
    echo "- 更新时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "- ${msg}"
    echo ""
    echo "日志: \`logs/detached_latest.log\`"
  } >"$STATUS_MD"
  echo "[STATUS] $msg"
}

run_t2() {
  local out_root="$1"
  local aux_type="$2"
  local aux_w="$3"
  write_status "开始 T2：type=${aux_type} weight=${aux_w} → ${out_root}"
  export BASELINE_OUT_ROOT="$out_root"
  export OPTIGENESIS_FRAME_AUX_TYPE="$aux_type"
  export OPTIGENESIS_FRAME_AUX_WEIGHT="$aux_w"
  # 子脚本内部会再 bind 数据；此处前台串行跑完 3 折
  bash "$T2_SCRIPT"
  write_status "完成 T2：type=${aux_type} weight=${aux_w} → ${out_root}"
}

echo "======================================"
echo "自动探路：ce@0.2 → 提数判断 → (可选) edl@0.3"
echo "GAP_THRESHOLD=${GAP_THRESHOLD}"
echo "======================================"

# 0) 绑定数据一次（子脚本还会绑，无妨）
"$PROJECT_ROOT/run_experiment.sh" "$PROJECT_ROOT/tsy_loho"

# 1) ce @ 0.2
run_t2 "$CE_OUT" "ce" "0.2"

# 2) 提数 + 判决
REPORT="${SCRIPT_DIR}/AUTO_PROBE_ce_w0.2_report.md"
DECISION_JSON="${PROJECT_ROOT}/outputs/paper_v4/baseline/AUTO_PROBE_decision_ce.json"
write_status "提取 ce@0.2 指标并与 edl@0.2 比较…"
"$PYTHON" "${SCRIPT_DIR}/eval_frameaux_t2_decide.py" \
  --candidate-root "$CE_OUT" \
  --ref-root "$REF_EDL02" \
  --gap-threshold "$GAP_THRESHOLD" \
  --candidate-name "ce@0.2" \
  --report "$REPORT" \
  --decision-json "$DECISION_JSON"

RUN03="$("$PYTHON" -c "import json; print('1' if json.load(open('${DECISION_JSON}'))['run_edl_w03'] else '0')")"

if [[ "$RUN03" == "1" ]]; then
  write_status "判决：ce@0.2 相对 edl@0.2 仍差得远 → 启动 edl@0.3"
  run_t2 "$EDL03_OUT" "edl" "0.3"

  REPORT3="${SCRIPT_DIR}/AUTO_PROBE_edl_w0.3_report.md"
  DECISION3="${PROJECT_ROOT}/outputs/paper_v4/baseline/AUTO_PROBE_decision_edl03.json"
  write_status "提取 edl@0.3 指标…"
  "$PYTHON" "${SCRIPT_DIR}/eval_frameaux_t2_decide.py" \
    --candidate-root "$EDL03_OUT" \
    --ref-root "$REF_EDL02" \
    --gap-threshold "$GAP_THRESHOLD" \
    --candidate-name "edl@0.3" \
    --report "$REPORT3" \
    --decision-json "$DECISION3"
  write_status "全部结束。见 ${REPORT} 与 ${REPORT3}；状态文件 ${STATUS_MD}"
else
  write_status "判决：ce@0.2 未明显差于 edl@0.2 → 不跑 edl@0.3。报告: ${REPORT}"
fi

echo "======================================"
echo "自动探路结束。"
echo "状态: $STATUS_MD"
echo "======================================"

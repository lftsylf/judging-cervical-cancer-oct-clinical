#!/usr/bin/env bash
# N=12 + FrameAux edl@0.3：在单卡上串行跑两次 T2
#   1) topk_p（k=5）
#   2) max_p_pool（训练版 max_p；hybrid 在探针中外部几乎=max_p，故用 max 代表）
#
# 用法（单卡）:
#   CUDA_VISIBLE_DEVICES=3 ./scripts/paper_v4_run/run_n12_topk5_then_maxp_t2_serial.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="/home/amax/anaconda3/bin:${PATH:-}"

T2="$ROOT/scripts/paper_v4_run/run_ours_uw_frameaux_t2_oct_only.sh"
STATUS="${ROOT}/outputs/paper_v4/baseline/N12_RULES_T2_STATUS.md"
GPU_TAG="${CUDA_VISIBLE_DEVICES:-?}"

write_status() {
  mkdir -p "$(dirname "$STATUS")"
  {
    echo "# N=12 topk5 → max_p_pool 串行状态"
    echo ""
    echo "- 更新: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "- CUDA_VISIBLE_DEVICES=${GPU_TAG}"
    echo "- $1"
  } >"$STATUS"
  echo "[STATUS] $1"
}

# 公共：N=12、无 expand、edl@0.3
export OPTIGENESIS_EXPAND_TIFF_PAGES=0
export OPTIGENESIS_MAX_PAGES_PER_TIFF=0
export OPTIGENESIS_BATCH_SIZE=4
export OPTIGENESIS_FRAME_AUX_TYPE=edl
export OPTIGENESIS_FRAME_AUX_WEIGHT=0.3
export OPTIGENESIS_FRAME_AGG_TEMP=0.5
export SKIP_COMPLETED=1
export SKIP_DATASET_BIND=1  # 总队列已绑 dataset；避免与 pages5 抢软链

write_status "开始 [1/2] N=12 topk_p k=5"
export OPTIGENESIS_FRAME_WEIGHT_SIGNAL=topk_p
export OPTIGENESIS_FRAME_TOPK_K=5
export BASELINE_OUT_ROOT="$ROOT/outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3_n12_topk5"
bash "$T2"
write_status "完成 [1/2] topk5 → 开始 [2/2] max_p_pool"

export OPTIGENESIS_FRAME_WEIGHT_SIGNAL=max_p_pool
unset OPTIGENESIS_FRAME_TOPK_K 2>/dev/null || true
export BASELINE_OUT_ROOT="$ROOT/outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3_n12_maxp"
bash "$T2"
write_status "全部完成：topk5 + max_p_pool（N=12）"

echo "完成。输出:"
echo "  $ROOT/outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3_n12_topk5/"
echo "  $ROOT/outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3_n12_maxp/"

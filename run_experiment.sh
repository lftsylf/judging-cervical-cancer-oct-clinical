#!/usr/bin/env bash
# =============================================================================
# 1) 仅绑定数据（默认，与以前一致）
#    ./run_experiment.sh /path/to/your/dataset
#
# 2) 绑定数据后，在后台跑训练（SSH 断开后仍继续；标准输出写入主日志）
#    ./run_experiment.sh --detach /path/to/your/dataset ./run_baseline_t1.sh
#    ./run_experiment.sh --detach /path/to/tsy_loho ./run_baseline_t0.sh
#
#    每个 seed 的明细仍由各训练脚本里的 tee 写入，例如：
#      <OUTPUT_DIR>/<hospital>/seed_<SEED>/logs/train_console.log
#
# 断线重连后看「整次任务」汇总输出（推荐存这条）：
#    tail -f "$(dirname "$(readlink -f "$0")")/logs/detached_latest.log"
#
# 或看某一折某一 seed（把路径换成你 BASELINE_OUT_ROOT 下的实际目录）：
#    tail -f /path/to/OptiGenesis_Lancet/outputs_baseline_t1_v2_resnet50/huaxi/seed_42/logs/train_console.log
#
# 更稳妥的多窗口长期任务：在 tmux 里跑（断线不影响会话）
#    tmux new -s optigen
#    cd /path/to/OptiGenesis_Lancet && ./run_baseline_t1.sh
#    # 断线前按 Ctrl-b 再按 d 脱离
#    # 重连后：tmux attach -t optigen
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

DETACH=false
if [[ "${1:-}" == "--detach" ]]; then
  DETACH=true
  shift
fi

if [[ -z "${1:-}" ]]; then
  echo "❌ 错误: 请提供目标数据集的完整路径作为参数。"
  echo "   用法: ./run_experiment.sh /path/to/your/dataset"
  echo "   断线续跑: ./run_experiment.sh --detach /path/to/dataset ./run_baseline_t1.sh"
  exit 1
fi

TARGET_DATASET_PATH="$1"
shift || true

if [[ ! -d "$TARGET_DATASET_PATH" ]]; then
  echo "❌ 错误: 数据集路径不存在: $TARGET_DATASET_PATH"
  exit 1
fi

if [[ "$DETACH" == true ]] && [[ $# -eq 0 ]]; then
  echo "❌ 错误: 使用 --detach 时，在数据集路径后还需写上要执行的命令，例如:"
  echo "   ./run_experiment.sh --detach $TARGET_DATASET_PATH ./run_baseline_t1.sh"
  exit 1
fi

if [[ "$DETACH" != true ]] && [[ $# -gt 0 ]]; then
  echo "⚠️  忽略多余参数: $*（若要在 SSH 断开后仍继续跑训练，请使用 --detach，见脚本头部说明）"
fi

DATASET_SYMLINK_PATH="$PROJECT_ROOT/dataset"

echo "Project Root: $PROJECT_ROOT"
echo "Target Dataset: $TARGET_DATASET_PATH"

if [[ -e "$DATASET_SYMLINK_PATH" ]]; then
  echo "🔗 正在删除旧的软链接或目录: $DATASET_SYMLINK_PATH"
  rm -rf "$DATASET_SYMLINK_PATH"
fi

echo "🔗 正在创建新的软链接，从 $DATASET_SYMLINK_PATH 指向 $TARGET_DATASET_PATH"
ln -s "$TARGET_DATASET_PATH" "$DATASET_SYMLINK_PATH"

echo "✅ 软链接创建成功。"

if [[ "$DETACH" != true ]]; then
  exit 0
fi

TRAIN_CMD=("$@")
LOGDIR="$PROJECT_ROOT/logs"
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOGDIR/detached_${STAMP}.log"
PID_FILE="$LOGDIR/detached_last.pid"
LATEST_LINK="$LOGDIR/detached_latest.log"

cmd_q=$(printf '%q ' "${TRAIN_CMD[@]}")
# nohup：忽略 SIGHUP，SSH 断开不杀训练进程
nohup bash -lc "cd $(printf %q "$PROJECT_ROOT") && exec $cmd_q" >>"$MASTER_LOG" 2>&1 &
DETACH_PID=$!
echo "$DETACH_PID" >"$PID_FILE"
ln -sf "$MASTER_LOG" "$LATEST_LINK"

echo ""
echo "✅ 已在后台启动 (nohup)，SSH 断开后训练会继续。"
echo "   主日志文件: $MASTER_LOG"
echo "   始终指向最近一次 detached 日志: $LATEST_LINK"
echo "   PID: $DETACH_PID （写入 $PID_FILE）"
echo ""
echo "━━━━━━━━ 断线重连后，在终端看实时输出（可收藏）━━━━━━━━"
echo "tail -f $(printf %q "$LATEST_LINK")"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

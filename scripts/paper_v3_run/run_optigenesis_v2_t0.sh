#!/usr/bin/env bash
# ⚠️ 已废弃：旧版 Swin + 多模态完全体。v2 OCT-only 完全体请用：
#   ./run_optigenesis_full_t0_oct_only.sh
echo "⚠️  本脚本为 v1 配置，已重定向到 run_optigenesis_full_t0_oct_only.sh" >&2
exec "$(cd "$(dirname "$0")" && pwd)/run_optigenesis_full_t0_oct_only.sh" "$@"

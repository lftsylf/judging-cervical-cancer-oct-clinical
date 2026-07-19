#!/usr/bin/env bash
# ⚠️ 已废弃：旧版 ViT-Small 湘雅续跑（多模态）。v2 请用 SKIP_COMPLETED 续跑：
#   SKIP_COMPLETED=1 ./run_comparison_vit_small_t0_oct_only.sh
echo "⚠️  本脚本为 v1 多模态配置，已重定向到 run_comparison_vit_small_t0_oct_only.sh" >&2
exec "$(cd "$(dirname "$0")" && pwd)/run_comparison_vit_small_t0_oct_only.sh" "$@"

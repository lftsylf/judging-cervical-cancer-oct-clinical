#!/usr/bin/env bash
# ⚠️ 已废弃：旧版 ViT-Small + 多模态 + batch=2。请改用 v2 单模态脚本：
#   ./run_comparison_vit_small_t0_oct_only.sh
echo "⚠️  本脚本为 v1 多模态配置，已重定向到 run_comparison_vit_small_t0_oct_only.sh" >&2
exec "$(cd "$(dirname "$0")" && pwd)/run_comparison_vit_small_t0_oct_only.sh" "$@"

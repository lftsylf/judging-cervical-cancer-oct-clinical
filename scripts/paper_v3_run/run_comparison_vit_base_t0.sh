#!/usr/bin/env bash
# ⚠️ 已废弃：v1 多模态 ViT-Base + v2 小样本不再纳入 ViT-Base。请改用：
#   ./run_comparison_resnet18_t0_oct_only.sh
echo "⚠️  ViT-Base 已移出 v2 对比矩阵，已重定向到 run_comparison_resnet18_t0_oct_only.sh" >&2
exec "$(cd "$(dirname "$0")" && pwd)/run_comparison_resnet18_t0_oct_only.sh" "$@"

#!/bin/bash

# --- 使用说明 ---
# 1. 确保此脚本有执行权限: chmod +x run_experiment.sh
# 2. 运行脚本并传入目标数据集的完整路径作为参数。
#
#    例如，要使用 "AnYang_devide" 数据集:
#    ./run_experiment.sh /ssd_data/CervixOCT_datasets/AnYang_devide
#
#    例如，要使用另一个名为 "New_Dataset" 的数据集:
#    ./run_experiment.sh /ssd_data/CervixOCT_datasets/New_Dataset
# ----------------

# --- 1. 参数检查 ---
# 检查是否提供了数据集路径参数
if [ -z "$1" ]; then
    echo "❌ 错误: 请提供目标数据集的完整路径作为参数。"
    echo "   用法: ./run_experiment.sh /path/to/your/dataset"
    exit 1
fi

# 获取传入的数据集路径
TARGET_DATASET_PATH=$1

# 检查目标数据集路径是否存在
if [ ! -d "$TARGET_DATASET_PATH" ]; then
    echo "❌ 错误: 数据集路径不存在: $TARGET_DATASET_PATH"
    exit 1
fi

# --- 2. 设置项目路径 ---
# 获取脚本所在的目录，即项目根目录
PROJECT_ROOT=$(dirname "$(readlink -f "$0")")
# 定义项目中数据文件夹的软链接名称
DATASET_SYMLINK_PATH="$PROJECT_ROOT/dataset"

echo "Project Root: $PROJECT_ROOT"
echo "Target Dataset: $TARGET_DATASET_PATH"

# --- 3. 创建软链接 ---
# 首先，删除旧的软链接或文件夹
if [ -e "$DATASET_SYMLINK_PATH" ]; then
    echo "🔗 正在删除旧的软链接: $DATASET_SYMLINK_PATH"
    rm -rf "$DATASET_SYMLINK_PATH"
fi

# 创建新的软链接，指向目标数据集
echo "🔗 正在创建新的软链接，从 $DATASET_SYMLINK_PATH 指向 $TARGET_DATASET_PATH"
ln -s "$TARGET_DATASET_PATH" "$DATASET_SYMLINK_PATH"

echo "✅ 软链接创建成功。"


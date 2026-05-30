# v2 单模态 T0 SOTA 对比（OCT-only）

与定稿 baseline（`run_baseline_t0_oct_only.sh`）对齐：**仅改 `OPTIGENESIS_BACKBONE`**（及 OOM 时的 batch）。  
每组 **3 折 × 5 seeds = 15 次**。主指标：**外部 ROC-AUC**（`train_console.log` 中「最佳权重 · 外部集」行）。

> **勿与旧多模态对比混表**：`outputs/对比试验/` 下结果为 v1 多模态（`USE_CLINICAL=1`、`BATCH=2`），协议不同，不可与本文 v2 OCT-only 结果并列。

## 为何用 ResNet18 而非 ViT-Base

小样本 LOHO 设定下，ViT-Base 参数量过大、易过拟合；ViT-Small 已作为 Transformer 对照。  
**ResNet18** 作为轻量 CNN 容量对照（与定稿 ResNet50 baseline 形成「小/中容量」对比），此前 T1 实验亦表明 ResNet18 明显弱于 ResNet50，更适合小数据叙事。  
旧 `run_comparison_vit_base_*` 脚本已重定向至 `run_comparison_resnet18_t0_oct_only.sh`。

## 协议矩阵（backbone × 超参）

| 项 | v2 OCT-only 对比 | 旧 v1 多模态对比（已归档） |
|----|------------------|---------------------------|
| 临床特征 | `USE_CLINICAL=0` | 默认 1（多模态） |
| LR / POS | `5e-5` / `1.25` | 未显式对齐 v2 |
| Batch | 4（Swin 为 2） | 2 |
| WMA / EMA / Aux / CORAL | 全关 | 全关 |
| 折 × seed | 3 × 5 = 15 | 3 × 5 = 15 |
| 脚本 | `run_comparison_*_t0_oct_only.sh` | `run_comparison_*_t0.sh`（已重定向） |

## 对比脚本与输出目录

| 脚本 | timm BACKBONE | Batch | 说明 |
|------|---------------|:-----:|------|
| `run_comparison_swin_tiny_t0_oct_only.sh` | `swin_tiny_patch4_window7_224` | 2 | v1 历史主骨干 |
| `run_comparison_swin_small_t0_oct_only.sh` | `swin_small_patch4_window7_224` | 2 | 旧 SOTA 对比 Swin |
| `run_comparison_convnext_small_t0_oct_only.sh` | `convnext_small` | **2** | 8GB 卡 batch=4 OOM |
| `run_comparison_vit_small_t0_oct_only.sh` | `vit_small_patch16_224` | **2** | 8GB 卡 batch=4 OOM |
| `run_comparison_resnet18_t0_oct_only.sh` | `resnet18` | 4 | 轻量 CNN，与 baseline 同 batch |

共享库：`run_comparison_t0_oct_only_lib.sh`  
串行入口：`run_comparison_t0_oct_only_all.sh`（5 组）  
**续跑剩余 3 组**（Swin 已完成后）：`run_comparison_t0_oct_only_remaining.sh`（先 ResNet18，再 ConvNeXt/ViT-S）

Swin 系列使用 `BATCH_SIZE=2`；可选 `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`（**PyTorch 2.0.x 勿用 expandable_segments**）。

## 论文表建议行（v2 OCT-only，勿混 v1）

| 行 | 方法 | 结果路径 |
|----|------|----------|
| Plain baseline | ResNet50，无 WMA/EMA/Aux | `outputs_baseline_t0_v2_resnet50_oct_only/`（≈0.618 外部 AUC） |
| 轻量 CNN 对照 | ResNet18 plain | `outputs_comparison_v2_oct_resnet18/` |
| SOTA 对比 | Swin / ConvNeXt / ViT-S 等 plain | `outputs/对比试验v2单模态/outputs_comparison_v2_oct_*/` |
| **OptiGenesis Full** | ResNet50 + WMA + EMA + Aux | `outputs/消融实验v2单模态/outputs_ablation_v2_oct_full/`（≈0.645 外部 AUC） |

## Detach 启动（断线仍继续）

```bash
export PATH="/home/amax/anaconda3/bin:$PATH"
cd /ssd_data/tsy_study_venv/OptiGenesis_Lancet

# 五组 SOTA 对比一次 detach 串行
./run_experiment.sh --detach \
  /ssd_data/tsy_study_venv/OptiGenesis_Lancet/tsy_loho \
  ./run_comparison_t0_oct_only_all.sh

# 断线后看总日志
tail -f logs/detached_latest.log

# 续跑（默认跳过已完成，日志含「训练完成！」）
SKIP_COMPLETED=1 ./run_comparison_t0_oct_only_all.sh
```

## 后处理

### Youden 阈值汇总（外部集）

```bash
python scripts/analyze_comparison_optimal_thresholds.py --v2-only
```

在各模型目录下生成 `outputs_comparison_v2_oct_*_summary.csv`（格式与旧 ViT 表一致）。  
分类指标基于 **Youden 最优阈值**，勿仅用 BalAcc@0.5。

### DeLong（湘雅外部 ROC-AUC）

```bash
python scripts/calculate_all_delong_pvalues.py --v2
```

Champion：`outputs_ablation_v2_oct_full`；Challenger：ResNet50 plain baseline + 各 v2 对比骨干（含 ResNet18）。

## 目录结构（训练完成后）

```
outputs/对比试验v2单模态/
├── README.md
├── outputs_comparison_v2_oct_swin_tiny/
├── outputs_comparison_v2_oct_swin_small/
├── outputs_comparison_v2_oct_convnext_small/
├── outputs_comparison_v2_oct_vit_small/
└── outputs_comparison_v2_oct_resnet18/
    └── {huaxi,liaoning,xiangya}/seed_*/logs/
        ├── train_console.log
        └── external_sample_predictions.csv
```

旧入口 `run_comparison_*_t0.sh`（无 `oct_only`）已重定向到对应 `*_t0_oct_only.sh`；`run_comparison_vit_base_*` 重定向至 ResNet18。

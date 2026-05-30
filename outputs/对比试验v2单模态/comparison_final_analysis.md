# v2 OCT-only T0 对比试验 — 终版汇总与分析

> 完成度：**75/75**（5 骨干 × 15 runs）  
> 数据文件：`comparison_v2_oct_detail.csv`、`comparison_v2_oct_summary.csv`、`comparison_v2_oct_by_hospital.csv`、`comparison_v2_oct_full_head2head.csv`  
> Youden 汇总：各目录下 `outputs_comparison_v2_oct_*_summary.csv`

## 1. 外部 ROC-AUC 总排名（15 runs mean±std）

| 排名 | 模型 | 外部 AUC | vs ResNet50 Baseline |
|:--:|------|----------|:--------------------:|
| 1 | **OptiGenesis Full** (ResNet50+WMA+EMA+Aux) | **0.6453±0.053** | **+0.027** |
| 2 | ViT-Small (plain) | 0.6381±0.073 | +0.020 |
| 3 | Swin-Tiny (plain) | 0.6275±0.042 | +0.009 |
| 4 | Swin-Small (plain) | 0.6262±0.048 | +0.008 |
| 5 | ConvNeXt-Small (plain) | 0.6253±0.069 | +0.007 |
| 6 | ResNet50 Baseline (plain) | 0.6183±0.037 | — |
| 7 | ResNet18 (plain) | 0.5941±0.052 | −0.024 |

**主结论**：Full 仍为总体最高；但 **ViT-Small plain（0.638）与 Full（0.645）差距仅 0.007**，且方差更大（std 0.073 vs 0.053）。

## 2. 分折外部 AUC

| 折 | Baseline | Full | ViT-S | Swin-T | Swin-S | ConvNeXt | ResNet18 |
|----|:--------:|:----:|:-----:|:------:|:------:|:--------:|:--------:|
| 华西 | 0.597 | 0.620 | 0.570 | 0.605 | **0.621** | 0.575 | 0.580 |
| 辽宁 | **0.637** | 0.616 | 0.638 | 0.634 | 0.597 | **0.649** | 0.575 |
| 湘雅 | 0.621 | 0.700 | **0.707** | 0.644 | 0.661 | 0.652 | 0.627 |
| **15次均值** | 0.618 | **0.645** | 0.638 | 0.628 | 0.626 | 0.625 | 0.594 |

- **无单一 plain backbone 全面占优**：华西 Swin-S、辽宁 ConvNeXt、湘雅 ViT-S 各折领先。
- **Full 在湘雅仍强**（0.700），但 **略低于 ViT-S plain（0.707）**。
- **辽宁折**：Full（0.616）低于 baseline（0.637）及多个 SOTA plain——与消融中 −Aux/−WMA 在辽宁的非单调现象一致，需在讨论中说明跨中心异质性。

## 3. Full vs 各 plain SOTA（15 次配对）

| 对比骨干 | Full 胜/负 | Mean Δ (Full − SOTA) | Baseline 胜/负 | Mean Δ (Baseline − SOTA) |
|----------|:----------:|:--------------------:|:--------------:|:------------------------:|
| Swin-Tiny | 8 / 7 | +0.018 | 7 / 8 | −0.009 |
| Swin-Small | 11 / 4 | +0.019 | 9 / 6 | −0.008 |
| ConvNeXt-Small | 9 / 6 | +0.020 | 7 / 8 | −0.007 |
| ViT-Small | 8 / 6 (1 平) | **+0.007** | 7 / 8 | −0.020 |
| ResNet18 | 14 / 1 | +0.051 | 9 / 6 | +0.024 |

- 换 **更强/不同 plain backbone** 普遍 **略优于** ResNet50 baseline（除 ResNet18），但 **仍未能稳定超过 Full**（除 ViT-S 在湘雅折均值上略高）。
- 与 ViT-S 的配对最接近：**Full 仅 8/15 略胜**，均值优势 +0.007，统计上需 DeLong 逐折说明。

## 4. DeLong（湘雅外部，seed=2024 示例）

Champion：OptiGenesis Full vs：

| Challenger | p 值 | Full AUC | Challenger AUC |
|------------|:----:|:--------:|:--------------:|
| ResNet50 Baseline | 0.016 * | 0.756 | 0.540 |
| ViT-Small | 0.659 ns | 0.756 | 0.728 |
| Swin-Small | 0.003 ** | 0.756 | 0.514 |
| ConvNeXt-Small | 0.127 ns | 0.756 | 0.621 |
| ResNet18 | 0.004 ** | 0.756 | 0.494 |

湘雅 seed_2024：**Full 显著优于 baseline 与轻量骨干**；与 **ViT-S plain 无显著差异**（p=0.66）。

## 5. Youden 阈值分类指标（外部集，参考）

主指标仍为 ROC-AUC；以下为 Youden 解耦后的 Adj Bal Acc（见各 `*_summary.csv`）：

| 模型 | 华西 | 辽宁 | 湘雅 |
|------|:----:|:----:|:----:|
| ViT-Small | 0.598 | 0.639 | **0.708** |
| Swin-Tiny | 0.619 | 0.636 | 0.643 |
| ResNet18 | 0.605 | 0.587 | 0.655 |
| Full（已有） | — | — | 湘雅 seed 间 Adj Bal 0.65–0.75 |

分类指标与 AUC 排序大体一致，但 **不宜仅用 BalAcc@0.5**（log 中大量 0.5 退化值）。

## 6. 论文表述建议

### 可以写的

1. **方法有效性**：Full 外部 AUC **0.645**，较 ResNet50 plain baseline **+0.027**（与消融一致）。
2. **非 backbone 竞赛**：plain SOTA 骨干（ViT-S / Swin / ConvNeXt）均值 **0.625–0.638**，多数略高于 baseline，但 **未全面超越 Full**；说明增益主要来自 **WMA+EMA+Aux**，而非单纯换骨干。
3. **轻量对照**：ResNet18 plain **0.594**，低于 ResNet50 baseline，支持「小容量 + 小样本 → 更差」的叙述。
4. **跨中心异质性**：三折领先模型不同，LOHO 下不宜过度解读单一折。

### 需要诚实报告的

1. **ViT-Small plain 总体第二（0.638）**，湘雅折均值 **略高于 Full**；Full vs ViT-S 配对 **8/15 胜、均值 +0.007**，DeLong 在代表性 seed 上 **不显著**。
2. **辽宁折** Full 并非最优，讨论中可联系 domain shift / 样本量 / 辅助监督的非单调性（与 −Aux 消融呼应）。

### 主表建议行

| 行 | 外部 AUC |
|----|----------|
| ResNet50 plain baseline | 0.618±0.037 |
| ResNet18 plain | 0.594±0.052 |
| Swin-Tiny / Swin-S / ConvNeXt / ViT-S plain | 0.625–0.638 |
| **OptiGenesis Full** | **0.645±0.053** |

## 7. 输出文件索引

| 文件 | 内容 |
|------|------|
| `comparison_v2_oct_detail.csv` | 75 行逐 run 外部 ROC/PR/BalAcc |
| `comparison_v2_oct_summary.csv` | 7 模型总体 + 分折 AUC |
| `comparison_v2_oct_by_hospital.csv` | 分折汇总 |
| `comparison_v2_oct_full_head2head.csv` | Full/Baseline vs 各 SOTA 配对 |
| `outputs_comparison_v2_oct_*/outputs_comparison_v2_oct_*_summary.csv` | Youden 分类指标 |

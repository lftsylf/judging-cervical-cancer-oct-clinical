# ViT-Small + Full（T0 · OCT-only）补充实验 — 结果与分析

> 完成度：**15/15**（3 折 × 5 seeds）  
> 协议：`vit_small_patch16_224` · WMA+EMA+Aux · LR=5e-5 · POS=1.25 · batch=2 · CORAL=0  
> 输出目录：`outputs_full_v2_oct_vit_small/`

## 1. 外部 ROC-AUC 总览（15 runs mean±std）

| 模型 | 外部 AUC | vs ViT-S plain | vs Full@ResNet50 |
|------|:--------:|:--------------:|:----------------:|
| **OptiGenesis Full (ResNet50)** | **0.6453±0.053** | +0.007 | — |
| **ViT-Small (plain)** | **0.6381±0.073** | — | −0.007 |
| **ViT-Small + Full** | **0.6008±0.053** | **−0.037** | **−0.045** |
| ResNet50 Baseline (plain) | 0.6183±0.037 | −0.020 | −0.027 |
| ResNet18 (plain) | 0.5941±0.052 | −0.044 | −0.051 |

**主结论**：在 ViT-S 上叠加 WMA+EMA+Aux **未带来增益，反而显著下降**（−0.037 vs ViT-S plain，−0.045 vs Full@ResNet50）。ViT-S+Full 甚至低于 ResNet50 plain baseline。

## 2. 分折外部 AUC

| 折 | ViT-S plain | ViT-S + Full | Δ (Full−plain) | Full@ResNet50 |
|----|:-----------:|:------------:|:--------------:|:-------------:|
| 华西 | 0.570 | 0.564 | −0.006 | 0.620 |
| 辽宁 | 0.638 | 0.613 | −0.025 | 0.616 |
| 湘雅 | **0.707** | **0.625** | **−0.081** | **0.700** |
| **15次均值** | **0.638** | **0.601** | **−0.037** | **0.645** |

- **湘雅折损失最大**（−0.081）：plain 在该折最强（0.707），加模块后反而掉到 0.625，且低于 ResNet50 Full（0.700）。
- 华西折基本持平（−0.006）；辽宁折中等幅度下降（−0.025）。

## 3. 15 次配对（同折同 seed）

| 对比 | 胜/负/平 | Mean Δ |
|------|:--------:|:------:|
| ViT-S Full vs ViT-S plain | 4 / 10 / 1 | **−0.037** |

**极端 seed**（湘雅 fold）：

| seed | ViT-S plain | ViT-S + Full | Δ |
|:----:|:-----------:|:------------:|:-:|
| 3407 | **0.822** | 0.515 | **−0.308** |
| 42 | 0.670 | 0.590 | −0.080 |
| 114514 | 0.692 | 0.608 | −0.084 |
| 123 | 0.621 | **0.705** | +0.083 |
| 2024 | 0.728 | 0.709 | −0.019 |

除 seed_123 外，湘雅 4/5 次 ViT-S Full 均低于 plain；seed_3407 出现崩溃式下降（0.822→0.515）。

## 4. Youden 阈值分类指标（外部集）

| 折 | ViT-S plain Adj Bal Acc | ViT-S + Full Adj Bal Acc |
|----|:-----------------------:|:------------------------:|
| 华西 | 0.598 | 0.599 |
| 辽宁 | 0.639 | 0.616 |
| 湘雅 | **0.708** | **0.635** |

分类指标与 AUC 趋势一致：湘雅折 ViT-S+Full 明显弱于 plain。

## 5. DeLong（湘雅，seed=2024）— 参考

| 对比 | p 值 | AUC A | AUC B |
|------|:----:|:-----:|:-----:|
| Full@ResNet50 vs ViT-S plain | 0.659 ns | 0.756 | 0.728 |
| ViT-S Full vs ViT-S plain | 0.059 ns | 0.709 | — |
| Full@ResNet50 vs ViT-S Full | 0.005 ** | 0.756 | 0.709 |

> **注意**：ViT-S Full 导出的 `external_sample_predictions.csv` 中，部分样本 `y_true` 与同折 plain / Full@ResNet50 不一致（15 runs 合计 646 处 mismatch），导致跨模型 DeLong 的 challenger AUC 可能被低估。**逐 run 外部 AUC（与训练 log 一致）仍可信**；配对 Δ 以同 run 原生 AUC 为准。

## 6. 解读与论文建议

### 可以写的

1. **模块增益具有 backbone 依赖性**：WMA+EMA+Aux 在 ResNet50 上有效（Full 0.645 vs baseline 0.618），但在 **ViT-S 上无效甚至有害**（0.601 vs plain 0.638）。
2. **更强 plain backbone ≠ 加模块更好**：ViT-S plain 已是 plain 组第二（0.638），叠加 Full 模块后 **跌至第六**（仅高于 ResNet18）。
3. **支持保留 ResNet50+Full 作为主模型**：不仅总体 AUC 最高，且是唯一验证过模块有效性的骨干配置。

### 建议谨慎或放 Supplement 的

1. 不必将 ViT-S+Full 作为主结果或新 champion；该实验更适合作为 **backbone 迁移性负结果**。
2. 若 reviewer 问「为何不换 ViT-S」：可答 plain ViT-S 已接近 Full@ResNet50（差距 0.007），且 **模块在 ViT-S 上未验证有效**；ResNet50 上模块贡献有完整消融支撑。
3. `y_true` 导出跨 run 不一致问题建议后续排查（不影响各 run 内部 AUC，但影响跨模型 DeLong）。

### 一句话摘要

> **ViT-S + Full（0.601）< ViT-S plain（0.638）< Full@ResNet50（0.645）** — Full 模块未能迁移到 ViT-S，ResNet50 仍是最佳骨干选择。

## 7. 输出文件

| 文件 | 说明 |
|------|------|
| `vit_small_full_detail.csv` | 15 runs 逐条 ext_roc / ext_pr / ext_bacc |
| `vit_small_full_summary.csv` | 三模型汇总（含 Full@ResNet50、ViT-S plain 对照） |
| `vit_small_full_vs_plain_paired.csv` | 15 次配对 Δ |
| `vit_small_full_vs_plain_by_hospital.csv` | 分折配对统计 |
| `vit_small_full_delong_xiangya.csv` | 湘雅 DeLong（含 label mismatch  caveat） |
| `outputs_full_v2_oct_vit_small/outputs_full_v2_oct_vit_small_summary.csv` | Youden 阈值汇总 |

# 实验迁移记录（来自 Obsidian）

## 1) 基本信息

- 实验名称：LOHO 基线/消融/对比实验迁移归档
- 日期：2026-04-08
- 负责人：lftsylf
- 关联分支：main
- 关联 commit：待补（建议后续每轮实验后填写）
- 实验类型：baseline + 消融 + 对比 + 方法规划

## 2) 研究主线（从原始笔记提炼）

- 主问题：小样本、多中心、类不平衡条件下的跨中心泛化与阈值失配。
- 三层改进路线：
  - 模型层：不确定性感知多模态融合（UGMF）
  - 域层：轻量域对齐/域泛化（CORAL/MMD/对抗）
  - 决策层：阈值迁移与概率校准（Youden/Platt/Isotonic）

## 3) 基线与协议口径

- 数据划分：LOHO 三折（xiangya / huaxi / liaoning）
- 统一选模规则：验证集 `AUC-ROC` 最高 epoch
- 主评估指标：external 上 `AUC-ROC / AUC-PR / Sens / Spec / MCC / BACC`
- 后处理定位：阈值迁移与校准属于部署层优化，不替代模型本体能力比较

## 4) 已记录的关键结果

### 4.1 baseline_outputs（旧 best 口径）

| Fold | Best Epoch | AUC-ROC | AUC-PR | BACC | Sens | Spec | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| xiangya | 19 | 0.7333 | 0.8676 | 0.6714 | 0.7761 | 0.5667 | 0.3370 |
| liaoning | 9 | 0.6236 | 0.4715 | 0.5000 | 0.0000 | 1.0000 | 0.0000 |
| huaxi | 35 | 0.6222 | 0.4654 | 0.5790 | 0.8125 | 0.3455 | 0.1683 |

### 4.2 消融与重跑结论（摘要）

- `ema_huaxi_15_outputs` 相对 `baseline_huaxi_15_outputs`：ROC-AUC、BACC、F1、MCC 提升，PR-AUC 略降。
- `aux_huaxi_outputs`：未观察到稳定提升。
- `aux_50_outputs`、`ema_outputs`、`uda_50_outputs`：按“新 best 口径”对比整体偏反面结果。
- `baseline_recheck_50_outputs`：辽宁改善，但华西/湘雅弱于第一版 baseline（旧 best）。

### 4.3 对比实验（ViT）

外部集宏平均（原记录）：

| 实验 | ROC-AUC | PR-AUC | Bal.Acc | Sens | Spec | F1+ | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_outputs | 0.6205 | 0.5783 | 0.5880 | 0.6983 | 0.4776 | 0.5777 | 0.1861 |
| baseline_recheck_50_outputs | 0.6189 | 0.5834 | 0.5813 | 0.5649 | 0.5978 | 0.5456 | 0.1714 |
| vit_small_patch16_224 | 0.6276 | 0.5853 | 0.5890 | 0.8316 | 0.3464 | 0.6243 | 0.1940 |
| vit_base_patch16_224 | 0.5620 | 0.5473 | 0.5335 | 0.8937 | 0.1732 | 0.6287 | 0.0643 |

结论摘要：

- ViT-Small 在宏平均上优于两版 baseline。
- ViT-Base 呈现高敏感度、低特异度，外部泛化不稳定。

## 5) 当前方法学注意事项

- 若用 external 特征参与域对齐但不使用其标签，需在论文中明确写为 UDA/transductive 设定。
- 建议主文主表优先放“训练预算一致、三折完整”的消融；不完整/探索性负结果放附录。
- 建议持续保留 `development/external_sample_predictions.csv` 以支撑阈值迁移与校准分析。

## 6) 下一步可执行计划

1. 固定协议：统一 50 epoch、统一 AUC-only 选模、统一三折顺序。
2. 最小增益路线：先做 EMA 全三折稳定复核，再决定是否推进双分支证据融合。
3. 域适应只作为第二阶段：先以 CORAL 小权重做 pilot，再扩展。
4. 将后续每次实验写成单独记录，并在 `EXPERIMENT_REGISTRY.md` 维护索引。


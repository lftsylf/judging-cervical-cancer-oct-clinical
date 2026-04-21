# OptiGenesis V2 消融实验进度记录

## 1) 当前实验目标

验证“完全体 V2”架构在 `Batch=2` 显存约束下的稳定性与下限表现。

- 架构开关：
  - `OPTIGENESIS_ENABLE_CORAL=0`
  - `OPTIGENESIS_ENABLE_EMA=1`
  - `OPTIGENESIS_ENABLE_AUX=1`
  - `OPTIGENESIS_USE_WMA=1`
  - `OPTIGENESIS_USE_CLINICAL=1`
  - `OPTIGENESIS_BATCH_SIZE=2`
  - `OPTIGENESIS_EPOCHS=30`

## 2) 已完成阶段

### 阶段 A：T2 单折探路（liaoning, seed=42）

- 脚本：`run_optigenesis_v2_t2_test.sh`
- 输出目录：`outputs_v2_t2_test/liaoning/seed_42`
- 主日志：`outputs_v2_t2_test/t2_test_master.log`
- 结果摘要：
  - 训练在 `Epoch 19/30` 触发 early stopping 后结束
  - `best_model.pth` 对应验证集 AUC：`0.6175`
  - 开发集（best 权重）：`ROC-AUC=0.4912, PR-AUC=0.5458, 平衡准确率=0.5055`
  - 外部集（best 权重）：`ROC-AUC=0.6175, PR-AUC=0.4637, 平衡准确率=0.4961`
- 结论（阶段性）：无 OOM，流程可稳定跑通，具备放大到全量 T0 的条件。

### 阶段 B：T0 全量跑批（3 中心 x 5 seeds）

- 脚本：`run_optigenesis_v2_t0.sh`
- 输出目录：`outputs_optigenesis_v2`
- 主日志：`outputs_optigenesis_v2/v2_t0_master.log`
- 已整理产物：
  - 单次汇总：`outputs_optigenesis_v2/v2_t0_runs_summary.csv`
  - 按医院均值/标准差：`outputs_optigenesis_v2/mean_std_by_hospital.csv`

## 3) 当前关键统计（按医院，5 seeds）

以下数值来自 `mean_std_by_hospital.csv`：

- **huaxi**
  - `external_roc_auc = 0.5704 +- 0.0322`
  - `external_pr_auc = 0.4265 +- 0.0312`
  - `external_balanced_acc = 0.5152 +- 0.0291`
- **liaoning**
  - `external_roc_auc = 0.5731 +- 0.0282`
  - `external_pr_auc = 0.4319 +- 0.0342`
  - `external_balanced_acc = 0.5186 +- 0.0304`
- **xiangya**
  - `external_roc_auc = 0.6432 +- 0.0664`
  - `external_pr_auc = 0.8183 +- 0.0439`
  - `external_balanced_acc = 0.5000 +- 0.0000`

> 备注：`xiangya` 的外部 PR-AUC 明显偏高，同时 `balanced_acc` 与 `MCC` 多个 seed 呈“阈值塌缩/单类预测”特征，后续建议结合阈值扫描与逐样本概率进一步核查。

## 4) 已知现象与风险点

- 大部分 run 在 early stopping 前结束，个别 run 跑满 30 epoch。
- `liaoning/seed_2024` 出现训练损失显著抬升（中后期高位波动），但流程未中断。
- `xiangya` 多个 seed 的 `max_mcc_ref=0.0`，提示分类阈值/类别判别存在退化风险。

## 5) 下一步计划（建议）

1. 基于每个 run 的 `development_sample_predictions.csv` 做阈值重选（Youden/F1/MCC）。
2. 对 `xiangya` 子集单独做校准分析（reliability + Brier + ECE）。
3. 固定 V2 配置后，追加重复实验或 bootstrap 统计，评估方差收敛趋势。
4. 输出对比基线（旧架构）的同口径表格，完成阶段性结论闭环。

## 6) 文件索引

- `run_optigenesis_v2_t2_test.sh`
- `run_optigenesis_v2_t0.sh`
- `outputs_v2_t2_test/t2_test_master.log`
- `outputs_optigenesis_v2/v2_t0_master.log`
- `outputs_optigenesis_v2/v2_t0_runs_summary.csv`
- `outputs_optigenesis_v2/mean_std_by_hospital.csv`

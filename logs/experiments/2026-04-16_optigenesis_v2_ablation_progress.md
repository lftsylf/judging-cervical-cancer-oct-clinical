# OptiGenesis V2 消融实验进度记录（T0 全程 · 已完成）

## 状态

- **进度**：15 / 15 次训练已全部跑完（3 医院 × 5 seeds）。
- **主日志**：`outputs_optigenesis_v2/v2_t0_master.log`
- **明细汇总**：`outputs_optigenesis_v2/v2_t0_runs_summary.csv`
- **按医院统计**：`outputs_optigenesis_v2/mean_std_by_hospital.csv`

## 实验协议（固定口径）

| 项目 | 设置 |
|------|------|
| 医院（LOHO） | `huaxi`、`liaoning`、`xiangya` |
| Seeds | `42`, `123`, `2024`, `3407`, `114514` |
| 最大 Epoch | 30 |
| 早停策略 | 验证集 ROC-AUC，`patience=10` |
| 选模策略 | 验证集 ROC-AUC 最高时保存 `best_model.pth` |
| 汇总字段 | `best_epoch`、`epochs_trained`、dev/external 指标、参考 F1/MCC |

## 按医院汇总（5 seeds，Mean ± Std）

数据来源：`outputs_optigenesis_v2/mean_std_by_hospital.csv`

### 华西（huaxi）

- `epochs_trained`：13.4000 ± 2.8810；`best_epoch`：3.4000 ± 2.8810
- 开发集：ROC-AUC 0.7019 ± 0.0729；PR-AUC 0.7114 ± 0.0823；平衡准确率 0.5770 ± 0.0574
- 外部集：ROC-AUC 0.5704 ± 0.0322；PR-AUC 0.4265 ± 0.0312；平衡准确率 0.5152 ± 0.0291
- `val_roc_auc_best`：0.5704 ± 0.0322；参考 `max_positive_f1_ref`：0.5396 ± 0.0033；参考 `max_mcc_ref`：0.1247 ± 0.0441

### 辽宁（liaoning）

- `epochs_trained`：17.6000 ± 7.4364；`best_epoch`：9.2000 ± 10.8490
- 开发集：ROC-AUC 0.5971 ± 0.1424；PR-AUC 0.6418 ± 0.1342；平衡准确率 0.5119 ± 0.0282
- 外部集：ROC-AUC 0.5731 ± 0.0282；PR-AUC 0.4319 ± 0.0342；平衡准确率 0.5186 ± 0.0304
- `val_roc_auc_best`：0.5731 ± 0.0282；参考 `max_positive_f1_ref`：0.4011 ± 0.2268；参考 `max_mcc_ref`：0.0888 ± 0.0594

### 湘雅（xiangya）

- `epochs_trained`：18.6000 ± 7.5366；`best_epoch`：8.6000 ± 7.5366
- 开发集：ROC-AUC 0.5816 ± 0.0663；PR-AUC 0.5949 ± 0.0467；平衡准确率 0.5000 ± 0.0000
- 外部集：ROC-AUC 0.6432 ± 0.0664；PR-AUC 0.8183 ± 0.0439；平衡准确率 0.5000 ± 0.0000
- `val_roc_auc_best`：0.6432 ± 0.0664；参考 `max_positive_f1_ref`：0.8171 ± 0.0000；参考 `max_mcc_ref`：0.0000 ± 0.0000

## 当前结论（阶段性）

- 外部 ROC-AUC 均值：`xiangya (0.6432)` > `liaoning (0.5731)` ≈ `huaxi (0.5704)`。
- `liaoning` 的 seed 方差较大（`best_epoch` 和开发集 AUC 波动明显），后续建议优先检查该中心的数据分布与阈值行为。
- `xiangya` 外部 PR-AUC 保持较高（0.8183 ± 0.0439），但平衡准确率固定在 0.5000，需关注阈值迁移/校准问题。

## 下一步建议

- [ ] 生成 v2 与 baseline/WMA 的并列表（统一指标、同一统计口径）。
- [ ] 对外部 ROC-AUC 做 seed 级 bootstrap 置信区间与配对检验。
- [ ] 增加阈值迁移与校准分析（基于 development/external 预测明细 CSV）。

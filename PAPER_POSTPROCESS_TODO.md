# 论文前后处理待办清单（训练已完成阶段）

> 适用范围：当前已完成 baseline、3 个消融、2 个对比，进入后处理与论文写作阶段。

## A. 先定规则（避免口径漂移）

- [x] **确定主 baseline 版本**（只能选一个作为正文主结果）
  - 候选：`baseline_outputs`（当前组合版）
  - 其他 baseline 版本统一放补充材料并解释差异来源
- [x] **固定选模与报告口径**
  - checkpoint 选择：`val ROC-AUC`（已采用）
  - 分类阈值策略：在 development 选阈值，再固定到 external
  - 主表指标：`ROC-AUC / PR-AUC / Bal.Acc / Accuracy / MCC / Sens / Spec / PPV / NPV`
- [x] **写入方法章节一句话**：
  - “早停/选模与阈值解耦：模型按 AUC 选，分类按预设阈值策略报告。”

## B. 阈值后处理（当前最关键）

### B1. 每折阈值选择（development）

- [x] 为 `xiangya / huaxi / liaoning` 分别扫描阈值 `0.00~1.00`（步长 0.01）
- [x] 选择规则二选一（已定主方案）：
  - [ ] Youden 最大（`sens + spec - 1`）
  - [x] 满足 `Sensitivity >= 0.85` 的**最大阈值**（临床偏召回，避免阈值=0 导致全阳性）
- [x] 输出每折阈值表：`threshold_table_<fold>.csv`

### B2. 阈值迁移（external）

- [x] 将每折 development 得到的阈值 `t*` 固定应用到对应 external
- [x] 计算并导出每折 external 指标：
  - `ROC-AUC, PR-AUC, Bal.Acc, Accuracy, MCC, Sens, Spec, PPV, NPV`
- [x] 输出 `t=0.5` vs `t=t*` 对照表（每折 + 三折均值）

### B3. 建议输出目录

- [x] 新建：`all_outputs/baseline_outputs/postprocess_threshold/`
- [x] 放入：
  - `threshold_table_xiangya.csv`
  - `threshold_table_huaxi.csv`
  - `threshold_table_liaoning.csv`
  - `external_metrics_at_0.5_vs_tstar.csv`
  - `external_metrics_at_0.5_vs_tstar_maxthr.csv`
  - `external_metrics_mean_0.5_vs_tstar.csv`
  - `summary_threshold_migration.md`

### B4. 已确认的后处理决策（论文可引用）

- 主方案（正文）：在 development 上选择满足 `Sensitivity >= 0.85` 的**最大阈值**，固定迁移到 external。
- 说明：若使用“最小阈值”会退化为接近 `t=0`，导致 external 近似全阳性（`Spec≈0`），不适合作为正文主策略。
- 结果文件：
  - `all_outputs/baseline_outputs/postprocess_threshold/external_metrics_at_0.5_vs_tstar_maxthr.csv`
  - `all_outputs/baseline_outputs/postprocess_threshold/summary_threshold_migration.md`

## C. 论文结果表与图（最低可投稿集）

- [ ] **主结果表**：baseline + 对比（2 个）+ 消融（3 个）
- [ ] **阈值前后对照表**：`0.5` vs `t*`（重点放辽宁）
- [ ] **图 1**：external ROC 曲线（3 折）
- [ ] **图 2**：external PR 曲线（3 折）
- [ ] **图 3**：阈值-指标曲线（`Sens/Spec/MCC`）

## D. 稳定性与统计（建议补）

- [ ] 统一输出三折 `mean ± std`
- [ ] 至少对 AUC 给出置信区间（bootstrap 或简化 CI）
- [ ] 在正文说明：样本量与中心间分布差异带来的不确定性

## E. 错误分析（医学审稿很看重）

- [ ] 挑选典型 `FP/FN` 案例（每折各 3-5 例）
- [ ] 给出临床解释线索（为什么错、可能改进方向）
- [ ] 形成 `Error Analysis` 小节素材

## F. 文档与可追溯性

- [ ] 在 `EXPERIMENT_REGISTRY.md` 补充后处理条目（阈值迁移实验）
- [ ] 新建详细记录：
  - `logs/experiments/2026-04-09_threshold_migration_baseline_outputs.md`
- [ ] 记录所有后处理脚本与输出目录，确保可复现

## G. 最终写作检查（提交前）

- [ ] Methods 中“阈值策略”与 Results 表格口径一致
- [ ] 所有主表数值可追溯到具体 csv/json 文件
- [ ] 讨论部分解释“消融多数反面”的合理性（分布偏移/阈值错配/样本结构）
- [ ] 附录放更多版本 baseline 对照，正文只保留主线

---

## 你现在可直接执行的下一步（建议顺序）

1. 先定阈值规则（Youden 或固定 Sensitivity）
2. 跑三折阈值迁移并导出 `0.5 vs t*` 总表
3. 更新主结果表（包含并集指标）
4. 开始写 Results 与 Discussion


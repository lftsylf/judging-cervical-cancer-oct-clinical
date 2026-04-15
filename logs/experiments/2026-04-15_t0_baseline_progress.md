# T0 Baseline 实验进度记录（第二阶段 · 已完成）

## 状态

- **进度**：15 / 15 次训练已全部跑完（3 折医院 × 5 个 seed）。
- **实验代号**：T0 Baseline 标杆。
- **记录日期**：2026-04-15（与汇总表一致时可更新本行）。

## 实验协议（固定口径）

| 项目 | 设置 |
|------|------|
| 折 | LOHO：`huaxi`、`liaoning`、`xiangya` 各为一折验证医院 |
| Seeds | `42`, `123`, `2024`, `3407`, `114514` |
| 最大 Epoch | `30`（`OPTIGENESIS_EPOCHS=30`） |
| Early Stopping | 验证集 `ROC-AUC`，`patience=10` |
| 选模 / `best_model.pth` | 验证集 `ROC-AUC` 最高时保存 |
| 骨干 / 开关 | Swin-Tiny；aux / EMA / CORAL 均关闭 |
| 批量脚本 | 仓库根目录 `run_baseline_t0.sh` |
| 挂机总日志 | 若仍保留：`outputs/t0_master.log`（路径以你服务器实际为准） |

## 产出目录- **单次运行**：`outputs/<hospital>/seed_<seed>/`  - `checkpoints/best_model.pth`  
  - `logs/training_history.json`、`development_sample_predictions.csv`、`external_sample_predictions.csv`、`train_console.log`（若脚本有 tee）
- **汇总表（推荐归档位置）**：
  - `outputs_baseline/t0_baseline_15runs_summary.csv` — 15 行明细（含 `best_epoch`、`trained_epochs`、dev/external 指标等）
  - `outputs_baseline/mean_std_by_hospital.csv` — 每家医院 5 seed 的均值与标准差

## 按医院汇总（5 seeds · Mean ± Std）

数据来自 `outputs_baseline/mean_std_by_hospital.csv`（小数保留与 CSV 一致）。

### 华西（huaxi）

- `best_epoch`：7.40 ± 3.91；`trained_epochs`：17.40 ± 3.91  
- 开发集 `ROC-AUC`：0.8244 ± 0.0391；`PR-AUC`：0.8256 ± 0.0692；平衡准确率：0.7354 ± 0.0411  
- 外部集 `ROC-AUC`：0.5737 ± 0.0154；`PR-AUC`：0.4397 ± 0.0368；平衡准确率：0.4916 ± 0.0187  
- `best_model`验证 AUC：0.5737 ± 0.0154  
- 全程最高阳性 F1（参考）：0.5451 ± 0.0028；全程最高 MCC（参考）：0.1150 ± 0.0182  

### 辽宁（liaoning）

- `best_epoch`：11.00 ± 5.39；`trained_epochs`：21.00 ± 5.39  
- 开发集 `ROC-AUC`：0.8255 ± 0.0697；`PR-AUC`：0.8468 ± 0.0543；平衡准确率：0.7117 ± 0.0881  
- 外部集 `ROC-AUC`：0.5872 ± 0.0074；`PR-AUC`：0.4618 ± 0.0180；平衡准确率：0.5683 ± 0.0094  
- `best_model` 验证 AUC：0.5872 ± 0.0074  
- 全程最高阳性 F1（参考）：0.5230 ± 0.0040；全程最高 MCC（参考）：0.1549 ± 0.0206  

### 湘雅（xiangya）

- `best_epoch`：8.40 ± 7.86；`trained_epochs`：18.20 ± 7.46  
- 开发集 `ROC-AUC`：0.6817 ± 0.1220；`PR-AUC`：0.6767 ± 0.1325；平衡准确率：0.6019 ± 0.1208  
- 外部集 `ROC-AUC`：0.6050 ± 0.0372；`PR-AUC`：0.7956 ± 0.0208；平衡准确率：0.5175 ± 0.0503  
- `best_model` 验证 AUC：0.6050 ± 0.0372  
- 全程最高阳性 F1（参考）：0.8171 ± 0.0000；全程最高 MCC（参考）：0.1406 ± 0.0979  

## 备注

- 明细表中 `history_path` 仍指向 `outputs/.../training_history.json`；若你之后整体迁移目录，请同步更新 CSV 或在本文件注明新根路径。
- 更完整的实现说明见：`logs/experiments/2026-04-09_t0_baseline_setup.md`。

## 建议的下一步

- [ ] 将三折外部集 `ROC-AUC`（或论文主表指标）汇总为「三折 Mean ± Std」或配合 bootstrap CI。  
- [ ] 在 `EXPERIMENT_REGISTRY.md` 增加一条 T0 完成记录，并附上 commit 与上述两个 CSV 路径。  
- [ ] 若正文需要阈值迁移结果，在 best 权重导出的 `development_sample_predictions.csv` 上跑既有后处理脚本。

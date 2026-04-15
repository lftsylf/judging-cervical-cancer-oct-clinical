# WMA Loss 消融实验进度记录（正式全程 · 已完成）

## 状态

- **进度**：15 / 15 次训练已全部跑完（3 折医院 × 5 个 seed）。
- **实验内容**：在 EDL 框架上启用 **WMA Loss**（重加权边距调整 + UNCE-MA / TCE-MA + 原 EDL KL 正则）；与 T0 Baseline 相同的 LOHO 协议，仅主损失由 Focal+EDL 切换为 WMA（`OPTIGENESIS_USE_WMA=1`）。
- **探路（T2）**：`liaoning` + `seed=42` 单次跑通，外部集 AUC 约 **0.6425**（详见 `outputs_wma_test/…`）。
- **记录更新**：WMA 正式全程汇总表与按医院 mean±std 已落盘（见下文路径）。

## 实验协议（固定口径）

| 项目 | 设置 |
|------|------|
| 折 | LOHO：`huaxi`、`liaoning`、`xiangya` 各为一折验证医院 |
| Seeds | `42`, `123`, `2024`, `3407`, `114514` |
| 最大 Epoch | `30`（`OPTIGENESIS_EPOCHS=30`） |
| Early Stopping | 验证集 `ROC-AUC`，`patience=10` |
| 选模 / `best_model.pth` | 验证集 `ROC-AUC` 最高时保存 |
| WMA | `export OPTIGENESIS_USE_WMA=1`；配置默认 `WMA_C=0.2`，`WMA_WARMUP_EPOCHS=10`（可用环境变量覆盖） |
| 骨干 / 其他开关 | Swin-Tiny；aux / EMA / CORAL 均关闭（与 `run_wma_full_t0.sh` 一致） |
| 批量脚本 | 仓库根目录 `run_wma_full_t0.sh`；单折探路 `run_wma_test_t2.sh` |
| 挂机总日志 | `outputs_wma_full/wma_full_master.log`（若使用 nohup 重定向） |

## 代码与实现位置（便于复现）

- **损失**：`training/losses.py` → `wma_loss`
- **训练集成**：`training/trainer.py`、`main.py`（训练集类计数 `N_P`、epoch 传入 WMA）
- **配置开关**：`configs/lancet_config.py` → `USE_WMA_LOSS`、`WMA_C`、`WMA_WARMUP_EPOCHS`、`WMA_TEMPERATURE`

## 产出目录

- **单次运行**：`outputs_wma_full/<hospital>/seed_<seed>/`
  - `checkpoints/best_model.pth`
  - `logs/training_history.json`、`development_sample_predictions.csv`、`external_sample_predictions.csv`、`train_console.log`
- **汇总表（当前归档）**：
  - `outputs_wma_full/wma_15runs_summary.csv` — 15 行明细（`best_epoch`、`trained_epochs`、dev/external、`best_auc`、参考 F1/MCC）
  - `outputs_wma_full/mean_std_by_hospital.csv` — 每家医院 5 seed 的均值 ± 标准差

## 按医院汇总（5 seeds · Mean ± Std）

数据来自 `outputs_wma_full/mean_std_by_hospital.csv`（小数与 CSV 一致）。

### 华西（huaxi）

- `best_epoch`：5.00 ± 8.40；`trained_epochs`：15.00 ± 8.40  
- 开发集 `ROC-AUC`：0.7534 ± 0.0580；`PR-AUC`：0.7720 ± 0.0436；平衡准确率：0.6252 ± 0.1213  
- 外部集 `ROC-AUC`：0.5749 ± 0.0228；`PR-AUC`：0.4572 ± 0.0183；平衡准确率：0.5057 ± 0.0083  
- `best_model` 验证 AUC：0.5749 ± 0.0228  
- 全程最高阳性 F1（参考）：0.5427 ± 0.0046；全程最高 MCC（参考）：0.1090 ± 0.0424  

### 辽宁（liaoning）

- `best_epoch`：5.20 ± 5.36；`trained_epochs`：15.20 ± 5.36  
- 开发集 `ROC-AUC`：0.7220 ± 0.0461；`PR-AUC`：0.7519 ± 0.0375；平衡准确率：0.5855 ± 0.0898  
- 外部集 `ROC-AUC`：0.5723 ± 0.0534；`PR-AUC`：0.4132 ± 0.0534；平衡准确率：0.5227 ± 0.0262  
- `best_model` 验证 AUC：0.5723 ± 0.0534  
- 全程最高阳性 F1（参考）：0.5194 ± 0.0124；全程最高 MCC（参考）：0.1048 ± 0.0524  

### 湘雅（xiangya）

- `best_epoch`：8.20 ± 6.72；`trained_epochs`：18.20 ± 6.72  
- 开发集 `ROC-AUC`：0.6512 ± 0.0690；`PR-AUC`：0.6716 ± 0.0903；平衡准确率：0.5281 ± 0.0629  
- 外部集 `ROC-AUC`：0.6628 ± 0.0395；`PR-AUC`：0.8389 ± 0.0267；平衡准确率：0.5132 ± 0.0296  
- `best_model` 验证 AUC：0.6628 ± 0.0395  
- 全程最高阳性 F1（参考）：0.8181 ± 0.0022；全程最高 MCC（参考）：0.1136 ± 0.1045  

## 与 T0 Baseline 的对照提示（写论文时可用）

- Baseline 进度与按医院汇总见：`logs/experiments/2026-04-15_t0_baseline_progress.md`（产出在 `outputs_baseline/`）。
- 本文件仅记录 **WMA 分支**；正式论文表建议在同一表格中并列 Baseline / WMA，并统一主指标（如三折外部 `ROC-AUC` 的 mean±std 或 CI）。

## 备注

- `best_epoch` 定义为：`training_history.json` 中验证集 `auc_roc` 最大的那条记录的 `epoch`；`trained_epochs` 为历史列表长度（实际跑完的 epoch 数，受早停与上限 30 约束）。
- 若之后迁移输出根目录，请同步更新本文件中的路径或 CSV 内相对引用。

## 建议的下一步

- [ ] 生成「三折合并」的 WMA vs Baseline 对照表（同一指标、同一格式化）。  
- [ ] 如需显著性：对外部 AUC 等做 bootstrap 或配对检验（按 seed 配对）。  
- [ ] 若需阈值/校准分析：在 `development_sample_predictions.csv` / `external_sample_predictions.csv` 上跑既有后处理脚本。  
- [ ] 在实验登记处（如 `EXPERIMENT_REGISTRY.md`）增加 WMA 全程完成记录，并附 commit 与上述 CSV 路径。

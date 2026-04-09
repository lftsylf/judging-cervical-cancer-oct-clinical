# EMA · 三中心 · 50 epoch（与 baseline 完整消融对齐）

- **目的**：在 **50 轮** 与仓库内 `baseline_outputs` 完整版一致的训练预算下，对 LOHO 三中心（辽宁肿瘤 / 华西 / 湘雅）跑 **EMA 消融**。
- **配置要点**：`EPOCHS = 50`，`ENABLE_MODEL_EMA = True`，`ENABLE_MULTIMODAL_AUX_LOSS = False`；通常由 `bash run_loho_3centers.sh` 依次设置 `HOSPITAL_NAME=liaoning|huaxi|xiangya`。
- **选模口径**：各中心子目录下以 `logs/training_history.json` 中 **`val.auc_roc` 最大值** 及对应 epoch 为准（验证集即该 fold 的外部 CSV）。
- **目录结构**：`{liaoning,huaxi,xiangya}/checkpoints/`、`.../logs/`。

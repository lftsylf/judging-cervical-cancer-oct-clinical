# T0 Baseline 挂机实验改动记录（3 折 × 5 Seed）

## 1. 本次目标

- 为第二阶段 T0 基线实验建立可复现、可批量、可挂机的执行方案。
- 统一协议：3 折医院（`huaxi / liaoning / xiangya`）× 5 个 seed（`42, 123, 2024, 3407, 114514`）。
- 最大训练轮数设为 `30`，同时保留 `Early Stopping`（基于验证集 `ROC-AUC`，`patience=10`）。

## 2. T0 Baseline 方法开关说明（用了什么 / 没用什么）

`run_baseline_t0.sh` 将 baseline 与后续消融（aux / EMA / CORAL / WMA 等）区分开，避免环境变量残留。下面仅描述 **T0 一次训练实际生效** 的配置。

### 2.1 启用（Baseline 主线）

| 模块 | 说明 |
|------|------|
| 视觉骨干 | `swin_tiny_patch4_window7_224`（脚本内 `OPTIGENESIS_BACKBONE` 固定） |
| 多模态融合 | `USE_CLINICAL=True`（默认；脚本未关闭）—— OCT 多帧 + 临床向量（如年龄、HPV、TCT）融合进同一网络 |
| 证据学习 / 不确定性 | `USE_UNCERTAINTY=True`（配置中开启）—— EDL 风格输出，用于概率与不确定度 |
| 损失与训练 | Focal + EDL 主损失路径、类别权重 `POS_WEIGHT`；AdamW + Cosine LR |
| 数据 | `WeightedRandomSampler` + 训练集强增强 |
| 选模与早停 | 验证集 `ROC-AUC` 保存 `best_model.pth`；`patience=10` Early Stopping；`seed_everything` 可复现 |

### 2.2 未启用（仓库有实现，但 T0 脚本显式关闭）

| 模块 | 环境变量 | T0 脚本中的设置 |
|------|----------|-----------------|
| CORAL 域适应 | `OPTIGENESIS_ENABLE_CORAL` | `export ...=0` |
| 多模态辅助监督（双头 aux） | `OPTIGENESIS_ENABLE_AUX` | `=0` |
| Model EMA | `OPTIGENESIS_ENABLE_EMA` | `=0` |
| WMA Loss（若当前 `lancet_config` 已接入） | `OPTIGENESIS_USE_WMA` | `=0` |

### 2.3 与常见表述的对应

- **「多模态」**：T0 指 **图像 + 临床特征融合**（主融合分支），**不是** aux 双头监督（aux 为单独消融）。
- **「域适应」**：代码中有 **CORAL**（`training/losses.py` + `ENABLE_DOMAIN_CORAL`），**T0 baseline 未打开**；需 UDA 实验时另设 `OPTIGENESIS_ENABLE_CORAL=1`（例如 `run_loho_uda_50.sh` 一类脚本）。

## 3. 代码改动清单

### 3.1 `main.py`

- 新增 `seed_everything(seed)`，锁定随机性来源：
  - Python `random`
  - `numpy`
  - `torch`（CPU/GPU）
  - `cudnn`（`deterministic=True`, `benchmark=False`）
  - `torch.use_deterministic_algorithms(True, warn_only=True)`
- 在 `main()` 开头调用：`seed_everything(Config.SEED)`。
- DataLoader 调用改为显式传 seed：
  - 训练集：`get_dataloader(..., seed=Config.SEED)`
  - 验证集：`get_dataloader(..., seed=Config.SEED + 1000)`
- 训练循环加入基于验证集 `auc_roc` 的 Early Stopping：
  - 连续 `10` 个 epoch 未提升则提前停止。
  - 与 `best_model.pth` 的保存口径一致（均以 `val ROC-AUC` 为准）。

### 3.2 `data/dataset_lancet.py`

- `get_dataloader` 新增参数：`seed=None`。
- 当传入 seed 时，新增：
  - `worker_init_fn`：为每个 worker 设置 `numpy/random/torch` 随机种子。
  - `generator.manual_seed(seed)`：固定 DataLoader 采样随机流。
- 目的：保证多进程数据增强与采样在同一环境下可复现。

### 3.3 `configs/lancet_config.py`

- `EPOCHS` 改为支持环境变量覆盖：
  - `OPTIGENESIS_EPOCHS`（默认 `50`）
- `SEED` 改为支持环境变量覆盖：
  - `OPTIGENESIS_SEED`（默认 `42`）
- 目的：支持脚本批量循环调用，无需手改配置文件。

### 3.4 新增脚本 `run_baseline_t0.sh`

- 功能：自动执行 3 折 × 5 seed 的完整训练。
- 脚本流程：
  1. 清理潜在环境残留（防止继承历史实验开关）。
  2. 显式锁定 baseline 关键开关（Swin + 关闭 aux/ema/coral）。
  3. 预处理步骤：`run_experiment.sh` + `prepare_loho_data.py`。
  4. 双层循环运行 `python main.py`。
- 输出目录规范：
  - `outputs/<hospital>/seed_<seed>/checkpoints`
  - `outputs/<hospital>/seed_<seed>/logs`
- 单次运行日志：
  - `outputs/<hospital>/seed_<seed>/logs/train_console.log`

## 4. 挂机执行方式（远程防断连）

推荐命令：

```bash
nohup bash -lc 'cd /ssd_data/tsy_study_venv/OptiGenesis_Lancet && ./run_baseline_t0.sh' > /ssd_data/tsy_study_venv/OptiGenesis_Lancet/outputs/t0_master.log 2>&1 &
```

查看总日志：

```bash
tail -f /ssd_data/tsy_study_venv/OptiGenesis_Lancet/outputs/t0_master.log
```

查看单组合日志（示例）：

```bash
tail -f /ssd_data/tsy_study_venv/OptiGenesis_Lancet/outputs/huaxi/seed_42/logs/train_console.log
```

## 5. 复现建议（用于论文统计）

- 对每个医院完成 5 个 seed 后，计算 `Mean ± Std`。
- 保持同一机器与软件环境（CUDA/PyTorch 版本）以减少非算法差异。
- 若后续扩展实验（如 EMA/CORAL），建议复用同一脚本模板，仅改方法开关，保证协议一致可比。

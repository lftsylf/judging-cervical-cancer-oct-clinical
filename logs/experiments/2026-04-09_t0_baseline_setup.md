# T0 Baseline 挂机实验改动记录（3 折 × 5 Seed）

## 1. 本次目标

- 为第二阶段 T0 基线实验建立可复现、可批量、可挂机的执行方案。
- 统一协议：3 折医院（`huaxi / liaoning / xiangya`）× 5 个 seed（`42, 123, 2024, 3407, 114514`）。
- 最大训练轮数设为 `30`，同时保留 `Early Stopping`（基于验证集 `ROC-AUC`，`patience=10`）。

## 2. 代码改动清单

### 2.1 `main.py`

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

### 2.2 `data/dataset_lancet.py`

- `get_dataloader` 新增参数：`seed=None`。
- 当传入 seed 时，新增：
  - `worker_init_fn`：为每个 worker 设置 `numpy/random/torch` 随机种子。
  - `generator.manual_seed(seed)`：固定 DataLoader 采样随机流。
- 目的：保证多进程数据增强与采样在同一环境下可复现。

### 2.3 `configs/lancet_config.py`

- `EPOCHS` 改为支持环境变量覆盖：
  - `OPTIGENESIS_EPOCHS`（默认 `50`）
- `SEED` 改为支持环境变量覆盖：
  - `OPTIGENESIS_SEED`（默认 `42`）
- 目的：支持脚本批量循环调用，无需手改配置文件。

### 2.4 新增脚本 `run_baseline_t0.sh`

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

## 3. 挂机执行方式（远程防断连）

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

## 4. 复现建议（用于论文统计）

- 对每个医院完成 5 个 seed 后，计算 `Mean ± Std`。
- 保持同一机器与软件环境（CUDA/PyTorch 版本）以减少非算法差异。
- 若后续扩展实验（如 EMA/CORAL），建议复用同一脚本模板，仅改方法开关，保证协议一致可比。

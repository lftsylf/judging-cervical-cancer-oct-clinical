# v2 单模态 T0 消融（OCT-only）

与定稿 baseline（`run_baseline_t0_oct_only.sh`）对齐：ResNet50、LR=5e-5、POS=1.25、BATCH=4、CORAL=0。  
每组 **3 折 × 5 seeds = 15 次**。主指标：**外部 ROC-AUC**。

## 完整 2³ 消融矩阵（WMA / EMA / Aux）

| 脚本 | WMA | EMA | Aux | 输出目录 | 状态 |
|------|:---:|:---:|:---:|----------|------|
| `run_baseline_t0_oct_only.sh` | 0 | 0 | 0 | `outputs_baseline_t0_v2_resnet50_oct_only/` | ✅ 定稿 |
| `run_ablation_only_wma_t0_oct_only.sh` | 1 | 0 | 0 | `outputs_ablation_v2_oct_only_wma/` | 待跑 |
| `run_ablation_only_ema_t0_oct_only.sh` | 0 | 1 | 0 | `outputs_ablation_v2_oct_only_ema/` | 待跑 |
| `run_ablation_only_aux_t0_oct_only.sh` | 0 | 0 | 1 | `outputs_ablation_v2_oct_only_aux/` | 待跑 |
| `run_ablation_no_wma_t0_oct_only.sh` | 0 | 1 | 1 | `outputs_ablation_v2_oct_no_wma/` | ✅ |
| `run_ablation_no_ema_t0_oct_only.sh` | 1 | 0 | 1 | `outputs_ablation_v2_oct_no_ema/` | ✅ |
| `run_ablation_no_aux_t0_oct_only.sh` | 1 | 1 | 0 | `outputs_ablation_v2_oct_no_aux/` | ✅ |
| `run_optigenesis_full_t0_oct_only.sh` | 1 | 1 | 1 | `outputs_ablation_v2_oct_full/` | ✅ |

## 串行入口

| 脚本 | 内容 |
|------|------|
| `run_ablation_t0_oct_only_all.sh` | Full + −WMA + −EMA + −Aux（四组） |
| `run_ablation_only_t0_oct_only_all.sh` | **仅 WMA / 仅 EMA / 仅 Aux（三组，补全矩阵）** |

## Detach 启动（断线仍继续）

```bash
export PATH="/home/amax/anaconda3/bin:$PATH"
cd /ssd_data/tsy_study_venv/OptiGenesis_Lancet

# 推荐：三组单模块一次 detach 串行
./run_experiment.sh --detach \
  /ssd_data/tsy_study_venv/OptiGenesis_Lancet/tsy_loho \
  ./run_ablation_only_t0_oct_only_all.sh

# 或只跑一组，例如仅 EMA
./run_experiment.sh --detach \
  /ssd_data/tsy_study_venv/OptiGenesis_Lancet/tsy_loho \
  ./run_ablation_only_ema_t0_oct_only.sh

# 断线后看总日志
tail -f logs/detached_latest.log

# 续跑（默认跳过已完成）
SKIP_COMPLETED=1 ./run_ablation_only_wma_t0_oct_only.sh
```

## 结果汇总

- 已完成四组（Full / −WMA / −EMA / −Aux）：`ablation_final_summary.csv`、`ablation_final_analysis.md`
- 单模块三组跑完后可同样从各目录 `*/logs/train_console.log` 提取「最佳权重 · 外部集」行

旧入口 `run_ablation_no_*_t0.sh`、`run_optigenesis_v2_t0.sh` 已重定向到 `*_oct_only.sh`。

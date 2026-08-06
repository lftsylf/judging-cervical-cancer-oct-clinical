# 湘雅内部 site-bag baseline（无稳定模块）· 2026-07-31

## 协议
- 内部：湘雅 97，8:2（`split-seed=20260731`）→ `train_xyi` / `val_xyi`
- 外部：华西 87 + 辽宁 196（磁盘有 OCT），分开终评 + pooled
- **n=2** 点位 × 全页；训优先病理阳点；兄妹 ID±1 共享阳点
- val/test：6 窗不重叠 → **mean**（主方法）或 **max**（消融）；硬判阈 0.5
- 方法：UW + FrameAux **MIL@0.3**；底座无 EMA / Aux / WMA / Attn

## 预处理对照（5 seeds，ext pooled ROC）
| 角色 | 目录（`baseline2/` 或旧名） | Ext ROC |
|--|--|--|
| ① max 消融 | `1[消融] xyi_sitebag_n2_uw_mil_t0` | 0.531±0.030 |
| **② mean 主底座** | `2[main] xyi_sitebag_n2_uw_mil_aggmean_t0` | **0.547±0.029** |
| ③ n12×page1 | `3[baseline] xyi_n12_page1_uw_mil_t0` | 0.510±0.028 |

## 稳定性（在 ② 上，5 seeds；输出在 `baseline/xyi_sitebag_n2_uw_mil_aggmean_*`）
四卡并行 wall：**~4.7 h**（2026-08-06 17:59 → 22:43）；单实验 5 seeds 串行约 **2.8–3.1 h/GPU**（~0.6 h/seed）。

| 配置 | Val ROC | Ext pooled | Huaxi | Liaoning | h/seed |
|--|--|--|--|--|--|
| ② mean（无稳定） | 0.754±0.042 | 0.547±0.029 | 0.521±0.020 | 0.561±0.037 | 0.42 |
| EMA0.99+Aux0.1 | 0.762±0.021 | 0.547±0.029 | 0.524±0.017 | 0.562±0.042 | 0.62 |
| **仅 EMA0.99** | 0.790±0.090 | **0.564±0.034** | 0.547±0.015 | 0.576±0.040 | 0.62 |
| 仅 Aux0.1 | 0.774±0.023 | 0.557±0.039 | 0.535±0.030 | 0.568±0.047 | 0.57 |
| 仅 WMA C=0.1 | 0.797±0.049 | 0.549±0.020 | 0.543±0.016 | 0.559±0.016 | 0.60 |

注：新协议数字不可与旧 LOHO #9（~0.556）直接比。当前外部最优为 **仅 EMA**；EMA+Aux 与底座持平。

## 启动
```bash
export PATH="/home/amax/anaconda3/bin:$PATH"
cd /ssd_data/tsy_study_venv/OptiGenesis_Lancet
# 底座 / 消融
./scripts/paper_v4_run/run_xyi_sitebag_n2_uw_mil_t0.sh
./scripts/paper_v4_run/run_xyi_ablation_quad_gpu.sh
# 稳定性四卡
./scripts/paper_v4_run/run_xyi_mean_stab_quad_gpu.sh
```

## 关键文件
- `data/sitebag_utils.py` / `data/prepare_xyi_sitebag_splits.py`
- `data/snapshots/paper_v4_xyi_sitebag/`
- `scripts/paper_v4_run/run_xyi_*.sh`
- Config：`OPTIGENESIS_SITEBAG=1` `SITEBAG_N=2` `SITEBAG_EVAL_AGG=mean|max`

## 切分规模（生成时）
- train 78（阳 54）/ val 19（阳 13）
- 阳且有点（含兄妹共享）湘雅 62/67
- external huaxi 87 / liaoning 196

## 输出根
`outputs/paper_v4/baseline{,2}/…`（不进 git）

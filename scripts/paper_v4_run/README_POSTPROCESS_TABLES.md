# paper_v4 / baseline2：阈值后处理 · 出表 · 绘图

> 协议锚：湘雅内训 / 华西+辽宁外测；**Ours = `4[final]`** Ext ROC **0.564±0.034**。  
> **表内指标列尚未最终定稿**——当前脚本会算齐 Youden 解耦列，但论文用哪些列请作者确认后再定。  
> 勿与旧 LOHO / Attn（现 `outputs/paper_v4/useless_旧baseline_LOHO_Attn/`）横比。

## 阈值怎么处理（对齐 v2/v3）

与旧版论文共用 `scripts/youden_threshold_utils.py`（**同一套算法**）：

1. 在给定 `(模型, seed, 划分)` 的 `y_true` / `prob_positive` 上搜阈值；  
2. 候选：分数唯一值的中点等，且要求 **Sens>0 且 Spec>0**（非退化）；  
3. 主目标：**Youden = Sens + Spec − 1**；  
4. 平局：先比 **F1(阳性)**，再比 **|t − 患病率|**，取更小者。  

| 策略 | 含义 | 与 v2/v3 |
|------|------|----------|
| **`youden_on_split`（默认）** | 在**当前评估划分**上 per-seed Youden（Ext pooled / 华西 / 辽宁 / Val 各自标定） | 与 v2 Table1/2「外部集逐 run Youden」同思路；本协议无 LOHO，湘雅只作 Val |
| `youden_on_val` | 湘雅 Val 标定 t* → 应用到外测 | 更贴近部署；可选附表 |
| `fixed0.5` | 固定 0.5 | 对齐训练日志；外部常不稳，**不建议主文** |

**ROC-AUC / PR-AUC 不依赖阈值**（全阈值扫描）。Sens/Spec/PPV/NPV/Youden 才依赖上述策略。

分中心文件：必须用 `external_huaxi_sample_predictions.csv` / `external_liaoning_*.csv`；不要从 pooled 按 `center` 切片。

## 一键命令

```bash
cd /ssd_data/tsy_study_venv/OptiGenesis_Lancet
export PATH="/home/amax/anaconda3/bin:$PATH"

python scripts/paper_v4_run/analyze_paper_v4_metrics.py
python scripts/paper_v4_run/analyze_paper_v4_metrics.py --threshold-policy youden_on_val

python scripts/paper_v4_run/plot_paper_v4_roc_pr.py --which comparison
python scripts/paper_v4_run/plot_paper_v4_roc_pr.py --which ablation
```

## 输出

| 路径 | 内容 |
|------|------|
| `outputs/paper_v4/tables/PAPER_TABLES_baseline2.md` | 主对比 / 消融 / 操作点草稿 / Wilcoxon |
| `outputs/paper_v4/tables/baseline2_metrics_*.csv` | 明细与汇总 |
| `figures/paper_v4/fig_ext_roc_pr_combined_*.{pdf,png}` | Ext ROC/PR |

注册表：`scripts/paper_v4_run/paper_v4_registry.py`（改方法路径后重跑即可）。

## 待作者确认

- 主表要不要 Sens/Spec/PPV/NPV，还是只报 ROC（±PR）  
- 操作点用 `youden_on_split` 还是 `youden_on_val`  
确认前请把阈值相关列标为「草稿」。

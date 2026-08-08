# paper_v4 / baseline2：阈值后处理 · 出表 · 绘图

> 协议锚：湘雅内训 / 华西+辽宁外测；**Ours = `4[final]`** Ext ROC **0.564±0.034**。  
> 主表指标：**ROC（±分中心/Val）+ Sens/Spec/PPV/NPV** 全部算齐保留；往 Word 粘多少列由版面宽度决定。  
> 勿与旧 LOHO / Attn（`outputs/paper_v4/useless_旧baseline_LOHO_Attn/`）横比。

## 两种 Youden 策略（不是一回事）

共同点：算法都是 `youden_threshold_utils.py`（Youden 主目标 → 平局 F1 → |t−患病率|）。  
不同点：**t\* 在哪张表上标定**。

### `youden_on_split`（默认，对齐 v2/v3 表注）

报哪一划分的 Sens/Spec，就在**同一划分、同一 seed** 上找 t\*：

| 你要报的数 | t\* 标定在 | 再评估在 |
|------------|------------|----------|
| Ext pooled 的 Sens/… | **外部合并集**（华西+辽宁，该 seed） | 同一外部合并集 |
| 华西 Sens/… | **华西**该 seed | 华西 |
| 辽宁 Sens/… | **辽宁**该 seed | 辽宁 |
| Val Sens/… | **湘雅 Val**该 seed | 湘雅 Val |

这是「操作点乐观估计」：阈值看见了该划分标签。v2/v3 LOHO 也是「每个外部折自己 Youden」。  
**不是**「在湘雅找 t 再套外测」。

### `youden_on_val`（部署叙事）

| 步骤 | 数据 |
|------|------|
| 1. 标定 | **仅湘雅 Val**（该 seed）→ 得到一个 t\* |
| 2. 应用 | **同一个 t\*** 接到 Ext pooled / 华西 / 辽宁 |

这才是「内部验证定阈值 → 外部测试报 Sens/Spec」。

```
youden_on_split:   Ext 上找 t_ext  → 用 t_ext 评 Ext
                   华西上找 t_hx   → 用 t_hx 评华西
                   （湘雅 Val 只用于报 Val 行，不传给外测）

youden_on_val:     湘雅 Val 找 t_val → 用 t_val 评 Ext / 华西 / 辽宁
```

**ROC / PR 两种策略结果相同**（不依赖阈值）。只有 Sens/Spec/PPV/NPV/Youden/t\* 不同。

### `fixed0.5`

固定 0.5；可对齐训练日志，外部常不稳，不建议主文。

## 分中心读数

必须用 `external_huaxi_sample_predictions.csv` / `external_liaoning_*.csv`；  
不要从 pooled `external_*.csv` 按 `center` 切片。

## 一键命令

```bash
cd /ssd_data/tsy_study_venv/OptiGenesis_Lancet
export PATH="/home/amax/anaconda3/bin:$PATH"

# 默认 = youden_on_split（与 v2/v3 同思路）
python scripts/paper_v4_run/analyze_paper_v4_metrics.py

# 部署叙事版（Val→外测）
python scripts/paper_v4_run/analyze_paper_v4_metrics.py --threshold-policy youden_on_val

python scripts/paper_v4_run/plot_paper_v4_roc_pr.py --which comparison
python scripts/paper_v4_run/plot_paper_v4_roc_pr.py --which ablation
```

## 输出

| 路径 | 内容 |
|------|------|
| `outputs/paper_v4/tables/PAPER_TABLES_baseline2.md` | 主对比 / 消融 / 操作点（Sens/Spec/PPV/NPV）/ Wilcoxon |
| `…/PAPER_TABLES_baseline2_youden_on_val.md` | Val 标定版操作点 |
| `figures/paper_v4/fig_ext_roc_pr_combined_*.{pdf,png}` | Ext ROC/PR |

注册表：`paper_v4_registry.py`。

## ECE / Brier（回应审稿人校准意见）

```bash
python scripts/paper_v4_run/analyze_paper_v4_ece_brier.py
```

| 产物 | 说明 |
|------|------|
| `tables/PAPER_ECE_BRIER_baseline2.md` | 主对比 ECE/Brier 表（华西/辽宁/Ext） |
| `figures/paper_v4/fig_reliability_huaxi_liaoning.*` | Ours 华西+辽宁可靠性图 |
| `figures/paper_v4/fig_ece_brier_bars.*` | Ext pooled 柱状对比 |

只含外测中心；**湘雅不进该图主面板**。不重训。

# 交接：baseline2 最终方法 + 对比实验窗口（无 Attn / 无旧 LOHO）

> **给谁用**：新开的 Cursor 对话框（对比实验实现、跑数、后处理、出表）。  
> **不要读**：旧 `outputs/paper_v4/baseline/` 里 Attn / LOHO / 0.544·0.556 那条线（协议不同，禁止横比）。  
> **仓库**：`/ssd_data/tsy_study_venv/OptiGenesis_Lancet`，分支 `feature/paper-v4`。  
> **Python**：`export PATH="/home/amax/anaconda3/bin:$PATH"`；数据软链 `dataset` → `tsy_loho`。  
> **切分快照**：`data/snapshots/paper_v4_xyi_sitebag/`（亦见 `outputs/paper_v4/baseline2/docs_xyi/`）。  
> **更短的论文向说明**：`scripts/paper_v4_run/PAPER_EXPERIMENT_BRIEF_for_Gemini.md`（含 §0 选定方法）。  
> **框架图规格**：`scripts/paper_v4_run/MODEL_FRAMEWORK_SPEC_for_Gemini.md`。

---

## 1. 当前选定的 Ours（最终方法）

**目录**：`outputs/paper_v4/baseline2/4[final] xyi_sitebag_n2_uw_mil_aggmean_ema099`

| 开关 | 值 |
|------|-----|
| SITEBAG | 1，N=2，EVAL_AGG=**mean**，读全页 |
| FRAME_AGG | **uncertainty_weighted**，τ=0.5，edl_u |
| FRAME_AUX | MIL@0.3，edl |
| EMA | **0.99** |
| WMA / Aux多模态 / Attn / CLINICAL | **全关** |
| 骨干 | resnet50；epochs 常 30；seeds 42 123 2024 3407 114514 |

**结果（5-seed ROC mean±std）**

| 划分 | ROC |
|------|-----|
| Val | 0.790±0.090 |
| **Ext pooled** | **0.564±0.034** |
| 华西 | 0.547±0.015 |
| 辽宁 | 0.576±0.040 |

### 流程要点（接上下文用）

**训练**：每患者 1 袋 = 2 点×全帧；阴随机 2 点；阳优先病理阳点，不足补阴点；帧 EDL → UWA；主损失证据分类 + MIL 辅损；EMA 影子权重用于 val/存盘/外测。

**推理**：12 点 → 6 不重叠窗；每窗 UWA 得 \(p^{(k)}\)；患者分 = mean(\(p^{(1..6)}\)）。  
**UWA = 窗内帧加权；Mean = 窗间汇总**——两层，勿混。

---

## 2. 协议（所有对比必须对齐）

- 内部：湘雅 97，8:2，`split-seed=20260731` → train 78 / val 19  
- 外部：华西 87 + 辽宁 196，只终评  
- OCT-only  
- 主报：**外部 pooled ROC-AUC**；可附表 Hx / Ln / Val  
- 生成切分：`python data/prepare_xyi_sitebag_splits.py --write --split-seed 20260731 --val-ratio 0.2`

---

## 3. baseline2 实验地图（已完成）

| 角色 | 目录关键词 | Ext ROC | 主文？ |
|------|------------|---------|--------|
| **Final / Ours** | `4[final] …_ema099` | **0.564±0.034** | ✅ |
| Baseline 旧输入 | `3[baseline] …_n12_page1` | 0.510±0.028 | ✅ |
| 消融 max 窗 | `1[消融mean->max]`（**含 EMA**，相对 4 单因子） | 0.544±0.035 | ✅ |
| 消融 无 EMA | `2[消融无ema] …_aggmean_t0` | 0.547±0.029 | ✅ |
| 消融 关 UWA | `…_equal_mil_…_ema099` | 0.535±0.033 | ✅ |
| 消融 关 FrameAux | `…_uw_noaux_…_ema099` | 0.519±0.018 | ✅ |
| **对比 WMA 损失** | `5[对比] …_wmaC01` | 0.549±0.020 | ✅（流程≈无EMA底座，换主损失） |
| 仅 Aux / EMA+Aux / 三件套 / EMA+WMA | `6`–`9` | 0.547–0.560 | 附录即可 |
| 未完成 first/broadcast | `useless/` | — | ❌ 不用 |

脚本入口（参考）：

- 最终类跑法：`scripts/paper_v4_run/run_xyi_mean_stab_one.sh`（METHOD=ema 等）  
- ④ 消融：`run_xyi_final4_ablation_one.sh`  
- 组合 WMA：`run_xyi_mean_stab_combo2_gpu.sh`  
- 配置环境变量见 `configs/lancet_config.py`（SITEBAG_* / FRAME_AGG / ENABLE_FRAME_AUX / ENABLE_EMA / USE_WMA）

---

## 4. 新窗口任务：对比实验（定义）

**对比** = 他人方法或公认模块，在**同一切分、同一指标**上复现，与 Ours(4) 比。  
**不是对比**：再改我们的窗数 / 是否全页 / max vs mean（已是消融）。

已有对比：**WMA 作主损失**（5）。还需 **1–2 个**可复现方法（MIL 经典变体、其他 evidential/uncertainty、宫颈 OCT 相关等——先检索再实现）。

公平设定建议锁：同一 CSV 切分、5 seeds、尽量 ResNet50、OCT-only、报 Ext pooled ROC。

输出建议目录：`outputs/paper_v4/baseline2/cmp_<methodname>/seed_*`（勿进 git 大权重；代码与表进 scripts/docs）。

---

## 5. 后处理 / 出表时常用路径

- 预测：`…/seed_*/logs/{val,external,external_huaxi,external_liaoning}_sample_predictions.csv`  
- 控制台指标行：`【最佳权重 · 内部验证 val】` / `外部终评 external`  
- **阈值 + 表 + 图（本仓库已接好）**：见 `scripts/paper_v4_run/README_POSTPROCESS_TABLES.md`  
  ```bash
  python scripts/paper_v4_run/analyze_paper_v4_metrics.py
  python scripts/paper_v4_run/plot_paper_v4_roc_pr.py --which comparison
  ```
  产物：`outputs/paper_v4/tables/`、`figures/paper_v4/`  
- 注册表改方法行：`scripts/paper_v4_run/paper_v4_registry.py`（对比实验更新后改路径并重跑即可）

---

## 6. Git

- 最近相关 commit 含 site-bag 代码与 `PAPER_EXPERIMENT_BRIEF_for_Gemini.md` 等；**不要 commit outputs/**。  
- 分支已与 `origin/feature/paper-v4` 对齐过；新改动另提交。

---

## 7. 给新对话框的开场白（可直接粘贴）

```text
请读 scripts/paper_v4_run/HANDOFF_COMPARISON_BASELINE2.md 与 PAPER_EXPERIMENT_BRIEF_for_Gemini.md §0。
协议：湘雅内训外测华西+辽宁；Ours = baseline2/4[final]（sitebag n=2×全页 + UW + MIL@0.3 + 六窗 mean + EMA0.99），Ext 0.564±0.034。
不要 Attn/旧 LOHO。本窗口任务：设计并实现对比实验（他人方法），对齐同一切分与 5 seeds，结果写入 baseline2/cmp_*，不要把消融当对比。
```

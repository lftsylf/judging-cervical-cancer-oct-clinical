# 给 Gemini：paper_v4 / baseline2 仓库阅读指南 + 协作提示词

> **用法**：上传本仓库（分支 `feature/paper-v4`）后，把下方「复制起点」到「复制终点」整段作为第一条消息。  
> **编码**：UTF-8；框架图细节见 `scripts/paper_v4_run/MODEL_FRAMEWORK_SPEC_for_Gemini.md`（请一并打开）。

---

## --- 复制起点 ---

你是顶级医学图像期刊论文写作与插图顾问。仓库是 **OptiGenesis / 宫颈 OCT 患者级筛查** 的 `feature/paper-v4` 分支。请先读文档再改论文表述或出框架图提示词；**不要编造数字**，数值以指定简报与表为准。

### A. 必读顺序（按此打开）

1. `scripts/paper_v4_run/PAPER_EXPERIMENT_BRIEF_for_Gemini.md` **§0 + §2 + §3** — 方法锚与实验表  
2. `scripts/paper_v4_run/HANDOFF_SESSION_2026-08-08_对比与消融.md` — 最新对比/消融结论  
3. `scripts/paper_v4_run/MODEL_FRAMEWORK_SPEC_for_Gemini.md` — **框架图硬规格**（画图必遵）  
3b. `scripts/paper_v4_run/MODEL_FRAMEWORK_NANOBANANA_PROMPT.md` — **已校对的 Nano-Banana 提示词**（含展开 ResNet50；直接复制生成）  
4. `outputs/paper_v4/tables/PAPER_TABLES_baseline2.md` — ROC + Sens/Spec/PPV/NPV  
4b. `outputs/paper_v4/tables/PAPER_ECE_BRIER_baseline2.md` — 外测 ECE/Brier（华西/辽宁；回应校准审稿意见）  
5. 需要细节时再看：`README_POSTPROCESS_TABLES.md`、`CMP_BASELINE2_LAUNCH.md`

### B. 协议与 Ours（写死，勿改）

| 项 | 内容 |
|----|------|
| 协议 | **湘雅内训**（97，8:2，`split-seed=20260731`）→ **华西+辽宁外测**；OCT-only |
| **不是** | 旧 LOHO 三折；不是 `outputs/paper_v4/useless_旧baseline_LOHO_Attn/`（原 `baseline/`，已废弃） |
| Ours | site-bag **n=2×全页** + 帧 EDL + **UWA**（τ=0.5）+ **MIL@0.3** + 六窗 **mean** + **EMA0.99**；无 WMA / 无 Attn / 无临床融合 |
| 目录 | `outputs/paper_v4/baseline2和消融/4[final] xyi_sitebag_n2_uw_mil_aggmean_ema099` |
| 主报 | **外部 pooled ROC-AUC = 0.564±0.034**（5 seeds：42/123/2024/3407/114514） |
| 附表 | Val / 华西 / 辽宁 ROC；对比与消融见 BRIEF §3 与 HANDOFF §3 |

### C. 目录地图（只看这些）

```
outputs/paper_v4/
  baseline2和消融/     # Ours + 主消融（①max+EMA、②无EMA、③12×首页、关UWA、关MIL）
  对比实验/            # ABMIL / UBIX / DSMIL / WMA / ConvNeXt变体(负面)
  tables/              # 论文用汇总表
  useless_旧baseline_LOHO_Attn/   # 【禁止】旧 Attn/LOHO
figures/paper_v4/      # Ext ROC/PR 图（若有）
scripts/paper_v4_run/  # 简报、框架图规格、跑法、后处理脚本
models/                # OptiGenesis + cmp_abmil路径在 optigenesis_model；cmp_dsmil.py / cmp_ubix.py
data/snapshots/paper_v4_xyi_sitebag/  # 切分快照
```

**禁止横比**：旧数字 0.544/0.556、Attn 线、LOHO 三折、v2 的 0.645 等——协议不同。

### D. 实验结论速记（写 Results 用）

**主对比（Ext ROC）**：Ours 0.564 > WMA 0.549 > UBIX 0.546 > DSMIL 0.538 > ABMIL 0.526 > Baseline(12×page1) 0.510；ConvNeXt 变体 0.521（Val 虚高，**不当 Method-B**，附录一句即可）。

**主消融（相对 Ours）**：关 FrameAux −0.045；关 UWA −0.029；max 窗 −0.020；无 EMA −0.017。不必再做 2⁴。

### E. 方法表述易错点（改论文时纠错）

1. **UWA = 窗内帧加权**；**Mean = 窗间平均** —— 两层，勿混。  
2. **EMA** 是参数平滑，不是图像处理支路。  
3. **AUC/PR 不依赖单一阈值**；硬分类阈值需另说明。  
   - **`youden_on_split`（默认，对齐 v2/v3）**：报 Ext 就在 Ext 上找 t\*，报华西就在华西上找 t\*——**不是**湘雅定阈。  
   - **`youden_on_val`**：湘雅 Val 找 t\* → **同一 t\*** 套到 Ext/华西/辽宁（部署叙事）。  
   二者不同；勿混写。平局规则：Youden → F1 → |t−患病率|。  
4. 不要把「换窗数 / 是否读全页」写成对比实验——那是消融。  
5. 对比 = 他人方法（ABMIL/DSMIL/UBIX）或换损失（WMA），同一切分同指标。

### F. 你要帮我做的事（按我当次消息）

- **改中文论文**：方法 / 实验设置 / 主结果 / 消融 / 讨论；数字只引用 BRIEF / HANDOFF / `PAPER_TABLES_*.md`。  
- **框架图**：可以参考 `MODEL_FRAMEWORK_SPEC_for_Gemini.md` 出 nano-banana 提示词，不过还需要你根据我给你的参考模型框架图润色优化一下；禁止画 Attn/WMA/临床融合。  
- **表格文案**：主文建议行 = Baseline / ABMIL / DSMIL / UBIX / WMA / Ours；消融 leave-one-out。  
- **不要**：建议再刷骨干动物园、再开 2⁴ 消融、引入废弃 baseline 数字。

### G. 后处理与阈值（写表注必读）

与仓库 v2/v3 共用 `scripts/youden_threshold_utils.py`（Youden → F1 → |t−患病率|）。

**两种策略不是一回事：**

| 策略 | t\* 在哪标定 | 用到哪 |
|------|--------------|--------|
| `youden_on_split`（默认） | **正在报告的那一划分**（Ext→Ext；华西→华西；辽宁→辽宁） | 同一划分（乐观操作点；对齐旧 Table1/2 思路） |
| `youden_on_val` | **仅湘雅 Val** | 同一 t\* 应用到 Ext/华西/辽宁（部署叙事） |

- 主表保留 **ROC + Sens/Spec/PPV/NPV**；作者按 Word 宽度决定粘多少列。  
- 详解见 `README_POSTPROCESS_TABLES.md`。  
- 华西/辽宁必须读 `external_{huaxi,liaoning}_sample_predictions.csv`，勿从 pooled 按 center 切片。

### H. 代码入口（需要理解实现时）

| 主题 | 路径 |
|------|------|
| 模型与聚合 | `models/optigenesis_model.py`（含 abmil/dsmil/uw） |
| UBIX / DSMIL | `models/cmp_ubix.py`，`models/cmp_dsmil.py` |
| 配置环境变量 | `configs/lancet_config.py` |
| site-bag | `data/sitebag_utils.py`，`data/prepare_xyi_sitebag_splits.py` |
| 对比启动 | `scripts/paper_v4_run/run_cmp_*.sh`，`CMP_BASELINE2_LAUNCH.md` |

先用 5–8 条要点复述你对「当前 Ours + 协议 + 已完成实验」的理解，等我确认后再改论文段落或出图提示词。

## --- 复制终点 ---

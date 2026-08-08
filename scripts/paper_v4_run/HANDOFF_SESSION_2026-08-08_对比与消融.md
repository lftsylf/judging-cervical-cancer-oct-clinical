# 交接：本会话完成内容（对比实验 + 对齐①消融）· 给新对话框

> **日期**：2026-08-07 ~ 08-08  
> **分支**：`feature/paper-v4` @ `bae6ec9`（已 push `origin`）  
> **协议**：仅 `baseline2` 湘雅 site-bag 线；**禁止**与旧 LOHO / Attn / 0.544·0.556 横比。  
> **Ours 锚**：`4[final]` Ext pooled ROC **0.564±0.034**。

---

## 0. 直接粘贴给新 Agent 的开场白

```text
请读 scripts/paper_v4_run/HANDOFF_SESSION_2026-08-08_对比与消融.md
与 PAPER_EXPERIMENT_BRIEF_for_Gemini.md §0。
协议：湘雅内训、华西+辽宁外测；Ours=4[final]（sitebag n=2×全页+UWA+MIL@0.3+六窗 mean+EMA0.99），Ext 0.564±0.034。
输出目录已改名：outputs/paper_v4/baseline2和消融/ ；对比在 outputs/paper_v4/对比实验/。
实验与对比代码已 push（bae6ec9）。本窗口任务：论文表述 / 后处理出表出图 / 勿再无必要刷骨干或 2^4 消融。
```

---

## 1. 用户本会话的要求与结论

| 要求 | 结论 / 执行 |
|------|-------------|
| 查看 baseline2 进度 | Ours=4[final]；消融齐；对比 initially 仅 WMA |
| Gemini 候选：ABMIL/UBIX/CLAM/TransMIL/DSMIL | **定 ABMIL+UBIX+DSMIL**；不做 CLAM/TransMIL（小袋） |
| 老师：换骨干试 Method-B | 选 **ConvNeXt-Tiny**；跑完 Ext **0.521**，Val 虚高 → **不当 Method-B**；**勿再主攻换骨干** |
| 四卡 detach 开跑对比 | `run_experiment.sh --detach`；输出到 `对比实验/` |
| 消融要不要 2⁴ | **不要**；leave-one-out 即可 |
| ① max 与 ④ 差 EMA | **重跑**「④ + 仅 mean→max」替换旧①；旧①进 useless |
| 看①结果 + 中文 commit + push | 已完成（见 §4、§5） |

---

## 2. Ours 方法（勿改锚）

- site-bag **n=2×全页** + 帧 EDL + **UWA**（τ=0.5, edl_u）+ **MIL@0.3** + 推理六窗 **mean** + **EMA0.99**
- **无** WMA / Attn / 临床融合
- 目录：`outputs/paper_v4/baseline2和消融/4[final] xyi_sitebag_n2_uw_mil_aggmean_ema099`

---

## 3. 本会话跑完的实验与关键数字

### 3.1 对比（`outputs/paper_v4/对比实验/`）

| 方法 | Ext ROC | 相对 Ours |
|------|---------|-----------|
| ABMIL | 0.526±0.014 | −0.038 |
| DSMIL | 0.538±0.027 | −0.026 |
| UBIX | 0.546±0.021 | −0.018 |
| Ours-ConvNeXt | 0.521±0.044 | −0.043（负面；附录） |
| （已有）WMA | 0.549±0.020 | −0.015 |

主文对比建议：**Baseline / ABMIL / DSMIL / UBIX / WMA / Ours(R50)**。

### 3.2 消融① 对齐版（替换旧无 EMA）

- 目录：`baseline2和消融/1[消融mean->max] xyi_sitebag_n2_uw_mil_aggmax_ema099`
- 设定：同 ④，仅 `SITEBAG_EVAL_AGG=max`
- **Val 0.715±0.042 | Ext 0.544±0.035 | Hx 0.525±0.047 | Ln 0.554±0.030**
- vs ④：Ext −0.020
- 旧①（无 EMA）→ `useless/【旧①无EMA…】`

### 3.3 其它主消融（本会话前已完成，目录在 `baseline2和消融/`）

| ID | Ext | 备注 |
|----|-----|------|
| ③ baseline 12×page1 | 0.510 | |
| ② 无 EMA | 0.547 | |
| equal 关 UWA | 0.535 | 现目录名常带 `10[消融关uw]` |
| noaux 关 MIL | 0.519 | 常带 `11[消融 无aux]` |

**不必**再做 2⁴ 全因子。

---

## 4. 本会话代码 / 工程改动

| 内容 | 路径 |
|------|------|
| ABMIL / DSMIL / UBIX 接入 | `models/optigenesis_model.py`，`models/cmp_dsmil.py`，`models/cmp_ubix.py` |
| 对比启动 | `scripts/paper_v4_run/run_cmp_{abmil,ubix,dsmil,ours_convnext,quad}_*.sh`，`cmp_env_common.inc.sh`，`CMP_BASELINE2_LAUNCH.md` |
| ① 对齐消融启动 | `run_xyi_ablation_aggmax_ema_{one,quad_gpu}.sh` |
| 简报 | `PAPER_EXPERIMENT_BRIEF_for_Gemini.md`（① 数字已更新） |
| 断线跑法 | `./run_experiment.sh --detach ./tsy_loho <script.sh>` → `logs/detached_latest.log` |

说明：`对比实验/logs/gpu*.log` 与 `seed_*/logs/train_console.log` 内容冗余；`.pid` 仅为进程号。

---

## 5. Git

| 项 | 值 |
|----|-----|
| 分支 | `feature/paper-v4` |
| 本会话 commit | **`bae6ec9`** `中午：接入 ABMIL/UBIX/DSMIL 对比与对齐版 max 窗消融。` |
| 远程 | 已 **`git push origin feature/paper-v4`** |
| 提交范围 | 仅代码/脚本/简报（15 files）；**未提交** `outputs/` 权重与大日志 |
| 工作区残留 | 大量无关删除（figures/case_study 等）、`Primary_care` submodule dirty、若干未跟踪后处理脚本——**不要误 commit** |

未跟踪但可能有用的后处理（若要用可另 commit）：

- `scripts/paper_v4_run/analyze_paper_v4_metrics.py`
- `scripts/paper_v4_run/plot_paper_v4_roc_pr.py`
- `scripts/paper_v4_run/paper_v4_registry.py`
- `scripts/paper_v4_run/README_POSTPROCESS_TABLES.md`
- 已有表：`outputs/paper_v4/tables/PAPER_TABLES_baseline2.md`

---

## 6. 目录地图（当前命名）

```
outputs/paper_v4/
  baseline2和消融/     # 原 baseline2（用户已改名）
    1[消融mean->max] …_aggmax_ema099/   # 新①
    2[消融无ema]/ 3[baseline]/ 4[final]/
    10[消融关uw]/ 11[消融 无aux]/
    useless/           # 旧①、未完成、6–9 等
  对比实验/
    10[对比] abmil_… / 11 ubix_… / 12 dsmil_… / 13 convnext_…
  tables/              # 自动论文表
figures/paper_v4/      # ROC/PR 图（若已生成）
```

注意：旧脚本里若仍写 `outputs/paper_v4/baseline2/`，需改成 **`baseline2和消融`**。

---

## 7. 建议新对话框优先做的事

1. **论文方法/实验/消融表述**（按 BRIEF + 上表；ConvNeXt 作负面敏感性一句）  
2. **后处理**：按 `README_POSTPROCESS_TABLES.md` 刷新表/图（确认 registry 指向新①与 `对比实验/`）  
3. 若 registry 里 Ln/路径有漂移，以各 `train_console.log` 的「最佳权重 · 外部终评」为准重算  
4. **不要**：再刷骨干动物园、不要 2⁴、不要引入 Attn/旧 LOHO 数字  

---

## 8. 公平对比锁死项（复现时）

- CSV：`data/snapshots/paper_v4_xyi_sitebag/`（或 dataset 软链下 train_xyi / val_xyi / external_*）  
- Seeds：42, 123, 2024, 3407, 114514  
- 骨干对比方法 = ResNet50；Ours 变体曾试 ConvNeXt（已否决）  
- UBIX：推理期 MC Dropout，**禁止**接 EDL-u  
- 主报 Ext pooled ROC；Sens/Spec 等依赖阈值策略（见后处理 README）

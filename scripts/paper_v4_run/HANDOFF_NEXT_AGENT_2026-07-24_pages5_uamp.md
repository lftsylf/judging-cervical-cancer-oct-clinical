# 交接提示词（给下一任 Agent）· 2026-07-24

> **用途**：用户把下文整段粘贴给新对话即可。  
> **本文件**：`scripts/paper_v4_run/HANDOFF_NEXT_AGENT_2026-07-24_pages5_uamp.md`  
> **更早总交接**（协议 / B1 / edl@0.3 / 锐化诊断）：`scripts/paper_v4_run/HANDOFF_NEXT_AGENT.md`  
> **上一轮长对话 transcript**：`agent-transcripts/273d4c39-3d47-49e7-b97d-6317037ec8cb`  
> **中间对话（锐化 / 换权重 / 口述 EDL）**：`.specstory/history/2026-07-22_12-44-36Z-edl-model-handoff-discussion.md`

工作区：`/ssd_data/tsy_study_venv/OptiGenesis_Lancet`  
分支：`feature/paper-v4`  
Python：`export PATH="/home/amax/anaconda3/bin:$PATH"`  
硬件：本机 **4× RTX 2080（各 8GB）**（用户口头说「四个服务器」= 四张卡，不是四台机器）  
中文 commit；**不要**提交 `outputs/` 大文件。

---

## 你要接手的现状（一句话）

在旧交接「edl@0.3 最好、外部≈0.577、但 \(n_{\mathrm{eff}}\approx12\)」之后，用户要求：**开 TIFF 时序页 + 放大不确定度差（`edl_u_amp`）重训 T2**。因 N=120 / batch=1 太慢已中断；后改成 **每 TIFF 最多 5 页（N≈60）+ batch=8 + 三折并行**，该轮 **已跑完**，外部均值 **≈0.512（劣于旧 0.577）**，UW 仍几乎没拉开（\(n_{\mathrm{eff}}\approx54/60\)）。

---

## 用户后来改了哪些要求（相对旧 HANDOFF）

请按「用户新指令」理解，不要回退到「必须读满 10 页 / N=120 / 单卡串行 / batch=1」：

1. **暂停**原先 N≈120、`BATCH=1`、单卡串行的 expand+uamp 跑法（太慢、显存可怜）。
2. **并行**：同一轮 T2 的 **三折（华西 / 辽宁 / 湘雅）各占一张 GPU 同时跑**；本机 4 卡，第 4 张可空或加 seed/对照。  
   - **不是**「4 卡 DDP 一起训同一折」（代码未接多卡训练；单卡 batch=8 已近满显存）。
3. **batch 加大**：N≈60 时默认 **`BATCH_SIZE=8`**（OOM 再降到 4）。
4. **先不要按 120 帧算**：华西 / 湘雅 TIFF 虽有约 10 页，**只取前面 5 页** → 与辽宁对齐，**N≈60**。  
   - `OPTIGENESIS_EXPAND_TIFF_PAGES=1`  
   - `OPTIGENESIS_MAX_PAGES_PER_TIFF=5`
5. **保留不确定性放大加权**（用户明确问过「有没有做」——有）：  
   - `OPTIGENESIS_FRAME_WEIGHT_SIGNAL=edl_u_amp`  
   - `score=(u_base−u)·scale`，再 `softmax(/τ)`  
   - `u_base=0.5`，`scale=10`，`τ=0.5`  
   - 帧弱监督仍为 **edl@0.3**（`ENABLE_FRAME_AUX=1`，`FRAME_AUX_WEIGHT=0.3`，`FRAME_AUX_TYPE=edl`）
6. 跑完后用户要过：**整轮墙钟时间 + 提数**；并解释 **华西为何慢、湘雅外部为何崩**。

---

## 本轮实验配置（已完成）

| 项 | 值 |
|----|-----|
| 脚本 | `scripts/paper_v4_run/run_expand_pages_edl03_uamp_t2_oct_only.sh` |
| 输出根 | `outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3_expand_uamp10_pages5/` |
| 汇总 CSV | 同上目录 `SUMMARY_seed42.csv` |
| seed | 42（仅 1 个） |
| 并行 | GPU0=华西，GPU1=辽宁，GPU2=湘雅 |
| OCT-only | `USE_CLINICAL=0`；无 WMA/EMA/多模态 Aux |

说明文档：`scripts/paper_v4_run/README_EXPAND_TIFF_PAGES.md`（已改为建议 pages≤5 / batch=4~8）。

旧的未完成目录（N 未截断、路径易混）：  
`outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3_expand_uamp10/` ← **不要当本轮终局结果**。

---

## 结果摘要（best 权重；选模=内部 val ROC）

**墙钟（三折并行）**：约 **10 h 26 min**（2026-07-23 20:18 → 07-24 06:45，以华西结束为准）

| 折 | 训练人数 | steps/epoch | 实际 epoch | 约耗时 | Val ROC | Ext ROC | Ext \(n_{\mathrm{eff}}\) |
|----|----------|-------------|------------|--------|---------|---------|---------------------------|
| 辽宁 | 148 | 19 | 11（早停） | ~3.4 h | 0.783 | 0.570 | ~52 |
| 湘雅 | 227 | 29 | 11（早停） | ~4.9 h | 0.685 | **0.390** | ~51 |
| 华西 | 235 | 30 | **30（跑满）** | ~10.4 h | 0.723 | 0.576 | ~58 |
| **均值** | | | | **~10.4 h** | **0.730** | **0.512** | **~54** |

对照（旧 **N=12**、首帧 TIFF、edl@0.3、无 expand）：外部均值 ROC **≈0.577**  
路径：`outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3/`

**结论**：pages5 + `edl_u_amp` **未超过**旧 edl@0.3；\(n_{\mathrm{eff}}\) 仍接近满帧数 → **UW 实质仍≈均匀**。

---

## 已解释过的两个现象（勿再当成未解 bug）

### 华西为什么慢？（不是「帧数更多」）

三折都是 N≈60。华西慢是因为：

1. 训练样本略多 → 每 epoch ~30 step（辽宁只有 19）；  
2. **val 持续缓慢上升，早停未触发，跑满 30 epoch**；辽宁/湘雅在 epoch 1 附近见顶后 **11 epoch 就停**。  
粗算 step 总量：华西 ~900 vs 辽宁 ~209 → 墙钟差一个量级合理。

### 湘雅外部为什么只有 ~0.39？

不是折跑错，是 **训崩 + 选到了崩之前的权重**：

1. **best = epoch 1**；之后 val ROC `0.685 → … → 0.56` 一路掉。  
2. 概率 **塌成近常数**（`p_std≈0.01`），val/external **几乎 100% 预测阳性**。  
3. 湘雅 **train 阴性多、external 阳性多**（类别先验反转）→ 全阳时 PR 仍可看、**ROC&lt;0.5 说明排序略反相关**。  
4. 旧 N=12 同折外部曾 **≈0.614**；本配方把排序弄坏了。

---

## 代码 / 能力备忘（本阶段已有）

- TIFF 多页：`data/dataset_lancet.py`（`EXPAND_TIFF_PAGES` / `MAX_PAGES_PER_TIFF`）+ `frame_mask` pad  
- 显存：`FRAME_ENCODE_CHUNK` + checkpoint（`models/optigenesis_model.py`）  
- 加权信号：`edl_u` / `edl_u_amp` / `maxprob` / `negent` 等  
- val 变长帧 stack 崩溃已修：`training/trainer.py` 的 `_pad_stack_2d`  
- 并行三折：expand 脚本里 `OPTIGENESIS_PARALLEL=1` + `CUDA_VISIBLE_DEVICES` 分卡  

**系统盘曾 100% 满**（`/` ENOSPC → Pyright 挂）；`/ssd_data` 仍充足。大缓存勿再堆 `/home`。

---

## 明确不要再做的

- 盲扫 FrameAux λ / 推理 τ / 手写换权重公式（锐化与后处理诊断已证伪「尖 τ 救 UW」）  
- 把「4 卡 DDP 训一折」当成默认提速手段（未实现且收益可疑）  
- 未征得用户同意就默认改回 **N=120 全页** 或改主方法为可学习注意力（D）/ 改监督（C）  
- 提交 outputs 权重进 git  

---

## 建议下一任优先征得用户同意后再做

按优先级（用户未拍板则先问）：

1. **对照**：同 pages5（N≈60），**关掉 `edl_u_amp`**，回到 `edl_u` + edl@0.3，看是否湘雅/均值能回到接近 0.577。  
2. 或 **pages5 + 无帧损 / 不同 seed**，确认不是 seed=42 单点翻车。  
3. 若仍 \(n_{\mathrm{eff}}\approx N\)：回到总交接里的 **D（可学习注意力）或 C（改监督）**，不要再堆手写 score。  
4. 湘雅专项：检查 epoch1 的 `frame_u_*` / `frame_w_*` 分布；外部先验反转是否要在叙事里单独写清。

---

## 给新 Agent 的可复制启动段

```text
请先阅读：
1) scripts/paper_v4_run/HANDOFF_NEXT_AGENT_2026-07-24_pages5_uamp.md（本轮用户改要求 + pages5+uamp 结果）
2) scripts/paper_v4_run/HANDOFF_NEXT_AGENT.md（总协议与更早诊断）

工作区 /ssd_data/tsy_study_venv/OptiGenesis_Lancet，分支 feature/paper-v4。
Python: export PATH="/home/amax/anaconda3/bin:$PATH"

本轮 pages5+edl_u_amp T2 已跑完，外部均值 ROC≈0.512 < 旧 N=12 edl@0.3 的 ≈0.577；
n_eff≈54/60，UW 仍未真正生效。华西慢因跑满30 epoch；湘雅因 epoch1 后概率塌缩且排序变差。

请先用 SUMMARY_seed42.csv 与三折 external_sample_predictions.csv 复核数字，
再按 HANDOFF 里「建议下一任」征得我同意后动手（默认先问要不要跑 pages5 + 关闭 uamp 的对照）。
```

---

## 一句话给用户复核

> 我让你改成：停掉 N=120/batch=1；三折三卡并行；batch 加大；华西/湘雅只取前 5 页（N≈60）；保留 `edl_u_amp`。  
> 该轮已跑完约 10.5h，外部均值 0.512，不如旧 0.577；放大 u 仍没拉开权重。

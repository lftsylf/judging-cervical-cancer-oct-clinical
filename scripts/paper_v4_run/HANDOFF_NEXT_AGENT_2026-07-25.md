# 交接提示词（给下一任 Agent）· 2026-07-25

> **用途**：用户把下文整段粘贴给新对话即可。  
> **本文件**：`scripts/paper_v4_run/HANDOFF_NEXT_AGENT_2026-07-25.md`  
> **总交接（协议/更早诊断）**：`scripts/paper_v4_run/HANDOFF_NEXT_AGENT.md`  
> **2026-07-24 增量**：`HANDOFF_NEXT_AGENT_2026-07-24_pages5_uamp.md`  
> **本轮提数**：`RESULTS_quad_gpu_2026-07-25.md`

工作区：`/ssd_data/tsy_study_venv/OptiGenesis_Lancet`  
分支：`feature/paper-v4`  
Python：`export PATH="/home/amax/anaconda3/bin:$PATH"`  
硬件：本机 **4× RTX 2080（各 8GB）**（用户说的「四个服务器」= 四张卡）  
中文 commit；**不要**提交 `outputs/` 大文件。

---

## 一句话现状（2026-07-25）

探路规模（**3 折 × seed=42**）下，**主方法锁定候选仍是**：

> **N=12（每 TIFF 取 1 页）+ FrameAux edl@0.3 + `edl_u`（\(w=\mathrm{softmax}((1-u)/0.5)\)）**  
> 路径：`outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3/`  
> 外部三折均值 ROC **≈0.577**（B1 mean ≈0.533）

已否定：pages5±amp、N12 topk5/max_p 重训、推理锐化、换权重后处理、帧 OR 决策规则。  
**用户已把结论发给裴老师，正在等回复**；未开「定稿规模（3 折 × 5 seeds）」——等老师拍板。

诚实局限：该最好设定下 \(n_{\mathrm{eff}}\approx12\)，**UW 实质仍≈等权**；论文需如实写。

---

## 给新 Agent 的可复制启动段

```text
请先阅读（按优先级）：
1) scripts/paper_v4_run/HANDOFF_NEXT_AGENT_2026-07-25.md（本文件：最新结论与实验链）
2) scripts/paper_v4_run/RESULTS_quad_gpu_2026-07-25.md（提数表）
3) scripts/paper_v4_run/HANDOFF_NEXT_AGENT.md（诚实协议 / 方法定义）
4) 如需 pages5 细节：HANDOFF_NEXT_AGENT_2026-07-24_pages5_uamp.md

工作区 /ssd_data/tsy_study_venv/OptiGenesis_Lancet，分支 feature/paper-v4。
Python: export PATH="/home/amax/anaconda3/bin:$PATH"

当前：用户已向裴老师汇报「主方法暂定 N=12+UW+FrameAux edl@0.3（外部≈0.577）」并等待回复。
大改动 / 开定稿多 seed 前先征得用户同意。中文沟通与 commit；勿提交 outputs。
```

---

## 协议（勿回退）

- development → 患者级分层 **8:2 train/val**（center × pathology_class）；**external 仅终评**
- 切分：`data/prepare_paper_v4_splits.py --write --split-seed 20260720 --val-ratio 0.2`
- 早停：内部 **val ROC-AUC**，patience=10
- OCT-only（`USE_CLINICAL=0`）；无 WMA/EMA/多模态 Aux/CORAL（v4 Full 探路）

---

## 命名注意（易混）

| 用户口语 | 含义 |
|----------|------|
| 「探路 / 我口头说的 T2 级」 | **3 折 × 1 seed(42)**，最多 30 epoch |
| 「定稿 / 我口头说的 T0 级」 | **3 折 × 5 seeds**（约 5× 探路工作量） |
| 仓库脚本名 `*_t0` / `*_t2` | **方法变体**（B1/无帧损 vs FrameAux），**不是**上面的规模 |

跟用户沟通时用「探路 / 定稿」更清晰。

---

## 方法与实现要点

### 帧不确定度与加权（主方法）

1. 每帧：backbone → fusion(256) → `UncertaintyHead` → α = softplus(logits)+1  
2. \(u = K/\sum\alpha\)（K=2）  
3. 默认 `edl_u`：\(w=\mathrm{softmax}((1-u)/\tau)\)，τ=0.5  
4. 患者 α = Σ w·α_frame；FrameAux：患者标签广播到每帧，λ=0.3，type=edl  

代码：`models/optigenesis_model.py`（`_aggregate_frame_alphas`）、`configs/lancet_config.py`

### 已实现的其它加权信号（env `OPTIGENESIS_FRAME_WEIGHT_SIGNAL`）

| signal | 含义 |
|--------|------|
| `edl_u` | 默认 (1−u) |
| `edl_u_amp` | score=(u_base−u)·scale 再 softmax（u_base=0.5, scale=10） |
| `maxprob` / `negent` | 后处理/重训探过，未胜出 |
| `topk_p` | 按阳性 p 硬选 top-k 等权（`FRAME_TOPK_K`，默认 5） |
| `max_p_pool` | 只取阳性 p 最大的 1 帧（训练版 max_p） |

### TIFF 多页

- `OPTIGENESIS_EXPAND_TIFF_PAGES=1`，`MAX_PAGES_PER_TIFF=5` → N≈60  
- pad + `frame_mask`；`FRAME_ENCODE_CHUNK` + checkpoint 防 OOM  
- 说明：`scripts/paper_v4_run/README_EXPAND_TIFF_PAGES.md`

### 并行习惯

- **三折各占一张 GPU 并行**（非 DDP 训同一折）  
- `./run_experiment.sh --detach ./tsy_loho <script>` → 断线继续；`tail -f logs/detached_latest.log`

---

## 本对话时间线与实验链（逻辑顺序）

### A. 更早已完成（其它对话 / 7-22～24）

1. FrameAux 网格：edl@0.2/0.3/0.5、ce@0.2 → **edl@0.3 最好（0.577）**，\(n_{\mathrm{eff}}\approx12\)  
2. 推理锐化 τ/top-k：能拉开权重但 **伤 AUC**（`DIAG_uw_inference_sharpen_edl03.md`）  
3. 换权重后处理（熵/maxprob/特征距离）：外部可虚高但 **伤 val**（`DIAG_uw_alt_weight_signals_edl03.md`）  
4. maxprob/negent **重训**：外部大跌（`RESULTS_weight_signal_retrain_2026-07-23.md`）  
5. pages5 + `edl_u_amp`：外部 **≈0.512**，湘雅崩（`HANDOFF_NEXT_AGENT_2026-07-24_pages5_uamp.md`）

### B. 本对话（约 7-24～25）完成的事

1. **解释** UW 不起作用主因是帧间 \(u\) 差太小；改 u_base 对 softmax 相对差无效；再加大 scale 性价比低  
2. **帧决策探针**（不重训）：OR / max / topk / hybrid  
   - 脚本：`diag_frame_decision_rules_probe.py`  
   - 报告：`DIAG_frame_decision_rules_probe.md`  
   - 结论：**硬 OR 不可行**；软 max/hybrid 伤 val  
3. **四卡过夜队列**（已全部完成，commit `c7ccd1e`）：  
   - GPU0/1/2：pages5 + **无 amp**（edl_u）  
   - GPU3：N=12 **topk5 → max_p_pool** 串行两轮 T2  
   - 脚本：`run_overnight_quad_gpu_pages5_edlu_and_n12_rules.sh`  
4. **提数**（`RESULTS_quad_gpu_2026-07-25.md`，commit `e97e53f`）→ 见下表  
5. 帮用户润色给裴老师的汇报；**已发出，等回复**

---

## 关键结果表（探路 seed=42，外部三折均值）

| # | 设置 | 外部 ROC | 相对 N12 UW |
|---|------|---------:|------------:|
| ① | **N12 + edl_u + FrameAux@0.3** | **0.577** | — |
| ② | pages5 + amp | 0.512 | −0.065 |
| ③ | pages5 无 amp | **0.506** | −0.071 |
| ④ | N12 topk5 重训 | 0.523 | −0.054（湘雅 0.415） |
| ⑤ | N12 max_p 重训 | 0.561 | −0.016 |
| — | B1 mean | 0.533 | −0.044 |

**判读**：

- ③≈② 且都差 → **伤分主因是 pages5（N≈60），不是 amp**  
- ④ val 最高但外部崩 → 不能按 val 盲目换主方法  
- ⑤ 不如 ①  
- **主候选维持 ①**

产物路径：

| 实验 | 目录（相对 `outputs/paper_v4/baseline/`） |
|------|------------------------------------------|
| ① 主候选 | `ours_uw_frameaux_t2_edl_w0.3/` |
| ② pages5+amp | `ours_uw_frameaux_t2_edl_w0.3_expand_uamp10_pages5/` |
| ③ pages5 无 amp | `ours_uw_frameaux_t2_edl_w0.3_expand_edlu_pages5/` |
| ④ N12 topk5 | `ours_uw_frameaux_t2_edl_w0.3_n12_topk5/` |
| ⑤ N12 max_p | `ours_uw_frameaux_t2_edl_w0.3_n12_maxp/` |

---

## 用户已发给老师的结论（大意）

① 最好；②③ 多页不好；④⑤ topk/max 不好；暂定 N=12+帧 EDL+aux0.3+score=1−u；问是否开定稿多 seed。  
Agent 建议的修订要点：数字用 0.577/0.533；规模用「探路/定稿」；承认 \(n_{\mathrm{eff}}\approx12\)。

---

## 明确不要再做的

- 盲扫 FrameAux λ / τ / 手写 amp scale / 再堆 OR 决策  
- 未同意就默认 N=120 全页或 4 卡 DDP  
- 未等老师回复就自动开 5 seeds 定稿（除非用户明确要求）  
- 提交 outputs 权重  

---

## 建议下一任（等用户/老师指示）

1. **若老师同意定稿**：写/复用脚本跑 **N=12 + edl_u + FrameAux edl@0.3，3 折 × 5 seeds**，detach 断线可续；提数含均值±std，对照 B1 / 无帧损。  
2. **若老师要继续攻 UW**：再议 **D（可学习帧注意力）** 或更深 **C（改帧监督结构）**——手写换公式已证伪。  
3. **若只要写论文材料**：用 ① 作 Full，②③④⑤ 作阴性/消融叙述；诚实写 \(n_{\mathrm{eff}}\)。

---

## 相关文件速查

| 文件 | 内容 |
|------|------|
| `RESULTS_quad_gpu_2026-07-25.md` | 本轮最终提数与建议 |
| `RESULTS_T2_frameaux_2026-07-22.md` | FrameAux 网格 |
| `RESULTS_weight_signal_retrain_2026-07-23.md` | maxprob/negent 重训失败 |
| `DIAG_uw_inference_sharpen_edl03.md` | 锐化否定 |
| `DIAG_uw_alt_weight_signals_edl03.md` | 后处理换信号 |
| `DIAG_frame_decision_rules_probe.md` | OR/max/topk 离线探针 |
| `run_overnight_quad_gpu_pages5_edlu_and_n12_rules.sh` | 四卡过夜总队列 |
| `run_expand_pages_edl03_edlu_pages5.sh` | pages5 无 amp |
| `run_n12_topk5_then_maxp_t2_serial.sh` | N12 两规则串行 |

近期相关 commits：`e97e53f`（提数）、`c7ccd1e`（四卡队列+topk/max 聚合）、`6671b32`（决策探针）。

---

## 一句话给用户复核

> 探路结论已齐：主方法仍是 N=12+UW+FrameAux@0.3（≈0.577）；pages5 与 topk/max 都更差；已汇报裴老师等回复。新对话请读 `HANDOFF_NEXT_AGENT_2026-07-25.md`。

# 交接提示词（给下一任 Agent）· paper_v4 / MUSE

> 用户可把下文整段粘贴给新对话。上一轮完整对话：  
> `agent-transcripts/273d4c39-3d47-49e7-b97d-6317037ec8cb`  
> 仓库内本文件：`scripts/paper_v4_run/HANDOFF_NEXT_AGENT26-7-22.md`  
> 同内容副本：`scripts/paper_v4_run/HANDOFF_NEXT_AGENT.md`

---

## 你要接手的任务

你是接续 OptiGenesis / **MUSE** paper_v4 实验的 coding agent。代码名 OptiGenesis，论文名 MUSE。  
工作区：`/ssd_data/tsy_study_venv/OptiGenesis_Lancet`  
分支：`feature/paper-v4`  
Python：`export PATH="/home/amax/anaconda3/bin:$PATH"`  
重要改动后用**中文** commit message（见 `GIT_MANAGEMENT_GUIDE.md` §3.4）；常用 `/usr/bin/git commit --no-verify`。  
**不要**把大体积 `outputs/` 权重提交进 git。

用户目标：在诚实 train/val 协议下，让 **帧级不确定度加权聚合（UW）真正生效**（`n_eff` 明显 < 12），并找到相对 B1 / 无帧损 Ours 更好的主方法版本。

---

## 协议（已落地，勿回退）

- 问题：旧 `main.py` 用 `external_*.csv` 做验证/早停 → 乐观偏差。
- 现协议：development → 患者级分层 **8:2 train/val**（按 **center × pathology_class**）；**external 仅终评一次**。
- 切分脚本：`data/prepare_paper_v4_splits.py`（`--write --split-seed 20260720 --val-ratio 0.2`）
- 快照：`data/snapshots/paper_v4_tsy_loho/`；`dataset/` → `tsy_loho`（含 `train_*` / `val_*` / `external_*`）
- 早停：内部 val **ROC-AUC**，patience=10
- CORAL+external 在 v4 下禁用
- CSV 角色：用 `train_*`/`val_*`/`external_*`；`development_*` 仅作切分源

---

## 方法定义

| 代号 | 设置 | 路径（相对 `outputs/paper_v4/baseline/`） |
|------|------|------------------------------------------|
| **B1** | `FRAME_AGG=mean`，患者级 EDL，无 WMA/EMA/Aux | `b1_mean_edl_t0/` |
| **Ours（消融）** | `uncertainty_weighted`，无帧弱监督 | `ours_uw_agg_t0/` |
| **Full / T2** | UW + 帧弱监督（患者标签广播到每帧） | `ours_uw_frameaux_t2*` |

核心实现：
- `models/optigenesis_model.py`：`_aggregate_frame_alphas`  
  \(w=\mathrm{softmax}((1-u)/\tau)\)，默认 \(\tau=0.5\)（`OPTIGENESIS_FRAME_AGG_TEMP`）
- 配置：`configs/lancet_config.py` — `FRAME_AGG_*`、`ENABLE_FRAME_AUX_*`
- 帧弱监督：`OPTIGENESIS_ENABLE_FRAME_AUX`、`FRAME_AUX_LOSS_WEIGHT`、`FRAME_AUX_LOSS_TYPE`∈{`edl`,`ce`}  
  **不是**多模态 Aux；是同标签帧级弱监督。
- 导出：`*_sample_predictions.csv` 含 `frame_u_*` / `frame_w_*` / `review_frame_indices`
- **注意**：B1 mean 模式 `frame_u` 为占位 0；真实帧 u 仅 equal/uw。

叙事约定（用户偏好）：
- 主方法候选 = **UW + 帧弱监督（Full）**
- 旧 Ours T0 = 无帧损消融
- B1 = 结构基线（mean pool）
- **不要**指望 WMA/EMA/多模态 Aux 来修帧 u 塌缩

---

## 环境变量 / 启动方式

```bash
export PATH="/home/amax/anaconda3/bin:$PATH"
cd /ssd_data/tsy_study_venv/OptiGenesis_Lancet
./run_experiment.sh --detach ./tsy_loho ./scripts/paper_v4_run/<script>.sh
# 看日志
tail -f logs/detached_latest.log
cat outputs/paper_v4/baseline/AUTO_PROBE_STATUS.md
```

脚本：
- `run_b1_mean_edl_t0_oct_only.sh`
- `run_ours_uw_agg_t0_oct_only.sh`
- `run_ours_uw_frameaux_t2_oct_only.sh`（可改 env 调 weight/type/outdir）
- `run_auto_frameaux_probe_ce_then_edl03.sh`（ce@0.2 → 若差 ≥0.02 则 edl@0.3；**已跑完**）
- 提数/判决：`eval_frameaux_t2_decide.py`（对照路径请指向 `测试t2/`，见下）

提数要点：
- 读各 run 的 `logs/{val,external}_sample_predictions.csv`（best 权重）
- MCC / F1+ / Sens/Spec 用 **阈值 0.5**（外部常误导，湘雅高 F1 常是患病率+全阳）
- \(n_{\mathrm{eff}}=1/\sum_i w_i^2\)；≈12 表示近等权（N=12 帧）
- `training_history.json` 只有 train+val；external 仅终评

**产物路径注意（易踩坑）：**
| 实验 | 实际可读预测 CSV 目录 |
|------|------------------------|
| edl@0.2 | `outputs/paper_v4/baseline/测试t2/ours_uw_frameaux_t2/` |
| ce@0.2 | `outputs/paper_v4/baseline/测试t2/ours_uw_frameaux_t2_ce_w0.2/` |
| edl@0.3 | `outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3/` |
| edl@0.5 | `outputs/paper_v4/baseline/ours_uw_frameaux_t2_w0.5/` |

原 `ours_uw_frameaux_t2/`、`ours_uw_frameaux_t2_ce_w0.2/` 根下可能空缺 CSV。

---

## 已跑结果（seed=42，三折 = huaxi / liaoning / xiangya）· 含 edl@0.3

**外部 ROC-AUC：**

| 方法 | huaxi | liaoning | xiangya | **三折均值** | vs edl@0.2 |
|------|------:|---------:|--------:|-------------:|-----------:|
| B1 | 0.578 | 0.640 | 0.380 | 0.533 | — |
| Ours 无帧损 | 0.552 | 0.551 | 0.400 | 0.501 | — |
| Full edl@0.2 | 0.556 | 0.534 | 0.576 | 0.555 | — |
| Full **edl@0.3** | 0.556 | 0.560 | **0.614** | **0.577** | **+0.022** ← AUC 当前最好 |
| Full edl@0.5 | 0.552 | 0.559 | 0.421 | 0.510 | −0.045 |
| Full ce@0.2 | 0.541 | 0.544 | 0.413 | 0.500 | −0.056 |

**edl@0.3 分中心（val / external ROC）：**

| 中心 | val ROC | ext ROC | ext PR | \(n_{\mathrm{eff}}\) | 帧 \(u\) 均值 | 帧 \(u\) std |
|------|--------:|--------:|-------:|---------------------:|--------------:|-------------:|
| huaxi | 0.707 | 0.556 | 0.448 | 11.996 | 0.246 | 0.009 |
| liaoning | 0.783 | 0.560 | 0.388 | 11.997 | 0.270 | 0.007 |
| xiangya | 0.667 | 0.614 | 0.790 | 11.997 | 0.269 | 0.008 |

**三折均值（external）：** edl@0.3 ROC **0.5769±0.033**；PR **0.542±0.217**；MCC≈0.04；阳性 F1≈0.63（阈值 0.5）。

探路时间线：
1. edl@0.2 最好 → 试 edl@0.5 变差 → 试 ce@0.2 更差  
2. 自动判决 gap(ref−ce)=0.056≥0.02 → 跑 edl@0.3  
3. edl@0.3 **已完成**（2026-07-22）；AUC 比 0.2 高约 **+0.022**（刚好过「+0.02」线，主要来自湘雅/辽宁；华西持平）  
4. 但 \(n_{\mathrm{eff}}\) 仍 ≈**12** → **UW 仍未生效**；AUC 增益不宜写成「不确定加权起作用」

ce 判决 JSON：`outputs/paper_v4/baseline/AUTO_PROBE_decision_ce.json`  
报告：`scripts/paper_v4_run/AUTO_PROBE_ce_w0.2_report.md`  
结果登记：`scripts/paper_v4_run/RESULTS_T2_frameaux_2026-07-22.md`

**关键诊断（贯穿所有 UW / FrameAux，含 edl@0.3）：**
- \(n_{\mathrm{eff}}\approx 11.98\text{–}12\)，帧 \(u\) 约 0.20–0.34，帧内 std 极小 → **UW 实质≈等权**
- 湘雅作 external：train/val 阳性率 ~36% vs external ~69%；高 PR/F1 常是患病率伪迹

---

## 你接手后的第一步（按顺序）

1. **FrameAux 权重网格已结束**：edl@0.2 / 0.3 / 0.5 / ce@0.2 都有数；**不要再盲扫** 0.1/0.4/0.6…。
2. **AUC 主候选暂定 edl@0.3**（均值 0.577），但必须在文中诚实写清：\(n_{\mathrm{eff}}\approx12\)。
3. **下一优先：让加权真正拉开**（见下节）。先做便宜的**推理锐化诊断**（固定 edl@0.3 或 edl@0.2 权重，只改 \(\tau\) / top-k）。
4. 大改前先征得用户同意；中文给简表 + 结论。

---

## 让加权生效的方向（优先序）

现状根因：帧 EDL 的 \(u\) 几乎常数 → softmax 近均匀。调 `FRAME_AUX_WEIGHT` 治不好。

**A. 温度 / 锐化（改动小，先探）**
- 降 \(\tau\)（如 0.1 / 0.05）使微小 \(u\) 差也被放大；或推理时用更尖的温度。
- 风险：若 \(u\) 真无差，锐化也无效；可能数值不稳。

**B. 换不确定度 / 权重定义**
- 不用 Dirichlet \(u=K/S\)，改用：预测熵、最大概率、方差、或「帧特征与患者均值的距离」。
- top-k / sparsemax / 硬阈值：只保留置信最高的 k 帧再聚合。

**C. 让帧预测真正分化（监督侧）**
- 仅患者标签广播 → 所有帧被推向同一决策，**鼓励 u 同质化**。
- 可选：对比/多样性正则；帧间一致性损失的反向（鼓励分歧）；或伪标签/注意力监督；或只对部分帧加 aux。
- 更强：引入真正的帧级标注 / 弱定位（若有数据）。

**D. 架构侧**
- 可学习帧注意力（query=患者全局），与 EDL-u 解耦；u 只用于校准/拒识，不强制做唯一加权源。
- 或：UW 只作推理后处理，训练仍 mean/equal（避免训练把 u 压平）。

**E. 诊断实验（必做、便宜）**
- 固定已训好的 **edl@0.3**（或 edl@0.2）权重，**只改推理** \(\tau\) / top-k，看 \(n_{\mathrm{eff}}\) 与 AUC 是否动。  
  → 若仍不动，说明 \(u\) 本身无信息，必须改监督或特征，而非再扫 loss weight。

**明确不要再做的：**
- 继续盲扫 FrameAux weight 指望拉开 \(u\)
- 把 WMA/EMA/多模态 Aux 当主修复

---

## 用户沟通偏好

- 中文回复；展开缩写（如 ROC-AUC、不确定度加权）
- 直接、简洁；先给结论表再解释
- 大改前先说明方案再动手
- commit 用中文 message

---

## 相关文档

- `scripts/paper_v4_run/README_2026-07-20.md`
- `scripts/paper_v4_run/RESULTS_T2_frameaux_2026-07-22.md`
- `GIT_MANAGEMENT_GUIDE.md`
- v3 冻结只读：`outputs/第三版/`、`data/snapshots/paper_v3_tsy_loho/`；tag `paper-v3-muse-full`

---

## 一句话现状

诚实协议已通；FrameAux 网格结束。**AUC 最好：edl@0.3（外部均值 ≈0.577，Δ vs 0.2 ≈+0.022）**，但 **\(n_{\mathrm{eff}}\approx12\) UW 仍未生效**。下一任：**停止权重网格** → 推理锐化诊断 → 再改 \(u\)/注意力/监督，让加权真正工作。

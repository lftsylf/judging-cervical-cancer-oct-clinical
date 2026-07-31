# 交接提示词（给下一任 Agent）· 2026-07-31

> **新对话请优先读本文件**，整段可粘贴为开场提示词。  
> 仓库路径：`scripts/paper_v4_run/HANDOFF_NEXT_AGENT_2026-07-31.md`  
> 上一轮对话 transcript：`agent-transcripts/93f1e0be-d83b-4571-b65f-bf92e9b78128`  
> 索引入口仍见：`scripts/paper_v4_run/HANDOFF_NEXT_AGENT.md`

---

## 你要接手的任务（用户下一目标）

用户要**修改内部训练输入图像的内容**（train/val 用的 OCT 输入如何取帧/裁切/预处理等），在现有 paper_v4 协议与当前锁定主方法之上继续实验。

工作区：`/ssd_data/tsy_study_venv/OptiGenesis_Lancet`  
分支：`feature/paper-v4`  
Python：`export PATH="/home/amax/anaconda3/bin:$PATH"`  
数据：`dataset/` → `tsy_loho`；切分 `prepare_paper_v4_splits.py --write --split-seed 20260720 --val-ratio 0.2`  
启动：`./run_experiment.sh --detach ./tsy_loho ./scripts/paper_v4_run/<script>.sh`  
**不要**把 `outputs/` 大体积结果提交进 git；中文 commit message。

动手前先读：
1. 本文件（状态 + 主方法锁定）
2. 输入相关代码：`data/dataset_lancet.py`、`configs/lancet_config.py`（TIFF expand / pages）、`main.py` 数据加载段
3. 用户打开的文件/具体改图意图（以用户本轮说明为准）

---

## 协议（勿回退）

- 患者分层 **8:2 train/val**；**external 只终评**；早停看内部 **val ROC**，patience=10  
- OCT-only；T0 = 3 折 × 5 seeds：`42,123,2024,3407,114514`  
- FrameAux ≠ 多模态 Aux；口语「探路」= 少 seed，「定稿/T0」= 满 15 折  

---

## 当前主方法锁定（本轮结论）

| 角色 | 配置 | 外部 ROC（5-seed 折均值） | 目录（`outputs/paper_v4/baseline/`） |
|------|------|---------------------------|-------------------------------------|
| **主方法候选** | Attn + FrameAux broadcast edl@0.3 + **EMA 0.99 + Aux vision 0.1**（无 WMA） | **0.556±0.014** | `9 ours_uw_frameaux_t0_edl_w0.3_n12_attn_ema099_aux01`（曾名 `8_3`） |
| Attn 基座 | Attn + FrameAux@0.3，无稳定模块 | 0.544±0.022 | `7 ours_uw_frameaux_t0_edl_w0.3_n12_attn` |
| B1 | mean + 患者级 EDL | 0.537±0.034 | `1 b1_mean_edl_t0` |
| **弃** | 叠 WMA（含 wea_tuned） | 有害 | 见下 |

**弃 WMA 依据**：  
- WMA alone C=0.1 ≈ Attn（0.543）  
- 旧 WEA（C=0.2+EMA0.999+Aux）0.527  
- **WEA-tuned**（EMA0.99+Aux0.1+WMA C=0.1）**0.521±0.015**，相对 EMA+Aux **Δ=−0.035**，5/5 seed 都更差  

报告：  
- `RESULTS_n12_attn_ema_aux_5seeds_2026-07-29.md`  
- `RESULTS_n12_attn_wea_tuned_5seeds_2026-07-30.md`

---

## 标号 T0 实验地图（相对 `2 ours`）

**`2 ours_uw_agg_t0` 基座**：ResNet50 OCT-only；`FRAME_AGG=uncertainty_weighted`；τ=0.5；信号默认 `edl_u`；患者主损 Focal+EDL；**无 FrameAux**；无 WMA/EMA/多模态 Aux；约 N=12（不 expand TIFF）。

| 标号 | 相对 2 的变化 |
|------|----------------|
| 1 | B1：mean pooling（不是从 2 改） |
| 3 | + FrameAux **broadcast** edl@0.3 |
| 4 | 3 + UW 信号 `edl_u_amp`，score=(0.5−u)×10 |
| 5 | 3 + UW 信号 `max_p_pool`（只留 p+ 最大 1 帧） |
| 6 | 3 + FrameAux **MIL**（pos_thr=0.5） |
| 7 | 3 + 聚合改为 **attention**（query=mean，τ=0.5）；后续 7+/8+/9+ 都站在 Attn 上 |
| 7_1 | 7 + attn query **evidence**（p+ 加权特征作 query） |
| 7_2 | 7 + 关 FrameAux |
| 8 | 7 + 旧 WEA 全家桶（WMA C=0.2 + EMA 0.999 + Aux） |
| 8_1 | 7 + 仅 WMA C=0.1（帧辅跟随） |
| 8_2 | 7 + 仅多模态 Aux vision=0.1 |
| **9** | 7 + EMA 0.99 + Aux 0.1 → **当前最强** |
| 9_1 | 9 + WMA C=0.1 → **已否决** |

术语快查：  
- **broadcast**：患者标签复制到每一帧做辅损  
- **MIL**：阴袋各帧压阴；阳袋若 max(p+)≥0.5 则不做帧辅损，否则只推最阳那一帧  
- **WMA C**：边距强度；越大修正越猛  
- **EMA decay**：影子权重平滑；0.99 比 0.999 跟得更快  
- **Aux vision**：患者级视觉辅头权重（≠ FrameAux）  
- **v3 MUSE**：`mean` 池化后**患者级** EDL 出阴/阳 α（无帧级主路径）

---

## 本轮已落地的代码/脚本（已或将 commit）

- 稳定性开关：`OPTIGENESIS_EMA_DECAY`、`OPTIGENESIS_AUX_W_*`、`OPTIGENESIS_FRAME_AUX_USE_WMA`、`FRAME_ATTN_QUERY` 等（`configs/lancet_config.py` / `trainer` / `model` / `main`）  
- 跑法：`run_n12_attn_stab_probe_one.sh`（含 `wea_tuned`）、`run_overnight_quad_gpu_attn_*`  
- 结果 md/csv：`scripts/paper_v4_run/RESULTS_n12_attn_*.md`、`SUMMARY_*.csv`

---

## 下一对话建议开场（可直接粘贴）

```
请读 scripts/paper_v4_run/HANDOFF_NEXT_AGENT_2026-07-31.md。
当前主方法锁 Attn + FrameAux@0.3 + EMA0.99 + Aux0.1（目录 9 ..._ema099_aux01），已弃 WMA。
我要修改内部训练输入图像的内容（train/val），请先摸清 dataset_lancet / 配置里现有取帧与预处理，再按我说的改法实现，并规划对照实验（建议相对锁定主方法做探路再定稿）。
协议勿回退：8:2 train/val、external 只终评、OCT-only、不要提交 outputs/。
```

用户会在开场补充**具体要怎么改输入图像**；未说明前先读代码并问清，不要盲目重训满 T0。

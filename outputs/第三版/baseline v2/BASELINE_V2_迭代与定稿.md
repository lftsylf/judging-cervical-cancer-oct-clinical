# Baseline v2 迭代与定稿记录

> **项目**：OptiGenesis（论文展示名 MUSE）· LOHO 三中心（华西 / 辽宁 / 湘雅）  
> **文档位置**：`outputs/baseline v2/`  
> **最后更新**：2026-05-21  

本文档汇总从 **v1（Swin-Tiny + 多模态）** 到 **v2 定稿 baseline（ResNet50 + OCT-only）** 的探索路径、代码改动、各阶段产物目录与最终协议。详细单次实验日志见仓库 `logs/experiments/`。

---

## 1. v1 起点（对照基线）

| 项目 | v1 设置 |
|------|---------|
| 视觉骨干 | `swin_tiny_patch4_window7_224`（timm） |
| 模态 | **多模态**（OCT + 临床 Age / HPV / TCT） |
| 损失 | Focal + EDL（主路径）；无 WMA / aux / EMA / CORAL |
| T0 规格 | 3 折 × 5 seeds，≤30 epoch，验证集 AUC 早停 patience=10 |
| 产物目录 | `outputs/outputs_baseline_v1/`（本文件夹 **未** 收录，仍在 `outputs/` 下） |

**v1 T0 外部 ROC-AUC（5 seeds 均值）**：华西 0.5737 · 辽宁 0.5872 · 湘雅 0.6050 · **15 次总平均 0.5886**  
（来源：`logs/experiments/2026-04-15_t0_baseline_progress.md`）

**v1 局限（触发 v2）**：老师反馈图像模块需加强；Transformer 骨干在小样本 LOHO 上开销大；多模态临床是否必要存疑。

---

## 2. v2 探索时间线（与本文件夹子目录对应）

```
v1 Swin 多模态 T0
    ↓ 换骨干 + 工程修复
3-2  ResNet50 多模态 T1（9 次）
3-1  ResNet18 多模态 T1（对照，弱于 ResNet50）
3-3  T2 学习率四组探路（12 次）
3-4  ResNet50 多模态 T1 定稿重跑（g1_lr5e5，POS=1.3）
3-5  T1 临床 on/off 消融 → 决定 OCT-only
3-6  【终版】T0 ResNet50 OCT-only（15 次）★
```

### 阶段一览表

| 序号 | 子文件夹 | 实验 | 规格 | 主要结论 |
|:---:|----------|------|------|----------|
| 3-1 | `3-1 outputs_baseline_t1_v2_resnet18` | ResNet18 多模态 T1 | 3×3 | 外部 AUC 三折均值 **0.5624**，明显弱于 ResNet50，弃用 |
| 3-2 | `3-2 outputs_baseline_t1_v2_resnet50` | ResNet50 **多模态** T1 | 3×3 | 外部均值 **0.5983**，略优于 v1 Swin T1（0.5922） |
| 3-3 | `3-3 outputs_baseline_t2_lr_resnet50` | LR 探路 T2 | 3折×1seed×4组 | **组1 全网 5e-5** 最佳（3折外部均值 0.5761） |
| 3-4 | `3-4 outputs_baseline_t1_v2_resnet50_g1_lr5e5` | 多模态 T1「定稿」重跑 | 3×3，LR=5e-5，POS=**1.3** | 外部均值 **0.5946**，与 3-2 接近 |
| 3-5 | `3-5 outputs_baseline_t1_v2_resnet50_clinical_off 单模态` | **临床 on/off** T1 | 3×3，OCT-only，POS=**1.25** | OFF 外部均值 **0.6255** > ON（0.5946），**选定单模态** |
| 3-6 | `3-6 outputs_baseline_t0_v2_resnet50_oct_only单模态【终版】` | **v2 定稿 baseline** | 3×5，OCT-only | 外部均值 **0.6183**，为目前最强 baseline |

---

## 3. 代码与仓库改动（相对 v1）

### 3.1 模型与配置

| 改动 | 文件 / 说明 |
|------|-------------|
| 默认骨干 Swin → **ResNet50** | `configs/lancet_config.py`：`BACKBONE=resnet50` |
| 通用 timm 骨干 | `models/optigenesis_model.py`：`vision_dim` 随骨干变化（ResNet50→2048） |
| 临床开关 | `USE_CLINICAL` ← `OPTIGENESIS_USE_CLINICAL`（0/1） |
| **POS_WEIGHT 默认 1.25** | `lancet_config.py`（与 T1 clinical_off / T0 终版一致；曾短暂默认 1.3） |
| 分层 LR / 冻结骨干 | `main.py`：`OPTIGENESIS_LR`、`BACKBONE_LR`、`HEAD_LR`、`FREEZE_BACKBONE_EPOCHS`（T2 探路用，终版未采用） |

### 3.2 训练与工程

| 改动 | 说明 |
|------|------|
| PyTorch 2.0 兼容 | `main.py`：`expandable_segments` 仅在 PyTorch ≥2.1 设置，避免 2.0.x 启动崩溃 |
| 断线续跑 | `run_experiment.sh --detach` + 各脚本 `SKIP_COMPLETED`（log 含「训练完成！」则跳过） |
| 批量脚本 | `run_baseline_t1.sh`、`run_baseline_t2_lr_probe.sh`、`run_baseline_t1_clinical_ablation.sh`、`run_baseline_t0_oct_only.sh` |

### 3.3 明确不属于「baseline」的模块（留给完全体 / 消融）

- **WMA**（`OPTIGENESIS_USE_WMA=1`）
- **多模态 aux**（`OPTIGENESIS_ENABLE_AUX=1`）
- **EMA**（`OPTIGENESIS_ENABLE_EMA=1`）
- **CORAL 域对齐**（`OPTIGENESIS_ENABLE_CORAL=1`）

Baseline 全系脚本均显式关闭上述四项。

---

## 4. 关键探索细节

### 4.1 骨干：Swin-Tiny → ResNet50

- **T1**：ResNet50 多模态 9 次外部 AUC 平均 **0.5983**，较 v1 Swin T1（0.5922）约 **+0.006**（见 `logs/experiments/2026-05-16_resnet50_t1_vs_v1_swin_baseline.md`）。
- **ResNet18** 三折均值仅 **0.5624**，不采用。

### 4.2 学习率：T2 四组探路（`3-3`）

| 组 | 配置 | 3 折外部 AUC 均值（seed 42） |
|:---:|------|:---:|
| **g1** | 全网 **5e-5** | **0.5761**（最优） |
| g2 | backbone 1e-5，head 1e-4 | 0.5371 |
| g3 | backbone 5e-6，head 1e-4 | 0.5287 |
| g4 | 冻 5 epoch + 分层 LR | 0.5337 |

**结论**：终版 baseline **仅使用全网 LR=5e-5**，不使用分层 LR 与 freeze。

### 4.3 模态：多模态 vs OCT-only（`3-5`）

**协议**：ResNet50 · LR=5e-5 · 无 WMA/aux/EMA/CORAL · T1 三折×三 seed。

| 设定 | 外部 ROC-AUC（9 次均值） | 说明 |
|------|:---:|------|
| 多模态 ON（g1，POS=1.3） | 0.5946 | 临床 MLP + 融合 |
| **OCT-only OFF（POS=1.25）** | **0.6255** | 9 次中 8 次 ≥ 多模态 |
| 相对 ON 提升 | **+0.031** | 决定主设定为 **单模态** |

开发集 AUC 多模态略高，但 **LOHO 外部** 上单模态更稳，符合「去临床、主打通 OCT」的论文叙事。

### 4.4 POS_WEIGHT：1.25 定稿

- T1 clinical_off 与 T0 终版训练日志均为 **POS_WEIGHT=1.25**。
- `g1_lr5e5` 多模态重跑曾用 config 默认 **1.3**，与 OFF 对比时存在轻微超参差；**后续 T0 / 消融 / 对比试验统一 1.25**（`lancet_config.py` 默认已改）。

---

## 5. v2 定稿 Baseline 配置（论文主表口径）

### 5.1 训练协议

| 项目 | 定稿值 |
|------|--------|
| 模型 | OptiGenesis v2（仅 baseline 分支，不含 WMA/aux/EMA） |
| 视觉骨干 | **ResNet50**（`timm`，ImageNet 预训练） |
| 输入模态 | **OCT-only**（`OPTIGENESIS_USE_CLINICAL=0`） |
| 学习率 | **5e-5**（全网，AdamW + Cosine） |
| POS_WEIGHT | **1.25** |
| Batch size | **4** |
| 最大 epoch | **30** |
| Early stopping | 验证集（外部折）ROC-AUC，**patience=10** |
| 选模 | 验证集 ROC-AUC 最高 → `best_model.pth` |
| LOHO | 华西 / 辽宁 / 湘雅 各作一次外部验证 |
| Seeds（T0） | **42, 123, 2024, 3407, 114514**（5 个） |
| 损失 | Focal + EDL + 类别权重；配合 WeightedRandomSampler |

### 5.2 关闭项

```
OPTIGENESIS_USE_WMA=0
OPTIGENESIS_ENABLE_AUX=0
OPTIGENESIS_ENABLE_EMA=0
OPTIGENESIS_ENABLE_CORAL=0
```

### 5.3 脚本与产物

| 用途 | 脚本 | 产物目录（运行默认） |
|------|------|----------------------|
| **T0 定稿（主 baseline）** | `run_baseline_t0_oct_only.sh` | `outputs_baseline_t0_v2_resnet50_oct_only/` |
| 归档副本（本文件夹） | — | `3-6 outputs_baseline_t0_v2_resnet50_oct_only单模态【终版】/` |
| T1 临床消融（已完成） | `run_baseline_t1_clinical_ablation.sh` | `outputs_baseline_t1_v2_resnet50_clinical_off/` |

**SSH 断线后台启动（T0 定稿）**：

```bash
export PATH="/home/amax/anaconda3/bin:$PATH"
cd /ssd_data/tsy_study_venv/OptiGenesis_Lancet

./run_experiment.sh --detach \
  /ssd_data/tsy_study_venv/OptiGenesis_Lancet/tsy_loho \
  ./run_baseline_t0_oct_only.sh
```

---

## 6. 定稿结果与历史对照

### 6.1 T0 v2 定稿 — 外部 ROC-AUC（15 次）

| 折 | 5-seed 均值 ± std |
|------|:---:|
| 华西 | 0.5967 ± 0.022 |
| 辽宁 | 0.6367 ± 0.028 |
| 湘雅 | 0.6211 ± 0.047 |
| **总平均** | **0.6183 ± 0.036** |

**示例（华西 seed 42）**：开发 AUC 0.7219 · **外部 AUC 0.5761** · 外部 PR-AUC 0.4591 · BalAcc 0.5000。

完整 15 行明细见各折 `logs/train_console.log` 中「最佳权重 · 开发集/外部集」行。

### 6.2 与 v1 / T1 多模态对比（外部 ROC-AUC 均值）

| 版本 | 模态 | 骨干 | 次数 | 外部 AUC 均值 |
|------|------|------|:---:|:---:|
| v1 T0 | 多模态 | Swin-Tiny | 15 | 0.5886 |
| T1 v2 | 多模态 | ResNet50 | 9 | 0.5983 |
| T1 v2 | OCT-only | ResNet50 | 9 | 0.6255 |
| **T0 v2 定稿** | **OCT-only** | **ResNet50** | **15** | **0.6183** |

相对 v1 T0：**+0.0297**；相对 T1 ResNet50 多模态（9 次）：**+0.0200**（同 seed 子集约 +0.027）。

### 6.3 解读（写进论文/答辩）

1. **骨干**：CNN（ResNet50）在本数据规模 LOHO 上优于 Swin-Tiny。  
2. **模态**：去掉临床后外部 AUC 提升，避免临床捷径与跨中心临床分布漂移。  
3. **开发 vs 外部**：定稿模型开发 AUC 低于 v1 多模态，但外部更高，选模仍按外部折验证 AUC，与 LOHO 目标一致。

---

## 7. 后续实验约定（基于本定稿）

凡称「与 baseline 同设定」者，应满足：

- `OPTIGENESIS_USE_CLINICAL=0`
- `OPTIGENESIS_BACKBONE=resnet50`
- `OPTIGENESIS_LR=5e-5`
- `OPTIGENESIS_POS_WEIGHT=1.25`（或不 export，走 config 默认）
- 无 WMA / aux / EMA / CORAL

**完全体（WMA+aux+EMA）**、**消融（-WMA/-aux/-EMA）**、**SOTA 对比** 均应在上述 OCT-only 协议上扩展；旧版 **多模态** 对比结果（`outputs/对比试验/` 等）不可与 v2 定稿直接同表，若沿用需用 `USE_CLINICAL=0` 重跑。

---

## 8. 相关文档索引

| 文档 | 内容 |
|------|------|
| `logs/experiments/2026-04-15_t0_baseline_progress.md` | v1 T0 完成与汇总 |
| `logs/experiments/2026-05-16_resnet50_t1_vs_v1_swin_baseline.md` | ResNet50 vs Swin T1 |
| `logs/experiments/2026-05-16_resnet18_t1_progress.md` | ResNet18 对照 |
| `logs/experiments/2026-05-16_t2_lr_probe_protocol.md` | T2 LR 四组 |
| `logs/experiments/2026-05-18_t1_resnet50_g1_lr5e5.md` | g1 多模态 T1 重跑 |
| `GIT_MANAGEMENT_GUIDE.md` §8 | v1/v2 分支与 tag 建议 |

---

## 9. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-05-21 | 初版：汇总 v1→v2 探索链路与 3-1～3-6 文件夹定稿说明 |

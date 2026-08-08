# baseline2 对比四实验 · 启动清单与接入点

> 输出根目录：`outputs/paper_v4/对比实验/`。ABMIL / UBIX / DSMIL / ConvNeXt 均已接入可跑。  
> 协议：湘雅内训 / 华西+辽宁外测 · sitebag n=2×全页 · 六窗 mean · OCT-only · seeds `42 123 2024 3407 114514`。  
> 禁止与旧 LOHO / Attn 数字横比。

---

## 0. 四卡 detach（断线继续）

```bash
cd /ssd_data/tsy_study_venv/OptiGenesis_Lancet
export PATH="/home/amax/anaconda3/bin:$PATH"

./run_experiment.sh --detach ./tsy_loho ./scripts/paper_v4_run/run_cmp_quad_gpu.sh

tail -f logs/detached_latest.log
tail -f outputs/paper_v4/对比实验/logs/gpu0_abmil.log
# gpu1_ubix / gpu2_dsmil / gpu3_convnext
```

| GPU | 方法 | 目录 |
|-----|------|------|
| 0 | ABMIL | `outputs/paper_v4/对比实验/10[对比] abmil_…/` |
| 1 | UBIX | `…/11[对比] ubix_…/` |
| 2 | DSMIL | `…/12[对比] dsmil_…/` |
| 3 | Ours+ConvNeXt-Tiny | `…/13[变体] final_…_convnext_tiny/` |

---

## 1. 公共环境变量（`cmp_env_common.inc.sh`）

| 变量 | 默认 | 含义 |
|------|------|------|
| `HOSPITAL_NAME` | `xyi` | 用 `train_xyi` / `val_xyi` / external |
| `OPTIGENESIS_USE_CLINICAL` | `0` | OCT-only |
| `OPTIGENESIS_LR` | `5e-5` | 全网 LR |
| `OPTIGENESIS_POS_WEIGHT` | `1.25` | |
| `OPTIGENESIS_EPOCHS` | `30` | |
| `OPTIGENESIS_SITEBAG` | `1` | |
| `OPTIGENESIS_SITEBAG_N` | `2` | |
| `OPTIGENESIS_SITEBAG_EVAL_AGG` | `mean` | 六窗平均 |
| `OPTIGENESIS_BATCH_SIZE` | `2` | |
| `OPTIGENESIS_USE_WMA` | `0` | |
| `OPTIGENESIS_ENABLE_AUX` | `0` | 多模态 Aux |
| `OPTIGENESIS_ENABLE_CORAL` | `0` | |
| `SEEDS` | 五粒 | 可覆盖 |
| `SKIP_COMPLETED` | `1` | 已有「训练完成！」则跳过 |
| `FORCE_RERUN` | `0` | `1` 强制重跑 |

---

## 2. 各方法环境变量 + 接入点

### 2.1 ABMIL — `run_cmp_abmil_one.sh`

| 变量 | 建议值 | 说明 |
|------|--------|------|
| `OPTIGENESIS_BACKBONE` | `resnet50` | |
| `OPTIGENESIS_FRAME_AGG` | `abmil`（正式）/ `attention`（INTERIM） | |
| `OPTIGENESIS_ABMIL_GATE` | `1` | gated attention（Ilse） |
| `OPTIGENESIS_ABMIL_ATTN_DIM` | `128` | |
| `OPTIGENESIS_ENABLE_FRAME_AUX` | `0` | 经典 ABMIL 无实例辅损 |
| `OPTIGENESIS_ENABLE_EMA` | `0` | 默认可与 4 不对齐 EMA；若要对齐改 `1` |
| `OPTIGENESIS_CMP_ABMIL_INTERIM` | `0` | `1`→现有 `attention` 冒烟 |

**代码接入点**

1. `models/optigenesis_model.py`
   - `AGG_MODES` 增加 `"abmil"`
   - `_aggregate_frame_alphas`：实现  
     \(a_i=\mathrm{softmax}(w^\top(\tanh(V h_i)\odot\mathrm{sig}(U h_i)))\)，对帧特征加权后再进分类头  
   - **不要**用 EDL-u 当注意力；可保留帧 EDL 仅作可选，或改为 CE 头（文中写清）
2. `configs/lancet_config.py`：读 `OPTIGENESIS_ABMIL_*`（可选）
3. 参考：现有 `FRAME_AGG=attention`（约 L115–227）是 Q-K 点积，**≠** ABMIL gated；论文表勿混用。

**参考实现**：Ilse ABMIL；或 `mahmoodlab/MIL-Lab` 的 `abmil.py`（只抄注意力头）。

---

### 2.2 UBIX — `run_cmp_ubix_one.sh`

| 变量 | 建议值 | 说明 |
|------|--------|------|
| `OPTIGENESIS_FRAME_AGG` | `equal` | 训练等权；UBIX 在**推理**降权 |
| `OPTIGENESIS_UBIX_ENABLE` | `1` | |
| `OPTIGENESIS_UBIX_MODE` | `soft` | `soft` / `hard` |
| `OPTIGENESIS_UBIX_U_SOURCE` | `mc_dropout` | **禁止 `edl_u`** |
| `OPTIGENESIS_UBIX_MC_T` | `16` | MC 次数 |
| `OPTIGENESIS_UBIX_DROPOUT_P` | `0.2` | |
| `OPTIGENESIS_ENABLE_FRAME_AUX` | `0` | |
| `OPTIGENESIS_ENABLE_EMA` | `0` | |

**代码接入点**

1. 新建 `models/cmp_ubix.py`
   - `estimate_instance_uncertainty(model, frames, T=16)` → MC Dropout 方差/熵  
   - `ubix_reweight(frame_logits_or_probs, u, mode=soft|hard)` → 池化前权重  
2. `training/trainer.py` + `scripts/predict.py`（或 eval 路径）  
   - **仅 val/test**：`UBIX_ENABLE=1` 时替换等权聚合  
   - 训练 forward **不走** UBIX（与原文 inference-time 一致）
3. Dropout：fusion / 分类头在 MC 时 `train()` 模式，骨干可 `eval()`  
4. 脚本门闩：存在 `models.cmp_ubix` 才允许开跑

**参考仓**：[qurAI-amsterdam/ubix-for-reliable-classification](https://github.com/qurAI-amsterdam/ubix-for-reliable-classification)

---

### 2.3 DSMIL — `run_cmp_dsmil_one.sh`

| 变量 | 建议值 | 说明 |
|------|--------|------|
| `OPTIGENESIS_FRAME_AGG` | `dsmil` | |
| `OPTIGENESIS_DSMIL_ATTN_DIM` | `128` | |
| `OPTIGENESIS_DSMIL_DROPOUT` | `0.25` | |
| `OPTIGENESIS_DSMIL_FUSE` | `mean` | 双流融合方式 |
| `OPTIGENESIS_ENABLE_FRAME_AUX` | `0` | |
| `OPTIGENESIS_ENABLE_EMA` | `0` | |

**代码接入点**

1. 新建 `models/cmp_dsmil.py`：`DSMILAggregator(in_dim=256, …)`  
   - 流 A：实例分类器 → max 关键实例  
   - 流 B：与关键实例的注意力相关聚合  
   - 输出 bag logit / 概率  
2. `models/optigenesis_model.py`：`AGG_MODES += ("dsmil",)`，在聚合分支调用上述模块  
3. 损失：患者级 CE（或保留现有 EDL 主损二选一，**文中写死一种**；建议 CE 更贴近原 DSMIL）

**参考仓**：[binli123/dsmil-wsi](https://github.com/binli123/dsmil-wsi) / MIL-Lab `dsmil.py`

---

### 2.4 Ours + ConvNeXt — `run_cmp_ours_convnext_one.sh` ✅ 可跑

| 变量 | 建议值 | 说明 |
|------|--------|------|
| `OPTIGENESIS_BACKBONE` | `convnext_tiny` | 可改 `convnext_small` |
| `OPTIGENESIS_FRAME_AGG` | `uncertainty_weighted` | |
| `OPTIGENESIS_FRAME_WEIGHT_SIGNAL` | `edl_u` | |
| `OPTIGENESIS_FRAME_AGG_TEMP` | `0.5` | |
| `OPTIGENESIS_ENABLE_FRAME_AUX` | `1` | MIL@0.3 |
| `OPTIGENESIS_FRAME_AUX_WEIGHT` | `0.3` | |
| `OPTIGENESIS_ENABLE_EMA` | `1` | `0.99` |
| `OPTIGENESIS_BATCH_SIZE` | `2` | OOM → `1` |
| `OPTIGENESIS_FRAME_ENCODE_CHUNK` | `8` | 默认已压 |

无需新模块；`timm.create_model` 已通。

---

## 3. 公平性检查清单（跑前打勾）

- [ ] 同一 `dataset/train_xyi.csv` 等（`split-seed=20260731`）
- [ ] 对比方法骨干均为 ResNet50；仅 #4 换 ConvNeXt
- [ ] 输入均为 sitebag n=2×全帧 + 六窗 mean（若 UBIX 改整袋，文中单列说明）
- [ ] UBIX 不确定度 = MC Dropout，**不是** EDL-u
- [ ] ABMIL 正式结果不用 `INTERIM=attention`
- [ ] 主报 Ext pooled ROC；附表 Hx / Ln / Val
- [ ] 勿把 #4 写进「他人方法」表；写「骨干变体 / Method-B」

---

## 4. 建议实现顺序（再开四机）

1. **先开 #4 ConvNeXt**（当晚出数）  
2. 实现 **ABMIL gated** → 开 #1（或先 `INTERIM` 冒烟流水线）  
3. 实现 **DSMIL** → 开 #3  
4. 实现 **UBIX**（推理路径）→ 开 #2（通常最费调试）

---

## 5. 相关文件

| 文件 | 作用 |
|------|------|
| `cmp_env_common.inc.sh` | 公共 export |
| `run_cmp_{abmil,ubix,dsmil,ours_convnext}_one.sh` | 四入口 |
| `PAPER_EXPERIMENT_BRIEF_for_Gemini.md` | 协议与 Ours 数字 |
| `HANDOFF_COMPARISON_BASELINE2.md` | 对比实验交接 |
| `models/optigenesis_model.py` | 聚合主战场 |
| `configs/lancet_config.py` | 环境变量入口 |

# 新对话提示词：编写 v2 OCT-only T0 对比试验脚本

> 用法：复制下方「复制用提示词」整段，粘贴到新的 Cursor 对话中执行。

---

## 复制用提示词

```
我在项目 /ssd_data/tsy_study_venv/OptiGenesis_Lancet（OptiGenesis / 论文 MUSE）上要做 **v2 单模态（OCT-only）T0 对比试验（SOTA backbone 对比）**。请按仓库现有约定编写/改写脚本，支持 `run_experiment.sh --detach` 与 `SKIP_COMPLETED=1`。

---

### 一、背景（必读）

**v2 定稿 baseline 协议**（所有对比必须与该协议对齐，仅改 `OPTIGENESIS_BACKBONE`）：
- `OPTIGENESIS_USE_CLINICAL=0`（OCT-only，不用临床）
- `OPTIGENESIS_BACKBONE=resnet50`（主方法定稿骨干）
- `OPTIGENESIS_LR=5e-5`，`OPTIGENESIS_POS_WEIGHT=1.25`
- `OPTIGENESIS_BATCH_SIZE=4`
- 关闭：`OPTIGENESIS_USE_WMA=0`，`OPTIGENESIS_ENABLE_EMA=0`，`OPTIGENESIS_ENABLE_AUX=0`，`OPTIGENESIS_ENABLE_CORAL=0`
- T0：3 折（huaxi / liaoning / xiangya）× 5 seeds（42, 123, 2024, 3407, 114514）= 15 次
- ≤30 epoch，验证集 ROC-AUC early stopping patience=10
- 模板脚本：`run_baseline_t0_oct_only.sh`；消融共享库：`run_ablation_t0_oct_only_lib.sh`（含 SKIP_COMPLETED、tee 日志、is_run_complete）

**主方法（论文 OptiGenesis Full）** 不在本批对比脚本里重复训练，已有：
- `run_optigenesis_full_t0_oct_only.sh` → `outputs/消融实验v2单模态/outputs_ablation_v2_oct_full/`（WMA+EMA+Aux，外部 AUC ≈0.645）
- 定稿 plain ResNet50 baseline：`run_baseline_t0_oct_only.sh` → `outputs_baseline_t0_v2_resnet50_oct_only/`（外部 AUC ≈0.618）

**旧版对比试验（多模态，不可直接用于 v2 论文表）**：
- 脚本：`run_comparison_{swin_small,convnext,vit_small,vit_base}_t0.sh`
- 问题：`USE_CLINICAL` 默认 1（多模态）、`BATCH_SIZE=2`、无 SKIP_COMPLETED
- 旧结果目录：`outputs_comparison_swin_small/`、`outputs_comparison_convnext_small/`、`outputs_comparison_vit_small/`、`outputs_comparison_vit_base/`，归档在 `outputs/对比试验/`
- 旧多模态外部 AUC 约：Swin-Small 0.598、ConvNeXt-Small 0.582（见 `outputs/对比试验/comparison_recent_sota_paper_table_external.md`）

**v1 曾被 ResNet50 替换的骨干**：
- `swin_tiny_patch4_window7_224`（v1 多模态主骨干，tag `paper-v1-swin-baseline`）
- 对比试验里还用过 **`swin_small_patch4_window7_224`**（旧 SOTA 对比），请 **两种 Swin 都纳入 v2 OCT-only 重跑**（tiny=历史主骨干；small=此前对比表里的 Swin）

---

### 二、任务：编写 v2 OCT-only 对比脚本

请参照 `run_ablation_t0_oct_only_lib.sh` + `run_baseline_t0_oct_only.sh`，实现：

1. **共享库** `run_comparison_t0_oct_only_lib.sh`  
   - 环境变量：`COMPARISON_TAG`（如 `swin_tiny` / `swin_small` / `convnext_small` / `vit_small` / `vit_base`）、`COMPARISON_BACKBONE`（timm 名）  
   - 输出根：`outputs/对比试验v2单模态/outputs_comparison_v2_oct_${COMPARISON_TAG}/`（或你建议的等价命名，需在 README 写清）  
   - 其余超参与 baseline 一致；`SKIP_COMPLETED` / `FORCE_RERUN` / `OPTIGENESIS_PYTHON` 与消融脚本一致  

2. **每个 backbone 一个入口脚本**（共 5 个，均为 **plain Focal+EDL**，无 WMA/EMA/Aux）：

   | 脚本建议名 | timm BACKBONE | 说明 |
   |------------|---------------|------|
   | `run_comparison_swin_tiny_t0_oct_only.sh` | `swin_tiny_patch4_window7_224` | v1 主骨干，论文历史对照 |
   | `run_comparison_swin_small_t0_oct_only.sh` | `swin_small_patch4_window7_224` | 旧对比试验 Swin，需重跑 |
   | `run_comparison_convnext_small_t0_oct_only.sh` | `convnext_small` | 旧对比已有 |
   | `run_comparison_vit_small_t0_oct_only.sh` | `vit_small_patch16_224` | 旧对比已有 |
   | `run_comparison_vit_base_t0_oct_only.sh` | `vit_base_patch16_224` | 可选，旧脚本有 |

3. **串行总入口** `run_comparison_t0_oct_only_all.sh`（5 个 backbone 依次跑）  

4. **旧脚本处理**：`run_comparison_*_t0.sh`（无 oct_only）改为打印废弃说明并 `exec` 到新脚本，或头部注释指向新脚本（与 `run_ablation_no_wma_t0.sh` 重定向方式一致）  

5. **OOM 策略**：默认 `BATCH_SIZE=4`；若 ViT-Base / Swin 显存不足，允许该模型单独 `BATCH_SIZE=2` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128`，在脚本注释和 README 标明，保证 **除 batch 外协议一致**。  

6. **Detach 示例**（写进每个脚本头部注释）：

   ```bash
   export PATH="/home/amax/anaconda3/bin:$PATH"
   cd /ssd_data/tsy_study_venv/OptiGenesis_Lancet
   ./run_experiment.sh --detach \
     /ssd_data/tsy_study_venv/OptiGenesis_Lancet/tsy_loho \
     ./run_comparison_t0_oct_only_all.sh
   tail -f logs/detached_latest.log
   ```

7. **后处理（脚本或 README 说明，可顺带改）**  
   - 扩展 `scripts/analyze_comparison_optimal_thresholds.py`，支持新输出目录，生成 `*_summary.csv`（Youden 阈值，与旧 ViT 表格式一致）  
   - 主指标：外部 ROC-AUC（log「最佳权重 · 外部集」）；分类指标勿仅用 BalAcc@0.5  
   - DeLong：`scripts/calculate_all_delong_pvalues.py` 需能指向 v2 目录（Full / baseline / 各对比骨干）  

8. **文档** `outputs/对比试验v2单模态/README.md`：  
   - 对比矩阵表（backbone × 协议）  
   - 与旧 `outputs/对比试验/` 多模态结果 **不可混表** 的说明  
   - 论文表建议行：ResNet50 plain baseline、各 SOTA backbone（OCT-only plain）、**OptiGenesis Full（ResNet50+WMA+EMA+Aux）** 引用已有消融路径  

---

### 三、验收标准

- `bash -n` 通过；`chmod +x`  
- 每个 backbone 15 次，日志含「训练完成！」可被 SKIP_COMPLETED 识别  
- 不与消融/Full 训练逻辑重复造轮子，最大化复用 `run_ablation_t0_oct_only_lib.sh` 结构  
- 不要改 `configs/lancet_config.py` 默认值，仅用 export 控制  

请先给出文件清单与目录结构，再实现脚本和 README；若认为 vit_base 显存过大可标为可选并单独 detach 跑。
```

---

## 本地索引

| 参考 | 路径 |
|------|------|
| v2 定稿 baseline 脚本 | `run_baseline_t0_oct_only.sh` |
| 消融共享库 | `run_ablation_t0_oct_only_lib.sh` |
| 旧多模态对比脚本 | `run_comparison_*_t0.sh` |
| 旧对比结果 | `outputs/对比试验/` |
| 消融终版分析 | `outputs/消融实验v2单模态/ablation_final_analysis.md` |
| 阈值后处理 | `scripts/analyze_comparison_optimal_thresholds.py` |
| DeLong | `scripts/calculate_all_delong_pvalues.py` |

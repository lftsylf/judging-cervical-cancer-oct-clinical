# TIFF 全时序展开（多页）· 使用说明

## 背景

- 每个点位是一个多页 TIFF：**辽宁常见 5 页**，**华西/湘雅常见 10 页**。
- 旧 loader 用 `PIL.open().convert('RGB')`，**只读第 1 页** → 每患者约 **12 帧**。
- 病灶有时只在少数时序页可见；老师要求用满时序。

## 开关（默认关闭，旧实验可比）

```bash
export OPTIGENESIS_EXPAND_TIFF_PAGES=1
export OPTIGENESIS_BATCH_SIZE=4          # 8GB+chunk 可跑；OOM 再降到 2
# 3 折并行（GPU 0/1/2）：脚本默认 OPTIGENESIS_PARALLEL=1
# export OPTIGENESIS_GPUS="0 1 2 3"     # 要用满 4 卡时改这里（第 4 个槽给多 seed）
# 可选：每 TIFF 最多取前 K 页（0=不截断）
export OPTIGENESIS_MAX_PAGES_PER_TIFF=0
```

展开后每患者帧数约：

| 中心 | 点位 TIFF | 页/ TIFF | 展开后 N |
|------|----------:|---------:|---------:|
| 辽宁 | 12 | 5 | **60** |
| 华西 / 湘雅 | 12 | 10 | **120** |

同一 LOHO 折的 train 常混有辽宁与华西/湘雅 → batch 内 N 不同，loader 会 **pad + `frame_mask`**，聚合/帧辅损只计有效帧。

### 页数不同会不会「弄坏」训练？

**不会因为 pad 泄漏或污染。** padding 全黑帧被 `frame_mask=0` 屏蔽，不进 softmax 聚合、不进帧辅损。

**会有的真实差异（不是 bug）**：辽宁有效帧≈60、华西/湘雅≈120。等权时更长序列更易稀释稀疏病灶；UW / `edl_u_amp` 的目标正是按 u 聚焦少数帧，减轻「帧数多=稀释重」。跨中心页数不同是采集协议差异，应用 mask 后算法上公平（每人都在自己的有效帧上归一化权重）。

## 启动探路

仅展开页（旧 edl_u）：

```bash
./run_experiment.sh --detach ./tsy_loho \
  ./scripts/paper_v4_run/run_expand_tiff_pages_t2_oct_only.sh
```

**展开页 + edl@0.3 + 放大 u 差（推荐本次）**：

```bash
./run_experiment.sh --detach ./tsy_loho \
  ./scripts/paper_v4_run/run_expand_pages_edl03_uamp_t2_oct_only.sh
```

输出：`outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3_expand_uamp10/`

### 倍率选 10 的原因（`edl_u_amp`）

\(\mathrm{score}=(u_{\mathrm{base}}-u)\cdot\mathrm{scale}\)，再 `softmax(score/τ)`，τ=0.5。

- 观测 \(u\) 多在 **0.20–0.33**；旧式 `(1−u)/τ` 下 Δu=0.05 → Δlogit≈0.1 → 近等权。
- **u_base=0.5**：相对该区间上沿，`(0.5−u)` 多为正；0.4 对 u≈0.33 过狠。
- **scale=10**：Δu=0.05 → Δscore=0.5 → `/0.5` → Δlogit≈**1**，权重比约 \(e\approx2.7\)，能拉开；scale 20–40 更接近硬选 top 帧。

环境变量：`OPTIGENESIS_FRAME_U_SCORE_BASE`、`OPTIGENESIS_FRAME_U_SCORE_SCALE`。

## 仍保持的协议

- 划分仍是 **患者整袋**：不会把同一患者的点位/时序页拆到 train 与 test。
- 标签仍是患者级；帧辅损仍是患者标签广播到有效帧。

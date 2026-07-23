# TIFF 全时序展开（多页）· 使用说明

## 背景

- 每个点位是一个多页 TIFF：**辽宁常见 5 页**，**华西/湘雅常见 10 页**。
- 旧 loader 用 `PIL.open().convert('RGB')`，**只读第 1 页** → 每患者约 **12 帧**。
- 病灶有时只在少数时序页可见；老师要求用满时序。

## 开关（默认关闭，旧实验可比）

```bash
export OPTIGENESIS_EXPAND_TIFF_PAGES=1
export OPTIGENESIS_BATCH_SIZE=1          # 强烈建议；N=60~120 易 OOM
# 可选：每 TIFF 最多取前 K 页（0=不截断）
export OPTIGENESIS_MAX_PAGES_PER_TIFF=0
```

展开后每患者帧数约：

| 中心 | 点位 TIFF | 页/ TIFF | 展开后 N |
|------|----------:|---------:|---------:|
| 辽宁 | 12 | 5 | **60** |
| 华西 / 湘雅 | 12 | 10 | **120** |

同一 LOHO 折的 train 常混有辽宁与华西/湘雅 → batch 内 N 不同，loader 会 **pad + `frame_mask`**，聚合/帧辅损只计有效帧。

## 启动探路

```bash
./run_experiment.sh --detach ./tsy_loho \
  ./scripts/paper_v4_run/run_expand_tiff_pages_t2_oct_only.sh
```

默认对齐 edl@0.3 帧辅损设定，输出目录：
`outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3_expand_pages/`

## 仍保持的协议

- 划分仍是 **患者整袋**：不会把同一患者的点位/时序页拆到 train 与 test。
- 标签仍是患者级；帧辅损仍是患者标签广播到有效帧。

# Attn 改进四卡探路（2026-07-26）

## Seeds
**42, 123, 3407** — 旧 attn 所有三 seed 组合中折均值最高（0.559），且略优于同子集 B1（0.554）。

## 四卡

| GPU | 代号 | 改动 | 输出 |
|----:|------|------|------|
| 0 | A | `FRAME_ATTN_QUERY=evidence`（p+ 加权特征作 query） | `ours_uw_frameaux_t2_edl_w0.3_n12_attn_eq/` |
| 1 | B | τ=0.2 | `ours_uw_frameaux_t2_edl_w0.3_n12_attn_tau02/` |
| 2 | C | 关闭 FrameAux | `ours_uw_frameaux_t2_edl_w0.3_n12_attn_noaux/` |
| 3 | D | 患者 LS ε=0.05 | `ours_uw_frameaux_t2_edl_w0.3_n12_attn_ls05/` |

其余与旧 attn 定稿一致：N=12、attention、broadcast@0.3（C 除外）、OCT-only、无 WMA/EMA。

## 启动（可断联）
```bash
./run_experiment.sh --detach ./tsy_loho \
  ./scripts/paper_v4_run/run_overnight_quad_gpu_attn_probe4.sh
```

## 看进度
```bash
tail -f logs/detached_latest.log
cat outputs/paper_v4/baseline/OVERNIGHT_ATTN_PROBE4_STATUS.md
tail -f logs/overnight_attn_probe_eq.log
```

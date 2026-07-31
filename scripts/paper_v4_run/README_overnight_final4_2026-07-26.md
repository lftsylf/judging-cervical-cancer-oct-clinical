# Paper v4 · 四卡定稿过夜（2026-07-26 启动）

> 背景：Full（broadcast+edl_u）定稿 5-seed 外部仅 **0.522±0.030**，探路 seed=42（0.577）偏乐观。  
> 故对探路曾否定的四条线做 **定稿规模（最多 5 seeds × 3 折）** 复核。

## 队列

| GPU | 代号 | 设定 | 输出 |
|----:|------|------|------|
| 0 | amp | N=12 + `edl_u_amp`×10 + FrameAux broadcast@0.3 | `ours_uw_frameaux_t2_edl_w0.3_n12_uamp10/` |
| 1 | maxp | N=12 + `max_p_pool` + broadcast@0.3 | `ours_uw_frameaux_t2_edl_w0.3_n12_maxp/` |
| 2 | mil | N=12 + `edl_u` + FrameAux **MIL**@0.3 | `ours_uw_frameaux_t2_edl_w0.3_n12_mil/` |
| 3 | attn | N=12 + **可学习 attention** + broadcast@0.3 | `ours_uw_frameaux_t2_edl_w0.3_n12_attn/` |

- Seeds：`42,123,2024,3407,114514`；①②③ 若 seed_42 已齐则自动只补后 4 个。  
- 脚本：`run_overnight_quad_gpu_final4methods.sh`  
- 新代码：`FRAME_AGG=attention`（query=患者全局融合特征，key=各帧；EDL \(u\) 仅导出）

## 看进度

```bash
tail -f logs/detached_latest.log
tail -f logs/overnight_final_amp.log   # /maxp /mil /attn
cat outputs/paper_v4/baseline/OVERNIGHT_FINAL4_STATUS.md
nvidia-smi
```

跑完后提数对照：`ours_uw_frameaux_t2_edl_w0.3/` 与 `b1_mean_edl_t0/` 的 5-seed 外部均值。

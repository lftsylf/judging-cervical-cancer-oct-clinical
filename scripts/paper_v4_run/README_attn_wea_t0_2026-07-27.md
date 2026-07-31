# Attn + WMA/EMA/Aux · T0（2026-07-27）

## 方法
旧 Attn 定稿设定 + 三稳定性模块：
- `FRAME_AGG=attention`，query=`mean`，τ=0.5
- FrameAux broadcast edl@0.3
- **WMA=1**（C=0.2, warmup=10）
- **EMA=1**
- **AUX=1**（OCT-only：视觉辅头；临床辅头因 `USE_CLINICAL=0` 自动跳过）
- Seeds：42, 123, 2024, 3407, 114514

输出：`outputs/paper_v4/baseline/ours_uw_frameaux_t0_edl_w0.3_n12_attn_wea/`

## 四卡
| GPU | seeds |
|----:|-------|
| 0 | 42 |
| 1 | 123 |
| 2 | 2024 |
| 3 | 3407 → 114514 |

## 启动
```bash
./run_experiment.sh --detach ./tsy_loho \
  ./scripts/paper_v4_run/run_overnight_quad_gpu_attn_wea_t0.sh
```

## 看进度
```bash
tail -f logs/detached_latest.log
cat outputs/paper_v4/baseline/OVERNIGHT_ATTN_WEA_T0_STATUS.md
tail -f logs/overnight_attn_wea_s42.log
```

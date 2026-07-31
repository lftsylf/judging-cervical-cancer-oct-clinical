# Attn + WEA-tuned T0（EMA0.99 + Aux0.1 + WMA C=0.1）

## 为何再跑

- 旧 WEA 全家桶（默认较强）外部 **0.527**，低于 Attn。
- 降强度后：EMA+Aux **0.556** 已超过 Attn；WMA alone **0.543 ≈ Attn**。
- 最后一刀：把调试后的 WMA（C=0.1）叠到 EMA+Aux 上，看能否再涨。

## 判定

| 结果 | 动作 |
|------|------|
| wea_tuned > EMA+Aux（约 +0.005+） | 可考虑升主方法 |
| ≈ EMA+Aux（|Δ|≲0.005） | **弃 WMA**，主方法锁 EMA+Aux |
| < EMA+Aux | 弃 WMA；确认叠 WMA 有害 |

## 启动

```bash
./run_experiment.sh --detach ./tsy_loho \
  ./scripts/paper_v4_run/run_overnight_quad_gpu_attn_wea_tuned_t0.sh
```

输出：`outputs/paper_v4/baseline/ours_uw_frameaux_t0_edl_w0.3_n12_attn_wea_tuned/`

# Attn 稳定性模块分开探路（降强度 · 2 seeds）

## 要不要分开？要
全家桶 WEA 已失败（0.527 vs 旧 Attn 0.544），分开才能归因。

## 要不要改配置？要
每个模块相对 v3 默认**降强度**后再探：

| 模块 | 旧默认 | 本探路 |
|------|--------|--------|
| WMA | C=0.2 | **C=0.1** |
| EMA | decay=0.999 | **0.99** |
| Aux | vision=0.2 | **0.1** |
| 第 4 刀 | — | **wma_pat**：患者 WMA，帧辅不跟 WMA |

## Seeds：**42 + 123**
- **42**：旧 Attn 最强（0.574），WEA 也曾涨 → 看单模块能否保住强 seed
- **123**：旧 Attn 中等（0.542），WEA 崩到 0.449 → 看降强度后还会不会崩

同子集旧 Attn 均值 ≈ **0.558**（(0.574+0.542)/2），作门槛。

## 四卡
| GPU | 方法 |
|----:|------|
| 0 | wma |
| 1 | ema |
| 2 | aux |
| 3 | wma_pat |

## 启动
```bash
./run_experiment.sh --detach ./tsy_loho \
  ./scripts/paper_v4_run/run_overnight_quad_gpu_attn_stab_probe2.sh
```
